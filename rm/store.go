package rm

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"sync"
	"time"
)

// Memory is one thing an agent remembers.
type Memory struct {
	ID          uint32    `json:"id"`
	Text        string    `json:"text"`
	Tags        []string  `json:"tags,omitempty"`
	Importance  float64   `json:"importance"` // 0..1, set at write time
	Created     time.Time `json:"created"`
	LastAccess  time.Time `json:"last_access"`
	AccessCount int       `json:"access_count"`
	Vec         []float32 `json:"-"`
}

// Store is the agent's long-term memory: hybrid retrieval (HNSW vector search
// fused with BM25 keyword search), Ebbinghaus-style time decay, and
// reinforcement - memories that get recalled become harder to forget.
//
// score = (wv*vecSim + wb*bm25norm) * decay(daysSinceAccess) * (1 + wi*importance)
// decay: exponential, default half-life 30 days. A recall that surfaces a
// memory reinforces it: access count +1, clock resets, importance +5% (cap 1).
type Store struct {
	mu           sync.RWMutex
	mem          map[uint32]*Memory
	hnsw         *HNSW
	bm25         *BM25
	emb          *Embedder
	nextID       uint32
	HalfLifeDays float64
	WVec, WBM25  float64
	WImportance  float64
	EfSearch     int
}

func NewStore() *Store {
	return &Store{
		mem:          map[uint32]*Memory{},
		hnsw:         NewHNSW(),
		bm25:         NewBM25(),
		emb:          NewEmbedder(),
		nextID:       1,
		HalfLifeDays: 30,
		WVec:         0.55,
		WBM25:        0.45,
		WImportance:  0.25,
		EfSearch:     200,
	}
}

// Remember stores a memory and indexes it for both vector and keyword recall.
func (s *Store) Remember(text string, tags []string, importance float64) *Memory {
	s.mu.Lock()
	defer s.mu.Unlock()
	if importance <= 0 {
		importance = 0.5
	}
	if importance > 1 {
		importance = 1
	}
	m := &Memory{
		ID: s.nextID, Text: text, Tags: tags, Importance: importance,
		Created: time.Now(), LastAccess: time.Now(),
		Vec: s.emb.Embed(text),
	}
	s.nextID++
	s.mem[m.ID] = m
	s.hnsw.Insert(m.ID, m.Vec)
	s.bm25.Add(m.ID, text)
	return m
}

// Forget deletes a memory from all indexes.
func (s *Store) Forget(id uint32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.mem[id]; !ok {
		return false
	}
	delete(s.mem, id)
	s.bm25.Remove(id)
	// HNSW entries become unreachable over time; a tombstone list keeps search
	// correct without graph surgery (filtered at recall time).
	return true
}

func (s *Store) decay(m *Memory, now time.Time) float64 {
	days := now.Sub(m.LastAccess).Hours() / 24
	if days < 0 {
		days = 0
	}
	return math.Pow(0.5, days/s.HalfLifeDays)
}

// Hit is one recalled memory with its score breakdown.
type Hit struct {
	Memory
	Score  float64 `json:"score"`
	VecSim float64 `json:"vec_sim"`
	BM25   float64 `json:"bm25"`
	Decay  float64 `json:"decay"`
}

// Recall finds the k most relevant memories for query: hybrid fusion of vector
// similarity and BM25, weighted by time decay and importance. Surfaced
// memories are reinforced (access count, clock, importance).
func (s *Store) Recall(query string, k int) []Hit {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recallLocked(query, k, true)
}

// RecallRaw is Recall without reinforcement (used by benchmarks).
func (s *Store) RecallRaw(query string, k int) []Hit {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recallLocked(query, k, false)
}

func (s *Store) recallLocked(query string, k int, reinforce bool) []Hit {
	if k <= 0 {
		return nil
	}
	qv := s.emb.Embed(query)
	cand := k * 5
	if cand < 50 {
		cand = 50
	}
	vecIDs := s.hnsw.Search(qv, cand, s.EfSearch)
	bmIDs := s.bm25.Search(query, cand)

	vecSim := map[uint32]float32{}
	for _, id := range vecIDs {
		m := s.mem[id]
		if m != nil {
			vecSim[id] = (Dot(qv, m.Vec) + 1) / 2 // to [0,1]
		}
	}
	bmScore := map[uint32]float64{}
	var bmMax float64
	for _, r := range bmIDs {
		if s.mem[r.ID] == nil {
			continue
		}
		bmScore[r.ID] = r.Score
		if r.Score > bmMax {
			bmMax = r.Score
		}
	}
	union := map[uint32]bool{}
	for id := range vecSim {
		union[id] = true
	}
	for id := range bmScore {
		union[id] = true
	}
	now := time.Now()
	hits := make([]Hit, 0, len(union))
	for id := range union {
		m := s.mem[id]
		if m == nil {
			continue
		}
		bm := 0.0
		if bmMax > 0 {
			bm = bmScore[id] / bmMax
		}
		d := s.decay(m, now)
		score := (s.WVec*float64(vecSim[id]) + s.WBM25*bm) * d * (1 + s.WImportance*m.Importance)
		hits = append(hits, Hit{Memory: *m, Score: score, VecSim: float64(vecSim[id]), BM25: bm, Decay: d})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if len(hits) > k {
		hits = hits[:k]
	}
	if reinforce {
		for i := range hits {
			m := s.mem[hits[i].ID]
			m.AccessCount++
			m.LastAccess = now
			m.Importance = math.Min(1, m.Importance+0.05)
			hits[i].Memory = *m
		}
	}
	return hits
}

// Stats for the dashboard.
type Stats struct {
	Memories      int     `json:"memories"`
	IndexedVec    int     `json:"indexed_vectors"`
	IndexedBM25   int     `json:"indexed_bm25"`
	AvgImportance float64 `json:"avg_importance"`
	TotalAccess   int     `json:"total_access"`
}

func (s *Store) Stats() Stats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	st := Stats{Memories: len(s.mem), IndexedVec: s.hnsw.Len(), IndexedBM25: s.bm25.Len()}
	var imp float64
	for _, m := range s.mem {
		imp += m.Importance
		st.TotalAccess += m.AccessCount
	}
	if len(s.mem) > 0 {
		st.AvgImportance = imp / float64(len(s.mem))
	}
	return st
}

// List returns all memories, newest first (dashboard feed).
func (s *Store) List(limit int) []Memory {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Memory, 0, len(s.mem))
	for _, m := range s.mem {
		out = append(out, *m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Created.After(out[j].Created) })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// snapshot is the on-disk format: memories carry their vectors, so the HNSW
// graph is rebuilt on load in O(n log n) - no fragile graph serialization.
// memDisk adds the embedding vector back for persistence (the public JSON of
// Memory hides it - API responses stay small).
type memDisk struct {
	Memory
	Vec []float32 `json:"vec"`
}

type snapshot struct {
	NextID   uint32    `json:"next_id"`
	Memories []memDisk `json:"memories"`
}

func (s *Store) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap := snapshot{NextID: s.nextID}
	for _, m := range s.mem {
		snap.Memories = append(snap.Memories, memDisk{*m, m.Vec})
	}
	b, err := json.Marshal(snap)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path) // atomic
}

func (s *Store) Load(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var snap snapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.mem = map[uint32]*Memory{}
	s.hnsw = NewHNSW()
	s.bm25 = NewBM25()
	s.nextID = snap.NextID
	for _, md := range snap.Memories {
		m := md.Memory
		m.Vec = md.Vec
		if m.Vec == nil {
			m.Vec = s.emb.Embed(m.Text)
		}
		s.mem[m.ID] = &m
		s.hnsw.Insert(m.ID, m.Vec)
		s.bm25.Add(m.ID, m.Text)
	}
	return nil
}

// SearchVector is vector-only recall (no BM25, no decay) - the HNSW fast path.
func (s *Store) SearchVector(vec []float32, k int) []uint32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := s.hnsw.Search(vec, k, s.EfSearch)
	out := ids[:0]
	for _, id := range ids {
		if s.mem[id] != nil {
			out = append(out, id)
		}
	}
	return out
}

// BruteForceVector is exact vector search over every memory - ground truth for
// recall benchmarks (O(n) per query by design).
func (s *Store) BruteForceVector(vec []float32, k int) []uint32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	type sd struct {
		id uint32
		d  float32
	}
	all := make([]sd, 0, len(s.mem))
	for id, m := range s.mem {
		all = append(all, sd{id, dist(vec, m.Vec)})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].d < all[j].d })
	if len(all) > k {
		all = all[:k]
	}
	out := make([]uint32, len(all))
	for i, x := range all {
		out[i] = x.id
	}
	return out
}
