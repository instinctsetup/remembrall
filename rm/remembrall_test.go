package rm

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"time"
)

func TestEmbedDeterministicNormalized(t *testing.T) {
	e := NewEmbedder()
	a := e.Embed("The user prefers dark roast coffee in the morning")
	b := e.Embed("The user prefers dark roast coffee in the morning")
	if len(a) != EmbedDim {
		t.Fatalf("dim %d", len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("embedding not deterministic")
		}
	}
	var n float64
	for _, x := range a {
		n += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(n)-1) > 1e-5 {
		t.Fatalf("not normalized: %f", math.Sqrt(n))
	}
}

func TestEmbedSimilarity(t *testing.T) {
	e := NewEmbedder()
	coffee := e.Embed("user likes dark roast coffee every morning")
	coffee2 := e.Embed("user enjoys dark roast coffee each morning")
	quantum := e.Embed("quantum chromodynamics lattice gauge theory")
	if Dot(coffee, coffee2) <= Dot(coffee, quantum) {
		t.Fatalf("related texts should be more similar: %f vs %f",
			Dot(coffee, coffee2), Dot(coffee, quantum))
	}
}

func bruteForce(vecs map[uint32][]float32, q []float32, k int) []uint32 {
	type sd struct {
		id uint32
		d  float32
	}
	all := make([]sd, 0, len(vecs))
	for id, v := range vecs {
		all = append(all, sd{id, dist(q, v)})
	}
	for i := 0; i < k && i < len(all); i++ {
		for j := i + 1; j < len(all); j++ {
			if all[j].d < all[i].d {
				all[i], all[j] = all[j], all[i]
			}
		}
	}
	out := make([]uint32, 0, k)
	for i := 0; i < k && i < len(all); i++ {
		out = append(out, all[i].id)
	}
	return out
}

func TestHNSWRecall(t *testing.T) {
	rnd := rand.New(rand.NewSource(7))
	const n = 3000
	vecs := map[uint32][]float32{}
	h := NewHNSW()
	for i := uint32(1); i <= n; i++ {
		v := make([]float32, 64)
		var norm float64
		for j := range v {
			v[j] = float32(rnd.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] = float32(float64(v[j]) / norm)
		}
		vecs[i] = v
		h.Insert(i, v)
	}
	// recall@10 over 50 random queries
	var total float64
	const nq = 50
	for qi := 0; qi < nq; qi++ {
		q := vecs[uint32(rnd.Intn(n)+1)]
		approx := h.Search(q, 10, 100)
		exact := bruteForce(vecs, q, 10)
		set := map[uint32]bool{}
		for _, id := range exact {
			set[id] = true
		}
		var hit float64
		for _, id := range approx {
			if set[id] {
				hit++
			}
		}
		total += hit / 10
	}
	recall := total / nq
	if recall < 0.95 {
		t.Fatalf("recall@10 = %.3f, want >= 0.95", recall)
	}
	t.Logf("recall@10 = %.3f", recall)
}

func TestBM25RanksExactTerms(t *testing.T) {
	ix := NewBM25()
	ix.Add(1, "the user prefers dark roast coffee in the morning")
	ix.Add(2, "the weather forecast predicts rain on tuesday")
	ix.Add(3, "coffee prices rose sharply in global markets this year")
	res := ix.Search("dark roast coffee", 2)
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("want doc 1 first, got %+v", res)
	}
	ix.Remove(1)
	res = ix.Search("dark roast coffee", 2)
	if len(res) > 0 && res[0].ID == 1 {
		t.Fatal("removed doc still searchable")
	}
}

func TestStoreRecallHybrid(t *testing.T) {
	s := NewStore()
	s.Remember("Mahesh prefers dark roast coffee before standup", []string{"pref"}, 0.8)
	s.Remember("Binance BTCUSDT order book depth snapshot schema", []string{"work"}, 0.6)
	s.Remember("Flight to Patna leaves at 7:40 PM on Friday", []string{"travel"}, 0.9)
	hits := s.Recall("what coffee does Mahesh like", 3)
	if len(hits) == 0 || hits[0].ID != 1 {
		t.Fatalf("want memory 1 first, got %+v", hits)
	}
	// exact-term query: BM25 should carry it
	hits = s.Recall("BTCUSDT schema", 3)
	if len(hits) == 0 || hits[0].ID != 2 {
		t.Fatalf("want memory 2 first, got %+v", hits)
	}
}

func TestDecayAndReinforcement(t *testing.T) {
	s := NewStore()
	old := s.Remember("old hobby: played chess competitively in school", nil, 0.5)
	fresh := s.Remember("old hobby notes: chess openings practice plan", nil, 0.5)
	// age the first memory 120 days
	s.mem[old.ID].LastAccess = time.Now().Add(-120 * 24 * time.Hour)
	_ = fresh
	d := s.decay(s.mem[old.ID], time.Now())
	if math.Abs(d-0.0625) > 1e-6 { // (1/2)^(120/30) = 1/16
		t.Fatalf("decay %f, want 0.0625", d)
	}
	imp0 := s.mem[old.ID].Importance
	s.Recall("chess", 2)
	if s.mem[old.ID].AccessCount == 0 {
		t.Fatal("recall did not reinforce access count")
	}
	if s.mem[old.ID].Importance <= imp0 {
		t.Fatal("recall did not reinforce importance")
	}
	if time.Since(s.mem[old.ID].LastAccess) > time.Minute {
		t.Fatal("recall did not reset the decay clock")
	}
}

func TestForget(t *testing.T) {
	s := NewStore()
	m := s.Remember("temporary fact to forget", nil, 0.5)
	if !s.Forget(m.ID) {
		t.Fatal("forget failed")
	}
	for _, h := range s.Recall("temporary fact", 5) {
		if h.ID == m.ID {
			t.Fatal("forgotten memory still recalled")
		}
	}
	if s.Forget(m.ID) {
		t.Fatal("double forget should return false")
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	s := NewStore()
	for i := 0; i < 200; i++ {
		s.Remember("memory number "+itoa(i)+" about topic "+itoa(i%10), nil, 0.5)
	}
	path := filepath.Join(t.TempDir(), "mem.json")
	if err := s.Save(path); err != nil {
		t.Fatal(err)
	}
	s2 := NewStore()
	if err := s2.Load(path); err != nil {
		t.Fatal(err)
	}
	if s2.Stats().Memories != 200 {
		t.Fatalf("loaded %d memories", s2.Stats().Memories)
	}
	h1 := s.RecallRaw("topic 3", 3)
	h2 := s2.RecallRaw("topic 3", 3)
	if len(h1) == 0 || len(h2) == 0 || h1[0].ID != h2[0].ID {
		t.Fatalf("recall mismatch after load: %v vs %v", h1, h2)
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b [20]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		p--
		b[p] = '-'
	}
	return string(b[p:])
}
