package rm

import (
	"math"
	"sort"
	"sync"
)

// BM25 keyword index over memory texts (Okapi BM25, k1=1.5, b=0.75) backed by
// an inverted index: a query touches only the posting lists of its terms.
// Complements vector recall: exact terms, names, IDs, dates.
type BM25 struct {
	mu       sync.RWMutex
	docs     map[uint32]int            // id -> doc length (tokens)
	postings map[string]map[uint32]int // term -> {id: tf}
	lenSum   int
	k1, b    float64
}

func NewBM25() *BM25 {
	return &BM25{docs: map[uint32]int{}, postings: map[string]map[uint32]int{}, k1: 1.5, b: 0.75}
}

func (ix *BM25) Add(id uint32, text string) {
	toks := tokenize(text)
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.removeLocked(id)
	ix.docs[id] = len(toks)
	ix.lenSum += len(toks)
	for _, t := range toks {
		post := ix.postings[t]
		if post == nil {
			post = map[uint32]int{}
			ix.postings[t] = post
		}
		post[id]++
	}
}

func (ix *BM25) removeLocked(id uint32) {
	dl, ok := ix.docs[id]
	if !ok {
		return
	}
	delete(ix.docs, id)
	ix.lenSum -= dl
	for t, post := range ix.postings {
		if _, ok := post[id]; ok {
			delete(post, id)
			if len(post) == 0 {
				delete(ix.postings, t)
			}
		}
	}
}

func (ix *BM25) Remove(id uint32) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.removeLocked(id)
}

func (ix *BM25) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.docs)
}

type ScoredID struct {
	ID    uint32
	Score float64
}

// Search returns up to k doc ids by BM25 score, descending.
func (ix *BM25) Search(query string, k int) []ScoredID {
	q := tokenize(query)
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	n := len(ix.docs)
	if n == 0 || len(q) == 0 {
		return nil
	}
	avgdl := float64(ix.lenSum) / float64(n)
	scores := map[uint32]float64{}
	for _, t := range q {
		post := ix.postings[t]
		if len(post) == 0 {
			continue
		}
		df := len(post)
		idf := math.Log(1 + (float64(n-df)+0.5)/(float64(df)+0.5))
		for id, tf := range post {
			dl := float64(ix.docs[id])
			num := float64(tf) * (ix.k1 + 1)
			den := float64(tf) + ix.k1*(1-ix.b+ix.b*dl/avgdl)
			scores[id] += idf * num / den
		}
	}
	out := make([]ScoredID, 0, len(scores))
	for id, s := range scores {
		out = append(out, ScoredID{id, s})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > k {
		out = out[:k]
	}
	return out
}
