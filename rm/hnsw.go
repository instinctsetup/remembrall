package rm

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// HNSW approximate nearest-neighbor index, implemented from scratch after
// Malkov & Yashunin (2016). Distance = 1 - cosine (vectors are normalized).
//
// Layers: node level ~ geometric distribution (mL = 1/ln(M)). Insertion greedily
// descends from the entry point, then links each layer to the M nearest
// candidates found by beam search (efConstruction wide). Search descends
// greedily to layer 0, then beams efSearch wide.

const (
	hnswM              = 16
	hnswMmax0          = 32 // max neighbors at layer 0
	hnswEfConstruction = 200
)

type hnswNode struct {
	id        uint32
	vec       []float32
	neighbors [][]uint32 // neighbors[layer]
}

type candidate struct {
	id   uint32
	dist float32
}

// minHeap / maxHeap over candidates.
type candHeap struct {
	c   []candidate
	max bool
}

func (h candHeap) Len() int { return len(h.c) }
func (h candHeap) Less(i, j int) bool {
	if h.max {
		return h.c[i].dist > h.c[j].dist
	}
	return h.c[i].dist < h.c[j].dist
}
func (h candHeap) Swap(i, j int) { h.c[i], h.c[j] = h.c[j], h.c[i] }
func (h *candHeap) Push(x any)   { h.c = append(h.c, x.(candidate)) }
func (h *candHeap) Pop() any     { old := h.c; n := len(old); it := old[n-1]; h.c = old[:n-1]; return it }

// HNSW is safe for concurrent reads; writes take the write lock.
type HNSW struct {
	mu       sync.RWMutex
	nodes    map[uint32]*hnswNode
	entry    int32 // -1 when empty
	maxLayer int
	rnd      *rand.Rand
	mL       float64
}

func NewHNSW() *HNSW {
	return &HNSW{nodes: map[uint32]*hnswNode{}, entry: -1, rnd: rand.New(rand.NewSource(42)), mL: 1 / math.Log(hnswM)}
}

func dist(a, b []float32) float32 { return 1 - Dot(a, b) }

// set during Insert so selectNeighbors can measure the node being inserted,
// which is registered in the map before linking but kept explicit here.
var (
	currentInsertID  uint32
	currentInsertVec []float32
)

func (h *HNSW) randomLevel() int {
	return int(-math.Log(h.rnd.Float64()+1e-12) * h.mL)
}

func (h *HNSW) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes)
}

// Insert adds id with vector vec.
func (h *HNSW) Insert(id uint32, vec []float32) {
	h.mu.Lock()
	defer h.mu.Unlock()
	level := h.randomLevel()
	n := &hnswNode{id: id, vec: vec, neighbors: make([][]uint32, level+1)}
	for i := range n.neighbors {
		n.neighbors[i] = []uint32{}
	}
	if h.entry < 0 {
		h.nodes[id] = n
		h.entry = int32(id)
		h.maxLayer = level
		return
	}
	h.nodes[id] = n // visible before linking so neighbor pruning can measure it
	ep := uint32(h.entry)
	// greedy descent above node's level
	for l := h.maxLayer; l > level; l-- {
		ep = h.greedyClosest(vec, ep, l)
	}
	// beam search + link at each layer <= level
	for l := min(level, h.maxLayer); l >= 0; l-- {
		cands := h.searchLayer(vec, ep, hnswEfConstruction, l)
		mmax := hnswM
		if l == 0 {
			mmax = hnswMmax0
		}
		currentInsertID, currentInsertVec = id, vec
		sel := h.selectNeighbors(vec, cands, hnswM)
		n.neighbors[l] = sel
		for _, nb := range sel {
			node := h.nodes[nb]
			if l >= len(node.neighbors) {
				continue
			}
			node.neighbors[l] = append(node.neighbors[l], id)
			if len(node.neighbors[l]) > mmax {
				// prune: keep mmax nearest to node
				d := make([]candidate, len(node.neighbors[l]))
				for i, x := range node.neighbors[l] {
					d[i] = candidate{x, dist(node.vec, h.nodes[x].vec)}
				}
				node.neighbors[l] = h.selectNeighbors(node.vec, d, mmax)
			}
		}
		if len(cands) > 0 {
			ep = cands[0].id
		}
	}
	if level > h.maxLayer {
		h.maxLayer = level
		h.entry = int32(id)
	}
}

func (h *HNSW) greedyClosest(vec []float32, ep uint32, layer int) uint32 {
	cur := ep
	for {
		node := h.nodes[cur]
		best := cur
		bestD := dist(vec, node.vec)
		if layer < len(node.neighbors) {
			for _, nb := range node.neighbors[layer] {
				d := dist(vec, h.nodes[nb].vec)
				if d < bestD {
					best, bestD = nb, d
				}
			}
		}
		if best == cur {
			return cur
		}
		cur = best
	}
}

// searchLayer beam-searches one layer, returning up to ef closest candidates,
// sorted ascending by distance.
func (h *HNSW) searchLayer(vec []float32, ep uint32, ef, layer int) []candidate {
	visited := map[uint32]bool{ep: true}
	cands := &candHeap{max: false} // to explore
	top := &candHeap{max: true}    // results (max-heap on dist)
	d0 := dist(vec, h.nodes[ep].vec)
	heap.Push(cands, candidate{ep, d0})
	heap.Push(top, candidate{ep, d0})
	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)
		worst := top.c[0].dist
		if c.dist > worst && top.Len() >= ef {
			break
		}
		node := h.nodes[c.id]
		if layer < len(node.neighbors) {
			for _, nb := range node.neighbors[layer] {
				if visited[nb] {
					continue
				}
				visited[nb] = true
				d := dist(vec, h.nodes[nb].vec)
				if d < worst || top.Len() < ef {
					heap.Push(cands, candidate{nb, d})
					heap.Push(top, candidate{nb, d})
					if top.Len() > ef {
						heap.Pop(top)
					}
				}
			}
		}
	}
	out := make([]candidate, 0, top.Len())
	for top.Len() > 0 {
		out = append(out, heap.Pop(top).(candidate))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].dist < out[j].dist })
	return out
}

// selectNeighbors implements the HNSW neighbor-selection heuristic (Malkov &
// Yashunin, alg 4): walk candidates nearest-first and keep one only if it is
// closer to the target vector than to every already-selected neighbor. This
// keeps the graph diverse across directions instead of stacking one cluster,
// which is what buys recall at high dimension.
func (h *HNSW) selectNeighbors(target []float32, cands []candidate, m int) []uint32 {
	if len(cands) <= m {
		out := make([]uint32, len(cands))
		for i, c := range cands {
			out[i] = c.id
		}
		return out
	}
	s := make([]candidate, len(cands))
	copy(s, cands)
	sort.Slice(s, func(i, j int) bool { return s[i].dist < s[j].dist })
	selected := make([]uint32, 0, m)
	for _, c := range s {
		if len(selected) >= m {
			break
		}
		cv := h.nodes[c.id].vec
		if c.id == currentInsertID {
			cv = currentInsertVec
		}
		diverse := true
		for _, sel := range selected {
			sv := h.nodes[sel].vec
			if sel == currentInsertID {
				sv = currentInsertVec
			}
			if dist(cv, sv) < c.dist {
				diverse = false
				break
			}
		}
		if diverse {
			selected = append(selected, c.id)
		}
	}
	// fall back to filling with nearest if the heuristic kept too few
	if len(selected) < m {
		in := map[uint32]bool{}
		for _, x := range selected {
			in[x] = true
		}
		for _, c := range s {
			if len(selected) >= m {
				break
			}
			if !in[c.id] {
				selected = append(selected, c.id)
			}
		}
	}
	return selected
}

// Search returns the k nearest ids by cosine distance.
func (h *HNSW) Search(vec []float32, k, efSearch int) []uint32 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entry < 0 || k <= 0 {
		return nil
	}
	if efSearch < k {
		efSearch = k
	}
	ep := uint32(h.entry)
	for l := h.maxLayer; l > 0; l-- {
		ep = h.greedyClosest(vec, ep, l)
	}
	cands := h.searchLayer(vec, ep, efSearch, 0)
	if len(cands) > k {
		cands = cands[:k]
	}
	out := make([]uint32, len(cands))
	for i, c := range cands {
		out[i] = c.id
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
