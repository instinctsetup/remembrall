package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/instinctsetup/remembrall/rm"
)

func main() {
	for _, n := range []int{30_000} {
		rnd := rand.New(rand.NewSource(7))
		h := rm.NewHNSW()
		vecs := make([][]float32, n+1)
		t0 := time.Now()
		for i := 1; i <= n; i++ {
			v := make([]float32, 384)
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
			h.Insert(uint32(i), v)
		}
		insDur := time.Since(t0)
		// exact top10 for 20 queries by brute force
		recs := map[int]float64{}
		t1 := time.Now()
		for qi := 0; qi < 20; qi++ {
			q := vecs[rnd.Intn(n)+1]
			type sd struct {
				id int
				d  float32
			}
			best := make([]sd, 0, 10)
			for i := 1; i <= n; i++ {
				d := 1 - rm.Dot(q, vecs[i])
				if len(best) < 10 || d < best[len(best)-1].d {
					best = append(best, sd{i, d})
					for j := len(best) - 1; j > 0 && best[j].d < best[j-1].d; j-- {
						best[j], best[j-1] = best[j-1], best[j]
					}
					if len(best) > 10 {
						best = best[:10]
					}
				}
			}
			set := map[int]bool{}
			for _, b := range best {
				set[b.id] = true
			}
			for _, ef := range []int{100, 400} {
				approx := h.Search(q, 10, ef)
				var hit float64
				for _, id := range approx {
					if set[int(id)] {
						hit++
					}
				}
				recs[ef] += hit / 10
			}
		}
		qDur := time.Since(t1)
		fmt.Printf("n=%d insert=%s (%.0f/s) recall@10 ef100=%.3f ef400=%.3f (bruteforce20q=%s)\n",
			n, insDur.Round(time.Millisecond), float64(n)/insDur.Seconds(), recs[100]/20, recs[400]/20, qDur.Round(time.Millisecond))
	}
}
