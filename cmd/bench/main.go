// cmd/bench measures insert throughput, recall latency, and recall@10 of the
// HNSW index vs exact brute force, on a synthetic agent-memory corpus.
// Method: single goroutine, sequential calls, in-process API, fixed seed.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/instinctsetup/remembrall/rm"
)

const nMem = 50_000

var subjects = []string{"user", "reviewer", "teammate", "manager", "professor", "doctor", "landlord", "mentor", "sister", "client", "recruiter", "neighbor"}
var verbs = []string{"prefers", "asked about", "scheduled", "cancelled", "confirmed", "mentioned", "booked", "postponed", "requested", "finished", "delegated", "flagged", "approved", "revised", "shared", "renewed"}
var objects = []string{
	"dark roast coffee", "morning standup", "flight tickets", "compaction benchmarks", "rent payment",
	"gym session", "code review", "order book replay", "interview prep", "medicine refill",
	"Rust load balancer", "vim keybindings", "quarterly report", "database migration", "latency budget",
	"vector index rebuild", "cache invalidation", "travel itinerary", "dinner reservation", "tax filing",
	"passport renewal", "dentist appointment", "sprint retrospective", "schema migration", "load test",
	"onboarding doc", "insurance claim", "book club", "cricket match", "hackathon submission",
	"visa appointment", "apartment viewing", "salary review", "conference talk", "open source contribution",
	"backup restore drill", "security audit", "team offsite", "budget forecast", "guitar practice",
}
var details = []string{"on Friday", "before noon", "with high priority", "next week", "every month", "at the office", "over UPI", "by email", "for the demo", "twice a week", "after lunch", "on Monday", "before the deadline", "with the Mumbai team", "under budget"}
var topics = []string{
	"golang", "kubernetes", "postgres", "kafka", "grpc", "terraform", "redis", "graphql", "wasm", "ebpf",
	"patna", "mumbai", "delhi", "bangalore", "hyderabad", "pune", "chennai", "kolkata", "jaipur", "lucknow",
	"cricket", "chess", "guitar", "photography", "cooking", "hiking", "swimming", "badminton", "cycling", "yoga",
	"bitcoin", "ethereum", "nifty", "options", "futures", "dividends", "mutual funds", "arbitrage", "staking", "custody",
	"grpc", "protobuf", "lambda", "dynamodb", "s3", "ec2", "cloudflare", "vercel", "supabase", "planetscale",
}

func genText(rnd *rand.Rand, i int) string {
	return fmt.Sprintf("%s %s %s %s about %s and %s",
		subjects[rnd.Intn(len(subjects))],
		verbs[rnd.Intn(len(verbs))],
		objects[rnd.Intn(len(objects))],
		details[rnd.Intn(len(details))],
		topics[rnd.Intn(len(topics))],
		topics[rnd.Intn(len(topics))])
}

func percentile(sorted []time.Duration, p float64) time.Duration {
	return sorted[int(float64(len(sorted)-1)*p)]
}

func main() {
	rnd := rand.New(rand.NewSource(42))
	st := rm.NewStore()

	// --- insert throughput ---
	start := time.Now()
	for i := 0; i < nMem; i++ {
		st.Remember(genText(rnd, i), nil, 0.5)
		if i%20000 == 0 && i > 0 {
			fmt.Printf("...%d memories in %s\n", i, time.Since(start).Round(time.Millisecond))
		}
	}
	wdur := time.Since(start)
	runtime.GC()

	// --- recall latency (hybrid path) ---
	queries := make([]string, 500)
	for i := range queries {
		queries[i] = verbs[rnd.Intn(len(verbs))] + " " + objects[rnd.Intn(len(objects))] + " " + topics[rnd.Intn(len(topics))]
	}
	lat := make([]time.Duration, len(queries))
	for i, q := range queries {
		t0 := time.Now()
		st.RecallRaw(q, 10)
		lat[i] = time.Since(t0)
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })

	// --- recall@10: HNSW vs exact brute force over all vectors (vector-only) ---
	emb := rm.NewEmbedder()
	var recTotal float64
	const nq = 30
	for qi := 0; qi < nq; qi++ {
		q := emb.Embed(queries[qi])
		approx := st.SearchVector(q, 10)
		exact := st.BruteForceVector(q, 10)
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
		recTotal += hit / 10
	}

	res := map[string]any{
		"platform":     runtime.GOOS + "/" + runtime.GOARCH + ", Go " + runtime.Version(),
		"method":       "single goroutine, sequential calls, in-process API (no HTTP), fixed seed, synthetic agent-memory corpus",
		"memories":     nMem,
		"insert_total": fmt.Sprintf("%.1fs for %d memories", wdur.Seconds(), nMem),
		"insert_ops":   fmt.Sprintf("%.0f memories/s", float64(nMem)/wdur.Seconds()),
		"recall_p50":   fmt.Sprintf("%.2f ms", float64(percentile(lat, 0.50).Microseconds())/1000),
		"recall_p99":   fmt.Sprintf("%.2f ms", float64(percentile(lat, 0.99).Microseconds())/1000),
		"recall_at_10": fmt.Sprintf("%.3f (HNSW vs exact brute force, %d queries)", recTotal/nq, nq),
		"embed_dim":    rm.EmbedDim,
	}
	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(out))
	if err := os.WriteFile("benchmarks.json", out, 0644); err != nil {
		log.Print(err)
	}
}
