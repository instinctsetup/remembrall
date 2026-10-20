// cmd/server exposes the memory store over REST with an embedded live dashboard.
package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/instinctsetup/remembrall/rm"
)

//go:embed web
var webFS embed.FS

type server struct {
	st             *rm.Store
	path           string
	lastQueryNanos atomic.Int64
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func (s *server) handleRemember(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Text       string   `json:"text"`
		Tags       []string `json:"tags"`
		Importance float64  `json:"importance"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if strings.TrimSpace(body.Text) == "" {
		http.Error(w, "text required", 400)
		return
	}
	m := s.st.Remember(body.Text, body.Tags, body.Importance)
	_ = s.st.Save(s.path)
	writeJSON(w, m)
}

func (s *server) handleRecall(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	k, _ := strconv.Atoi(r.URL.Query().Get("k"))
	if k <= 0 {
		k = 5
	}
	start := time.Now()
	hits := s.st.Recall(q, k)
	s.lastQueryNanos.Store(time.Since(start).Nanoseconds())
	_ = s.st.Save(s.path)
	writeJSON(w, map[string]any{"query": q, "hits": hits})
}

func (s *server) handleForget(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.ParseUint(r.PathValue("id"), 10, 32)
	ok := s.st.Forget(uint32(id))
	_ = s.st.Save(s.path)
	writeJSON(w, map[string]bool{"forgotten": ok})
}

func (s *server) handleList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.st.List(50))
}

func (s *server) handleStats(w http.ResponseWriter, r *http.Request) {
	st := s.st.Stats()
	writeJSON(w, map[string]any{
		"memories":        st.Memories,
		"indexed_vectors": st.IndexedVec,
		"avg_importance":  fmt.Sprintf("%.2f", st.AvgImportance),
		"total_access":    st.TotalAccess,
		"last_query_ms":   fmt.Sprintf("%.2f", float64(s.lastQueryNanos.Load())/1e6),
	})
}

// sample corpus: realistic agent memories
var sampleTopics = []string{
	"prefers dark roast coffee before morning standup",
	"flight to Patna departs at 7:40 PM on Friday",
	"reviewer asked to split the compaction logic into its own package",
	"birthday is February 28, plans a small dinner with college friends",
	"uses Go for systems projects and Python for scripting",
	"BTCUSDT order book replay pipeline needs depth snapshots every 100ms",
	"prefers concise answers with the numbers first",
	"gym schedule changed to Monday Wednesday Friday evenings",
	"interview loop at a trading firm covers low latency C++ and market microstructure",
	"mother's medicines refill on the first of every month",
	"project cipherbase benchmarks must be measured never estimated",
	"favorite editor keybindings: vim mode with relative line numbers",
	"rent is due on the 5th, paid via UPI to the landlord",
	"learning Rust by building a small HTTP load balancer",
	"allergic to peanuts, checks labels when ordering snacks",
}

func (s *server) handleSeed(w http.ResponseWriter, r *http.Request) {
	go func() {
		rnd := rand.New(rand.NewSource(99))
		for i := 0; i < 200; i++ {
			t := sampleTopics[rnd.Intn(len(sampleTopics))]
			s.st.Remember(fmt.Sprintf("%s (note %d)", t, i), []string{"seed"}, 0.3+rnd.Float64()*0.6)
			time.Sleep(5 * time.Millisecond)
		}
		_ = s.st.Save(s.path)
	}()
	writeJSON(w, map[string]bool{"seeding": true})
}

func main() {
	addr := flag.String("addr", ":8091", "listen address")
	data := flag.String("data", "remembrall-data/memories.json", "memory file")
	flag.Parse()
	st := rm.NewStore()
	if err := st.Load(*data); err == nil {
		log.Printf("loaded %d memories from %s", st.Stats().Memories, *data)
	}
	s := &server{st: st, path: *data}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/remember", s.handleRemember)
	mux.HandleFunc("GET /api/recall", s.handleRecall)
	mux.HandleFunc("DELETE /api/memory/{id}", s.handleForget)
	mux.HandleFunc("GET /api/memories", s.handleList)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("POST /api/seed", s.handleSeed)
	sub, _ := fs.Sub(webFS, "web")
	mux.Handle("/", http.FileServer(http.FS(sub)))
	log.Printf("remembrall dashboard on http://localhost%s (data in %s)", *addr, *data)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
