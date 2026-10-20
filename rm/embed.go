package rm

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// Embedder produces fixed-dimension dense vectors locally - no model download,
// no API. Signed feature hashing over token unigrams and bigrams: each n-gram
// votes for one dimension with a +/- sign from a second hash. Captures lexical
// similarity; swap in a neural embedder behind the same interface for semantic
// similarity (documented in README).
const EmbedDim = 384

type Embedder struct{}

func NewEmbedder() *Embedder { return &Embedder{} }

func tokenize(text string) []string {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := words[:0]
	for _, w := range words {
		if len(w) >= 2 {
			out = append(out, w)
		}
	}
	return out
}

// Embed maps text to an L2-normalized 384-dim vector.
func (e *Embedder) Embed(text string) []float32 {
	toks := tokenize(text)
	v := make([]float32, EmbedDim)
	if len(toks) == 0 {
		return v
	}
	add := func(feat string, w float32) {
		h := fnv.New64a()
		h.Write([]byte(feat))
		hv := h.Sum64()
		dim := hv % EmbedDim
		if (hv>>32)&1 == 1 {
			v[dim] += w
		} else {
			v[dim] -= w
		}
	}
	for i, t := range toks {
		add("u:"+t, 1.0)
		if i+1 < len(toks) {
			add("b:"+t+" "+toks[i+1], 1.5) // bigrams weigh more
		}
	}
	// L2 normalize
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Sqrt(n)
	if n > 0 {
		for i := range v {
			v[i] = float32(float64(v[i]) / n)
		}
	}
	return v
}

// dot is cosine similarity on normalized vectors.
func Dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
