// go-cache/main.go
// In-memory LRU cache for StreamRec recommendation results.
// The Python FastAPI server calls this before computing recommendations.
//
// Endpoints:
//   GET  /cache/:key      → returns cached JSON or 404
//   POST /cache/:key      → stores JSON payload
//   DELETE /cache/:key    → evicts a key
//   GET  /cache/stats     → hit rate, size, uptime

package main

import (
	"container/list"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ── LRU Cache ────────────────────────────────────────────────────────────────

type entry struct {
	key       string
	value     []byte
	createdAt time.Time
}

type LRUCache struct {
	mu       sync.RWMutex
	cap      int
	ttl      time.Duration
	ll       *list.List
	items    map[string]*list.Element
	hits     int64
	misses   int64
	startedAt time.Time
}

func NewLRUCache(capacity int, ttl time.Duration) *LRUCache {
	return &LRUCache{
		cap:       capacity,
		ttl:       ttl,
		ll:        list.New(),
		items:     make(map[string]*list.Element),
		startedAt: time.Now(),
	}
}

func (c *LRUCache) Get(key string) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	el, ok := c.items[key]
	if !ok {
		c.misses++
		return nil, false
	}
	e := el.Value.(*entry)
	if c.ttl > 0 && time.Since(e.createdAt) > c.ttl {
		c.ll.Remove(el)
		delete(c.items, key)
		c.misses++
		return nil, false
	}
	c.ll.MoveToFront(el)
	c.hits++
	return e.value, true
}

func (c *LRUCache) Set(key string, value []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		el.Value.(*entry).value = value
		el.Value.(*entry).createdAt = time.Now()
		return
	}
	if c.ll.Len() >= c.cap {
		// evict least recently used
		oldest := c.ll.Back()
		if oldest != nil {
			c.ll.Remove(oldest)
			delete(c.items, oldest.Value.(*entry).key)
		}
	}
	el := c.ll.PushFront(&entry{key: key, value: value, createdAt: time.Now()})
	c.items[key] = el
}

func (c *LRUCache) Delete(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	el, ok := c.items[key]
	if !ok {
		return false
	}
	c.ll.Remove(el)
	delete(c.items, key)
	return true
}

func (c *LRUCache) Stats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(c.hits) / float64(total)
	}
	return map[string]interface{}{
		"size":       c.ll.Len(),
		"capacity":   c.cap,
		"hits":       c.hits,
		"misses":     c.misses,
		"hit_rate":   fmt.Sprintf("%.2f%%", hitRate*100),
		"uptime_s":   time.Since(c.startedAt).Seconds(),
	}
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

var cache = NewLRUCache(1000, 5*time.Minute)

func cacheHandler(w http.ResponseWriter, r *http.Request) {
	// strip /cache/ prefix to get the key
	key := strings.TrimPrefix(r.URL.Path, "/cache/")
	if key == "" || key == "stats" {
		if key == "stats" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(cache.Stats())
			return
		}
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		val, ok := cache.Get(key)
		if !ok {
			http.Error(w, "cache miss", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Cache", "HIT")
		w.Write(val)

	case http.MethodPost:
		var body interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		raw, _ := json.Marshal(body)
		cache.Set(key, raw)
		w.WriteHeader(http.StatusCreated)

	case http.MethodDelete:
		if cache.Delete(key) {
			w.WriteHeader(http.StatusNoContent)
		} else {
			http.Error(w, "key not found", http.StatusNotFound)
		}

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "ok",
		"uptime_s": time.Since(cache.startedAt).Seconds(),
	})
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/cache/", cacheHandler)
	mux.HandleFunc("/health", healthHandler)

	port := "8080"
	log.Printf("StreamRec Go cache running on :%s (cap=1000, ttl=5m)", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
