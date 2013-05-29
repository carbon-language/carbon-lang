// A loadable module with a large thread local section, which would require
// allocation of a new TLS storage chunk when loaded with dlopen(). We use it
// to test the reachability of such chunks in LSan tests.

// This must be large enough that it doesn't fit into preallocated static TLS
// space (see STATIC_TLS_SURPLUS in glibc).
__thread void *huge_thread_local_array[(1 << 20) / sizeof(void *)]; // NOLINT

extern "C" void **StoreToTLS(void *p) {
  huge_thread_local_array[0] = p;
  return &huge_thread_local_array[0];
}
