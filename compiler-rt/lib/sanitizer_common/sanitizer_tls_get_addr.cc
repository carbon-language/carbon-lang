//===-- sanitizer_tls_get_addr.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Handle the __tls_get_addr call.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_tls_get_addr.h"

#include "sanitizer_flags.h"
#include "sanitizer_platform_interceptors.h"

namespace __sanitizer {
#if SANITIZER_INTERCEPT_TLS_GET_ADDR

// The actual parameter that comes to __tls_get_addr
// is a pointer to a struct with two words in it:
struct TlsGetAddrParam {
  uptr dso_id;
  uptr offset;
};

// Glibc starting from 2.19 allocates tls using __signal_safe_memalign,
// which has such header.
struct Glibc_2_19_tls_header {
  uptr size;
  uptr start;
};

// This must be static TLS
__attribute__((tls_model("initial-exec")))
static __thread DTLS dtls;

// Make sure we properly destroy the DTLS objects:
// this counter should never get too large.
static atomic_uintptr_t number_of_live_dtls;

static const uptr kDestroyedThread = -1;

static inline void DTLS_Resize(uptr new_size) {
  if (dtls.dtv_size >= new_size) return;
  new_size = RoundUpToPowerOfTwo(new_size);
  new_size = Max(new_size, 4096UL / sizeof(DTLS::DTV));
  DTLS::DTV *new_dtv =
      (DTLS::DTV *)MmapOrDie(new_size * sizeof(DTLS::DTV), "DTLS_Resize");
  CHECK_LT(atomic_fetch_add(&number_of_live_dtls, 1, memory_order_relaxed),
           1 << 20);
  if (dtls.dtv_size)
    internal_memcpy(new_dtv, dtls.dtv, dtls.dtv_size * sizeof(DTLS::DTV));
  DTLS_Destroy();
  dtls.dtv = new_dtv;
  dtls.dtv_size = new_size;
}

void DTLS_Destroy() {
  if (!dtls.dtv_size) return;
  uptr s = dtls.dtv_size;
  dtls.dtv_size = kDestroyedThread;  // Do this before unmap for AS-safety.
  UnmapOrDie(dtls.dtv, s * sizeof(DTLS::DTV));
  atomic_fetch_sub(&number_of_live_dtls, 1, memory_order_relaxed);
}

void DTLS_on_tls_get_addr(void *arg_void, void *res) {
  TlsGetAddrParam *arg = reinterpret_cast<TlsGetAddrParam *>(arg_void);
  uptr dso_id = arg->dso_id;
  if (dtls.dtv_size == kDestroyedThread) return;
  DTLS_Resize(dso_id + 1);
  if (dtls.dtv[dso_id].beg)
    return;
  uptr tls_size = 0;
  uptr tls_beg = reinterpret_cast<uptr>(res) - arg->offset;
  // Don't do anything fancy in this function, in particular don't print.
  // Some versions of libstdc++ are miscompiled and call this function
  // with mis-aligned stack:
  // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=58066
  // Printf uses SSE instructions that crash with mis-aligned accesses.
  // VPrintf(2, "__tls_get_addr: %p {%p,%p} => %p; tls_beg: %p; sp: %p\n", arg,
  //        arg->dso_id, arg->offset, res, tls_beg, &tls_beg);
  if (dtls.last_memalign_ptr == tls_beg) {
    tls_size = dtls.last_memalign_size;
    // VPrintf(2, "__tls_get_addr: glibc <=2.18 suspected; tls={%p,%p}\n",
    //        tls_beg, tls_size);
  } else if ((tls_beg % 4096) == sizeof(Glibc_2_19_tls_header)) {
    // We may want to check gnu_get_libc_version().
    Glibc_2_19_tls_header *header = (Glibc_2_19_tls_header *)tls_beg - 1;
    tls_size = header->size;
    tls_beg = header->start;
    //VPrintf(2, "__tls_get_addr: glibc >=2.19 suspected; tls={%p %p}\n",
    //        tls_beg, tls_size);
  } else {
    //VPrintf(2, "__tls_get_addr: Can't guess glibc version\n");
    // This may happen inside the DTOR of main thread, so just ignore it.
    tls_size = 0;
  }
  dtls.dtv[dso_id].beg = tls_beg;
  dtls.dtv[dso_id].size = tls_size;
}

void DTLS_on_libc_memalign(void *ptr, uptr size) {
  // VPrintf(2, "DTLS_on_libc_memalign: %p %p\n", ptr, size);
  dtls.last_memalign_ptr = reinterpret_cast<uptr>(ptr);
  dtls.last_memalign_size = size;
}

DTLS *DTLS_Get() { return &dtls; }

#else
void DTLS_on_libc_memalign(void *ptr, uptr size) {}
void DTLS_on_tls_get_addr(void *arg, void *res) {}
DTLS *DTLS_Get() { return 0; }
void DTLS_Destroy() {}
#endif  // SANITIZER_INTERCEPT_TLS_GET_ADDR

}  // namespace __sanitizer
