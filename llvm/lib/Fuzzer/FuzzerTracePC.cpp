//===- FuzzerTracePC.cpp - PC tracing--------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Trace PCs.
// This module implements __sanitizer_cov_trace_pc, a callback required
// for -fsanitize-coverage=trace-pc instrumentation.
//
// Experimental and not yet tuned for performance.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"

namespace fuzzer {
static const size_t kMapSize = 65371; // Prime.
static uint8_t CurMap[kMapSize];
static uint8_t CombinedMap[kMapSize];
static size_t CombinedMapSize;
static thread_local uintptr_t Prev;

void PcMapResetCurrent() {
  if (Prev) {
    Prev = 0;
    memset(CurMap, 0, sizeof(CurMap));
  }
}

// TODO: speed this up.
void PcMapMergeCurrentToCombined() {
  if (!Prev) return;
  uintptr_t Res = 0;
  for (size_t i = 0; i < kMapSize; i++) {
    uint8_t p = (CombinedMap[i] |= CurMap[i]);
    CurMap[i] = 0;
    Res += p != 0;
  }
  CombinedMapSize = Res;
}

size_t PcMapCombinedSize() { return CombinedMapSize; }

static void HandlePC(uintptr_t PC) {
  // We take 12 bits of PC and mix it with the previous PCs.
  uintptr_t Idx = (Prev << 5) ^ (PC & 4095);
  CurMap[Idx % kMapSize] = 1;
  Prev = Idx;
}

} // namespace fuzzer

extern "C" void __sanitizer_cov_trace_pc() {
  fuzzer::HandlePC(reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
}
//uintptr_t __sanitizer_get_total_unique_coverage() { return 0; }
//uintptr_t __sanitizer_get_number_of_counters() { return 0; }
