//===-- tsan_sync_test.cc ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_sync.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "gtest/gtest.h"

#include <stdlib.h>
#include <stdint.h>
#include <map>

namespace __tsan {

TEST(Sync, Table) {
  const uintptr_t kIters = 512*1024;
  const uintptr_t kRange = 10000;

  ScopedInRtl in_rtl;
  ThreadState *thr = cur_thread();
  uptr pc = 0;

  SyncTab tab;
  SyncVar *golden[kRange] = {};
  unsigned seed = 0;
  for (uintptr_t i = 0; i < kIters; i++) {
    uintptr_t addr = rand_r(&seed) % (kRange - 1) + 1;
    if (rand_r(&seed) % 2) {
      // Get or add.
      SyncVar *v = tab.GetAndLock(thr, pc, addr, true);
      EXPECT_TRUE(golden[addr] == 0 || golden[addr] == v);
      EXPECT_EQ(v->addr, addr);
      golden[addr] = v;
      v->mtx.Unlock();
    } else {
      // Remove.
      SyncVar *v = tab.GetAndRemove(thr, pc, addr);
      EXPECT_EQ(golden[addr], v);
      if (v) {
        EXPECT_EQ(v->addr, addr);
        golden[addr] = 0;
        DestroyAndFree(v);
      }
    }
  }
  for (uintptr_t addr = 0; addr < kRange; addr++) {
    if (golden[addr] == 0)
      continue;
    SyncVar *v = tab.GetAndRemove(thr, pc, addr);
    EXPECT_EQ(v, golden[addr]);
    EXPECT_EQ(v->addr, addr);
    DestroyAndFree(v);
  }
}

}  // namespace __tsan
