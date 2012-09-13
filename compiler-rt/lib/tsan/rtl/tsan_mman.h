//===-- tsan_mman.h ---------------------------------------------*- C++ -*-===//
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
#ifndef TSAN_MMAN_H
#define TSAN_MMAN_H

#include "tsan_defs.h"

namespace __tsan {

const uptr kDefaultAlignment = 16;

void InitializeAllocator();
void AlloctorThreadFinish(ThreadState *thr);

// For user allocations.
void *user_alloc(ThreadState *thr, uptr pc, uptr sz,
                 uptr align = kDefaultAlignment);
// Does not accept NULL.
void user_free(ThreadState *thr, uptr pc, void *p);
void *user_realloc(ThreadState *thr, uptr pc, void *p, uptr sz);
void *user_alloc_aligned(ThreadState *thr, uptr pc, uptr sz, uptr align);
// Given the pointer p into a valid allocated block,
// returns the descriptor of the block.
MBlock *user_mblock(ThreadState *thr, void *p);

enum MBlockType {
  MBlockScopedBuf,
  MBlockString,
  MBlockStackTrace,
  MBlockShadowStack,
  MBlockSync,
  MBlockClock,
  MBlockThreadContex,
  MBlockDeadInfo,
  MBlockRacyStacks,
  MBlockRacyAddresses,
  MBlockAtExit,
  MBlockFlag,
  MBlockReport,
  MBlockReportMop,
  MBlockReportThread,
  MBlockReportMutex,
  MBlockReportLoc,
  MBlockReportStack,
  MBlockSuppression,
  MBlockExpectRace,
  MBlockSignal,

  // This must be the last.
  MBlockTypeCount
};

// For internal data structures.
void *internal_alloc(MBlockType typ, uptr sz);
void internal_free(void *p);

template<typename T>
void DestroyAndFree(T *&p) {
  p->~T();
  internal_free(p);
  p = 0;
}

}  // namespace __tsan
#endif  // TSAN_MMAN_H
