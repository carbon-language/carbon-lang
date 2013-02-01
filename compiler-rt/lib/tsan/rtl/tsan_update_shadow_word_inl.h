//===-- tsan_update_shadow_word_inl.h ---------------------------*- C++ -*-===//
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
// Body of the hottest inner loop.
// If we wrap this body into a function, compilers (both gcc and clang)
// produce sligtly less efficient code.
//===----------------------------------------------------------------------===//
do {
  StatInc(thr, StatShadowProcessed);
  const unsigned kAccessSize = 1 << kAccessSizeLog;
  unsigned off = cur.ComputeSearchOffset();
  u64 *sp = &shadow_mem[(idx + off) % kShadowCnt];
  old = LoadShadow(sp);
  if (old.IsZero()) {
    StatInc(thr, StatShadowZero);
    if (store_word)
      StoreIfNotYetStored(sp, &store_word);
    // The above StoreIfNotYetStored could be done unconditionally
    // and it even shows 4% gain on synthetic benchmarks (r4307).
    break;
  }
  // is the memory access equal to the previous?
  if (Shadow::Addr0AndSizeAreEqual(cur, old)) {
    StatInc(thr, StatShadowSameSize);
    // same thread?
    if (Shadow::TidsAreEqual(old, cur)) {
      StatInc(thr, StatShadowSameThread);
      if (OldIsInSameSynchEpoch(old, thr)) {
        if (old.IsRWNotWeaker(kAccessIsWrite, kIsAtomic)) {
          // found a slot that holds effectively the same info
          // (that is, same tid, same sync epoch and same size)
          StatInc(thr, StatMopSame);
          return;
        }
        StoreIfNotYetStored(sp, &store_word);
        break;
      }
      if (old.IsRWWeakerOrEqual(kAccessIsWrite, kIsAtomic))
        StoreIfNotYetStored(sp, &store_word);
      break;
    }
    StatInc(thr, StatShadowAnotherThread);
    if (HappensBefore(old, thr)) {
      StoreIfNotYetStored(sp, &store_word);
      break;
    }
    if (old.IsBothReadsOrAtomic(kAccessIsWrite, kIsAtomic))
      break;
    goto RACE;
  }
  // Do the memory access intersect?
  // In Go all memory accesses are 1 byte, so there can be no intersections.
  if (kCppMode && Shadow::TwoRangesIntersect(old, cur, kAccessSize)) {
    StatInc(thr, StatShadowIntersect);
    if (Shadow::TidsAreEqual(old, cur)) {
      StatInc(thr, StatShadowSameThread);
      break;
    }
    StatInc(thr, StatShadowAnotherThread);
    if (old.IsBothReadsOrAtomic(kAccessIsWrite, kIsAtomic))
      break;
    if (HappensBefore(old, thr))
      break;
    goto RACE;
  }
  // The accesses do not intersect.
  StatInc(thr, StatShadowNotIntersect);
  break;
} while (0);
