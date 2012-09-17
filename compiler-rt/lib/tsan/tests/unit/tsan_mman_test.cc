//===-- tsan_mman_test.cc -------------------------------------------------===//
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
#include "tsan_mman.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"

namespace __tsan {

TEST(Mman, Internal) {
  ScopedInRtl in_rtl;
  char *p = (char*)internal_alloc(MBlockScopedBuf, 10);
  EXPECT_NE(p, (char*)0);
  char *p2 = (char*)internal_alloc(MBlockScopedBuf, 20);
  EXPECT_NE(p2, (char*)0);
  EXPECT_NE(p2, p);
  for (int i = 0; i < 10; i++) {
    p[i] = 42;
  }
  for (int i = 0; i < 20; i++) {
    ((char*)p2)[i] = 42;
  }
  internal_free(p);
  internal_free(p2);
}

TEST(Mman, User) {
  ScopedInRtl in_rtl;
  ThreadState *thr = cur_thread();
  uptr pc = 0;
  char *p = (char*)user_alloc(thr, pc, 10);
  EXPECT_NE(p, (char*)0);
  char *p2 = (char*)user_alloc(thr, pc, 20);
  EXPECT_NE(p2, (char*)0);
  EXPECT_NE(p2, p);
  MBlock *b = user_mblock(thr, p);
  EXPECT_NE(b, (MBlock*)0);
  EXPECT_EQ(b->size, (uptr)10);
  MBlock *b2 = user_mblock(thr, p2);
  EXPECT_NE(b2, (MBlock*)0);
  EXPECT_EQ(b2->size, (uptr)20);
  for (int i = 0; i < 10; i++) {
    p[i] = 42;
    EXPECT_EQ(b, user_mblock(thr, p + i));
  }
  for (int i = 0; i < 20; i++) {
    ((char*)p2)[i] = 42;
    EXPECT_EQ(b2, user_mblock(thr, p2 + i));
  }
  user_free(thr, pc, p);
  user_free(thr, pc, p2);
}

TEST(Mman, UserRealloc) {
  ScopedInRtl in_rtl;
  ThreadState *thr = cur_thread();
  uptr pc = 0;
  {
    void *p = user_realloc(thr, pc, 0, 0);
    // Strictly saying this is incorrect, realloc(NULL, N) is equivalent to
    // malloc(N), thus must return non-NULL pointer.
    EXPECT_EQ(p, (void*)0);
  }
  {
    void *p = user_realloc(thr, pc, 0, 100);
    EXPECT_NE(p, (void*)0);
    memset(p, 0xde, 100);
    user_free(thr, pc, p);
  }
  {
    void *p = user_alloc(thr, pc, 100);
    EXPECT_NE(p, (void*)0);
    memset(p, 0xde, 100);
    void *p2 = user_realloc(thr, pc, p, 0);
    EXPECT_EQ(p2, (void*)0);
  }
  {
    void *p = user_realloc(thr, pc, 0, 100);
    EXPECT_NE(p, (void*)0);
    memset(p, 0xde, 100);
    void *p2 = user_realloc(thr, pc, p, 10000);
    EXPECT_NE(p2, (void*)0);
    for (int i = 0; i < 100; i++)
      EXPECT_EQ(((char*)p2)[i], (char)0xde);
    memset(p2, 0xde, 10000);
    user_free(thr, pc, p2);
  }
  {
    void *p = user_realloc(thr, pc, 0, 10000);
    EXPECT_NE(p, (void*)0);
    memset(p, 0xde, 10000);
    void *p2 = user_realloc(thr, pc, p, 10);
    EXPECT_NE(p2, (void*)0);
    for (int i = 0; i < 10; i++)
      EXPECT_EQ(((char*)p2)[i], (char)0xde);
    user_free(thr, pc, p2);
  }
}

}  // namespace __tsan
