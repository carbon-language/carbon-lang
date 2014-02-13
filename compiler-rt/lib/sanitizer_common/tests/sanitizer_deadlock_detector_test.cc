//===-- sanitizer_deadlock_detector_test.cc -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// Tests for sanitizer_deadlock_detector.h
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_deadlock_detector.h"

#include "sanitizer_test_utils.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <vector>
#include <set>

using namespace __sanitizer;
using namespace std;

template <class BV>
void TestDeadlockDetector() {
  // Can't allocate on stack.
  DeadlockDetector<BV> *dp = new DeadlockDetector<BV>;
  DeadlockDetector<BV> &d = *dp;
  DeadlockDetectorTLS dtls;
  d.clear();
  set<uptr> s;
  for (size_t i = 0; i < d.size() * 3; i++) {
    uptr node = d.newNode(0);
    EXPECT_TRUE(s.insert(node).second);
  }

  d.clear();
  s.clear();
  // Add size() nodes.
  for (size_t i = 0; i < d.size(); i++) {
    uptr node = d.newNode(0);
    EXPECT_TRUE(s.insert(node).second);
  }
  // Remove all nodes.
  for (set<uptr>::iterator it = s.begin(); it != s.end(); ++it)
    d.removeNode(*it);
  // The nodes should be reused.
  for (size_t i = 0; i < d.size(); i++) {
    uptr node = d.newNode(0);
    EXPECT_FALSE(s.insert(node).second);
  }

  // Cycle: n1->n2->n1
  {
    d.clear();
    dtls.clear();
    uptr n1 = d.newNode(1);
    uptr n2 = d.newNode(2);
    EXPECT_FALSE(d.onLock(&dtls, n1));
    EXPECT_FALSE(d.onLock(&dtls, n2));
    d.onUnlock(&dtls, n2);
    d.onUnlock(&dtls, n1);

    EXPECT_FALSE(d.onLock(&dtls, n2));
    EXPECT_TRUE(d.onLock(&dtls, n1));
    d.onUnlock(&dtls, n1);
    d.onUnlock(&dtls, n2);
  }

  // Cycle: n1->n2->n3->n1
  {
    d.clear();
    dtls.clear();
    uptr n1 = d.newNode(1);
    uptr n2 = d.newNode(2);
    uptr n3 = d.newNode(3);

    EXPECT_FALSE(d.onLock(&dtls, n1));
    EXPECT_FALSE(d.onLock(&dtls, n2));
    d.onUnlock(&dtls, n2);
    d.onUnlock(&dtls, n1);

    EXPECT_FALSE(d.onLock(&dtls, n2));
    EXPECT_FALSE(d.onLock(&dtls, n3));
    d.onUnlock(&dtls, n3);
    d.onUnlock(&dtls, n2);

    EXPECT_FALSE(d.onLock(&dtls, n3));
    EXPECT_TRUE(d.onLock(&dtls, n1));
    d.onUnlock(&dtls, n1);
    d.onUnlock(&dtls, n3);
  }

  delete dp;
}

TEST(SanitizerCommon, DeadlockDetector) {
  TestDeadlockDetector<BasicBitVector<> >();
  TestDeadlockDetector<TwoLevelBitVector<2> >();
}
