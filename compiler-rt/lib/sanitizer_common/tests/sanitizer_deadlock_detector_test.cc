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

typedef BasicBitVector<u8> BV1;
typedef BasicBitVector<> BV2;
typedef TwoLevelBitVector<> BV3;
typedef TwoLevelBitVector<3, BasicBitVector<u8> > BV4;

// Poor man's unique_ptr.
template<class BV>
struct ScopedDD {
  ScopedDD() {
    dp = new DeadlockDetector<BV>;
    dp->clear();
    dtls.clear();
  }
  ~ScopedDD() { delete dp; }
  DeadlockDetector<BV> *dp;
  DeadlockDetectorTLS<BV> dtls;
};

template <class BV>
void BasicTest() {
  ScopedDD<BV> sdd;
  DeadlockDetector<BV> &d = *sdd.dp;
  DeadlockDetectorTLS<BV> &dtls = sdd.dtls;
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
}

TEST(DeadlockDetector, BasicTest) {
  BasicTest<BV1>();
  BasicTest<BV2>();
  BasicTest<BV3>();
  BasicTest<BV4>();
}

template <class BV>
void RemoveNodeTest() {
  ScopedDD<BV> sdd;
  DeadlockDetector<BV> &d = *sdd.dp;
  DeadlockDetectorTLS<BV> &dtls = sdd.dtls;

  uptr l0 = d.newNode(0);
  uptr l1 = d.newNode(1);
  uptr l2 = d.newNode(2);
  uptr l3 = d.newNode(3);
  uptr l4 = d.newNode(4);
  uptr l5 = d.newNode(5);

  // l0=>l1=>l2
  d.onLock(&dtls, l0);
  d.onLock(&dtls, l1);
  d.onLock(&dtls, l2);
  d.onUnlock(&dtls, l1);
  d.onUnlock(&dtls, l0);
  d.onUnlock(&dtls, l2);
  // l3=>l4=>l5
  d.onLock(&dtls, l3);
  d.onLock(&dtls, l4);
  d.onLock(&dtls, l5);
  d.onUnlock(&dtls, l4);
  d.onUnlock(&dtls, l3);
  d.onUnlock(&dtls, l5);

  set<uptr> locks;
  locks.insert(l0);
  locks.insert(l1);
  locks.insert(l2);
  locks.insert(l3);
  locks.insert(l4);
  locks.insert(l5);
  for (uptr i = 6; i < d.size(); i++) {
    uptr lt = d.newNode(i);
    locks.insert(lt);
    d.onLock(&dtls, lt);
    d.onUnlock(&dtls, lt);
    d.removeNode(lt);
  }
  EXPECT_EQ(locks.size(), d.size());
  // l2=>l0
  EXPECT_FALSE(d.onLock(&dtls, l2));
  EXPECT_TRUE(d.onLock(&dtls, l0));
  d.onUnlock(&dtls, l2);
  d.onUnlock(&dtls, l0);
  // l4=>l3
  EXPECT_FALSE(d.onLock(&dtls, l4));
  EXPECT_TRUE(d.onLock(&dtls, l3));
  d.onUnlock(&dtls, l4);
  d.onUnlock(&dtls, l3);

  EXPECT_EQ(d.size(), d.testOnlyGetEpoch());

  d.removeNode(l2);
  d.removeNode(l3);
  locks.clear();
  // make sure no edges from or to l0,l1,l4,l5 left.
  for (uptr i = 4; i < d.size(); i++) {
    uptr lt = d.newNode(i);
    locks.insert(lt);
    uptr a, b;
    // l0 => lt?
    a = l0; b = lt;
    EXPECT_FALSE(d.onLock(&dtls, a));
    EXPECT_FALSE(d.onLock(&dtls, b));
    d.onUnlock(&dtls, a);
    d.onUnlock(&dtls, b);
    // l1 => lt?
    a = l1; b = lt;
    EXPECT_FALSE(d.onLock(&dtls, a));
    EXPECT_FALSE(d.onLock(&dtls, b));
    d.onUnlock(&dtls, a);
    d.onUnlock(&dtls, b);
    // lt => l4?
    a = lt; b = l4;
    EXPECT_FALSE(d.onLock(&dtls, a));
    EXPECT_FALSE(d.onLock(&dtls, b));
    d.onUnlock(&dtls, a);
    d.onUnlock(&dtls, b);
    // lt => l5?
    a = lt; b = l5;
    EXPECT_FALSE(d.onLock(&dtls, a));
    EXPECT_FALSE(d.onLock(&dtls, b));
    d.onUnlock(&dtls, a);
    d.onUnlock(&dtls, b);

    d.removeNode(lt);
  }
  // Still the same epoch.
  EXPECT_EQ(d.size(), d.testOnlyGetEpoch());
  EXPECT_EQ(locks.size(), d.size() - 4);
  // l2 and l3 should have ben reused.
  EXPECT_EQ(locks.count(l2), 1U);
  EXPECT_EQ(locks.count(l3), 1U);
}

TEST(DeadlockDetector, RemoveNodeTest) {
  RemoveNodeTest<BV1>();
  RemoveNodeTest<BV2>();
  RemoveNodeTest<BV3>();
  RemoveNodeTest<BV4>();
}

template <class BV>
void MultipleEpochsTest() {
  ScopedDD<BV> sdd;
  DeadlockDetector<BV> &d = *sdd.dp;
  DeadlockDetectorTLS<BV> &dtls = sdd.dtls;

  set<uptr> locks;
  for (uptr i = 0; i < d.size(); i++) {
    EXPECT_TRUE(locks.insert(d.newNode(i)).second);
  }
  EXPECT_EQ(d.testOnlyGetEpoch(), d.size());
  for (uptr i = 0; i < d.size(); i++) {
    EXPECT_TRUE(locks.insert(d.newNode(i)).second);
    EXPECT_EQ(d.testOnlyGetEpoch(), d.size() * 2);
  }
  locks.clear();

  uptr l0 = d.newNode(0);
  uptr l1 = d.newNode(0);
  d.onLock(&dtls, l0);
  d.onLock(&dtls, l1);
  d.onUnlock(&dtls, l0);
  EXPECT_EQ(d.testOnlyGetEpoch(), 3 * d.size());
  for (uptr i = 0; i < d.size(); i++) {
    EXPECT_TRUE(locks.insert(d.newNode(i)).second);
  }
  EXPECT_EQ(d.testOnlyGetEpoch(), 4 * d.size());

  // Can not handle the locks from the previous epoch.
  // The user should update the lock id.
  EXPECT_DEATH(d.onLock(&dtls, l0), "CHECK failed.*current_epoch_");
  EXPECT_DEATH(d.onUnlock(&dtls, l1), "CHECK failed.*current_epoch_");
}

TEST(DeadlockDetector, MultipleEpochsTest) {
  MultipleEpochsTest<BV1>();
  MultipleEpochsTest<BV2>();
  MultipleEpochsTest<BV3>();
  MultipleEpochsTest<BV4>();
}


