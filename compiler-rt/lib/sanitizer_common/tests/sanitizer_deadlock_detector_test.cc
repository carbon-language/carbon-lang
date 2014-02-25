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
void RunBasicTest() {
  uptr path[10];
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
    EXPECT_EQ(0U, d.findPathToHeldLock(&dtls, n1, path, 1));
    EXPECT_EQ(2U, d.findPathToHeldLock(&dtls, n1, path, 10));
    EXPECT_EQ(2U, d.findPathToHeldLock(&dtls, n1, path, 2));
    EXPECT_EQ(path[0], n1);
    EXPECT_EQ(path[1], n2);
    EXPECT_EQ(d.getData(n1), 1U);
    EXPECT_EQ(d.getData(n2), 2U);
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
    EXPECT_EQ(0U, d.findPathToHeldLock(&dtls, n1, path, 2));
    EXPECT_EQ(3U, d.findPathToHeldLock(&dtls, n1, path, 10));
    EXPECT_EQ(path[0], n1);
    EXPECT_EQ(path[1], n2);
    EXPECT_EQ(path[2], n3);
    EXPECT_EQ(d.getData(n1), 1U);
    EXPECT_EQ(d.getData(n2), 2U);
    EXPECT_EQ(d.getData(n3), 3U);
    d.onUnlock(&dtls, n1);
    d.onUnlock(&dtls, n3);
  }
}

TEST(DeadlockDetector, BasicTest) {
  RunBasicTest<BV1>();
  RunBasicTest<BV2>();
  RunBasicTest<BV3>();
  RunBasicTest<BV4>();
}

template <class BV>
void RunRemoveNodeTest() {
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
  RunRemoveNodeTest<BV1>();
  RunRemoveNodeTest<BV2>();
  RunRemoveNodeTest<BV3>();
  RunRemoveNodeTest<BV4>();
}

template <class BV>
void RunMultipleEpochsTest() {
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
  RunMultipleEpochsTest<BV1>();
  RunMultipleEpochsTest<BV2>();
  RunMultipleEpochsTest<BV3>();
  RunMultipleEpochsTest<BV4>();
}

template <class BV>
void RunCorrectEpochFlush() {
  ScopedDD<BV> sdd;
  DeadlockDetector<BV> &d = *sdd.dp;
  DeadlockDetectorTLS<BV> &dtls = sdd.dtls;
  vector<uptr> locks1;
  for (uptr i = 0; i < d.size(); i++)
    locks1.push_back(d.newNode(i));
  EXPECT_EQ(d.testOnlyGetEpoch(), d.size());
  d.onLock(&dtls, locks1[3]);
  d.onLock(&dtls, locks1[4]);
  d.onLock(&dtls, locks1[5]);

  // We have a new epoch, old locks in dtls will have to be forgotten.
  uptr l0 = d.newNode(0);
  EXPECT_EQ(d.testOnlyGetEpoch(), d.size() * 2);
  uptr l1 = d.newNode(0);
  EXPECT_EQ(d.testOnlyGetEpoch(), d.size() * 2);
  d.onLock(&dtls, l0);
  d.onLock(&dtls, l1);
  EXPECT_TRUE(d.testOnlyHasEdge(0, 1));
  EXPECT_FALSE(d.testOnlyHasEdge(1, 0));
  EXPECT_FALSE(d.testOnlyHasEdge(3, 0));
  EXPECT_FALSE(d.testOnlyHasEdge(4, 0));
  EXPECT_FALSE(d.testOnlyHasEdge(5, 0));
}

TEST(DeadlockDetector, CorrectEpochFlush) {
  RunCorrectEpochFlush<BV1>();
  RunCorrectEpochFlush<BV2>();
}
