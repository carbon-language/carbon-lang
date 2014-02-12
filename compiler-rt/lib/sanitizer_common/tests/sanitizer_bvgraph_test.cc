//===-- sanitizer_bvgraph_test.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// Tests for sanitizer_bvgraph.h.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_bvgraph.h"

#include "sanitizer_test_utils.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <vector>
#include <set>

using namespace __sanitizer;
using namespace std;

class SimpleGraph {
 public:
  bool addEdge(uptr from, uptr to) {
    return s_.insert(idx(from, to)).second;
  }
 private:
  uptr idx(uptr from, uptr to) {
    CHECK_LE(from|to, 1 << 16);
    return (from << 16) + to;
  }
  set<uptr> s_;
};

TEST(SanitizerCommon, BVGraph) {
  typedef TwoLevelBitVector<> BV;
  BVGraph<BV> g;
  g.clear();
  BV target;
  SimpleGraph s_g;
  set<uptr> s;
  set<uptr> s_target;
  int num_reachable = 0;
  for (int it = 0; it < 3000; it++) {
    target.clear();
    s_target.clear();
    for (int t = 0; t < 4; t++) {
      uptr idx = (uptr)my_rand() % g.size();
      EXPECT_EQ(target.setBit(idx), s_target.insert(idx).second);
    }
    uptr from = my_rand() % g.size();
    uptr to = my_rand() % g.size();
    EXPECT_EQ(g.addEdge(from, to), s_g.addEdge(from, to));
    EXPECT_TRUE(g.hasEdge(from, to));
    for (int i = 0; i < 10; i++) {
      from = my_rand() % g.size();
      bool is_reachable = g.isReachable(from, target);
      if (is_reachable) {
        // printf("reachable: %d %zd\n", it, from);
        num_reachable++;
      }
    }
  }
  EXPECT_GT(num_reachable, 0);
}

TEST(SanitizerCommon, BVGraph_isReachable) {
  typedef TwoLevelBitVector<> BV;
  BVGraph<BV> g;
  g.clear();
  BV target;
  target.clear();
  uptr t0 = 100;
  uptr t1 = g.size() - 1;
  target.setBit(t0);
  target.setBit(t1);

  uptr f0 = 0;
  uptr f1 = 99;
  uptr f2 = 200;
  uptr f3 = g.size() - 2;

  EXPECT_FALSE(g.isReachable(f0, target));
  EXPECT_FALSE(g.isReachable(f1, target));
  EXPECT_FALSE(g.isReachable(f2, target));
  EXPECT_FALSE(g.isReachable(f3, target));

  g.addEdge(f0, f1);
  g.addEdge(f1, f2);
  g.addEdge(f2, f3);
  EXPECT_FALSE(g.isReachable(f0, target));
  EXPECT_FALSE(g.isReachable(f1, target));
  EXPECT_FALSE(g.isReachable(f2, target));
  EXPECT_FALSE(g.isReachable(f3, target));

  g.addEdge(f1, t0);
  EXPECT_TRUE(g.isReachable(f0, target));
  EXPECT_TRUE(g.isReachable(f1, target));
  EXPECT_FALSE(g.isReachable(f2, target));
  EXPECT_FALSE(g.isReachable(f3, target));

  g.addEdge(f3, t1);
  EXPECT_TRUE(g.isReachable(f0, target));
  EXPECT_TRUE(g.isReachable(f1, target));
  EXPECT_TRUE(g.isReachable(f2, target));
  EXPECT_TRUE(g.isReachable(f3, target));
}
