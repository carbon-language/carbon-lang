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


template<class G>
void PrintGraph(const G &g) {
  for (uptr i = 0; i < g.size(); i++) {
    for (uptr j = 0; j < g.size(); j++) {
      fprintf(stderr, "%d", g.hasEdge(i, j));
    }
    fprintf(stderr, "\n");
  }
}

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
        uptr path[BV::kSize];
        uptr len;
        for (len = 1; len < BV::kSize; len++) {
          if (g.findPath(from, target, path, len) == len)
            break;
        }
        EXPECT_LT(len, BV::kSize);
        EXPECT_TRUE(target.getBit(path[len - 1]));
        // fprintf(stderr, "reachable: %zd; path %zd {%zd %zd %zd}\n",
        //        from, len, path[0], path[1], path[2]);
        num_reachable++;
      }
    }
  }
  EXPECT_GT(num_reachable, 0);
}

template <class BV>
void Test_isReachable() {
  uptr path[5];
  BVGraph<BV> g;
  g.clear();
  BV target;
  target.clear();
  uptr t0 = 0;
  uptr t1 = g.size() - 1;
  target.setBit(t0);
  target.setBit(t1);

  uptr f0 = 1;
  uptr f1 = 2;
  uptr f2 = g.size() / 2;
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
  EXPECT_EQ(g.findPath(f0, target, path, ARRAY_SIZE(path)), 3U);
  EXPECT_EQ(path[0], f0);
  EXPECT_EQ(path[1], f1);
  EXPECT_EQ(path[2], t0);
  EXPECT_EQ(g.findPath(f1, target, path, ARRAY_SIZE(path)), 2U);
  EXPECT_EQ(path[0], f1);
  EXPECT_EQ(path[1], t0);

  g.addEdge(f3, t1);
  EXPECT_TRUE(g.isReachable(f0, target));
  EXPECT_TRUE(g.isReachable(f1, target));
  EXPECT_TRUE(g.isReachable(f2, target));
  EXPECT_TRUE(g.isReachable(f3, target));
}

TEST(SanitizerCommon, BVGraph_isReachable) {
  Test_isReachable<BasicBitVector<u8> >();
  Test_isReachable<BasicBitVector<> >();
  Test_isReachable<TwoLevelBitVector<> >();
  Test_isReachable<TwoLevelBitVector<3, BasicBitVector<u8> > >();
}

template <class BV>
void LongCycle() {
  BVGraph<BV> g;
  g.clear();
  vector<uptr> path_vec(g.size());
  uptr *path = path_vec.data();
  uptr start = 5;
  for (uptr i = start; i < g.size() - 1; i++) {
    g.addEdge(i, i + 1);
    for (uptr j = 0; j < start; j++)
      g.addEdge(i, j);
  }
  //  Bad graph that looks like this:
  // 00000000000000
  // 00000000000000
  // 00000000000000
  // 00000000000000
  // 00000000000000
  // 11111010000000
  // 11111001000000
  // 11111000100000
  // 11111000010000
  // 11111000001000
  // 11111000000100
  // 11111000000010
  // 11111000000001
  // if (g.size() <= 64) PrintGraph(g);
  BV target;
  for (uptr i = start + 1; i < g.size(); i += 11) {
    // if ((i & (i - 1)) == 0) fprintf(stderr, "Path: : %zd\n", i);
    target.clear();
    target.setBit(i);
    EXPECT_TRUE(g.isReachable(start, target));
    EXPECT_EQ(g.findPath(start, target, path, g.size()), i - start + 1);
  }
}

TEST(SanitizerCommon, BVGraph_LongCycle) {
  LongCycle<BasicBitVector<u8> >();
  LongCycle<BasicBitVector<> >();
  LongCycle<TwoLevelBitVector<> >();
  LongCycle<TwoLevelBitVector<2, BasicBitVector<u8> > >();
}
