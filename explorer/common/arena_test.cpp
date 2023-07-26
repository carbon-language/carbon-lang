// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/common/arena.h"

#include <gtest/gtest.h>

namespace Carbon {

class ReportDestruction {
 public:
  explicit ReportDestruction(bool* destroyed) : destroyed_(destroyed) {}

  ~ReportDestruction() { *destroyed_ = true; }

 private:
  bool* destroyed_;
};

TEST(ArenaTest, BasicAllocation) {
  bool destroyed = false;
  {
    Arena arena;
    (void)arena.New<ReportDestruction>(&destroyed);
  }
  EXPECT_TRUE(destroyed);
}

struct CanonicalizedDummy {
  explicit CanonicalizedDummy(int) {}
  explicit CanonicalizedDummy(int*) {}
  explicit CanonicalizedDummy(int, int*) {}
  using EnableCanonicalizedAllocation = void;
};

TEST(ArenaTest, Canonicalization) {
  Arena arena;
  auto* dummy1 = arena.New<CanonicalizedDummy>(1);
  EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(dummy1)>>);
  auto* dummy2 = arena.New<CanonicalizedDummy>(1);
  EXPECT_TRUE(dummy1 == dummy2);
}

TEST(ArenaTest, CanonicalizationArgMismatch) {
  Arena arena;
  auto* dummy1 = arena.New<CanonicalizedDummy>(1);
  auto* dummy2 = arena.New<CanonicalizedDummy>(2);
  EXPECT_TRUE(dummy1 != dummy2);
}

TEST(ArenaTest, CanonicalizationDifferentArenas) {
  Arena arena1;
  Arena arena2;
  auto* dummy1 = arena1.New<CanonicalizedDummy>(1);
  auto* dummy2 = arena2.New<CanonicalizedDummy>(1);

  EXPECT_TRUE(dummy1 != dummy2);
}

TEST(ArenaTest, CanonicalizationShallow) {
  Arena arena;
  int i1 = 1;
  int i2 = 1;
  auto* dummy1 = arena.New<CanonicalizedDummy>(&i1);
  auto* dummy2 = arena.New<CanonicalizedDummy>(&i2);
  EXPECT_TRUE(dummy1 != dummy2);
}

TEST(ArenaTest, CanonicalizationMultipleArgs) {
  Arena arena;
  int i;
  auto* dummy1 = arena.New<CanonicalizedDummy>(1, &i);
  auto* dummy2 = arena.New<CanonicalizedDummy>(1, &i);
  EXPECT_TRUE(dummy1 == dummy2);
}

}  // namespace Carbon
