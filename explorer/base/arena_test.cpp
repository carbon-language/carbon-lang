// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/base/arena.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

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
  explicit CanonicalizedDummy(std::vector<int>, std::nullopt_t) {}
  using EnableCanonicalizedAllocation = void;
};

TEST(ArenaTest, Canonicalize) {
  Arena arena;
  auto* dummy1 = arena.New<CanonicalizedDummy>(1);
  EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(dummy1)>>);
  auto* dummy2 = arena.New<CanonicalizedDummy>(1);
  EXPECT_TRUE(dummy1 == dummy2);
}

TEST(ArenaTest, CanonicalizeArgMismatch) {
  Arena arena;
  auto* dummy1 = arena.New<CanonicalizedDummy>(1);
  auto* dummy2 = arena.New<CanonicalizedDummy>(2);
  EXPECT_TRUE(dummy1 != dummy2);
}

TEST(ArenaTest, CanonicalizeDifferentArenas) {
  Arena arena1;
  Arena arena2;
  auto* dummy1 = arena1.New<CanonicalizedDummy>(1);
  auto* dummy2 = arena2.New<CanonicalizedDummy>(1);

  EXPECT_TRUE(dummy1 != dummy2);
}

TEST(ArenaTest, CanonicalizeIsShallow) {
  Arena arena;
  int i1 = 1;
  int i2 = 1;
  auto* dummy1 = arena.New<CanonicalizedDummy>(&i1);
  auto* dummy2 = arena.New<CanonicalizedDummy>(&i2);
  EXPECT_TRUE(dummy1 != dummy2);
}

TEST(ArenaTest, CanonicalizeMultipleArgs) {
  Arena arena;
  int i;
  auto* dummy1 = arena.New<CanonicalizedDummy>(1, &i);
  auto* dummy2 = arena.New<CanonicalizedDummy>(1, &i);
  EXPECT_TRUE(dummy1 == dummy2);
}

TEST(ArenaTest, CanonicalizeStdTypes) {
  Arena arena;
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {1, 2, 3};
  std::vector<int> v3 = {1, 2, 3, 4};

  auto* dummy1 = arena.New<CanonicalizedDummy>(v1, std::nullopt);
  auto* dummy2 = arena.New<CanonicalizedDummy>(v2, std::nullopt);
  EXPECT_TRUE(dummy1 == dummy2);

  auto* dummy3 = arena.New<CanonicalizedDummy>(v3, std::nullopt);
  EXPECT_TRUE(dummy1 != dummy3);
}

}  // namespace Carbon
