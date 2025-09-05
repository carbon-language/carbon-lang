// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/emplace_by_calling.h"

#include <gtest/gtest.h>

#include <list>

namespace Carbon {
namespace {

struct NoncopyableType {
  NoncopyableType() = default;
  NoncopyableType(const NoncopyableType&) = delete;
  auto operator=(const NoncopyableType&) -> NoncopyableType& = delete;
};

auto Make() -> NoncopyableType { return NoncopyableType(); }

TEST(EmplaceByCalling, Noncopyable) {
  std::list<NoncopyableType> list;
  // This should compile.
  list.emplace_back(EmplaceByCalling(Make));
}

TEST(EmplaceByCalling, NoncopyableInAggregate) {
  struct Aggregate {
    int a, b, c;
    NoncopyableType noncopyable;
  };

  std::list<Aggregate> list;
  // This should compile.
  list.emplace_back(EmplaceByCalling(
      [] { return Aggregate{.a = 1, .b = 2, .c = 3, .noncopyable = Make()}; }));
}

class CopyCounter {
 public:
  explicit CopyCounter(int* counter) : counter_(counter) {}
  CopyCounter(const CopyCounter& other) : counter_(other.counter_) {
    ++*counter_;
  }

 private:
  int* counter_;
};

TEST(EmplaceByCalling, NoCopies) {
  std::vector<CopyCounter> vec;
  vec.reserve(10);
  int copies = 0;
  for (int i = 0; i != 10; ++i) {
    vec.emplace_back(EmplaceByCalling([&] { return CopyCounter(&copies); }));
  }
  EXPECT_EQ(0, copies);
}

}  // namespace
}  // namespace Carbon
