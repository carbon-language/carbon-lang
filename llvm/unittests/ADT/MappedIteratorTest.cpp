//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <class T, class Fn>
auto map_range(const T &range, Fn fn)
    -> decltype(make_range(map_iterator(range.begin(), fn),
                           map_iterator(range.end(), fn))) {
  return make_range(map_iterator(range.begin(), fn),
                    map_iterator(range.end(), fn));
}

static char add1(char C) { return C + 1; }

TEST(MappedIterator, FnTest) {
  std::string S("abc");
  std::string T;

  for (char C : map_range(S, add1)) {
    T.push_back(C);
  }

  EXPECT_STREQ("bcd", T.c_str());
}

TEST(MappedIterator, LambdaTest) {
  std::string S("abc");
  std::string T;

  for (char C : map_range(S, [](char C) { return C + 1; })) {
    T.push_back(C);
  }

  EXPECT_STREQ("bcd", T.c_str());
}
}
