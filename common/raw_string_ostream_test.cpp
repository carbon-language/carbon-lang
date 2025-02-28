// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_string_ostream.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

using ::testing::HasSubstr;
using ::testing::StrEq;

TEST(RawStringOstreamTest, Basics) {
  RawStringOstream os;

  os << "test 1";
  EXPECT_THAT(os.TakeStr(), StrEq("test 1"));

  os << "test"
     << " "
     << "2";
  EXPECT_THAT(os.TakeStr(), StrEq("test 2"));

  os << "test";
  os << " ";
  os << "3";
  EXPECT_THAT(os.TakeStr(), StrEq("test 3"));
}

TEST(RawStringOstreamTest, MultipleStreams) {
  RawStringOstream os1;
  RawStringOstream os2;

  os1 << "test ";
  os2 << "test stream 2";
  os1 << "stream 1";
  EXPECT_THAT(os1.TakeStr(), StrEq("test stream 1"));
  EXPECT_THAT(os2.TakeStr(), StrEq("test stream 2"));
}

TEST(RawStringOstreamTest, MultipleLines) {
  RawStringOstream os;

  os << "test line 1\n";
  os << "test line 2\n";
  os << "test line 3\n";
  EXPECT_THAT(os.TakeStr(), StrEq("test line 1\ntest line 2\ntest line 3\n"));
}

TEST(RawStringOstreamTest, Substring) {
  RawStringOstream os;

  os << "test line 1\n";
  os << "test line 2\n";
  os << "test line 3\n";
  EXPECT_THAT(os.TakeStr(), HasSubstr("test line 2"));
}

TEST(RawStringOstreamTest, Pwrite) {
  RawStringOstream os;
  os << "test line 1\n";
  os.pwrite("splat", 5, 1);
  os << "test line 2\n";
  EXPECT_THAT(os.TakeStr(), HasSubstr("tsplatine 1\ntest line 2\n"));
}

}  // namespace
}  // namespace Carbon::Testing
