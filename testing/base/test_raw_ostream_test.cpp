// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/test_raw_ostream.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

using ::testing::HasSubstr;
using ::testing::StrEq;

TEST(TestRawOstreamTest, Basics) {
  TestRawOstream os;

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

TEST(TestRawOstreamTest, MultipleStreams) {
  TestRawOstream os1;
  TestRawOstream os2;

  os1 << "test ";
  os2 << "test stream 2";
  os1 << "stream 1";
  EXPECT_THAT(os1.TakeStr(), StrEq("test stream 1"));
  EXPECT_THAT(os2.TakeStr(), StrEq("test stream 2"));
}

TEST(TestRawOstreamTest, MultipleLines) {
  TestRawOstream os;

  os << "test line 1\n";
  os << "test line 2\n";
  os << "test line 3\n";
  EXPECT_THAT(os.TakeStr(), StrEq("test line 1\ntest line 2\ntest line 3\n"));
}

TEST(TestRawOstreamTest, Substring) {
  TestRawOstream os;

  os << "test line 1\n";
  os << "test line 2\n";
  os << "test line 3\n";
  EXPECT_THAT(os.TakeStr(), HasSubstr("test line 2"));
}

}  // namespace
}  // namespace Carbon::Testing
