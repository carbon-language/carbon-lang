//===- unittests/ErrorOrTest.cpp - ErrorOr.h tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/ErrorOr.h"

#include "gtest/gtest.h"

#include <memory>

using namespace lld;
using namespace llvm;

namespace {

ErrorOr<int> t1() {return 1;}
ErrorOr<int> t2() {return make_error_code(errc::invalid_argument);}

TEST(ErrorOr, SimpleValue) {
  auto a = t1();
  EXPECT_TRUE(a);
  EXPECT_EQ(1, *a);

  a = t2();
  EXPECT_FALSE(a);
  EXPECT_EQ(errc::invalid_argument, a);
  EXPECT_DEBUG_DEATH(*a, "Cannot get value when an error exists");
}

ErrorOr<std::unique_ptr<int>> t3() {
  return std::unique_ptr<int>(new int(3));
}

TEST(ErrorOr, Types) {
  int x;
  ErrorOr<int&> a(x);
  *a = 42;
  EXPECT_EQ(42, *a);

  // Move only types.
  EXPECT_EQ(3, **t3());
}
} // end anon namespace

struct InvalidArgError {
  InvalidArgError() {}
  InvalidArgError(std::string S) : ArgName(S) {}
  std::string ArgName;
};

namespace lld {
template<>
struct ErrorOrUserDataTraits<InvalidArgError> : std::true_type {
  static error_code error() {
    return make_error_code(errc::invalid_argument);
  }
};
} // end namespace lld

ErrorOr<int> t4() {
  return InvalidArgError("adena");
}

namespace {
TEST(ErrorOr, UserErrorData) {
  auto a = t4();
  EXPECT_EQ(errc::invalid_argument, a);
  EXPECT_EQ("adena", t4().getError<InvalidArgError>().ArgName);
}
} // end anon namespace
