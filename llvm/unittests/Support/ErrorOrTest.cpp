//===- unittests/ErrorOrTest.cpp - ErrorOr.h tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ErrorOr.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

ErrorOr<int> t1() {return 1;}
ErrorOr<int> t2() { return errc::invalid_argument; }

TEST(ErrorOr, SimpleValue) {
  ErrorOr<int> a = t1();
  EXPECT_TRUE(a);
  EXPECT_EQ(1, *a);

  a = t2();
  EXPECT_FALSE(a);
  EXPECT_EQ(errc::invalid_argument, a.getError());
#ifdef EXPECT_DEBUG_DEATH
  EXPECT_DEBUG_DEATH(*a, "Cannot get value when an error exists");
#endif
}

#if LLVM_HAS_CXX11_STDLIB
ErrorOr<std::unique_ptr<int> > t3() {
  return std::unique_ptr<int>(new int(3));
}
#endif

TEST(ErrorOr, Types) {
  int x;
  ErrorOr<int&> a(x);
  *a = 42;
  EXPECT_EQ(42, x);

#if LLVM_HAS_CXX11_STDLIB
  // Move only types.
  EXPECT_EQ(3, **t3());
#endif
}

struct B {};
struct D : B {};

TEST(ErrorOr, Covariant) {
  ErrorOr<B*> b(ErrorOr<D*>(0));
  b = ErrorOr<D*>(0);

#if LLVM_HAS_CXX11_STDLIB
  ErrorOr<std::unique_ptr<B> > b1(ErrorOr<std::unique_ptr<D> >(0));
  b1 = ErrorOr<std::unique_ptr<D> >(0);
#endif
}
} // end anon namespace
