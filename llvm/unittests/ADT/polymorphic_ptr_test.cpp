//===- llvm/unittest/ADT/polymorphic_ptr.h - polymorphic_ptr<T> tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/polymorphic_ptr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct S {
  S(int x) : x(x) {}
  S *clone() { return new S(*this); }
  int x;
};

// A function that forces the return of a copy.
polymorphic_ptr<S> dummy_copy(const polymorphic_ptr<S> &arg) { return arg; }

TEST(polymorphic_ptr_test, Basic) {
  polymorphic_ptr<S> null;
  EXPECT_FALSE((bool)null);
  EXPECT_TRUE(!null);
  EXPECT_EQ((S*)0, null.get());

  S *s = new S(42);
  polymorphic_ptr<S> p(s);
  EXPECT_TRUE((bool)p);
  EXPECT_FALSE(!p);
  EXPECT_TRUE(p != null);
  EXPECT_FALSE(p == null);
  EXPECT_TRUE(p == s);
  EXPECT_TRUE(s == p);
  EXPECT_FALSE(p != s);
  EXPECT_FALSE(s != p);
  EXPECT_EQ(s, &*p);
  EXPECT_EQ(s, p.operator->());
  EXPECT_EQ(s, p.get());
  EXPECT_EQ(42, p->x);

  EXPECT_EQ(s, p.take());
  EXPECT_FALSE((bool)p);
  EXPECT_TRUE(!p);
  p = s;
  EXPECT_TRUE((bool)p);
  EXPECT_FALSE(!p);
  EXPECT_EQ(s, &*p);
  EXPECT_EQ(s, p.operator->());
  EXPECT_EQ(s, p.get());
  EXPECT_EQ(42, p->x);

  polymorphic_ptr<S> p2((llvm_move(p)));
  EXPECT_FALSE((bool)p);
  EXPECT_TRUE(!p);
  EXPECT_TRUE((bool)p2);
  EXPECT_FALSE(!p2);
  EXPECT_EQ(s, &*p2);

  using std::swap;
  swap(p, p2);
  EXPECT_TRUE((bool)p);
  EXPECT_FALSE(!p);
  EXPECT_EQ(s, &*p);
  EXPECT_FALSE((bool)p2);
  EXPECT_TRUE(!p2);

  // Force copies and that everything survives.
  polymorphic_ptr<S> p3 = dummy_copy(polymorphic_ptr<S>(p));
  EXPECT_TRUE((bool)p3);
  EXPECT_FALSE(!p3);
  EXPECT_NE(s, &*p3);
  EXPECT_EQ(42, p3->x);
}

}
