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
template <typename T>
T dummy_copy(const T &arg) { return arg; }

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

  // Force copies of null without trying to dereference anything.
  polymorphic_ptr<S> null_copy = dummy_copy(polymorphic_ptr<S>(null));
  EXPECT_FALSE((bool)null_copy);
  EXPECT_TRUE(!null_copy);
  EXPECT_EQ(null, null_copy);
}

struct Base {
  virtual ~Base() {}
  virtual Base *clone() = 0;
  virtual StringRef name() { return "Base"; }
};

struct DerivedA : Base {
  virtual DerivedA *clone() { return new DerivedA(); }
  virtual StringRef name() { return "DerivedA"; }
};
struct DerivedB : Base {
  virtual DerivedB *clone() { return new DerivedB(); }
  virtual StringRef name() { return "DerivedB"; }
};

TEST(polymorphic_ptr_test, Polymorphism) {
  polymorphic_ptr<Base> a(new DerivedA());
  polymorphic_ptr<Base> b(new DerivedB());

  EXPECT_EQ("DerivedA", a->name());
  EXPECT_EQ("DerivedB", b->name());

  polymorphic_ptr<Base> copy = dummy_copy(a);
  EXPECT_NE(a, copy);
  EXPECT_EQ("DerivedA", copy->name());

  copy = dummy_copy(b);
  EXPECT_NE(b, copy);
  EXPECT_EQ("DerivedB", copy->name());

  // Test creating a copy out of a temporary directly.
  copy = dummy_copy<polymorphic_ptr<Base> >(new DerivedA());
  EXPECT_NE(a, copy);
  EXPECT_EQ("DerivedA", copy->name());
}

}
