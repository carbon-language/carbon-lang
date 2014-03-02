//===- llvm/unittest/ADT/OwningPtrTest.cpp - OwningPtr unit tests -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

struct TrackDestructor {
  static unsigned Destructions;
  int val;
  explicit TrackDestructor(int val) : val(val) {}
  ~TrackDestructor() { ++Destructions; }
  static void ResetCounts() { Destructions = 0; }

private:
  TrackDestructor(const TrackDestructor &other) LLVM_DELETED_FUNCTION;
  TrackDestructor &
  operator=(const TrackDestructor &other) LLVM_DELETED_FUNCTION;
  TrackDestructor(TrackDestructor &&other) LLVM_DELETED_FUNCTION;
  TrackDestructor &operator=(TrackDestructor &&other) LLVM_DELETED_FUNCTION;
};

unsigned TrackDestructor::Destructions = 0;

// Test fixture
class OwningPtrTest : public testing::Test {};

TEST_F(OwningPtrTest, DefaultConstruction) {
  TrackDestructor::ResetCounts();
  {
    OwningPtr<TrackDestructor> O;
    EXPECT_FALSE(O);
    EXPECT_TRUE(!O);
    EXPECT_FALSE(O.get());
    EXPECT_FALSE(O.isValid());
  }
  EXPECT_EQ(0u, TrackDestructor::Destructions);
}

TEST_F(OwningPtrTest, PtrConstruction) {
  TrackDestructor::ResetCounts();
  {
    OwningPtr<TrackDestructor> O(new TrackDestructor(3));
    EXPECT_TRUE((bool)O);
    EXPECT_FALSE(!O);
    EXPECT_TRUE(O.get());
    EXPECT_TRUE(O.isValid());
    EXPECT_EQ(3, (*O).val);
    EXPECT_EQ(3, O->val);
    EXPECT_EQ(0u, TrackDestructor::Destructions);
  }
  EXPECT_EQ(1u, TrackDestructor::Destructions);
}

TEST_F(OwningPtrTest, Reset) {
  TrackDestructor::ResetCounts();
  OwningPtr<TrackDestructor> O(new TrackDestructor(3));
  EXPECT_EQ(0u, TrackDestructor::Destructions);
  O.reset();
  EXPECT_FALSE((bool)O);
  EXPECT_TRUE(!O);
  EXPECT_FALSE(O.get());
  EXPECT_FALSE(O.isValid());
  EXPECT_EQ(1u, TrackDestructor::Destructions);
}

TEST_F(OwningPtrTest, Take) {
  TrackDestructor::ResetCounts();
  TrackDestructor *T = 0;
  {
    OwningPtr<TrackDestructor> O(new TrackDestructor(3));
    T = O.take();
    EXPECT_FALSE((bool)O);
    EXPECT_TRUE(!O);
    EXPECT_FALSE(O.get());
    EXPECT_FALSE(O.isValid());
    EXPECT_TRUE(T);
    EXPECT_EQ(3, T->val);
    EXPECT_EQ(0u, TrackDestructor::Destructions);
  }
  delete T;
  EXPECT_EQ(1u, TrackDestructor::Destructions);
}

TEST_F(OwningPtrTest, MoveConstruction) {
  TrackDestructor::ResetCounts();
  {
    OwningPtr<TrackDestructor> A(new TrackDestructor(3));
    OwningPtr<TrackDestructor> B = std::move(A);
    EXPECT_FALSE((bool)A);
    EXPECT_TRUE(!A);
    EXPECT_FALSE(A.get());
    EXPECT_FALSE(A.isValid());
    EXPECT_TRUE((bool)B);
    EXPECT_FALSE(!B);
    EXPECT_TRUE(B.get());
    EXPECT_TRUE(B.isValid());
    EXPECT_EQ(3, (*B).val);
    EXPECT_EQ(3, B->val);
    EXPECT_EQ(0u, TrackDestructor::Destructions);
  }
  EXPECT_EQ(1u, TrackDestructor::Destructions);
}

TEST_F(OwningPtrTest, MoveAssignment) {
  TrackDestructor::ResetCounts();
  {
    OwningPtr<TrackDestructor> A(new TrackDestructor(3));
    OwningPtr<TrackDestructor> B(new TrackDestructor(4));
    B = std::move(A);
    EXPECT_FALSE(A);
    EXPECT_TRUE(!A);
    EXPECT_FALSE(A.get());
    EXPECT_FALSE(A.isValid());
    EXPECT_TRUE((bool)B);
    EXPECT_FALSE(!B);
    EXPECT_TRUE(B.get());
    EXPECT_TRUE(B.isValid());
    EXPECT_EQ(3, (*B).val);
    EXPECT_EQ(3, B->val);
    EXPECT_EQ(1u, TrackDestructor::Destructions);
  }
  EXPECT_EQ(2u, TrackDestructor::Destructions);
}

TEST_F(OwningPtrTest, Swap) {
  TrackDestructor::ResetCounts();
  {
    OwningPtr<TrackDestructor> A(new TrackDestructor(3));
    OwningPtr<TrackDestructor> B(new TrackDestructor(4));
    B.swap(A);
    EXPECT_TRUE((bool)A);
    EXPECT_FALSE(!A);
    EXPECT_TRUE(A.get());
    EXPECT_TRUE(A.isValid());
    EXPECT_EQ(4, (*A).val);
    EXPECT_EQ(4, A->val);
    EXPECT_TRUE((bool)B);
    EXPECT_FALSE(!B);
    EXPECT_TRUE(B.get());
    EXPECT_TRUE(B.isValid());
    EXPECT_EQ(3, (*B).val);
    EXPECT_EQ(3, B->val);
    EXPECT_EQ(0u, TrackDestructor::Destructions);
  }
  EXPECT_EQ(2u, TrackDestructor::Destructions);
  TrackDestructor::ResetCounts();
  {
    OwningPtr<TrackDestructor> A(new TrackDestructor(3));
    OwningPtr<TrackDestructor> B(new TrackDestructor(4));
    swap(A, B);
    EXPECT_TRUE((bool)A);
    EXPECT_FALSE(!A);
    EXPECT_TRUE(A.get());
    EXPECT_TRUE(A.isValid());
    EXPECT_EQ(4, (*A).val);
    EXPECT_EQ(4, A->val);
    EXPECT_TRUE((bool)B);
    EXPECT_FALSE(!B);
    EXPECT_TRUE(B.get());
    EXPECT_TRUE(B.isValid());
    EXPECT_EQ(3, (*B).val);
    EXPECT_EQ(3, B->val);
    EXPECT_EQ(0u, TrackDestructor::Destructions);
  }
  EXPECT_EQ(2u, TrackDestructor::Destructions);
}

}
