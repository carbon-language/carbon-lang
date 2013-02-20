//===- llvm/unittest/ADT/OptionalTest.cpp - Optional unit tests -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/Optional.h"
using namespace llvm;

namespace {

struct NonDefaultConstructible {
  static unsigned CopyConstructions;
  static unsigned Destructions;
  static unsigned CopyAssignments;
  explicit NonDefaultConstructible(int) {
  }
  NonDefaultConstructible(const NonDefaultConstructible&) {
    ++CopyConstructions;
  }
  NonDefaultConstructible &operator=(const NonDefaultConstructible&) {
    ++CopyAssignments;
    return *this;
  }
  ~NonDefaultConstructible() {
    ++Destructions;
  }
  static void ResetCounts() {
    CopyConstructions = 0;
    Destructions = 0;
    CopyAssignments = 0;
  }
};

unsigned NonDefaultConstructible::CopyConstructions = 0;
unsigned NonDefaultConstructible::Destructions = 0;
unsigned NonDefaultConstructible::CopyAssignments = 0;

// Test fixture
class OptionalTest : public testing::Test {
};

TEST_F(OptionalTest, NonDefaultConstructibleTest) {
  Optional<NonDefaultConstructible> O;
  EXPECT_FALSE(O);
}

TEST_F(OptionalTest, ResetTest) {
  NonDefaultConstructible::ResetCounts();
  Optional<NonDefaultConstructible> O(NonDefaultConstructible(3));
  EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
  NonDefaultConstructible::ResetCounts();
  O.Reset();
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, InitializationLeakTest) {
  NonDefaultConstructible::ResetCounts();
  Optional<NonDefaultConstructible>(NonDefaultConstructible(3));
  EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, CopyConstructionTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A(NonDefaultConstructible(3));
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    Optional<NonDefaultConstructible> B(A);
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, ConstructingCopyAssignmentTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A(NonDefaultConstructible(3));
    Optional<NonDefaultConstructible> B;
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, CopyingCopyAssignmentTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A(NonDefaultConstructible(3));
    Optional<NonDefaultConstructible> B(NonDefaultConstructible(4));
    EXPECT_EQ(2u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(1u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, DeletingCopyAssignmentTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A;
    Optional<NonDefaultConstructible> B(NonDefaultConstructible(3));
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, NullCopyConstructionTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A;
    Optional<NonDefaultConstructible> B;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
}

} // end anonymous namespace

