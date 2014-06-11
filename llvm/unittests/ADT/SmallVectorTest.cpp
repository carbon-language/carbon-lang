//===- llvm/unittest/ADT/SmallVectorTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallVector unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "gtest/gtest.h"
#include <list>
#include <stdarg.h>

using namespace llvm;

namespace {

/// A helper class that counts the total number of constructor and
/// destructor calls.
class Constructable {
private:
  static int numConstructorCalls;
  static int numMoveConstructorCalls;
  static int numCopyConstructorCalls;
  static int numDestructorCalls;
  static int numAssignmentCalls;
  static int numMoveAssignmentCalls;
  static int numCopyAssignmentCalls;

  bool constructed;
  int value;

public:
  Constructable() : constructed(true), value(0) {
    ++numConstructorCalls;
  }

  Constructable(int val) : constructed(true), value(val) {
    ++numConstructorCalls;
  }

  Constructable(const Constructable & src) : constructed(true) {
    value = src.value;
    ++numConstructorCalls;
    ++numCopyConstructorCalls;
  }

  Constructable(Constructable && src) : constructed(true) {
    value = src.value;
    ++numConstructorCalls;
    ++numMoveConstructorCalls;
  }

  ~Constructable() {
    EXPECT_TRUE(constructed);
    ++numDestructorCalls;
    constructed = false;
  }

  Constructable & operator=(const Constructable & src) {
    EXPECT_TRUE(constructed);
    value = src.value;
    ++numAssignmentCalls;
    ++numCopyAssignmentCalls;
    return *this;
  }

  Constructable & operator=(Constructable && src) {
    EXPECT_TRUE(constructed);
    value = src.value;
    ++numAssignmentCalls;
    ++numMoveAssignmentCalls;
    return *this;
  }

  int getValue() const {
    return abs(value);
  }

  static void reset() {
    numConstructorCalls = 0;
    numMoveConstructorCalls = 0;
    numCopyConstructorCalls = 0;
    numDestructorCalls = 0;
    numAssignmentCalls = 0;
    numMoveAssignmentCalls = 0;
    numCopyAssignmentCalls = 0;
  }

  static int getNumConstructorCalls() {
    return numConstructorCalls;
  }

  static int getNumMoveConstructorCalls() {
    return numMoveConstructorCalls;
  }

  static int getNumCopyConstructorCalls() {
    return numCopyConstructorCalls;
  }

  static int getNumDestructorCalls() {
    return numDestructorCalls;
  }

  static int getNumAssignmentCalls() {
    return numAssignmentCalls;
  }

  static int getNumMoveAssignmentCalls() {
    return numMoveAssignmentCalls;
  }

  static int getNumCopyAssignmentCalls() {
    return numCopyAssignmentCalls;
  }

  friend bool operator==(const Constructable & c0, const Constructable & c1) {
    return c0.getValue() == c1.getValue();
  }

  friend bool LLVM_ATTRIBUTE_UNUSED
  operator!=(const Constructable & c0, const Constructable & c1) {
    return c0.getValue() != c1.getValue();
  }
};

int Constructable::numConstructorCalls;
int Constructable::numCopyConstructorCalls;
int Constructable::numMoveConstructorCalls;
int Constructable::numDestructorCalls;
int Constructable::numAssignmentCalls;
int Constructable::numCopyAssignmentCalls;
int Constructable::numMoveAssignmentCalls;

struct NonCopyable {
  NonCopyable() {}
  NonCopyable(NonCopyable &&) {}
  NonCopyable &operator=(NonCopyable &&) { return *this; }
private:
  NonCopyable(const NonCopyable &) LLVM_DELETED_FUNCTION;
  NonCopyable &operator=(const NonCopyable &) LLVM_DELETED_FUNCTION;
};

LLVM_ATTRIBUTE_USED void CompileTest() {
  SmallVector<NonCopyable, 0> V;
  V.resize(42);
}

// Test fixture class
template <typename VectorT>
class SmallVectorTest : public testing::Test {
protected:
  VectorT theVector;
  VectorT otherVector;

  void SetUp() {
    Constructable::reset();
  }

  void assertEmpty(VectorT & v) {
    // Size tests
    EXPECT_EQ(0u, v.size());
    EXPECT_TRUE(v.empty());

    // Iterator tests
    EXPECT_TRUE(v.begin() == v.end());
  }

  // Assert that theVector contains the specified values, in order.
  void assertValuesInOrder(VectorT & v, size_t size, ...) {
    EXPECT_EQ(size, v.size());

    va_list ap;
    va_start(ap, size);
    for (size_t i = 0; i < size; ++i) {
      int value = va_arg(ap, int);
      EXPECT_EQ(value, v[i].getValue());
    }

    va_end(ap);
  }

  // Generate a sequence of values to initialize the vector.
  void makeSequence(VectorT & v, int start, int end) {
    for (int i = start; i <= end; ++i) {
      v.push_back(Constructable(i));
    }
  }
};

typedef ::testing::Types<SmallVector<Constructable, 0>,
                         SmallVector<Constructable, 1>,
                         SmallVector<Constructable, 2>,
                         SmallVector<Constructable, 4>,
                         SmallVector<Constructable, 5>
                         > SmallVectorTestTypes;
TYPED_TEST_CASE(SmallVectorTest, SmallVectorTestTypes);

// New vector test.
TYPED_TEST(SmallVectorTest, EmptyVectorTest) {
  SCOPED_TRACE("EmptyVectorTest");
  this->assertEmpty(this->theVector);
  EXPECT_TRUE(this->theVector.rbegin() == this->theVector.rend());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
}

// Simple insertions and deletions.
TYPED_TEST(SmallVectorTest, PushPopTest) {
  SCOPED_TRACE("PushPopTest");

  // Track whether the vector will potentially have to grow.
  bool RequiresGrowth = this->theVector.capacity() < 3;

  // Push an element
  this->theVector.push_back(Constructable(1));

  // Size tests
  this->assertValuesInOrder(this->theVector, 1u, 1);
  EXPECT_FALSE(this->theVector.begin() == this->theVector.end());
  EXPECT_FALSE(this->theVector.empty());

  // Push another element
  this->theVector.push_back(Constructable(2));
  this->assertValuesInOrder(this->theVector, 2u, 1, 2);

  // Insert at beginning
  this->theVector.insert(this->theVector.begin(), this->theVector[1]);
  this->assertValuesInOrder(this->theVector, 3u, 2, 1, 2);

  // Pop one element
  this->theVector.pop_back();
  this->assertValuesInOrder(this->theVector, 2u, 2, 1);

  // Pop remaining elements
  this->theVector.pop_back();
  this->theVector.pop_back();
  this->assertEmpty(this->theVector);

  // Check number of constructor calls. Should be 2 for each list element,
  // one for the argument to push_back, one for the argument to insert,
  // and one for the list element itself.
  if (!RequiresGrowth) {
    EXPECT_EQ(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(5, Constructable::getNumDestructorCalls());
  } else {
    // If we had to grow the vector, these only have a lower bound, but should
    // always be equal.
    EXPECT_LE(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(Constructable::getNumConstructorCalls(),
              Constructable::getNumDestructorCalls());
  }
}

// Clear test.
TYPED_TEST(SmallVectorTest, ClearTest) {
  SCOPED_TRACE("ClearTest");

  this->theVector.reserve(2);
  this->makeSequence(this->theVector, 1, 2);
  this->theVector.clear();

  this->assertEmpty(this->theVector);
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

// Resize smaller test.
TYPED_TEST(SmallVectorTest, ResizeShrinkTest) {
  SCOPED_TRACE("ResizeShrinkTest");

  this->theVector.reserve(3);
  this->makeSequence(this->theVector, 1, 3);
  this->theVector.resize(1);

  this->assertValuesInOrder(this->theVector, 1u, 1);
  EXPECT_EQ(6, Constructable::getNumConstructorCalls());
  EXPECT_EQ(5, Constructable::getNumDestructorCalls());
}

// Resize bigger test.
TYPED_TEST(SmallVectorTest, ResizeGrowTest) {
  SCOPED_TRACE("ResizeGrowTest");

  this->theVector.resize(2);

  EXPECT_EQ(2, Constructable::getNumConstructorCalls());
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
  EXPECT_EQ(2u, this->theVector.size());
}

TYPED_TEST(SmallVectorTest, ResizeWithElementsTest) {
  this->theVector.resize(2);

  Constructable::reset();

  this->theVector.resize(4);

  size_t Ctors = Constructable::getNumConstructorCalls();
  EXPECT_TRUE(Ctors == 2 || Ctors == 4);
  size_t MoveCtors = Constructable::getNumMoveConstructorCalls();
  EXPECT_TRUE(MoveCtors == 0 || MoveCtors == 2);
  size_t Dtors = Constructable::getNumDestructorCalls();
  EXPECT_TRUE(Dtors == 0 || Dtors == 2);
}

// Resize with fill value.
TYPED_TEST(SmallVectorTest, ResizeFillTest) {
  SCOPED_TRACE("ResizeFillTest");

  this->theVector.resize(3, Constructable(77));
  this->assertValuesInOrder(this->theVector, 3u, 77, 77, 77);
}

// Overflow past fixed size.
TYPED_TEST(SmallVectorTest, OverflowTest) {
  SCOPED_TRACE("OverflowTest");

  // Push more elements than the fixed size.
  this->makeSequence(this->theVector, 1, 10);

  // Test size and values.
  EXPECT_EQ(10u, this->theVector.size());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(i+1, this->theVector[i].getValue());
  }

  // Now resize back to fixed size.
  this->theVector.resize(1);

  this->assertValuesInOrder(this->theVector, 1u, 1);
}

// Iteration tests.
TYPED_TEST(SmallVectorTest, IterationTest) {
  this->makeSequence(this->theVector, 1, 2);

  // Forward Iteration
  typename TypeParam::iterator it = this->theVector.begin();
  EXPECT_TRUE(*it == this->theVector.front());
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());
  ++it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_TRUE(*it == this->theVector.back());
  EXPECT_EQ(2, it->getValue());
  ++it;
  EXPECT_TRUE(it == this->theVector.end());
  --it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_EQ(2, it->getValue());
  --it;
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());

  // Reverse Iteration
  typename TypeParam::reverse_iterator rit = this->theVector.rbegin();
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
  ++rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  ++rit;
  EXPECT_TRUE(rit == this->theVector.rend());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
}

// Swap test.
TYPED_TEST(SmallVectorTest, SwapTest) {
  SCOPED_TRACE("SwapTest");

  this->makeSequence(this->theVector, 1, 2);
  std::swap(this->theVector, this->otherVector);

  this->assertEmpty(this->theVector);
  this->assertValuesInOrder(this->otherVector, 2u, 1, 2);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
  SCOPED_TRACE("AppendTest");

  this->makeSequence(this->otherVector, 2, 3);

  this->theVector.push_back(Constructable(1));
  this->theVector.append(this->otherVector.begin(), this->otherVector.end());

  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// Append repeated test
TYPED_TEST(SmallVectorTest, AppendRepeatedTest) {
  SCOPED_TRACE("AppendRepeatedTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.append(2, Constructable(77));
  this->assertValuesInOrder(this->theVector, 3u, 1, 77, 77);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignTest) {
  SCOPED_TRACE("AssignTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.assign(2, Constructable(77));
  this->assertValuesInOrder(this->theVector, 2u, 77, 77);
}

// Move-assign test
TYPED_TEST(SmallVectorTest, MoveAssignTest) {
  SCOPED_TRACE("MoveAssignTest");

  // Set up our vector with a single element, but enough capacity for 4.
  this->theVector.reserve(4);
  this->theVector.push_back(Constructable(1));
  
  // Set up the other vector with 2 elements.
  this->otherVector.push_back(Constructable(2));
  this->otherVector.push_back(Constructable(3));

  // Move-assign from the other vector.
  this->theVector = std::move(this->otherVector);

  // Make sure we have the right result.
  this->assertValuesInOrder(this->theVector, 2u, 2, 3);

  // Make sure the # of constructor/destructor calls line up. There
  // are two live objects after clearing the other vector.
  this->otherVector.clear();
  EXPECT_EQ(Constructable::getNumConstructorCalls()-2, 
            Constructable::getNumDestructorCalls());

  // There shouldn't be any live objects any more.
  this->theVector.clear();
  EXPECT_EQ(Constructable::getNumConstructorCalls(), 
            Constructable::getNumDestructorCalls());
}

// Erase a single element
TYPED_TEST(SmallVectorTest, EraseTest) {
  SCOPED_TRACE("EraseTest");

  this->makeSequence(this->theVector, 1, 3);
  this->theVector.erase(this->theVector.begin());
  this->assertValuesInOrder(this->theVector, 2u, 2, 3);
}

// Erase a range of elements
TYPED_TEST(SmallVectorTest, EraseRangeTest) {
  SCOPED_TRACE("EraseRangeTest");

  this->makeSequence(this->theVector, 1, 3);
  this->theVector.erase(this->theVector.begin(), this->theVector.begin() + 2);
  this->assertValuesInOrder(this->theVector, 1u, 3);
}

// Insert a single element.
TYPED_TEST(SmallVectorTest, InsertTest) {
  SCOPED_TRACE("InsertTest");

  this->makeSequence(this->theVector, 1, 3);
  typename TypeParam::iterator I =
    this->theVector.insert(this->theVector.begin() + 1, Constructable(77));
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);
}

// Insert a copy of a single element.
TYPED_TEST(SmallVectorTest, InsertCopy) {
  SCOPED_TRACE("InsertTest");

  this->makeSequence(this->theVector, 1, 3);
  Constructable C(77);
  typename TypeParam::iterator I =
      this->theVector.insert(this->theVector.begin() + 1, C);
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);
}

// Insert repeated elements.
TYPED_TEST(SmallVectorTest, InsertRepeatedTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 1, 4);
  Constructable::reset();
  auto I =
      this->theVector.insert(this->theVector.begin() + 1, 2, Constructable(16));
  // Move construct the top element into newly allocated space, and optionally
  // reallocate the whole buffer, move constructing into it.
  // FIXME: This is inefficient, we shouldn't move things into newly allocated
  // space, then move them up/around, there should only be 2 or 4 move
  // constructions here.
  EXPECT_TRUE(Constructable::getNumMoveConstructorCalls() == 2 ||
              Constructable::getNumMoveConstructorCalls() == 6);
  // Move assign the next two to shift them up and make a gap.
  EXPECT_EQ(1, Constructable::getNumMoveAssignmentCalls());
  // Copy construct the two new elements from the parameter.
  EXPECT_EQ(2, Constructable::getNumCopyAssignmentCalls());
  // All without any copy construction.
  EXPECT_EQ(0, Constructable::getNumCopyConstructorCalls());
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 16, 16, 2, 3, 4);
}


TYPED_TEST(SmallVectorTest, InsertRepeatedAtEndTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 1, 4);
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.end(), 2, Constructable(16));
  // Just copy construct them into newly allocated space
  EXPECT_EQ(2, Constructable::getNumCopyConstructorCalls());
  // Move everything across if reallocation is needed.
  EXPECT_TRUE(Constructable::getNumMoveConstructorCalls() == 0 ||
              Constructable::getNumMoveConstructorCalls() == 4);
  // Without ever moving or copying anything else.
  EXPECT_EQ(0, Constructable::getNumCopyAssignmentCalls());
  EXPECT_EQ(0, Constructable::getNumMoveAssignmentCalls());

  EXPECT_EQ(this->theVector.begin() + 4, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 2, 3, 4, 16, 16);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedEmptyTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 10, 15);

  // Empty insert.
  EXPECT_EQ(this->theVector.end(),
            this->theVector.insert(this->theVector.end(),
                                   0, Constructable(42)));
  EXPECT_EQ(this->theVector.begin() + 1,
            this->theVector.insert(this->theVector.begin() + 1,
                                   0, Constructable(42)));
}

// Insert range.
TYPED_TEST(SmallVectorTest, InsertRangeTest) {
  SCOPED_TRACE("InsertRangeTest");

  Constructable Arr[3] =
    { Constructable(77), Constructable(77), Constructable(77) };

  this->makeSequence(this->theVector, 1, 3);
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.begin() + 1, Arr, Arr + 3);
  // Move construct the top 3 elements into newly allocated space.
  // Possibly move the whole sequence into new space first.
  // FIXME: This is inefficient, we shouldn't move things into newly allocated
  // space, then move them up/around, there should only be 2 or 3 move
  // constructions here.
  EXPECT_TRUE(Constructable::getNumMoveConstructorCalls() == 2 ||
              Constructable::getNumMoveConstructorCalls() == 5);
  // Copy assign the lower 2 new elements into existing space.
  EXPECT_EQ(2, Constructable::getNumCopyAssignmentCalls());
  // Copy construct the third element into newly allocated space.
  EXPECT_EQ(1, Constructable::getNumCopyConstructorCalls());
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 77, 77, 77, 2, 3);
}


TYPED_TEST(SmallVectorTest, InsertRangeAtEndTest) {
  SCOPED_TRACE("InsertRangeTest");

  Constructable Arr[3] =
    { Constructable(77), Constructable(77), Constructable(77) };

  this->makeSequence(this->theVector, 1, 3);

  // Insert at end.
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.end(), Arr, Arr+3);
  // Copy construct the 3 elements into new space at the top.
  EXPECT_EQ(3, Constructable::getNumCopyConstructorCalls());
  // Don't copy/move anything else.
  EXPECT_EQ(0, Constructable::getNumCopyAssignmentCalls());
  // Reallocation might occur, causing all elements to be moved into the new
  // buffer.
  EXPECT_TRUE(Constructable::getNumMoveConstructorCalls() == 0 ||
              Constructable::getNumMoveConstructorCalls() == 3);
  EXPECT_EQ(0, Constructable::getNumMoveAssignmentCalls());
  EXPECT_EQ(this->theVector.begin() + 3, I);
  this->assertValuesInOrder(this->theVector, 6u,
                            1, 2, 3, 77, 77, 77);
}

TYPED_TEST(SmallVectorTest, InsertEmptyRangeTest) {
  SCOPED_TRACE("InsertRangeTest");

  this->makeSequence(this->theVector, 1, 3);

  // Empty insert.
  EXPECT_EQ(this->theVector.end(),
            this->theVector.insert(this->theVector.end(),
                                   this->theVector.begin(),
                                   this->theVector.begin()));
  EXPECT_EQ(this->theVector.begin() + 1,
            this->theVector.insert(this->theVector.begin() + 1,
                                   this->theVector.begin(),
                                   this->theVector.begin()));
}

// Comparison tests.
TYPED_TEST(SmallVectorTest, ComparisonTest) {
  SCOPED_TRACE("ComparisonTest");

  this->makeSequence(this->theVector, 1, 3);
  this->makeSequence(this->otherVector, 1, 3);

  EXPECT_TRUE(this->theVector == this->otherVector);
  EXPECT_FALSE(this->theVector != this->otherVector);

  this->otherVector.clear();
  this->makeSequence(this->otherVector, 2, 4);

  EXPECT_FALSE(this->theVector == this->otherVector);
  EXPECT_TRUE(this->theVector != this->otherVector);
}

// Constant vector tests.
TYPED_TEST(SmallVectorTest, ConstVectorTest) {
  const TypeParam constVector;

  EXPECT_EQ(0u, constVector.size());
  EXPECT_TRUE(constVector.empty());
  EXPECT_TRUE(constVector.begin() == constVector.end());
}

// Direct array access.
TYPED_TEST(SmallVectorTest, DirectVectorTest) {
  EXPECT_EQ(0u, this->theVector.size());
  this->theVector.reserve(4);
  EXPECT_LE(4u, this->theVector.capacity());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  this->theVector.push_back(1);
  this->theVector.push_back(2);
  this->theVector.push_back(3);
  this->theVector.push_back(4);
  EXPECT_EQ(4u, this->theVector.size());
  EXPECT_EQ(8, Constructable::getNumConstructorCalls());
  EXPECT_EQ(1, this->theVector[0].getValue());
  EXPECT_EQ(2, this->theVector[1].getValue());
  EXPECT_EQ(3, this->theVector[2].getValue());
  EXPECT_EQ(4, this->theVector[3].getValue());
}

TYPED_TEST(SmallVectorTest, IteratorTest) {
  std::list<int> L;
  this->theVector.insert(this->theVector.end(), L.begin(), L.end());
}

struct notassignable {
  int &x;
  notassignable(int &x) : x(x) {}
};

TEST(SmallVectorCustomTest, NoAssignTest) {
  int x = 0;
  SmallVector<notassignable, 2> vec;
  vec.push_back(notassignable(x));
  x = 42;
  EXPECT_EQ(42, vec.pop_back_val().x);
}

struct MovedFrom {
  bool hasValue;
  MovedFrom() : hasValue(true) {
  }
  MovedFrom(MovedFrom&& m) : hasValue(m.hasValue) {
    m.hasValue = false;
  }
  MovedFrom &operator=(MovedFrom&& m) {
    hasValue = m.hasValue;
    m.hasValue = false;
    return *this;
  }
};

TEST(SmallVectorTest, MidInsert) {
  SmallVector<MovedFrom, 3> v;
  v.push_back(MovedFrom());
  v.insert(v.begin(), MovedFrom());
  for (MovedFrom &m : v)
    EXPECT_TRUE(m.hasValue);
}

}
