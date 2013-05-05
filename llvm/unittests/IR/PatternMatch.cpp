//===---- llvm/unittest/IR/PatternMatch.cpp - PatternMatch unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/NoFolder.h"
#include "llvm/Support/PatternMatch.h"
#include "gtest/gtest.h"

using namespace llvm::PatternMatch;

namespace llvm {
namespace {

/// Ordered floating point minimum/maximum tests.

static void m_OrdFMin_expect_match_and_delete(Value *Cmp, Value *Select,
                                              Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR)).match(Select));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  delete Select;
  delete Cmp;
}

static void m_OrdFMin_expect_nomatch_and_delete(Value *Cmp, Value *Select,
                                                Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR)).match(Select));
  delete Select;
  delete Cmp;
}

static void m_OrdFMax_expect_match_and_delete(Value *Cmp, Value *Select,
                                              Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR)).match(Select));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  delete Select;
  delete Cmp;
}

static void m_OrdFMax_expect_nomatch_and_delete(Value *Cmp, Value *Select,
                                                Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR)).match(Select));
  delete Select;
  delete Cmp;
}



TEST(PatternMatchTest, FloatingPointOrderedMin) {
  LLVMContext &C(getGlobalContext());
  IRBuilder<true, NoFolder> Builder(C);

  Type *FltTy = Builder.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);

  // Test OLT.
  Value *Cmp = Builder.CreateFCmpOLT(L, R);
  Value *Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMin_expect_match_and_delete(Cmp, Select, L, R);

  // Test OLE.
  Cmp = Builder.CreateFCmpOLE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMin_expect_match_and_delete(Cmp, Select, L, R);

  // Test no match on OGE.
  Cmp = Builder.CreateFCmpOGE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMin_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test no match on OGT.
  Cmp = Builder.CreateFCmpOGT(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMin_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test match on OGE with inverted select.
  Cmp = Builder.CreateFCmpOGE(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_OrdFMin_expect_match_and_delete(Cmp, Select, L, R);

  // Test match on OGT with inverted select.
  Cmp = Builder.CreateFCmpOGT(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_OrdFMin_expect_match_and_delete(Cmp, Select, L, R);
}

TEST(PatternMatchTest, FloatingPointOrderedMax) {
  LLVMContext &C(getGlobalContext());
  IRBuilder<true, NoFolder> Builder(C);

  Type *FltTy = Builder.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);

  // Test OGT.
  Value *Cmp = Builder.CreateFCmpOGT(L, R);
  Value *Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMax_expect_match_and_delete(Cmp, Select, L, R);

  // Test OGE.
  Cmp = Builder.CreateFCmpOGE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMax_expect_match_and_delete(Cmp, Select, L, R);

  // Test no match on OLE.
  Cmp = Builder.CreateFCmpOLE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMax_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test no match on OLT.
  Cmp = Builder.CreateFCmpOLT(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_OrdFMax_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test match on OLE with inverted select.
  Cmp = Builder.CreateFCmpOLE(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_OrdFMax_expect_match_and_delete(Cmp, Select, L, R);

  // Test match on OLT with inverted select.
  Cmp = Builder.CreateFCmpOLT(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_OrdFMax_expect_match_and_delete(Cmp, Select, L, R);
}

/// Unordered floating point minimum/maximum tests.

static void m_UnordFMin_expect_match_and_delete(Value *Cmp, Value *Select,
                                              Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR)).match(Select));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  delete Select;
  delete Cmp;
}

static void m_UnordFMin_expect_nomatch_and_delete(Value *Cmp, Value *Select,
                                                Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR)).match(Select));
  delete Select;
  delete Cmp;
}

static void m_UnordFMax_expect_match_and_delete(Value *Cmp, Value *Select,
                                              Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR)).match(Select));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  delete Select;
  delete Cmp;
}

static void m_UnordFMax_expect_nomatch_and_delete(Value *Cmp, Value *Select,
                                                Value *L, Value *R) {
  Value *MatchL, *MatchR;
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR)).match(Select));
  delete Select;
  delete Cmp;
}

TEST(PatternMatchTest, FloatingPointUnorderedMin) {
  LLVMContext &C(getGlobalContext());
  IRBuilder<true, NoFolder> Builder(C);

  Type *FltTy = Builder.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);

  // Test ULT.
  Value *Cmp = Builder.CreateFCmpULT(L, R);
  Value *Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMin_expect_match_and_delete(Cmp, Select, L, R);

  // Test ULE.
  Cmp = Builder.CreateFCmpULE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMin_expect_match_and_delete(Cmp, Select, L, R);

  // Test no match on UGE.
  Cmp = Builder.CreateFCmpUGE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMin_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test no match on UGT.
  Cmp = Builder.CreateFCmpUGT(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMin_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test match on UGE with inverted select.
  Cmp = Builder.CreateFCmpUGE(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_UnordFMin_expect_match_and_delete(Cmp, Select, L, R);

  // Test match on UGT with inverted select.
  Cmp = Builder.CreateFCmpUGT(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_UnordFMin_expect_match_and_delete(Cmp, Select, L, R);
}

TEST(PatternMatchTest, FloatingPointUnorderedMax) {
  LLVMContext &C(getGlobalContext());
  IRBuilder<true, NoFolder> Builder(C);

  Type *FltTy = Builder.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);

  // Test UGT.
  Value *Cmp = Builder.CreateFCmpUGT(L, R);
  Value *Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMax_expect_match_and_delete(Cmp, Select, L, R);

  // Test UGE.
  Cmp = Builder.CreateFCmpUGE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMax_expect_match_and_delete(Cmp, Select, L, R);

  // Test no match on ULE.
  Cmp = Builder.CreateFCmpULE(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMax_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test no match on ULT.
  Cmp = Builder.CreateFCmpULT(L, R);
  Select = Builder.CreateSelect(Cmp, L, R);
  m_UnordFMax_expect_nomatch_and_delete(Cmp, Select, L, R);

  // Test match on ULE with inverted select.
  Cmp = Builder.CreateFCmpULE(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_UnordFMax_expect_match_and_delete(Cmp, Select, L, R);

  // Test match on ULT with inverted select.
  Cmp = Builder.CreateFCmpULT(L, R);
  Select = Builder.CreateSelect(Cmp, R, L);
  m_UnordFMax_expect_match_and_delete(Cmp, Select, L, R);
}

} // anonymous namespace.
} // llvm namespace.
