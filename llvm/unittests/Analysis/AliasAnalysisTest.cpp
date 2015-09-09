//===--- AliasAnalysisTest.cpp - Mixed TBAA unit tests --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class AliasAnalysisTest : public testing::Test {
protected:
  LLVMContext C;
  Module M;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<BasicAAResult> BAR;
  std::unique_ptr<AAResults> AAR;

  AliasAnalysisTest() : M("AliasAnalysisTest", C), TLI(TLII) {}

  AAResults &getAAResults(Function &F) {
    // Reset the Function AA results first to clear out any references.
    AAR.reset(new AAResults());

    // Build the various AA results and register them.
    AC.reset(new AssumptionCache(F));
    BAR.reset(new BasicAAResult(M.getDataLayout(), TLI, *AC));
    AAR->addAAResult(*BAR);

    return *AAR;
  }
};

TEST_F(AliasAnalysisTest, getModRefInfo) {
  // Setup function.
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), std::vector<Type *>(), false);
  auto *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  auto *BB = BasicBlock::Create(C, "entry", F);
  auto IntType = Type::getInt32Ty(C);
  auto PtrType = Type::getInt32PtrTy(C);
  auto *Value = ConstantInt::get(IntType, 42);
  auto *Addr = ConstantPointerNull::get(PtrType);

  auto *Store1 = new StoreInst(Value, Addr, BB);
  auto *Load1 = new LoadInst(Addr, "load", BB);
  auto *Add1 = BinaryOperator::CreateAdd(Value, Value, "add", BB);
  auto *VAArg1 = new VAArgInst(Addr, PtrType, "vaarg", BB);
  auto *CmpXChg1 = new AtomicCmpXchgInst(Addr, ConstantInt::get(IntType, 0),
                                         ConstantInt::get(IntType, 1),
                                         Monotonic, Monotonic, CrossThread, BB);
  auto *AtomicRMW =
      new AtomicRMWInst(AtomicRMWInst::Xchg, Addr, ConstantInt::get(IntType, 1),
                        Monotonic, CrossThread, BB);

  ReturnInst::Create(C, nullptr, BB);

  auto &AA = getAAResults(*F);

  // Check basic results
  EXPECT_EQ(AA.getModRefInfo(Store1, MemoryLocation()), MRI_Mod);
  EXPECT_EQ(AA.getModRefInfo(Store1), MRI_Mod);
  EXPECT_EQ(AA.getModRefInfo(Load1, MemoryLocation()), MRI_Ref);
  EXPECT_EQ(AA.getModRefInfo(Load1), MRI_Ref);
  EXPECT_EQ(AA.getModRefInfo(Add1, MemoryLocation()), MRI_NoModRef);
  EXPECT_EQ(AA.getModRefInfo(Add1), MRI_NoModRef);
  EXPECT_EQ(AA.getModRefInfo(VAArg1, MemoryLocation()), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(VAArg1), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(CmpXChg1, MemoryLocation()), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(CmpXChg1), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(AtomicRMW, MemoryLocation()), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(AtomicRMW), MRI_ModRef);
}

} // end anonymous namspace
} // end llvm namespace
