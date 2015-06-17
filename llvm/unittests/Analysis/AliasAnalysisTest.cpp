//===--- AliasAnalysisTest.cpp - Mixed TBAA unit tests --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class AliasAnalysisTest : public testing::Test {
protected:
  AliasAnalysisTest() : M("AliasAnalysisTBAATest", C) {}

  // This is going to check that calling getModRefInfo without a location, and
  // with a default location, first, doesn't crash, and second, gives the right
  // answer.
  void CheckModRef(Instruction *I, AliasAnalysis::ModRefResult Result) {
    static char ID;
    class CheckModRefTestPass : public FunctionPass {
    public:
      CheckModRefTestPass(Instruction *I, AliasAnalysis::ModRefResult Result)
          : FunctionPass(ID), ExpectResult(Result), I(I) {}
      static int initialize() {
        PassInfo *PI = new PassInfo("CheckModRef testing pass", "", &ID,
                                    nullptr, true, true);
        PassRegistry::getPassRegistry()->registerPass(*PI, false);
        initializeAliasAnalysisAnalysisGroup(*PassRegistry::getPassRegistry());
        initializeBasicAliasAnalysisPass(*PassRegistry::getPassRegistry());
        return 0;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
        AU.addRequiredTransitive<AliasAnalysis>();
      }
      bool runOnFunction(Function &) override {
        AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
        EXPECT_EQ(AA.getModRefInfo(I, MemoryLocation()), ExpectResult);
        EXPECT_EQ(AA.getModRefInfo(I), ExpectResult);
        return false;
      }
      AliasAnalysis::ModRefResult ExpectResult;
      Instruction *I;
    };
    static int initialize = CheckModRefTestPass::initialize();
    (void)initialize;
    CheckModRefTestPass *P = new CheckModRefTestPass(I, Result);
    legacy::PassManager PM;
    PM.add(createBasicAliasAnalysisPass());
    PM.add(P);
    PM.run(M);
  }

  LLVMContext C;
  Module M;
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

  // Check basic results
  CheckModRef(Store1, AliasAnalysis::ModRefResult::Mod);
  CheckModRef(Load1, AliasAnalysis::ModRefResult::Ref);
  CheckModRef(Add1, AliasAnalysis::ModRefResult::NoModRef);
  CheckModRef(VAArg1, AliasAnalysis::ModRefResult::ModRef);
  CheckModRef(CmpXChg1, AliasAnalysis::ModRefResult::ModRef);
  CheckModRef(AtomicRMW, AliasAnalysis::ModRefResult::ModRef);
}

} // end anonymous namspace
} // end llvm namespace
