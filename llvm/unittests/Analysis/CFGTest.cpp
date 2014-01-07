//===- CFGTest.cpp - CFG tests --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFG.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// This fixture assists in running the isPotentiallyReachable utility four ways
// and ensuring it produces the correct answer each time.
class IsPotentiallyReachableTest : public testing::Test {
protected:
  void ParseAssembly(const char *Assembly) {
    M.reset(new Module("Module", getGlobalContext()));

    SMDiagnostic Error;
    bool Parsed = ParseAssemblyString(Assembly, M.get(),
                                      Error, M->getContext()) == M.get();

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    if (!Parsed) {
      // A failure here means that the test itself is buggy.
      report_fatal_error(os.str().c_str());
    }

    Function *F = M->getFunction("test");
    if (F == NULL)
      report_fatal_error("Test must have a function named @test");

    A = B = NULL;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (I->hasName()) {
        if (I->getName() == "A")
          A = &*I;
        else if (I->getName() == "B")
          B = &*I;
      }
    }
    if (A == NULL)
      report_fatal_error("@test must have an instruction %A");
    if (B == NULL)
      report_fatal_error("@test must have an instruction %B");
  }

  void ExpectPath(bool ExpectedResult) {
    static char ID;
    class IsPotentiallyReachableTestPass : public FunctionPass {
     public:
      IsPotentiallyReachableTestPass(bool ExpectedResult,
                                     Instruction *A, Instruction *B)
          : FunctionPass(ID), ExpectedResult(ExpectedResult), A(A), B(B) {}

      static int initialize() {
        PassInfo *PI = new PassInfo("isPotentiallyReachable testing pass",
                                    "", &ID, 0, true, true);
        PassRegistry::getPassRegistry()->registerPass(*PI, false);
        initializeLoopInfoPass(*PassRegistry::getPassRegistry());
        initializeDominatorTreePass(*PassRegistry::getPassRegistry());
        return 0;
      }

      void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.setPreservesAll();
        AU.addRequired<LoopInfo>();
        AU.addRequired<DominatorTree>();
      }

      bool runOnFunction(Function &F) {
        if (!F.hasName() || F.getName() != "test")
          return false;

        LoopInfo *LI = &getAnalysis<LoopInfo>();
        DominatorTree *DT = &getAnalysis<DominatorTree>();
        EXPECT_EQ(isPotentiallyReachable(A, B, 0, 0), ExpectedResult);
        EXPECT_EQ(isPotentiallyReachable(A, B, DT, 0), ExpectedResult);
        EXPECT_EQ(isPotentiallyReachable(A, B, 0, LI), ExpectedResult);
        EXPECT_EQ(isPotentiallyReachable(A, B, DT, LI), ExpectedResult);
        return false;
      }
      bool ExpectedResult;
      Instruction *A, *B;
    };

    static int initialize = IsPotentiallyReachableTestPass::initialize();
    (void)initialize;

    IsPotentiallyReachableTestPass *P =
        new IsPotentiallyReachableTestPass(ExpectedResult, A, B);
    PassManager PM;
    PM.add(P);
    PM.run(*M);
  }
private:
  OwningPtr<Module> M;
  Instruction *A, *B;
};

}

TEST_F(IsPotentiallyReachableTest, SameBlockNoPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  bitcast i8 undef to i8\n"
      "  %B = bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}\n");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, SameBlockPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  %B = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}\n");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, SameBlockNoLoop) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  br label %middle\n"
      "middle:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br label %nextblock\n"
      "nextblock:\n"
      "  ret void\n"
      "}\n");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, StraightNoPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %exit\n"
      "exit:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, StraightPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br label %exit\n"
      "exit:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, DestUnreachable) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  br label %midblock\n"
      "midblock:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "unreachable:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %midblock\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, BranchToReturn) {
  ParseAssembly(
      "define void @test(i1 %x) {\n"
      "entry:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br i1 %x, label %block1, label %block2\n"
      "block1:\n"
      "  ret void\n"
      "block2:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, SimpleLoop1) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, SimpleLoop2) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %loop\n"
      "loop:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, SimpleLoop3) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop, label %exit\n"
      "exit:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(false);
}


TEST_F(IsPotentiallyReachableTest, OneLoopAfterTheOther1) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop1\n"
      "loop1:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop1, label %loop1exit\n"
      "loop1exit:\n"
      "  br label %loop2\n"
      "loop2:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %x, label %loop2, label %loop2exit\n"
      "loop2exit:"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, OneLoopAfterTheOther2) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop1\n"
      "loop1:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop1, label %loop1exit\n"
      "loop1exit:\n"
      "  br label %loop2\n"
      "loop2:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %x, label %loop2, label %loop2exit\n"
      "loop2exit:"
      "  ret void\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, OneLoopAfterTheOtherInsideAThirdLoop) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %outerloop3\n"
      "outerloop3:\n"
      "  br label %innerloop1\n"
      "innerloop1:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %innerloop1, label %innerloop1exit\n"
      "innerloop1exit:\n"
      "  br label %innerloop2\n"
      "innerloop2:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %x, label %innerloop2, label %innerloop2exit\n"
      "innerloop2exit:"
      "  ;; In outer loop3 now.\n"
      "  %z = call i1 @switch()\n"
      "  br i1 %z, label %outerloop3, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, BranchInsideLoop) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %nextloopblock, label %exit\n"
      "nextloopblock:\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %y, label %left, label %right\n"
      "left:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br label %loop\n"
      "right:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %loop\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}
