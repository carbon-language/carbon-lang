//===- llvm/unittest/IR/PassManager.cpp - PassManager tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestAnalysisPass {
public:
  typedef Function IRUnitT;

  struct Result {
    Result(int Count) : InstructionCount(Count) {}
    bool invalidate(Function *) { return true; }
    int InstructionCount;
  };

  /// \brief Returns an opaque, unique ID for this pass type.
  static void *ID() { return (void *)&PassID; }

  /// \brief Run the analysis pass over the function and return a result.
  Result run(Function *F) {
    int Count = 0;
    for (Function::iterator BBI = F->begin(), BBE = F->end(); BBI != BBE; ++BBI)
      for (BasicBlock::iterator II = BBI->begin(), IE = BBI->end(); II != IE;
           ++II)
        ++Count;
    return Result(Count);
  }

private:
  /// \brief Private static data to provide unique ID.
  static char PassID;
};

char TestAnalysisPass::PassID;

struct TestModulePass {
  TestModulePass(int &RunCount) : RunCount(RunCount) {}

  bool run(Module *M) {
    ++RunCount;
    return true;
  }

  int &RunCount;
};

struct TestFunctionPass {
  TestFunctionPass(AnalysisManager &AM, int &RunCount, int &AnalyzedInstrCount)
      : AM(AM), RunCount(RunCount), AnalyzedInstrCount(AnalyzedInstrCount) {
  }

  bool run(Function *F) {
    ++RunCount;

    const TestAnalysisPass::Result &AR = AM.getResult<TestAnalysisPass>(F);
    AnalyzedInstrCount += AR.InstructionCount;

    return true;
  }

  AnalysisManager &AM;
  int &RunCount;
  int &AnalyzedInstrCount;
};

Module *parseIR(const char *IR) {
  LLVMContext &C = getGlobalContext();
  SMDiagnostic Err;
  return ParseAssemblyString(IR, 0, Err, C);
}

class PassManagerTest : public ::testing::Test {
protected:
  OwningPtr<Module> M;

public:
  PassManagerTest()
      : M(parseIR("define void @f() {\n"
                  "entry:\n"
                  "  call void @g()\n"
                  "  call void @h()\n"
                  "  ret void\n"
                  "}\n"
                  "define void @g() {\n"
                  "  ret void\n"
                  "}\n"
                  "define void @h() {\n"
                  "  ret void\n"
                  "}\n")) {}
};

TEST_F(PassManagerTest, Basic) {
  AnalysisManager AM(M.get());
  AM.registerAnalysisPass(TestAnalysisPass());

  ModulePassManager MPM(M.get(), &AM);
  FunctionPassManager FPM(&AM);

  // Count the runs over a module.
  int ModulePassRunCount = 0;
  MPM.addPass(TestModulePass(ModulePassRunCount));

  // Count the runs over a Function.
  int FunctionPassRunCount = 0;
  int AnalyzedInstrCount = 0;
  FPM.addPass(TestFunctionPass(AM, FunctionPassRunCount, AnalyzedInstrCount));
  MPM.addPass(FPM);

  MPM.run();
  EXPECT_EQ(1, ModulePassRunCount);
  EXPECT_EQ(3, FunctionPassRunCount);
  EXPECT_EQ(5, AnalyzedInstrCount);
}

}
