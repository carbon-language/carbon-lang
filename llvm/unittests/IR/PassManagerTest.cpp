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
    int InstructionCount;
  };

  /// \brief Returns an opaque, unique ID for this pass type.
  static void *ID() { return (void *)&PassID; }

  TestAnalysisPass(int &Runs) : Runs(Runs) {}

  /// \brief Run the analysis pass over the function and return a result.
  Result run(Function *F) {
    ++Runs;
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

  int &Runs;
};

char TestAnalysisPass::PassID;

struct TestModulePass {
  TestModulePass(int &RunCount) : RunCount(RunCount) {}

  PreservedAnalyses run(Module *M) {
    ++RunCount;
    return PreservedAnalyses::none();
  }

  int &RunCount;
};

struct TestPreservingModulePass {
  PreservedAnalyses run(Module *M) {
    return PreservedAnalyses::all();
  }
};

struct TestMinPreservingModulePass {
  PreservedAnalyses run(Module *M) {
    PreservedAnalyses PA;
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
  }
};

struct TestFunctionPass {
  TestFunctionPass(int &RunCount, int &AnalyzedInstrCount)
      : RunCount(RunCount), AnalyzedInstrCount(AnalyzedInstrCount) {}

  PreservedAnalyses run(Function *F, FunctionAnalysisManager *AM) {
    ++RunCount;

    const TestAnalysisPass::Result &AR = AM->getResult<TestAnalysisPass>(F);
    AnalyzedInstrCount += AR.InstructionCount;

    return PreservedAnalyses::all();
  }

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
  FunctionAnalysisManager FAM;
  int AnalysisRuns = 0;
  FAM.registerPass(TestAnalysisPass(AnalysisRuns));

  ModuleAnalysisManager MAM;
  MAM.registerPass(FunctionAnalysisManagerModuleProxy(FAM));

  ModulePassManager MPM;

  // Count the runs over a Function.
  FunctionPassManager FPM1;
  int FunctionPassRunCount1 = 0;
  int AnalyzedInstrCount1 = 0;
  FPM1.addPass(TestFunctionPass(FunctionPassRunCount1, AnalyzedInstrCount1));
  MPM.addPass(createModuleToFunctionPassAdaptor(FPM1));

  // Count the runs over a module.
  int ModulePassRunCount = 0;
  MPM.addPass(TestModulePass(ModulePassRunCount));

  // Count the runs over a Function in a separate manager.
  FunctionPassManager FPM2;
  int FunctionPassRunCount2 = 0;
  int AnalyzedInstrCount2 = 0;
  FPM2.addPass(TestFunctionPass(FunctionPassRunCount2, AnalyzedInstrCount2));
  MPM.addPass(createModuleToFunctionPassAdaptor(FPM2));

  // A third function pass manager but with only preserving intervening passes.
  MPM.addPass(TestPreservingModulePass());
  FunctionPassManager FPM3;
  int FunctionPassRunCount3 = 0;
  int AnalyzedInstrCount3 = 0;
  FPM3.addPass(TestFunctionPass(FunctionPassRunCount3, AnalyzedInstrCount3));
  MPM.addPass(createModuleToFunctionPassAdaptor(FPM3));

  // A fourth function pass manager but with a minimal intervening passes.
  MPM.addPass(TestMinPreservingModulePass());
  FunctionPassManager FPM4;
  int FunctionPassRunCount4 = 0;
  int AnalyzedInstrCount4 = 0;
  FPM4.addPass(TestFunctionPass(FunctionPassRunCount4, AnalyzedInstrCount4));
  MPM.addPass(createModuleToFunctionPassAdaptor(FPM4));

  MPM.run(M.get(), &MAM);

  // Validate module pass counters.
  EXPECT_EQ(1, ModulePassRunCount);

  // Validate both function pass counter sets.
  EXPECT_EQ(3, FunctionPassRunCount1);
  EXPECT_EQ(5, AnalyzedInstrCount1);
  EXPECT_EQ(3, FunctionPassRunCount2);
  EXPECT_EQ(5, AnalyzedInstrCount2);
  EXPECT_EQ(3, FunctionPassRunCount3);
  EXPECT_EQ(5, AnalyzedInstrCount3);
  EXPECT_EQ(3, FunctionPassRunCount4);
  EXPECT_EQ(5, AnalyzedInstrCount4);

  // Validate the analysis counters.
  EXPECT_EQ(9, AnalysisRuns);
}
}
