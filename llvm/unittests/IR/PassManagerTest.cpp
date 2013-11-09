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

struct TestModulePass {
  TestModulePass(int &RunCount) : RunCount(RunCount) {}

  bool run(Module *M) {
    ++RunCount;
    return true;
  }

  int &RunCount;
};

struct TestFunctionPass {
  TestFunctionPass(int &RunCount) : RunCount(RunCount) {}

  bool run(Function *F) {
    ++RunCount;
    return true;
  }

  int &RunCount;
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
  ModulePassManager MPM(M.get());
  FunctionPassManager FPM;

  // Count the runs over a module.
  int ModulePassRunCount = 0;
  MPM.addPass(TestModulePass(ModulePassRunCount));

  // Count the runs over a Function.
  int FunctionPassRunCount = 0;
  FPM.addPass(TestFunctionPass(FunctionPassRunCount));
  MPM.addPass(FPM);

  MPM.run();
  EXPECT_EQ(1, ModulePassRunCount);
  EXPECT_EQ(3, FunctionPassRunCount);
}

}
