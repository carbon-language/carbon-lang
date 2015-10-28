//===--------------- OrcCAPITest.cpp - Unit tests Orc C API ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "gtest/gtest.h"
#include "llvm-c/OrcBindings.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace llvm {

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef);

class OrcCAPIExecutionTest : public testing::Test, public OrcExecutionTest {
protected:

  std::unique_ptr<Module> createTestModule(const Triple &TT) {
    ModuleBuilder MB(getGlobalContext(), TT.str(), "");
    Function *TestFunc = MB.createFunctionDecl<int()>("testFunc");
    Function *Main = MB.createFunctionDecl<int(int, char*[])>("main");

    Main->getBasicBlockList().push_back(BasicBlock::Create(getGlobalContext()));
    IRBuilder<> B(&Main->back());
    Value* Result = B.CreateCall(TestFunc);
    B.CreateRet(Result);

    return MB.takeModule();
  }

  typedef int (*MainFnTy)(void);

  static int myTestFuncImpl(void) {
    return 42;
  }

  static char *testFuncName;

  static uint64_t myResolver(const char *Name, void *Ctx) {
    if (!strncmp(Name, testFuncName, 8))
      return (uint64_t)&myTestFuncImpl;
    return 0;
  }

};

char *OrcCAPIExecutionTest::testFuncName = 0;

TEST_F(OrcCAPIExecutionTest, TestEagerIRCompilation) {
  auto TM = getHostTargetMachineIfSupported();

  if (!TM)
    return;

  std::unique_ptr<Module> M = createTestModule(TM->getTargetTriple());

  LLVMOrcJITStackRef JIT =
    LLVMOrcCreateInstance(wrap(TM.get()), LLVMGetGlobalContext());

  LLVMOrcGetMangledSymbol(JIT, &testFuncName, "testFunc");

  LLVMOrcModuleHandle H =
    LLVMOrcAddEagerlyCompiledIR(JIT, wrap(M.get()), myResolver, 0);
  MainFnTy MainFn = (MainFnTy)LLVMOrcGetSymbolAddress(JIT, "main");
  int Result = MainFn();
  EXPECT_EQ(Result, 42)
    << "Eagerly JIT'd code did not return expected result";

  LLVMOrcRemoveModule(JIT, H);

  LLVMOrcDisposeMangledSymbol(testFuncName);
  LLVMOrcDisposeInstance(JIT);
}

TEST_F(OrcCAPIExecutionTest, TestLazyIRCompilation) {
  auto TM = getHostTargetMachineIfSupported();

  if (!TM)
    return;

  std::unique_ptr<Module> M = createTestModule(TM->getTargetTriple());

  LLVMOrcJITStackRef JIT =
    LLVMOrcCreateInstance(wrap(TM.get()), LLVMGetGlobalContext());

  LLVMOrcGetMangledSymbol(JIT, &testFuncName, "testFunc");
  LLVMOrcModuleHandle H =
    LLVMOrcAddLazilyCompiledIR(JIT, wrap(M.get()), myResolver, 0);
  MainFnTy MainFn = (MainFnTy)LLVMOrcGetSymbolAddress(JIT, "main");
  int Result = MainFn();
  EXPECT_EQ(Result, 42)
    << "Lazily JIT'd code did not return expected result";

  LLVMOrcRemoveModule(JIT, H);

  LLVMOrcDisposeMangledSymbol(testFuncName);
  LLVMOrcDisposeInstance(JIT);
}

}
