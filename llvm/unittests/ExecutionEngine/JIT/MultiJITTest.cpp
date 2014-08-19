//===- MultiJITTest.cpp - Unit tests for instantiating multiple JITs ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;

namespace {

// ARM, PowerPC and SystemZ tests disabled pending fix for PR10783.
#if !defined(__arm__) && !defined(__powerpc__) && !defined(__s390__) \
                      && !defined(__aarch64__)

std::unique_ptr<Module> loadAssembly(LLVMContext &Context,
                                     const char *Assembly) {
  SMDiagnostic Error;
  std::unique_ptr<Module> Ret = parseAssemblyString(Assembly, Error, Context);
  std::string errMsg;
  raw_string_ostream os(errMsg);
  Error.print("", os);
  EXPECT_TRUE((bool)Ret) << os.str();
  return std::move(Ret);
}

std::unique_ptr<Module> createModule1(LLVMContext &Context1, Function *&FooF1) {
  std::unique_ptr<Module> Ret =  loadAssembly(Context1,
                   "define i32 @add1(i32 %ArgX1) { "
                   "entry: "
                   "  %addresult = add i32 1, %ArgX1 "
                   "  ret i32 %addresult "
                   "} "
                   " "
                   "define i32 @foo1() { "
                   "entry: "
                   "  %add1 = call i32 @add1(i32 10) "
                   "  ret i32 %add1 "
                   "} ");
  FooF1 = Ret->getFunction("foo1");
  return std::move(Ret);
}

std::unique_ptr<Module> createModule2(LLVMContext &Context2, Function *&FooF2) {
  std::unique_ptr<Module> Ret = loadAssembly(Context2,
                   "define i32 @add2(i32 %ArgX2) { "
                   "entry: "
                   "  %addresult = add i32 2, %ArgX2 "
                   "  ret i32 %addresult "
                   "} "
                   " "
                   "define i32 @foo2() { "
                   "entry: "
                   "  %add2 = call i32 @add2(i32 10) "
                   "  ret i32 %add2 "
                   "} ");
  FooF2 = Ret->getFunction("foo2");
  return std::move(Ret);
}

TEST(MultiJitTest, EagerMode) {
  LLVMContext Context1;
  Function *FooF1 = nullptr;
  std::unique_ptr<Module> M1 = createModule1(Context1, FooF1);

  LLVMContext Context2;
  Function *FooF2 = nullptr;
  std::unique_ptr<Module> M2 = createModule2(Context2, FooF2);

  // Now we create the JIT in eager mode
  std::unique_ptr<ExecutionEngine> EE1(EngineBuilder(std::move(M1)).create());
  EE1->DisableLazyCompilation(true);
  std::unique_ptr<ExecutionEngine> EE2(EngineBuilder(std::move(M2)).create());
  EE2->DisableLazyCompilation(true);

  // Call the `foo' function with no arguments:
  std::vector<GenericValue> noargs;
  GenericValue gv1 = EE1->runFunction(FooF1, noargs);
  GenericValue gv2 = EE2->runFunction(FooF2, noargs);

  // Import result of execution:
  EXPECT_EQ(gv1.IntVal, 11);
  EXPECT_EQ(gv2.IntVal, 12);

  EE1->freeMachineCodeForFunction(FooF1);
  EE2->freeMachineCodeForFunction(FooF2);
}

TEST(MultiJitTest, LazyMode) {
  LLVMContext Context1;
  Function *FooF1 = nullptr;
  std::unique_ptr<Module> M1 = createModule1(Context1, FooF1);

  LLVMContext Context2;
  Function *FooF2 = nullptr;
  std::unique_ptr<Module> M2 = createModule2(Context2, FooF2);

  // Now we create the JIT in lazy mode
  std::unique_ptr<ExecutionEngine> EE1(EngineBuilder(std::move(M1)).create());
  EE1->DisableLazyCompilation(false);
  std::unique_ptr<ExecutionEngine> EE2(EngineBuilder(std::move(M2)).create());
  EE2->DisableLazyCompilation(false);

  // Call the `foo' function with no arguments:
  std::vector<GenericValue> noargs;
  GenericValue gv1 = EE1->runFunction(FooF1, noargs);
  GenericValue gv2 = EE2->runFunction(FooF2, noargs);

  // Import result of execution:
  EXPECT_EQ(gv1.IntVal, 11);
  EXPECT_EQ(gv2.IntVal, 12);

  EE1->freeMachineCodeForFunction(FooF1);
  EE2->freeMachineCodeForFunction(FooF2);
}

extern "C" {
  extern void *getPointerToNamedFunction(const char *Name);
}

TEST(MultiJitTest, JitPool) {
  LLVMContext Context1;
  Function *FooF1 = nullptr;
  std::unique_ptr<Module> M1 = createModule1(Context1, FooF1);

  LLVMContext Context2;
  Function *FooF2 = nullptr;
  std::unique_ptr<Module> M2 = createModule2(Context2, FooF2);

  // Now we create two JITs
  std::unique_ptr<ExecutionEngine> EE1(EngineBuilder(std::move(M1)).create());
  std::unique_ptr<ExecutionEngine> EE2(EngineBuilder(std::move(M2)).create());

  Function *F1 = EE1->FindFunctionNamed("foo1");
  void *foo1 = EE1->getPointerToFunction(F1);

  Function *F2 = EE2->FindFunctionNamed("foo2");
  void *foo2 = EE2->getPointerToFunction(F2);

  // Function in M1
  EXPECT_EQ(getPointerToNamedFunction("foo1"), foo1);

  // Function in M2
  EXPECT_EQ(getPointerToNamedFunction("foo2"), foo2);

  // Symbol search
  intptr_t
    sa = (intptr_t)getPointerToNamedFunction("getPointerToNamedFunction");
  EXPECT_TRUE(sa != 0);
  intptr_t fa = (intptr_t)&getPointerToNamedFunction;
  EXPECT_TRUE(fa != 0);
#ifdef __i386__
  // getPointerToNamedFunction might be indirect jump on Win32 --enable-shared.
  // FF 25 <disp32>: jmp *(pointer to IAT)
  if (sa != fa && memcmp((char *)fa, "\xFF\x25", 2) == 0) {
    fa = *(intptr_t *)(fa + 2); // Address to IAT
    EXPECT_TRUE(fa != 0);
    fa = *(intptr_t *)fa;       // Bound value of IAT
  }
#elif defined(__x86_64__)
  // getPointerToNamedFunction might be indirect jump
  // on Win32 x64 --enable-shared.
  // FF 25 <pcrel32>: jmp *(RIP + pointer to IAT)
  if (sa != fa && memcmp((char *)fa, "\xFF\x25", 2) == 0) {
    fa += *(int32_t *)(fa + 2) + 6;     // Address to IAT(RIP)
    fa = *(intptr_t *)fa;               // Bound value of IAT
  }
#endif
  EXPECT_TRUE(sa == fa);
}
#endif  // !defined(__arm__) && !defined(__powerpc__) && !defined(__s390__)

}  // anonymous namespace
