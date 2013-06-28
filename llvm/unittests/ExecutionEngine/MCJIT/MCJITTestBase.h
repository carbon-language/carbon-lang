//===- MCJITTestBase.h - Common base class for MCJIT Unit tests  ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements common functionality required by the MCJIT unit tests,
// as well as logic to skip tests on unsupported architectures and operating
// systems.
//
//===----------------------------------------------------------------------===//


#ifndef MCJIT_TEST_BASE_H
#define MCJIT_TEST_BASE_H

#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "MCJITTestAPICommon.h"

namespace llvm {

/// Helper class that can build very simple Modules
class TrivialModuleBuilder {
protected:
  LLVMContext Context;
  IRBuilder<> Builder;
  std::string BuilderTriple;

  TrivialModuleBuilder(const std::string &Triple)
    : Builder(Context), BuilderTriple(Triple) {}

  Module *createEmptyModule(StringRef Name = StringRef()) {
    Module * M = new Module(Name, Context);
    M->setTargetTriple(Triple::normalize(BuilderTriple));
    return M;
  }

  template<typename FuncType>
  Function *startFunction(Module *M, StringRef Name) {
    Function *Result = Function::Create(
      TypeBuilder<FuncType, false>::get(Context),
      GlobalValue::ExternalLinkage, Name, M);

    BasicBlock *BB = BasicBlock::Create(Context, Name, Result);
    Builder.SetInsertPoint(BB);

    return Result;
  }

  void endFunctionWithRet(Function *Func, Value *RetValue) {
    Builder.CreateRet(RetValue);
  }

  // Inserts a simple function that invokes Callee and takes the same arguments:
  //    int Caller(...) { return Callee(...); }
  template<typename Signature>
  Function *insertSimpleCallFunction(Module *M, Function *Callee) {
    Function *Result = startFunction<Signature>(M, "caller");

    SmallVector<Value*, 1> CallArgs;

    Function::arg_iterator arg_iter = Result->arg_begin();
    for(;arg_iter != Result->arg_end(); ++arg_iter)
      CallArgs.push_back(arg_iter);

    Value *ReturnCode = Builder.CreateCall(Callee, CallArgs);
    Builder.CreateRet(ReturnCode);
    return Result;
  }

  // Inserts a function named 'main' that returns a uint32_t:
  //    int32_t main() { return X; }
  // where X is given by returnCode
  Function *insertMainFunction(Module *M, uint32_t returnCode) {
    Function *Result = startFunction<int32_t(void)>(M, "main");

    Value *ReturnVal = ConstantInt::get(Context, APInt(32, returnCode));
    endFunctionWithRet(Result, ReturnVal);

    return Result;
  }

  // Inserts a function
  //    int32_t add(int32_t a, int32_t b) { return a + b; }
  // in the current module and returns a pointer to it.
  Function *insertAddFunction(Module *M, StringRef Name = "add") {
    Function *Result = startFunction<int32_t(int32_t, int32_t)>(M, Name);

    Function::arg_iterator args = Result->arg_begin();
    Value *Arg1 = args;
    Value *Arg2 = ++args;
    Value *AddResult = Builder.CreateAdd(Arg1, Arg2);

    endFunctionWithRet(Result, AddResult);

    return Result;
  }

  // Inserts an declaration to a function defined elsewhere
  Function *insertExternalReferenceToFunction(Module *M, StringRef Name,
                                              FunctionType *FuncTy) {
    Function *Result = Function::Create(FuncTy,
                                        GlobalValue::ExternalLinkage,
                                        Name, M);
    return Result;
  }

  // Inserts an declaration to a function defined elsewhere
  Function *insertExternalReferenceToFunction(Module *M, Function *Func) {
    Function *Result = Function::Create(Func->getFunctionType(),
                                        GlobalValue::AvailableExternallyLinkage,
                                        Func->getName(), M);
    return Result;
  }

  // Inserts a global variable of type int32
  GlobalVariable *insertGlobalInt32(Module *M,
                                    StringRef name,
                                    int32_t InitialValue) {
    Type *GlobalTy = TypeBuilder<types::i<32>, true>::get(Context);
    Constant *IV = ConstantInt::get(Context, APInt(32, InitialValue));
    GlobalVariable *Global = new GlobalVariable(*M,
                                                GlobalTy,
                                                false,
                                                GlobalValue::ExternalLinkage,
                                                IV,
                                                name);
    return Global;
  }
};

class MCJITTestBase : public MCJITTestAPICommon, public TrivialModuleBuilder {
protected:
  
  MCJITTestBase()
    : TrivialModuleBuilder(HostTriple)
    , OptLevel(CodeGenOpt::None)
    , RelocModel(Reloc::Default)
    , CodeModel(CodeModel::Default)
    , MArch("")
    , MM(new SectionMemoryManager)
  {
    // The architectures below are known to be compatible with MCJIT as they
    // are copied from test/ExecutionEngine/MCJIT/lit.local.cfg and should be
    // kept in sync.
    SupportedArchs.push_back(Triple::aarch64);
    SupportedArchs.push_back(Triple::arm);
    SupportedArchs.push_back(Triple::x86);
    SupportedArchs.push_back(Triple::x86_64);

    // Some architectures have sub-architectures in which tests will fail, like
    // ARM. These two vectors will define if they do have sub-archs (to avoid
    // extra work for those who don't), and if so, if they are listed to work
    HasSubArchs.push_back(Triple::arm);
    SupportedSubArchs.push_back("armv6");
    SupportedSubArchs.push_back("armv7");

    // The operating systems below are known to be incompatible with MCJIT as
    // they are copied from the test/ExecutionEngine/MCJIT/lit.local.cfg and
    // should be kept in sync.
    UnsupportedOSs.push_back(Triple::Cygwin);
    UnsupportedOSs.push_back(Triple::Darwin);
  }

  void createJIT(Module *M) {

    // Due to the EngineBuilder constructor, it is required to have a Module
    // in order to construct an ExecutionEngine (i.e. MCJIT)
    assert(M != 0 && "a non-null Module must be provided to create MCJIT");

    EngineBuilder EB(M);
    std::string Error;
    TheJIT.reset(EB.setEngineKind(EngineKind::JIT)
                 .setUseMCJIT(true) /* can this be folded into the EngineKind enum? */
                 .setMCJITMemoryManager(MM)
                 .setErrorStr(&Error)
                 .setOptLevel(CodeGenOpt::None)
                 .setAllocateGVsWithCode(false) /*does this do anything?*/
                 .setCodeModel(CodeModel::JITDefault)
                 .setRelocationModel(Reloc::Default)
                 .setMArch(MArch)
                 .setMCPU(sys::getHostCPUName())
                 //.setMAttrs(MAttrs)
                 .create());
    // At this point, we cannot modify the module any more.
    assert(TheJIT.get() != NULL && "error creating MCJIT with EngineBuilder");
  }

  CodeGenOpt::Level OptLevel;
  Reloc::Model RelocModel;
  CodeModel::Model CodeModel;
  StringRef MArch;
  SmallVector<std::string, 1> MAttrs;
  OwningPtr<ExecutionEngine> TheJIT;
  RTDyldMemoryManager *MM;

  OwningPtr<Module> M;
};

} // namespace llvm

#endif // MCJIT_TEST_H
