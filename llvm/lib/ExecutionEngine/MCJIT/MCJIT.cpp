//===-- MCJIT.cpp - MC-based Just-in-Time Compiler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCJIT.h"
#include "MCJITMemoryManager.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

namespace {

static struct RegisterJIT {
  RegisterJIT() { MCJIT::Register(); }
} JITRegistrator;

}

extern "C" void LLVMLinkInMCJIT() {
}

ExecutionEngine *MCJIT::createJIT(Module *M,
                                  std::string *ErrorStr,
                                  JITMemoryManager *JMM,
                                  CodeGenOpt::Level OptLevel,
                                  bool GVsWithCode,
                                  TargetMachine *TM) {
  // Try to register the program as a source of symbols to resolve against.
  //
  // FIXME: Don't do this here.
  sys::DynamicLibrary::LoadLibraryPermanently(0, NULL);

  // If the target supports JIT code generation, create the JIT.
  if (TargetJITInfo *TJ = TM->getJITInfo())
    return new MCJIT(M, TM, *TJ, new MCJITMemoryManager(JMM, M), OptLevel,
                     GVsWithCode);

  if (ErrorStr)
    *ErrorStr = "target does not support JIT code generation";
  return 0;
}

MCJIT::MCJIT(Module *m, TargetMachine *tm, TargetJITInfo &tji,
             RTDyldMemoryManager *MM, CodeGenOpt::Level OptLevel,
             bool AllocateGVsWithCode)
  : ExecutionEngine(m), TM(tm), MemMgr(MM), M(m), OS(Buffer), Dyld(MM) {

  PM.add(new TargetData(*TM->getTargetData()));

  // Turn the machine code intermediate representation into bytes in memory
  // that may be executed.
  if (TM->addPassesToEmitMC(PM, Ctx, OS, CodeGenOpt::Default, false)) {
    report_fatal_error("Target does not support MC emission!");
  }

  // Initialize passes.
  // FIXME: When we support multiple modules, we'll want to move the code
  // gen and finalization out of the constructor here and do it more
  // on-demand as part of getPointerToFunction().
  PM.run(*M);
  // Flush the output buffer so the SmallVector gets its data.
  OS.flush();

  // Load the object into the dynamic linker.
  // FIXME: It would be nice to avoid making yet another copy.
  MemoryBuffer *MB = MemoryBuffer::getMemBufferCopy(StringRef(Buffer.data(),
                                                              Buffer.size()));
  if (Dyld.loadObject(MB))
    report_fatal_error(Dyld.getErrorString());
  // Resolve any relocations.
  Dyld.resolveRelocations();
}

MCJIT::~MCJIT() {
  delete MemMgr;
}

void *MCJIT::getPointerToBasicBlock(BasicBlock *BB) {
  report_fatal_error("not yet implemented");
  return 0;
}

void *MCJIT::getPointerToFunction(Function *F) {
  if (F->isDeclaration() || F->hasAvailableExternallyLinkage()) {
    bool AbortOnFailure = !F->hasExternalWeakLinkage();
    void *Addr = getPointerToNamedFunction(F->getName(), AbortOnFailure);
    addGlobalMapping(F, Addr);
    return Addr;
  }

  // FIXME: Should we be using the mangler for this? Probably.
  StringRef BaseName = F->getName();
  Twine Name;
  if (BaseName[0] == '\1')
    Name = BaseName.substr(1);
  else
    Name = TM->getMCAsmInfo()->getGlobalPrefix() + BaseName;
  return (void*)Dyld.getSymbolAddress(Name.str());
}

void *MCJIT::recompileAndRelinkFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

void MCJIT::freeMachineCodeForFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

GenericValue MCJIT::runFunction(Function *F,
                                const std::vector<GenericValue> &ArgValues) {
  assert(F && "Function *F was null at entry to run()");

  void *FPtr = getPointerToFunction(F);
  assert(FPtr && "Pointer to fn's code was null after getPointerToFunction");
  const FunctionType *FTy = F->getFunctionType();
  const Type *RetTy = FTy->getReturnType();

  assert((FTy->getNumParams() == ArgValues.size() ||
          (FTy->isVarArg() && FTy->getNumParams() <= ArgValues.size())) &&
         "Wrong number of arguments passed into function!");
  assert(FTy->getNumParams() == ArgValues.size() &&
         "This doesn't support passing arguments through varargs (yet)!");

  // Handle some common cases first.  These cases correspond to common `main'
  // prototypes.
  if (RetTy->isIntegerTy(32) || RetTy->isVoidTy()) {
    switch (ArgValues.size()) {
    case 3:
      if (FTy->getParamType(0)->isIntegerTy(32) &&
          FTy->getParamType(1)->isPointerTy() &&
          FTy->getParamType(2)->isPointerTy()) {
        int (*PF)(int, char **, const char **) =
          (int(*)(int, char **, const char **))(intptr_t)FPtr;

        // Call the function.
        GenericValue rv;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                 (char **)GVTOP(ArgValues[1]),
                                 (const char **)GVTOP(ArgValues[2])));
        return rv;
      }
      break;
    case 2:
      if (FTy->getParamType(0)->isIntegerTy(32) &&
          FTy->getParamType(1)->isPointerTy()) {
        int (*PF)(int, char **) = (int(*)(int, char **))(intptr_t)FPtr;

        // Call the function.
        GenericValue rv;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                 (char **)GVTOP(ArgValues[1])));
        return rv;
      }
      break;
    case 1:
      if (FTy->getNumParams() == 1 &&
          FTy->getParamType(0)->isIntegerTy(32)) {
        GenericValue rv;
        int (*PF)(int) = (int(*)(int))(intptr_t)FPtr;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue()));
        return rv;
      }
      break;
    }
  }

  // Handle cases where no arguments are passed first.
  if (ArgValues.empty()) {
    GenericValue rv;
    switch (RetTy->getTypeID()) {
    default: llvm_unreachable("Unknown return type for function call!");
    case Type::IntegerTyID: {
      unsigned BitWidth = cast<IntegerType>(RetTy)->getBitWidth();
      if (BitWidth == 1)
        rv.IntVal = APInt(BitWidth, ((bool(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 8)
        rv.IntVal = APInt(BitWidth, ((char(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 16)
        rv.IntVal = APInt(BitWidth, ((short(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 32)
        rv.IntVal = APInt(BitWidth, ((int(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 64)
        rv.IntVal = APInt(BitWidth, ((int64_t(*)())(intptr_t)FPtr)());
      else
        llvm_unreachable("Integer types > 64 bits not supported");
      return rv;
    }
    case Type::VoidTyID:
      rv.IntVal = APInt(32, ((int(*)())(intptr_t)FPtr)());
      return rv;
    case Type::FloatTyID:
      rv.FloatVal = ((float(*)())(intptr_t)FPtr)();
      return rv;
    case Type::DoubleTyID:
      rv.DoubleVal = ((double(*)())(intptr_t)FPtr)();
      return rv;
    case Type::X86_FP80TyID:
    case Type::FP128TyID:
    case Type::PPC_FP128TyID:
      llvm_unreachable("long double not supported yet");
      return rv;
    case Type::PointerTyID:
      return PTOGV(((void*(*)())(intptr_t)FPtr)());
    }
  }

  assert("Full-featured argument passing not supported yet!");
  return GenericValue();
}
