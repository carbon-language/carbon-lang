//===-- JIT.cpp - MC-based Just-in-Time Compiler --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCJIT.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/System/DynamicLibrary.h"

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
                                  CodeModel::Model CMM,
                                  StringRef MArch,
                                  StringRef MCPU,
                                  const SmallVectorImpl<std::string>& MAttrs) {
  // Try to register the program as a source of symbols to resolve against.
  //
  // FIXME: Don't do this here.
  sys::DynamicLibrary::LoadLibraryPermanently(0, NULL);

  // Pick a target either via -march or by guessing the native arch.
  //
  // FIXME: This should be lifted out of here, it isn't something which should
  // be part of the JIT policy, rather the burden for this selection should be
  // pushed to clients.
  TargetMachine *TM = MCJIT::selectTarget(M, MArch, MCPU, MAttrs, ErrorStr);
  if (!TM || (ErrorStr && ErrorStr->length() > 0)) return 0;
  TM->setCodeModel(CMM);

  // If the target supports JIT code generation, create the JIT.
  if (TargetJITInfo *TJ = TM->getJITInfo())
    return new MCJIT(M, *TM, *TJ, JMM, OptLevel, GVsWithCode);

  if (ErrorStr)
    *ErrorStr = "target does not support JIT code generation";
  return 0;
}

MCJIT::MCJIT(Module *M, TargetMachine &tm, TargetJITInfo &tji,
             JITMemoryManager *JMM, CodeGenOpt::Level OptLevel,
             bool AllocateGVsWithCode)
  : ExecutionEngine(M) {
}

MCJIT::~MCJIT() {
}

void *MCJIT::getPointerToBasicBlock(BasicBlock *BB) {
  report_fatal_error("not yet implemented");
  return 0;
}

void *MCJIT::getPointerToFunction(Function *F) {
  report_fatal_error("not yet implemented");
  return 0;
}

void *MCJIT::recompileAndRelinkFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

void MCJIT::freeMachineCodeForFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

GenericValue MCJIT::runFunction(Function *F,
                                const std::vector<GenericValue> &ArgValues) {
  report_fatal_error("not yet implemented");
  return GenericValue();
}
