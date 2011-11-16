//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeX86Target() {
  // Register the target.
  RegisterTargetMachine<X86_32TargetMachine> X(TheX86_32Target);
  RegisterTargetMachine<X86_64TargetMachine> Y(TheX86_64Target);
}


X86_32TargetMachine::X86_32TargetMachine(const Target &T, StringRef TT,
                                         StringRef CPU, StringRef FS,
                                         Reloc::Model RM, CodeModel::Model CM,
                                         CodeGenOpt::Level OL)
  : X86TargetMachine(T, TT, CPU, FS, RM, CM, OL, false),
    DataLayout(getSubtargetImpl()->isTargetDarwin() ?
               "e-p:32:32-f64:32:64-i64:32:64-f80:128:128-f128:128:128-"
               "n8:16:32-S128" :
               (getSubtargetImpl()->isTargetCygMing() ||
                getSubtargetImpl()->isTargetWindows()) ?
               "e-p:32:32-f64:64:64-i64:64:64-f80:32:32-f128:128:128-"
               "n8:16:32-S32" :
               "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-f128:128:128-"
               "n8:16:32-S128"),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this),
    JITInfo(*this) {
}


X86_64TargetMachine::X86_64TargetMachine(const Target &T, StringRef TT,
                                         StringRef CPU, StringRef FS,
                                         Reloc::Model RM, CodeModel::Model CM,
                                         CodeGenOpt::Level OL)
  : X86TargetMachine(T, TT, CPU, FS, RM, CM, OL, true),
    DataLayout("e-p:64:64-s:64-f64:64:64-i64:64:64-f80:128:128-f128:128:128-"
               "n8:16:32:64-S128"),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this),
    JITInfo(*this) {
}

/// X86TargetMachine ctor - Create an X86 target.
///
X86TargetMachine::X86TargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, RM, CM, OL),
    Subtarget(TT, CPU, FS, StackAlignmentOverride, is64Bit),
    FrameLowering(*this, Subtarget),
    ELFWriterInfo(is64Bit, true) {
  // Determine the PICStyle based on the target selected.
  if (getRelocationModel() == Reloc::Static) {
    // Unless we're in PIC or DynamicNoPIC mode, set the PIC style to None.
    Subtarget.setPICStyle(PICStyles::None);
  } else if (Subtarget.is64Bit()) {
    // PIC in 64 bit mode is always rip-rel.
    Subtarget.setPICStyle(PICStyles::RIPRel);
  } else if (Subtarget.isTargetCygMing()) {
    Subtarget.setPICStyle(PICStyles::None);
  } else if (Subtarget.isTargetDarwin()) {
    if (getRelocationModel() == Reloc::PIC_)
      Subtarget.setPICStyle(PICStyles::StubPIC);
    else {
      assert(getRelocationModel() == Reloc::DynamicNoPIC);
      Subtarget.setPICStyle(PICStyles::StubDynamicNoPIC);
    }
  } else if (Subtarget.isTargetELF()) {
    Subtarget.setPICStyle(PICStyles::GOT);
  }

  // default to hard float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Hard;    
}

//===----------------------------------------------------------------------===//
// Command line options for x86
//===----------------------------------------------------------------------===//
static cl::opt<bool>
UseVZeroUpper("x86-use-vzeroupper",
  cl::desc("Minimize AVX to SSE transition penalty"),
  cl::init(false));

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool X86TargetMachine::addInstSelector(PassManagerBase &PM) {
  // Install an instruction selector.
  PM.add(createX86ISelDag(*this, getOptLevel()));

  // For 32-bit, prepend instructions to set the "global base reg" for PIC.
  if (!Subtarget.is64Bit())
    PM.add(createGlobalBaseRegPass());

  return false;
}

bool X86TargetMachine::addPreRegAlloc(PassManagerBase &PM) {
  PM.add(createX86MaxStackAlignmentHeuristicPass());
  return false;  // -print-machineinstr shouldn't print after this.
}

bool X86TargetMachine::addPostRegAlloc(PassManagerBase &PM) {
  PM.add(createX86FloatingPointStackifierPass());
  return true;  // -print-machineinstr should print after this.
}

bool X86TargetMachine::addPreEmitPass(PassManagerBase &PM) {
  bool ShouldPrint = false;
  if (getOptLevel() != CodeGenOpt::None && Subtarget.hasXMMInt()) {
    PM.add(createExecutionDependencyFixPass(&X86::VR128RegClass));
    ShouldPrint = true;
  }

  if (Subtarget.hasAVX() && UseVZeroUpper) {
    PM.add(createX86IssueVZeroUpperPass());
    ShouldPrint = true;
  }

  return ShouldPrint;
}

bool X86TargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      JITCodeEmitter &JCE) {
  PM.add(createX86JITCodeEmitterPass(*this, JCE));

  return false;
}
