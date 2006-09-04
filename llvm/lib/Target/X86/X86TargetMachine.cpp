//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <iostream>
using namespace llvm;

/// X86TargetMachineModule - Note that this is used on hosts that cannot link
/// in a library unless there are references into the library.  In particular,
/// it seems that it is not possible to get things to work on Win32 without
/// this.  Though it is unused, do not remove it.
extern "C" int X86TargetMachineModule;
int X86TargetMachineModule = 0;

namespace {
  // Register the target.
  RegisterTarget<X86TargetMachine> X("x86", "  IA-32 (Pentium and above)");
}

unsigned X86TargetMachine::getJITMatchQuality() {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  return 10;
#else
  return 0;
#endif
}

unsigned X86TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "i[3-9]86-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && TT[0] == 'i' && TT[2] == '8' && TT[3] == '6' &&
      TT[4] == '-' && TT[1] - '3' < 6)
    return 20;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

/// X86TargetMachine ctor - Create an ILP32 architecture model
///
X86TargetMachine::X86TargetMachine(const Module &M, const std::string &FS)
  : Subtarget(M, FS), DataLayout("e-p:32:32-d:32-l:32"),
    FrameInfo(TargetFrameInfo::StackGrowsDown,
              Subtarget.getStackAlignment(), -4),
    InstrInfo(*this), JITInfo(*this), TLInfo(*this) {
  if (getRelocationModel() == Reloc::Default)
    if (Subtarget.isTargetDarwin())
      setRelocationModel(Reloc::DynamicNoPIC);
    else
      setRelocationModel(Reloc::PIC_);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool X86TargetMachine::addInstSelector(FunctionPassManager &PM, bool Fast) {
  // Install an instruction selector.
  PM.add(createX86ISelDag(*this, Fast));
  return false;
}

bool X86TargetMachine::addPostRegAlloc(FunctionPassManager &PM, bool Fast) {
  PM.add(createX86FloatingPointStackifierPass());
  return true;  // -print-machineinstr should print after this.
}

bool X86TargetMachine::addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                          std::ostream &Out) {
  PM.add(createX86CodePrinterPass(Out, *this));
  return false;
}

bool X86TargetMachine::addObjectWriter(FunctionPassManager &PM, bool Fast,
                                       std::ostream &Out) {
  if (Subtarget.isTargetELF()) {
    addX86ELFObjectWriterPass(PM, Out, *this);
    return false;
  }
  return true;
}

bool X86TargetMachine::addCodeEmitter(FunctionPassManager &PM, bool Fast,
                                      MachineCodeEmitter &MCE) {
  // FIXME: Move this to TargetJITInfo!
  setRelocationModel(Reloc::Static);
  
  PM.add(createX86CodeEmitterPass(*this, MCE));
  return false;
}
