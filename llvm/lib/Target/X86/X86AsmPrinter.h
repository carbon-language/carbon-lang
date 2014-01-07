//===-- X86AsmPrinter.h - X86 implementation of AsmPrinter ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86ASMPRINTER_H
#define X86ASMPRINTER_H

#include "X86.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCStreamer;

class LLVM_LIBRARY_VISIBILITY X86AsmPrinter : public AsmPrinter {
  const X86Subtarget *Subtarget;
  StackMaps SM;

 public:
  explicit X86AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), SM(*this) {
    Subtarget = &TM.getSubtarget<X86Subtarget>();
  }

  virtual const char *getPassName() const LLVM_OVERRIDE {
    return "X86 Assembly / Object Emitter";
  }

  const X86Subtarget &getSubtarget() const { return *Subtarget; }

  virtual void EmitStartOfAsmFile(Module &M) LLVM_OVERRIDE;

  virtual void EmitEndOfAsmFile(Module &M) LLVM_OVERRIDE;

  virtual void EmitInstruction(const MachineInstr *MI) LLVM_OVERRIDE;

  virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &OS) LLVM_OVERRIDE;
  virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                     unsigned AsmVariant, const char *ExtraCode,
                                     raw_ostream &OS) LLVM_OVERRIDE;

  virtual bool runOnMachineFunction(MachineFunction &F) LLVM_OVERRIDE;
};

} // end namespace llvm

#endif
