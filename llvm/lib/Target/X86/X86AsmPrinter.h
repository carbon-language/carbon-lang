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

#include "X86Subtarget.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class MCStreamer;
class MCSymbol;

class LLVM_LIBRARY_VISIBILITY X86AsmPrinter : public AsmPrinter {
  const X86Subtarget *Subtarget;
  StackMaps SM;

  void GenerateExportDirective(const MCSymbol *Sym, bool IsData);

 public:
  explicit X86AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), SM(*this) {
    Subtarget = &TM.getSubtarget<X86Subtarget>();
  }

  const char *getPassName() const override {
    return "X86 Assembly / Object Emitter";
  }

  const X86Subtarget &getSubtarget() const { return *Subtarget; }

  void EmitStartOfAsmFile(Module &M) override;

  void EmitEndOfAsmFile(Module &M) override;

  void EmitInstruction(const MachineInstr *MI) override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &OS) override;

  /// \brief Return the symbol for the specified constant pool entry.
  MCSymbol *GetCPISymbol(unsigned CPID) const override;

  bool runOnMachineFunction(MachineFunction &F) override;
};

} // end namespace llvm

#endif
