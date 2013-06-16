// AArch64AsmPrinter.h - Print machine code to an AArch64 .s file -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64 assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64ASMPRINTER_H
#define LLVM_AARCH64ASMPRINTER_H

#include "AArch64.h"
#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCOperand;

class LLVM_LIBRARY_VISIBILITY AArch64AsmPrinter : public AsmPrinter {

  /// Subtarget - Keep a pointer to the AArch64Subtarget around so that we can
  /// make the right decision when printing asm code for different targets.
  const AArch64Subtarget *Subtarget;

  // emitPseudoExpansionLowering - tblgen'erated.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

  public:
  explicit AArch64AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {
    Subtarget = &TM.getSubtarget<AArch64Subtarget>();
  }

  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  MCOperand lowerSymbolOperand(const MachineOperand &MO,
                               const MCSymbol *Sym) const;

  void EmitInstruction(const MachineInstr *MI);
  void EmitEndOfAsmFile(Module &M);

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O);

  /// printSymbolicAddress - Given some kind of reasonably bare symbolic
  /// reference, print out the appropriate asm string to represent it. If
  /// appropriate, a relocation-specifier will be produced, composed of a
  /// general class derived from the MO parameter and an instruction-specific
  /// suffix, provided in Suffix. E.g. ":got_lo12:" if a Suffix of "lo12" is
  /// given.
  bool printSymbolicAddress(const MachineOperand &MO,
                            bool PrintImmediatePrefix,
                            StringRef Suffix, raw_ostream &O);

  virtual const char *getPassName() const {
    return "AArch64 Assembly Printer";
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);
};
} // end namespace llvm

#endif
