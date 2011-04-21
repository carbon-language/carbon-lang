//===-- ARMAsmPrinter.h - Print machine code to an ARM .s file ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ARM Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMASMPRINTER_H
#define ARMASMPRINTER_H

#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

namespace ARM {
  enum DW_ISA {
    DW_ISA_ARM_thumb = 1,
    DW_ISA_ARM_arm = 2
  };
}

class LLVM_LIBRARY_VISIBILITY ARMAsmPrinter : public AsmPrinter {

  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when printing asm code for different targets.
  const ARMSubtarget *Subtarget;

  /// AFI - Keep a pointer to ARMFunctionInfo for the current
  /// MachineFunction.
  ARMFunctionInfo *AFI;

  /// MCP - Keep a pointer to constantpool entries of the current
  /// MachineFunction.
  const MachineConstantPool *MCP;

public:
  explicit ARMAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), AFI(NULL), MCP(NULL) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

  virtual const char *getPassName() const {
    return "ARM Assembly Printer";
  }

  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                    const char *Modifier = 0);

  virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O);
  virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                                     unsigned AsmVariant,
                                     const char *ExtraCode, raw_ostream &O);

  void EmitJumpTable(const MachineInstr *MI);
  void EmitJump2Table(const MachineInstr *MI);
  virtual void EmitInstruction(const MachineInstr *MI);
  bool runOnMachineFunction(MachineFunction &F);

  virtual void EmitConstantPool() {} // we emit constant pools customly!
  virtual void EmitFunctionEntryLabel();
  void EmitStartOfAsmFile(Module &M);
  void EmitEndOfAsmFile(Module &M);

private:
  // Helpers for EmitStartOfAsmFile() and EmitEndOfAsmFile()
  void emitAttributes();

  // Helper for ELF .o only
  void emitARMAttributeSection();

  // Generic helper used to emit e.g. ARMv5 mul pseudos
  void EmitPatchedInstruction(const MachineInstr *MI, unsigned TargetOpc);

  void EmitUnwindingInstruction(const MachineInstr *MI);

public:
  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);

  MachineLocation getDebugValueLocation(const MachineInstr *MI) const;

  /// EmitDwarfRegOp - Emit dwarf register operation.
  virtual void EmitDwarfRegOp(const MachineLocation &MLoc) const;

  virtual unsigned getISAEncoding() {
    // ARM/Darwin adds ISA to the DWARF info for each function.
    if (!Subtarget->isTargetDarwin())
      return 0;
    return Subtarget->isThumb() ?
      llvm::ARM::DW_ISA_ARM_thumb : llvm::ARM::DW_ISA_ARM_arm;
  }

  MCSymbol *GetARMSetPICJumpTableLabel2(unsigned uid, unsigned uid2,
                                        const MachineBasicBlock *MBB) const;
  MCSymbol *GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const;

  MCSymbol *GetARMSJLJEHLabel(void) const;

  MCSymbol *GetARMGVSymbol(const GlobalValue *GV);
  
  /// EmitMachineConstantPoolValue - Print a machine constantpool value to
  /// the .s file.
  virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV);
};
} // end namespace llvm

#endif
