//===-- ARMAsmPrinter.h - ARM implementation of AsmPrinter ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ARMASMPRINTER_H
#define ARMASMPRINTER_H

#include "ARMSubtarget.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ARMFunctionInfo;
class MCOperand;
class MachineConstantPool;
class MachineOperand;

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

  /// InConstantPool - Maintain state when emitting a sequence of constant
  /// pool entries so we can properly mark them as data regions.
  bool InConstantPool;
public:
  explicit ARMAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), AFI(NULL), MCP(NULL), InConstantPool(false) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

  const char *getPassName() const override {
    return "ARM Assembly / Object Emitter";
  }

  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                    const char *Modifier = 0);

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O) override;

  void emitInlineAsmEnd(const MCSubtargetInfo &StartInfo,
                        const MCSubtargetInfo *EndInfo) const override;

  void EmitJumpTable(const MachineInstr *MI);
  void EmitJump2Table(const MachineInstr *MI);
  void EmitInstruction(const MachineInstr *MI) override;
  bool runOnMachineFunction(MachineFunction &F) override;

  void EmitConstantPool() override {
    // we emit constant pools customly!
  }
  void EmitFunctionBodyEnd() override;
  void EmitFunctionEntryLabel() override;
  void EmitStartOfAsmFile(Module &M) override;
  void EmitEndOfAsmFile(Module &M) override;
  void EmitXXStructor(const Constant *CV) override;

  // lowerOperand - Convert a MachineOperand into the equivalent MCOperand.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp);

private:
  // Helpers for EmitStartOfAsmFile() and EmitEndOfAsmFile()
  void emitAttributes();

  // Generic helper used to emit e.g. ARMv5 mul pseudos
  void EmitPatchedInstruction(const MachineInstr *MI, unsigned TargetOpc);

  void EmitUnwindingInstruction(const MachineInstr *MI);

  // emitPseudoExpansionLowering - tblgen'erated.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

public:
  unsigned getISAEncoding() override {
    // ARM/Darwin adds ISA to the DWARF info for each function.
    if (!Subtarget->isTargetMachO())
      return 0;
    return Subtarget->isThumb() ?
      ARM::DW_ISA_ARM_thumb : ARM::DW_ISA_ARM_arm;
  }

private:
  MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol);
  MCSymbol *GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const;

  MCSymbol *GetARMSJLJEHLabel() const;

  MCSymbol *GetARMGVSymbol(const GlobalValue *GV, unsigned char TargetFlags);

public:
  /// EmitMachineConstantPoolValue - Print a machine constantpool value to
  /// the .s file.
  void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;
};
} // end namespace llvm

#endif
