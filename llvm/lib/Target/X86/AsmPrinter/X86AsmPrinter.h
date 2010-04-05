//===-- X86AsmPrinter.h - Convert X86 LLVM code to assembly -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AT&T assembly code printer class.
//
//===----------------------------------------------------------------------===//

#ifndef X86ASMPRINTER_H
#define X86ASMPRINTER_H

#include "../X86.h"
#include "../X86MachineFunctionInfo.h"
#include "../X86TargetMachine.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineJumpTableInfo;
class MCContext;
class MCInst;
class MCStreamer;
class MCSymbol;

class VISIBILITY_HIDDEN X86AsmPrinter : public AsmPrinter {
  const X86Subtarget *Subtarget;
 public:
  explicit X86AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {
    Subtarget = &TM.getSubtarget<X86Subtarget>();
  }

  virtual const char *getPassName() const {
    return "X86 AT&T-Style Assembly Printer";
  }
  
  const X86Subtarget &getSubtarget() const { return *Subtarget; }

  virtual void EmitStartOfAsmFile(Module &M);

  virtual void EmitEndOfAsmFile(Module &M);
  
  virtual void EmitInstruction(const MachineInstr *MI);
  
  void printSymbolOperand(const MachineOperand &MO, raw_ostream &O);

  // These methods are used by the tablegen'erated instruction printer.
  void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O,
                    const char *Modifier = 0);
  void print_pcrel_imm(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);

  bool printAsmMRegister(const MachineOperand &MO, char Mode, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &OS);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &OS);

  void printMachineInstruction(const MachineInstr *MI);
  void printSSECC(const MachineInstr *MI, unsigned Op, raw_ostream &O);
  void printMemReference(const MachineInstr *MI, unsigned Op, raw_ostream &O,
                         const char *Modifier=NULL);
  void printLeaMemReference(const MachineInstr *MI, unsigned Op, raw_ostream &O,
                            const char *Modifier=NULL);

  void printPICLabel(const MachineInstr *MI, unsigned Op, raw_ostream &O);

  void PrintPICBaseSymbol(raw_ostream &O) const;
  
  bool runOnMachineFunction(MachineFunction &F);
  
  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);
};

} // end namespace llvm

#endif
