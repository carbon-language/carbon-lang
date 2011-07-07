//===-- MipsAsmPrinter.h - Mips LLVM assembly writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Mips Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSASMPRINTER_H
#define MIPSASMPRINTER_H

#include "MipsSubtarget.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class MCStreamer;
class MachineInstr;
class raw_ostream;
class MachineBasicBlock;
class Module;

class LLVM_LIBRARY_VISIBILITY MipsAsmPrinter : public AsmPrinter {
  const MipsSubtarget *Subtarget;
  
public:
  explicit MipsAsmPrinter(TargetMachine &TM,  MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {
    Subtarget = &TM.getSubtarget<MipsSubtarget>();
  }

  virtual const char *getPassName() const {
    return "Mips Assembly Printer";
  }

  // These two methods are autogen'd by tablegen.
  void printInstruction(const MachineInstr *MI, raw_ostream &O);
  static const char *getRegisterName(unsigned RegNo);

  void EmitInstruction(const MachineInstr *MI);
  void printSavedRegsBitmask(raw_ostream &O);
  void printHex32(unsigned int Value, raw_ostream &O);
  void emitFrameDirective();
  const char *getCurrentABIString() const;
  virtual void EmitFunctionEntryLabel();
  virtual void EmitFunctionBodyStart();
  virtual void EmitFunctionBodyEnd();
  virtual bool isBlockOnlyReachableByFallthrough(const MachineBasicBlock*
                                                 MBB) const;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O);
  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
  void printUnsignedImm(const MachineInstr *MI, int opNum, raw_ostream &O);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                       const char *Modifier = 0);
  void printFCCOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                       const char *Modifier = 0);
  void EmitStartOfAsmFile(Module &M);
  virtual MachineLocation getDebugValueLocation(const MachineInstr *MI) const;
  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);
};
}

#endif

