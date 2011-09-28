//===-- PTXAsmPrinter.h - Print machine code to a PTX file ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PTX Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef PTXASMPRINTER_H
#define PTXASMPRINTER_H

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCOperand;

class LLVM_LIBRARY_VISIBILITY PTXAsmPrinter : public AsmPrinter {
public:
  explicit PTXAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {}

  const char *getPassName() const { return "PTX Assembly Printer"; }

  bool doFinalization(Module &M);

  virtual void EmitStartOfAsmFile(Module &M);

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual void EmitFunctionBodyStart();
  virtual void EmitFunctionBodyEnd();

  virtual void EmitInstruction(const MachineInstr *MI);

  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS,
                       const char *Modifier = 0);
  void printReturnOperand(const MachineInstr *MI, int opNum, raw_ostream &OS,
                          const char *Modifier = 0);
  void printPredicateOperand(const MachineInstr *MI, raw_ostream &O);

  void printCall(const MachineInstr *MI, raw_ostream &O);

  unsigned GetOrCreateSourceID(StringRef FileName,
                               StringRef DirName);

  MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol);
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp);

  // autogen'd.
  void printInstruction(const MachineInstr *MI, raw_ostream &OS);
  static const char *getRegisterName(unsigned RegNo);

private:
  void EmitVariableDeclaration(const GlobalVariable *gv);
  void EmitFunctionDeclaration();

  StringMap<unsigned> SourceIdMap;
}; // class PTXAsmPrinter
} // namespace llvm

#endif

