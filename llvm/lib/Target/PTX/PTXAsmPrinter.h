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
  virtual void EmitFunctionBodyStart();
  virtual void EmitFunctionBodyEnd();
  virtual void EmitFunctionEntryLabel();
  virtual void EmitInstruction(const MachineInstr *MI);

  unsigned GetOrCreateSourceID(StringRef FileName,
                               StringRef DirName);

  MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol);
  MCOperand lowerOperand(const MachineOperand &MO);

private:
  void EmitVariableDeclaration(const GlobalVariable *gv);
  void EmitFunctionDeclaration();

  StringMap<unsigned> SourceIdMap;
}; // class PTXAsmPrinter
} // namespace llvm

#endif

