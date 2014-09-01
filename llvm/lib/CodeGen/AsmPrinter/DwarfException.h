//===-- DwarfException.h - Dwarf Exception Framework -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf exception info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXCEPTION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXCEPTION_H

#include "EHStreamer.h"
#include "llvm/CodeGen/AsmPrinter.h"

namespace llvm {
class MachineFunction;
class ARMTargetStreamer;

class DwarfCFIException : public EHStreamer {
  /// shouldEmitPersonality - Per-function flag to indicate if .cfi_personality
  /// should be emitted.
  bool shouldEmitPersonality;

  /// shouldEmitLSDA - Per-function flag to indicate if .cfi_lsda
  /// should be emitted.
  bool shouldEmitLSDA;

  /// shouldEmitMoves - Per-function flag to indicate if frame moves info
  /// should be emitted.
  bool shouldEmitMoves;

  AsmPrinter::CFIMoveType moveTypeModule;

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfCFIException(AsmPrinter *A);
  virtual ~DwarfCFIException();

  /// endModule - Emit all exception information that should come after the
  /// content.
  void endModule() override;

  /// beginFunction - Gather pre-function exception information.  Assumes being
  /// emitted immediately after the function entry point.
  void beginFunction(const MachineFunction *MF) override;

  /// endFunction - Gather and emit post-function exception information.
  void endFunction(const MachineFunction *) override;
};

class ARMException : public EHStreamer {
  void emitTypeInfos(unsigned TTypeEncoding) override;
  ARMTargetStreamer &getTargetStreamer();

  /// shouldEmitCFI - Per-function flag to indicate if frame CFI info
  /// should be emitted.
  bool shouldEmitCFI;

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  ARMException(AsmPrinter *A);
  virtual ~ARMException();

  /// endModule - Emit all exception information that should come after the
  /// content.
  void endModule() override;

  /// beginFunction - Gather pre-function exception information.  Assumes being
  /// emitted immediately after the function entry point.
  void beginFunction(const MachineFunction *MF) override;

  /// endFunction - Gather and emit post-function exception information.
  void endFunction(const MachineFunction *) override;
};
} // End of namespace llvm

#endif
