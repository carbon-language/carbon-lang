//===-- WinException.h - Windows Exception Handling ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing windows exception info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_WIN64EXCEPTION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_WIN64EXCEPTION_H

#include "EHStreamer.h"

namespace llvm {
class Function;
class GlobalValue;
class MachineFunction;
class MCExpr;
struct WinEHFuncInfo;

class WinException : public EHStreamer {
  /// Per-function flag to indicate if personality info should be emitted.
  bool shouldEmitPersonality = false;

  /// Per-function flag to indicate if the LSDA should be emitted.
  bool shouldEmitLSDA = false;

  /// Per-function flag to indicate if frame moves info should be emitted.
  bool shouldEmitMoves = false;

  /// True if this is a 64-bit target and we should use image relative offsets.
  bool useImageRel32 = false;

  void emitCSpecificHandlerTable();

  void emitCXXFrameHandler3Table(const MachineFunction *MF);

  void extendIP2StateTable(const MachineFunction *MF, const Function *ParentF,
                           WinEHFuncInfo &FuncInfo);

  const MCExpr *create32bitRef(const MCSymbol *Value);
  const MCExpr *create32bitRef(const GlobalValue *GV);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  WinException(AsmPrinter *A);
  ~WinException() override;

  /// Emit all exception information that should come after the content.
  void endModule() override;

  /// Gather pre-function exception information.  Assumes being emitted
  /// immediately after the function entry point.
  void beginFunction(const MachineFunction *MF) override;

  /// Gather and emit post-function exception information.
  void endFunction(const MachineFunction *) override;
};
}

#endif

