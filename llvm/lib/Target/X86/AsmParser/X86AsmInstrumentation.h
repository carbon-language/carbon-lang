//===- X86AsmInstrumentation.h - Instrument X86 inline assembly *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_ASM_INSTRUMENTATION_H
#define X86_ASM_INSTRUMENTATION_H

#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace llvm {

class MCContext;
class MCInst;
class MCInstrInfo;
class MCParsedAsmOperand;
class MCStreamer;
class MCSubtargetInfo;
class MCTargetOptions;

class X86AsmInstrumentation;

X86AsmInstrumentation *
CreateX86AsmInstrumentation(const MCTargetOptions &MCOptions,
                            const MCContext &Ctx, const MCSubtargetInfo &STI);

class X86AsmInstrumentation {
public:
  virtual ~X86AsmInstrumentation();

  // Tries to instrument and emit instruction.
  virtual void InstrumentAndEmitInstruction(
      const MCInst &Inst,
      SmallVectorImpl<std::unique_ptr<MCParsedAsmOperand>> &Operands,
      MCContext &Ctx, const MCInstrInfo &MII, MCStreamer &Out);

protected:
  friend X86AsmInstrumentation *
  CreateX86AsmInstrumentation(const MCTargetOptions &MCOptions,
                              const MCContext &Ctx, const MCSubtargetInfo &STI);

  X86AsmInstrumentation(const MCSubtargetInfo &STI);

  void EmitInstruction(MCStreamer &Out, const MCInst &Inst);

  const MCSubtargetInfo &STI;
};

} // End llvm namespace

#endif // X86_ASM_INSTRUMENTATION_H
