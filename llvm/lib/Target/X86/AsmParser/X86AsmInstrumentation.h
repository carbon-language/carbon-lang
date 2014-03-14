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

namespace llvm {

class MCContext;
class MCInst;
class MCParsedAsmOperand;
class MCStreamer;
class MCSubtargetInfo;

class X86AsmInstrumentation;

X86AsmInstrumentation *CreateX86AsmInstrumentation(MCSubtargetInfo &STI);

class X86AsmInstrumentation {
public:
  virtual ~X86AsmInstrumentation();

  // Instruments Inst. Should be called just before the original
  // instruction is sent to Out.
  virtual void InstrumentInstruction(
      const MCInst &Inst, SmallVectorImpl<MCParsedAsmOperand *> &Operands,
      MCContext &Ctx, MCStreamer &Out);

protected:
  friend X86AsmInstrumentation *
  CreateX86AsmInstrumentation(MCSubtargetInfo &STI);

  X86AsmInstrumentation();
};

}  // End llvm namespace

#endif  // X86_ASM_INSTRUMENTATION_H
