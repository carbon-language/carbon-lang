//===-- X86AsmInstrumentation.cpp - Instrument X86 inline assembly C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86AsmInstrumentation.h"
#include "X86Operand.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"

namespace llvm {
namespace {

bool IsStackReg(unsigned Reg) {
  return Reg == X86::RSP || Reg == X86::ESP || Reg == X86::SP;
}

std::string FuncName(unsigned AccessSize, bool IsWrite) {
  return std::string("__sanitizer_sanitize_") + (IsWrite ? "store" : "load") +
         (utostr(AccessSize));
}

class X86AddressSanitizer : public X86AsmInstrumentation {
public:
  X86AddressSanitizer(const MCSubtargetInfo &STI) : STI(STI) {}
  virtual ~X86AddressSanitizer() {}

  // X86AsmInstrumentation implementation:
  virtual void InstrumentInstruction(
      const MCInst &Inst, SmallVectorImpl<MCParsedAsmOperand *> &Operands,
      MCContext &Ctx, MCStreamer &Out) override {
    InstrumentMOV(Inst, Operands, Ctx, Out);
  }

  // Should be implemented differently in x86_32 and x86_64 subclasses.
  virtual void InstrumentMemOperandImpl(X86Operand *Op, unsigned AccessSize,
                                        bool IsWrite, MCContext &Ctx,
                                        MCStreamer &Out) = 0;

  void InstrumentMemOperand(MCParsedAsmOperand *Op, unsigned AccessSize,
                            bool IsWrite, MCContext &Ctx, MCStreamer &Out);
  void InstrumentMOV(const MCInst &Inst,
                     SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                     MCContext &Ctx, MCStreamer &Out);
  void EmitInstruction(MCStreamer &Out, const MCInst &Inst) {
    Out.EmitInstruction(Inst, STI);
  }

protected:
  const MCSubtargetInfo &STI;
};

void X86AddressSanitizer::InstrumentMemOperand(
    MCParsedAsmOperand *Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  assert(Op && Op->isMem() && "Op should be a memory operand.");
  assert((AccessSize & (AccessSize - 1)) == 0 && AccessSize <= 16 &&
         "AccessSize should be a power of two, less or equal than 16.");

  X86Operand *MemOp = static_cast<X86Operand *>(Op);
  // FIXME: get rid of this limitation.
  if (IsStackReg(MemOp->getMemBaseReg()) || IsStackReg(MemOp->getMemIndexReg()))
    return;

  InstrumentMemOperandImpl(MemOp, AccessSize, IsWrite, Ctx, Out);
}

void X86AddressSanitizer::InstrumentMOV(
    const MCInst &Inst, SmallVectorImpl<MCParsedAsmOperand *> &Operands,
    MCContext &Ctx, MCStreamer &Out) {
  // Access size in bytes.
  unsigned AccessSize = 0;
  unsigned long OpIx = Operands.size();
  switch (Inst.getOpcode()) {
  case X86::MOV8mi:
  case X86::MOV8mr:
    AccessSize = 1;
    OpIx = 2;
    break;
  case X86::MOV8rm:
    AccessSize = 1;
    OpIx = 1;
    break;
  case X86::MOV16mi:
  case X86::MOV16mr:
    AccessSize = 2;
    OpIx = 2;
    break;
  case X86::MOV16rm:
    AccessSize = 2;
    OpIx = 1;
    break;
  case X86::MOV32mi:
  case X86::MOV32mr:
    AccessSize = 4;
    OpIx = 2;
    break;
  case X86::MOV32rm:
    AccessSize = 4;
    OpIx = 1;
    break;
  case X86::MOV64mi32:
  case X86::MOV64mr:
    AccessSize = 8;
    OpIx = 2;
    break;
  case X86::MOV64rm:
    AccessSize = 8;
    OpIx = 1;
    break;
  case X86::MOVAPDmr:
  case X86::MOVAPSmr:
    AccessSize = 16;
    OpIx = 2;
    break;
  case X86::MOVAPDrm:
  case X86::MOVAPSrm:
    AccessSize = 16;
    OpIx = 1;
    break;
  }
  if (OpIx >= Operands.size())
    return;

  const bool IsWrite = (OpIx != 1);
  InstrumentMemOperand(Operands[OpIx], AccessSize, IsWrite, Ctx, Out);
}

class X86AddressSanitizer32 : public X86AddressSanitizer {
public:
  X86AddressSanitizer32(const MCSubtargetInfo &STI)
      : X86AddressSanitizer(STI) {}
  virtual ~X86AddressSanitizer32() {}

  virtual void InstrumentMemOperandImpl(X86Operand *Op, unsigned AccessSize,
                                        bool IsWrite, MCContext &Ctx,
                                        MCStreamer &Out) override;
};

void X86AddressSanitizer32::InstrumentMemOperandImpl(
    X86Operand *Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  // FIXME: emit .cfi directives for correct stack unwinding.
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::EAX));
  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(X86::EAX));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::EAX));
  {
    const std::string Func = FuncName(AccessSize, IsWrite);
    const MCSymbol *FuncSym = Ctx.GetOrCreateSymbol(StringRef(Func));
    const MCSymbolRefExpr *FuncExpr =
        MCSymbolRefExpr::Create(FuncSym, MCSymbolRefExpr::VK_PLT, Ctx);
    EmitInstruction(Out, MCInstBuilder(X86::CALLpcrel32).addExpr(FuncExpr));
  }
  EmitInstruction(Out, MCInstBuilder(X86::ADD32ri).addReg(X86::ESP)
                           .addReg(X86::ESP).addImm(4));
  EmitInstruction(Out, MCInstBuilder(X86::POP32r).addReg(X86::EAX));
}

class X86AddressSanitizer64 : public X86AddressSanitizer {
public:
  X86AddressSanitizer64(const MCSubtargetInfo &STI)
      : X86AddressSanitizer(STI) {}
  virtual ~X86AddressSanitizer64() {}

  virtual void InstrumentMemOperandImpl(X86Operand *Op, unsigned AccessSize,
                                        bool IsWrite, MCContext &Ctx,
                                        MCStreamer &Out) override;
};

void X86AddressSanitizer64::InstrumentMemOperandImpl(
    X86Operand *Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  // FIXME: emit .cfi directives for correct stack unwinding.
  // Set %rsp below current red zone (128 bytes wide)
  EmitInstruction(Out, MCInstBuilder(X86::SUB64ri32).addReg(X86::RSP)
                           .addReg(X86::RSP).addImm(128));
  EmitInstruction(Out, MCInstBuilder(X86::PUSH64r).addReg(X86::RDI));
  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA64r);
    Inst.addOperand(MCOperand::CreateReg(X86::RDI));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  {
    const std::string Func = FuncName(AccessSize, IsWrite);
    const MCSymbol *FuncSym = Ctx.GetOrCreateSymbol(StringRef(Func));
    const MCSymbolRefExpr *FuncExpr =
        MCSymbolRefExpr::Create(FuncSym, MCSymbolRefExpr::VK_PLT, Ctx);
    EmitInstruction(Out, MCInstBuilder(X86::CALL64pcrel32).addExpr(FuncExpr));
  }
  EmitInstruction(Out, MCInstBuilder(X86::POP64r).addReg(X86::RDI));
  EmitInstruction(Out, MCInstBuilder(X86::ADD64ri32).addReg(X86::RSP)
                           .addReg(X86::RSP).addImm(128));
}

} // End anonymous namespace

X86AsmInstrumentation::X86AsmInstrumentation() {}
X86AsmInstrumentation::~X86AsmInstrumentation() {}

void X86AsmInstrumentation::InstrumentInstruction(
    const MCInst &Inst, SmallVectorImpl<MCParsedAsmOperand *> &Operands,
    MCContext &Ctx, MCStreamer &Out) {}

X86AsmInstrumentation *
CreateX86AsmInstrumentation(const MCTargetOptions &MCOptions, const MCContext &Ctx,
                            const MCSubtargetInfo &STI) {
  if (MCOptions.SanitizeAddress) {
    if ((STI.getFeatureBits() & X86::Mode32Bit) != 0)
      return new X86AddressSanitizer32(STI);
    if ((STI.getFeatureBits() & X86::Mode64Bit) != 0)
      return new X86AddressSanitizer64(STI);
  }
  return new X86AsmInstrumentation();
}

} // End llvm namespace
