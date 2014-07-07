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
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
namespace {

static cl::opt<bool> ClAsanInstrumentAssembly(
    "asan-instrument-assembly",
    cl::desc("instrument assembly with AddressSanitizer checks"), cl::Hidden,
    cl::init(false));

bool IsStackReg(unsigned Reg) {
  return Reg == X86::RSP || Reg == X86::ESP || Reg == X86::SP;
}

std::string FuncName(unsigned AccessSize, bool IsWrite) {
  return std::string("__asan_report_") + (IsWrite ? "store" : "load") +
         utostr(AccessSize);
}

class X86AddressSanitizer : public X86AsmInstrumentation {
public:
  X86AddressSanitizer(const MCSubtargetInfo &STI) : STI(STI) {}
  virtual ~X86AddressSanitizer() {}

  // X86AsmInstrumentation implementation:
  virtual void InstrumentInstruction(
      const MCInst &Inst, OperandVector &Operands, MCContext &Ctx,
      const MCInstrInfo &MII, MCStreamer &Out) override {
    InstrumentMOV(Inst, Operands, Ctx, MII, Out);
  }

  // Should be implemented differently in x86_32 and x86_64 subclasses.
  virtual void InstrumentMemOperandSmallImpl(
      X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
      MCStreamer &Out) = 0;
  virtual void InstrumentMemOperandLargeImpl(
      X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
      MCStreamer &Out) = 0;

  void InstrumentMemOperand(MCParsedAsmOperand &Op, unsigned AccessSize,
                            bool IsWrite, MCContext &Ctx, MCStreamer &Out);
  void InstrumentMOV(const MCInst &Inst, OperandVector &Operands,
                     MCContext &Ctx, const MCInstrInfo &MII, MCStreamer &Out);
  void EmitInstruction(MCStreamer &Out, const MCInst &Inst) {
    Out.EmitInstruction(Inst, STI);
  }

  void EmitLabel(MCStreamer &Out, MCSymbol *Label) { Out.EmitLabel(Label); }

protected:
  const MCSubtargetInfo &STI;
};

void X86AddressSanitizer::InstrumentMemOperand(
    MCParsedAsmOperand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  assert(Op.isMem() && "Op should be a memory operand.");
  assert((AccessSize & (AccessSize - 1)) == 0 && AccessSize <= 16 &&
         "AccessSize should be a power of two, less or equal than 16.");

  X86Operand &MemOp = static_cast<X86Operand &>(Op);
  // FIXME: get rid of this limitation.
  if (IsStackReg(MemOp.getMemBaseReg()) || IsStackReg(MemOp.getMemIndexReg()))
    return;

  // FIXME: take into account load/store alignment.
  if (AccessSize < 8)
    InstrumentMemOperandSmallImpl(MemOp, AccessSize, IsWrite, Ctx, Out);
  else
    InstrumentMemOperandLargeImpl(MemOp, AccessSize, IsWrite, Ctx, Out);
}

void X86AddressSanitizer::InstrumentMOV(
    const MCInst &Inst, OperandVector &Operands, MCContext &Ctx,
    const MCInstrInfo &MII, MCStreamer &Out) {
  // Access size in bytes.
  unsigned AccessSize = 0;

  switch (Inst.getOpcode()) {
  case X86::MOV8mi:
  case X86::MOV8mr:
  case X86::MOV8rm:
    AccessSize = 1;
    break;
  case X86::MOV16mi:
  case X86::MOV16mr:
  case X86::MOV16rm:
    AccessSize = 2;
    break;
  case X86::MOV32mi:
  case X86::MOV32mr:
  case X86::MOV32rm:
    AccessSize = 4;
    break;
  case X86::MOV64mi32:
  case X86::MOV64mr:
  case X86::MOV64rm:
    AccessSize = 8;
    break;
  case X86::MOVAPDmr:
  case X86::MOVAPSmr:
  case X86::MOVAPDrm:
  case X86::MOVAPSrm:
    AccessSize = 16;
    break;
  default:
    return;
  }

  const bool IsWrite = MII.get(Inst.getOpcode()).mayStore();
  for (unsigned Ix = 0; Ix < Operands.size(); ++Ix) {
    assert(Operands[Ix]);
    MCParsedAsmOperand &Op = *Operands[Ix];
    if (Op.isMem())
      InstrumentMemOperand(Op, AccessSize, IsWrite, Ctx, Out);
  }
}

class X86AddressSanitizer32 : public X86AddressSanitizer {
public:
  static const long kShadowOffset = 0x20000000;

  X86AddressSanitizer32(const MCSubtargetInfo &STI)
      : X86AddressSanitizer(STI) {}
  virtual ~X86AddressSanitizer32() {}

  virtual void InstrumentMemOperandSmallImpl(
      X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
      MCStreamer &Out) override;
  virtual void InstrumentMemOperandLargeImpl(
      X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
      MCStreamer &Out) override;

 private:
  void EmitCallAsanReport(MCContext &Ctx, MCStreamer &Out, unsigned AccessSize,
                          bool IsWrite, unsigned AddressReg) {
    EmitInstruction(Out, MCInstBuilder(X86::CLD));
    EmitInstruction(Out, MCInstBuilder(X86::MMX_EMMS));

    EmitInstruction(Out, MCInstBuilder(X86::AND64ri8).addReg(X86::ESP)
                             .addReg(X86::ESP).addImm(-16));
    EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(AddressReg));


    const std::string& Fn = FuncName(AccessSize, IsWrite);
    MCSymbol *FnSym = Ctx.GetOrCreateSymbol(StringRef(Fn));
    const MCSymbolRefExpr *FnExpr =
        MCSymbolRefExpr::Create(FnSym, MCSymbolRefExpr::VK_PLT, Ctx);
    EmitInstruction(Out, MCInstBuilder(X86::CALLpcrel32).addExpr(FnExpr));
  }
};

void X86AddressSanitizer32::InstrumentMemOperandSmallImpl(
    X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::EAX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::ECX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::EDX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSHF32));

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(X86::EAX));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(
      Out, MCInstBuilder(X86::MOV32rr).addReg(X86::ECX).addReg(X86::EAX));
  EmitInstruction(Out, MCInstBuilder(X86::SHR32ri).addReg(X86::ECX)
                           .addReg(X86::ECX).addImm(3));

  {
    MCInst Inst;
    Inst.setOpcode(X86::MOV8rm);
    Inst.addOperand(MCOperand::CreateReg(X86::CL));
    const MCExpr *Disp = MCConstantExpr::Create(kShadowOffset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::ECX, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(Out,
                  MCInstBuilder(X86::TEST8rr).addReg(X86::CL).addReg(X86::CL));
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitInstruction(
      Out, MCInstBuilder(X86::MOV32rr).addReg(X86::EDX).addReg(X86::EAX));
  EmitInstruction(Out, MCInstBuilder(X86::AND32ri).addReg(X86::EDX)
                           .addReg(X86::EDX).addImm(7));

  switch (AccessSize) {
  case 1:
    break;
  case 2: {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(X86::EDX));

    const MCExpr *Disp = MCConstantExpr::Create(1, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::EDX, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
    break;
  }
  case 4:
    EmitInstruction(Out, MCInstBuilder(X86::ADD32ri8).addReg(X86::EDX)
                             .addReg(X86::EDX).addImm(3));
    break;
  default:
    assert(false && "Incorrect access size");
    break;
  }

  EmitInstruction(
      Out, MCInstBuilder(X86::MOVSX32rr8).addReg(X86::ECX).addReg(X86::CL));
  EmitInstruction(
      Out, MCInstBuilder(X86::CMP32rr).addReg(X86::EDX).addReg(X86::ECX));
  EmitInstruction(Out, MCInstBuilder(X86::JL_4).addExpr(DoneExpr));

  EmitCallAsanReport(Ctx, Out, AccessSize, IsWrite, X86::EAX);
  EmitLabel(Out, DoneSym);

  EmitInstruction(Out, MCInstBuilder(X86::POPF32));
  EmitInstruction(Out, MCInstBuilder(X86::POP32r).addReg(X86::EDX));
  EmitInstruction(Out, MCInstBuilder(X86::POP32r).addReg(X86::ECX));
  EmitInstruction(Out, MCInstBuilder(X86::POP32r).addReg(X86::EAX));
}

void X86AddressSanitizer32::InstrumentMemOperandLargeImpl(
    X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::EAX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSH32r).addReg(X86::ECX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSHF32));

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(X86::EAX));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  EmitInstruction(
      Out, MCInstBuilder(X86::MOV32rr).addReg(X86::ECX).addReg(X86::EAX));
  EmitInstruction(Out, MCInstBuilder(X86::SHR32ri).addReg(X86::ECX)
                           .addReg(X86::ECX).addImm(3));
  {
    MCInst Inst;
    switch (AccessSize) {
      case 8:
        Inst.setOpcode(X86::CMP8mi);
        break;
      case 16:
        Inst.setOpcode(X86::CMP16mi);
        break;
      default:
        assert(false && "Incorrect access size");
        break;
    }
    const MCExpr *Disp = MCConstantExpr::Create(kShadowOffset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::ECX, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    Inst.addOperand(MCOperand::CreateImm(0));
    EmitInstruction(Out, Inst);
  }
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitCallAsanReport(Ctx, Out, AccessSize, IsWrite, X86::EAX);
  EmitLabel(Out, DoneSym);

  EmitInstruction(Out, MCInstBuilder(X86::POPF32));
  EmitInstruction(Out, MCInstBuilder(X86::POP32r).addReg(X86::ECX));
  EmitInstruction(Out, MCInstBuilder(X86::POP32r).addReg(X86::EAX));
}

class X86AddressSanitizer64 : public X86AddressSanitizer {
public:
  static const long kShadowOffset = 0x7fff8000;

  X86AddressSanitizer64(const MCSubtargetInfo &STI)
      : X86AddressSanitizer(STI) {}
  virtual ~X86AddressSanitizer64() {}

  virtual void InstrumentMemOperandSmallImpl(
      X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
      MCStreamer &Out) override;
  virtual void InstrumentMemOperandLargeImpl(
      X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
      MCStreamer &Out) override;

private:
  void EmitAdjustRSP(MCContext &Ctx, MCStreamer &Out, long Offset) {
    MCInst Inst;
    Inst.setOpcode(X86::LEA64r);
    Inst.addOperand(MCOperand::CreateReg(X86::RSP));

    const MCExpr *Disp = MCConstantExpr::Create(Offset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::RSP, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  void EmitCallAsanReport(MCContext &Ctx, MCStreamer &Out, unsigned AccessSize,
                          bool IsWrite) {
    EmitInstruction(Out, MCInstBuilder(X86::CLD));
    EmitInstruction(Out, MCInstBuilder(X86::MMX_EMMS));

    EmitInstruction(Out, MCInstBuilder(X86::AND64ri8).addReg(X86::RSP)
                             .addReg(X86::RSP).addImm(-16));

    const std::string& Fn = FuncName(AccessSize, IsWrite);
    MCSymbol *FnSym = Ctx.GetOrCreateSymbol(StringRef(Fn));
    const MCSymbolRefExpr *FnExpr =
        MCSymbolRefExpr::Create(FnSym, MCSymbolRefExpr::VK_PLT, Ctx);
    EmitInstruction(Out, MCInstBuilder(X86::CALL64pcrel32).addExpr(FnExpr));
  }
};

void X86AddressSanitizer64::InstrumentMemOperandSmallImpl(
    X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  EmitAdjustRSP(Ctx, Out, -128);
  EmitInstruction(Out, MCInstBuilder(X86::PUSH64r).addReg(X86::RAX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSH64r).addReg(X86::RCX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSH64r).addReg(X86::RDI));
  EmitInstruction(Out, MCInstBuilder(X86::PUSHF64));
  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA64r);
    Inst.addOperand(MCOperand::CreateReg(X86::RDI));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  EmitInstruction(
      Out, MCInstBuilder(X86::MOV64rr).addReg(X86::RAX).addReg(X86::RDI));
  EmitInstruction(Out, MCInstBuilder(X86::SHR64ri).addReg(X86::RAX)
                           .addReg(X86::RAX).addImm(3));
  {
    MCInst Inst;
    Inst.setOpcode(X86::MOV8rm);
    Inst.addOperand(MCOperand::CreateReg(X86::AL));
    const MCExpr *Disp = MCConstantExpr::Create(kShadowOffset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::RAX, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(Out,
                  MCInstBuilder(X86::TEST8rr).addReg(X86::AL).addReg(X86::AL));
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitInstruction(
      Out, MCInstBuilder(X86::MOV32rr).addReg(X86::ECX).addReg(X86::EDI));
  EmitInstruction(Out, MCInstBuilder(X86::AND32ri).addReg(X86::ECX)
                           .addReg(X86::ECX).addImm(7));

  switch (AccessSize) {
  case 1:
    break;
  case 2: {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(X86::ECX));

    const MCExpr *Disp = MCConstantExpr::Create(1, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::ECX, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
    break;
  }
  case 4:
    EmitInstruction(Out, MCInstBuilder(X86::ADD32ri8).addReg(X86::ECX)
                             .addReg(X86::ECX).addImm(3));
    break;
  default:
    assert(false && "Incorrect access size");
    break;
  }

  EmitInstruction(
      Out, MCInstBuilder(X86::MOVSX32rr8).addReg(X86::EAX).addReg(X86::AL));
  EmitInstruction(
      Out, MCInstBuilder(X86::CMP32rr).addReg(X86::ECX).addReg(X86::EAX));
  EmitInstruction(Out, MCInstBuilder(X86::JL_4).addExpr(DoneExpr));

  EmitCallAsanReport(Ctx, Out, AccessSize, IsWrite);
  EmitLabel(Out, DoneSym);

  EmitInstruction(Out, MCInstBuilder(X86::POPF64));
  EmitInstruction(Out, MCInstBuilder(X86::POP64r).addReg(X86::RDI));
  EmitInstruction(Out, MCInstBuilder(X86::POP64r).addReg(X86::RCX));
  EmitInstruction(Out, MCInstBuilder(X86::POP64r).addReg(X86::RAX));
  EmitAdjustRSP(Ctx, Out, 128);
}

void X86AddressSanitizer64::InstrumentMemOperandLargeImpl(
    X86Operand &Op, unsigned AccessSize, bool IsWrite, MCContext &Ctx,
    MCStreamer &Out) {
  EmitAdjustRSP(Ctx, Out, -128);
  EmitInstruction(Out, MCInstBuilder(X86::PUSH64r).addReg(X86::RAX));
  EmitInstruction(Out, MCInstBuilder(X86::PUSHF64));

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA64r);
    Inst.addOperand(MCOperand::CreateReg(X86::RAX));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  EmitInstruction(Out, MCInstBuilder(X86::SHR64ri).addReg(X86::RAX)
                           .addReg(X86::RAX).addImm(3));
  {
    MCInst Inst;
    switch (AccessSize) {
    case 8:
      Inst.setOpcode(X86::CMP8mi);
      break;
    case 16:
      Inst.setOpcode(X86::CMP16mi);
      break;
    default:
      assert(false && "Incorrect access size");
      break;
    }
    const MCExpr *Disp = MCConstantExpr::Create(kShadowOffset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, X86::RAX, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    Inst.addOperand(MCOperand::CreateImm(0));
    EmitInstruction(Out, Inst);
  }

  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitCallAsanReport(Ctx, Out, AccessSize, IsWrite);
  EmitLabel(Out, DoneSym);

  EmitInstruction(Out, MCInstBuilder(X86::POPF64));
  EmitInstruction(Out, MCInstBuilder(X86::POP64r).addReg(X86::RAX));
  EmitAdjustRSP(Ctx, Out, 128);
}

} // End anonymous namespace

X86AsmInstrumentation::X86AsmInstrumentation() {}
X86AsmInstrumentation::~X86AsmInstrumentation() {}

void X86AsmInstrumentation::InstrumentInstruction(
    const MCInst &Inst, OperandVector &Operands, MCContext &Ctx,
    const MCInstrInfo &MII, MCStreamer &Out) {}

X86AsmInstrumentation *
CreateX86AsmInstrumentation(const MCTargetOptions &MCOptions,
                            const MCContext &Ctx, const MCSubtargetInfo &STI) {
  Triple T(STI.getTargetTriple());
  const bool hasCompilerRTSupport = T.isOSLinux();
  if (ClAsanInstrumentAssembly && hasCompilerRTSupport &&
      MCOptions.SanitizeAddress) {
    if ((STI.getFeatureBits() & X86::Mode32Bit) != 0)
      return new X86AddressSanitizer32(STI);
    if ((STI.getFeatureBits() & X86::Mode64Bit) != 0)
      return new X86AddressSanitizer64(STI);
  }
  return new X86AsmInstrumentation();
}

} // End llvm namespace
