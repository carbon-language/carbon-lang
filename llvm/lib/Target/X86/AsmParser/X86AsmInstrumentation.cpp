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
#include "X86RegisterInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineValueType.h"
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

bool IsSmallMemAccess(unsigned AccessSize) { return AccessSize < 8; }

std::string FuncName(unsigned AccessSize, bool IsWrite) {
  return std::string("__asan_report_") + (IsWrite ? "store" : "load") +
         utostr(AccessSize);
}

class X86AddressSanitizer : public X86AsmInstrumentation {
public:
  struct RegisterContext {
    RegisterContext(unsigned AddressReg, unsigned ShadowReg,
                    unsigned ScratchReg)
        : AddressReg(AddressReg), ShadowReg(ShadowReg), ScratchReg(ScratchReg) {
    }

    unsigned addressReg(MVT::SimpleValueType VT) const {
      return getX86SubSuperRegister(AddressReg, VT);
    }

    unsigned shadowReg(MVT::SimpleValueType VT) const {
      return getX86SubSuperRegister(ShadowReg, VT);
    }

    unsigned scratchReg(MVT::SimpleValueType VT) const {
      return getX86SubSuperRegister(ScratchReg, VT);
    }

    const unsigned AddressReg;
    const unsigned ShadowReg;
    const unsigned ScratchReg;
  };

  X86AddressSanitizer(const MCSubtargetInfo &STI)
      : X86AsmInstrumentation(STI), RepPrefix(false) {}
  virtual ~X86AddressSanitizer() {}

  // X86AsmInstrumentation implementation:
  virtual void InstrumentAndEmitInstruction(const MCInst &Inst,
                                            OperandVector &Operands,
                                            MCContext &Ctx,
                                            const MCInstrInfo &MII,
                                            MCStreamer &Out) override {
    InstrumentMOVS(Inst, Operands, Ctx, MII, Out);
    if (RepPrefix)
      EmitInstruction(Out, MCInstBuilder(X86::REP_PREFIX));

    InstrumentMOV(Inst, Operands, Ctx, MII, Out);

    RepPrefix = (Inst.getOpcode() == X86::REP_PREFIX);
    if (!RepPrefix)
      EmitInstruction(Out, Inst);
  }

  // Should be implemented differently in x86_32 and x86_64 subclasses.
  virtual void StoreFlags(MCStreamer &Out) = 0;

  virtual void RestoreFlags(MCStreamer &Out) = 0;

  // Adjusts up stack and saves all registers used in instrumentation.
  virtual void InstrumentMemOperandPrologue(const RegisterContext &RegCtx,
                                            MCContext &Ctx,
                                            MCStreamer &Out) = 0;

  // Restores all registers used in instrumentation and adjusts stack.
  virtual void InstrumentMemOperandEpilogue(const RegisterContext &RegCtx,
                                            MCContext &Ctx,
                                            MCStreamer &Out) = 0;

  virtual void InstrumentMemOperandSmall(X86Operand &Op, unsigned AccessSize,
                                         bool IsWrite,
                                         const RegisterContext &RegCtx,
                                         MCContext &Ctx, MCStreamer &Out) = 0;
  virtual void InstrumentMemOperandLarge(X86Operand &Op, unsigned AccessSize,
                                         bool IsWrite,
                                         const RegisterContext &RegCtx,
                                         MCContext &Ctx, MCStreamer &Out) = 0;

  virtual void InstrumentMOVSImpl(unsigned AccessSize, MCContext &Ctx,
                                  MCStreamer &Out) = 0;

  void InstrumentMemOperand(X86Operand &Op, unsigned AccessSize, bool IsWrite,
                            const RegisterContext &RegCtx, MCContext &Ctx,
                            MCStreamer &Out);
  void InstrumentMOVSBase(unsigned DstReg, unsigned SrcReg, unsigned CntReg,
                          unsigned AccessSize, MCContext &Ctx, MCStreamer &Out);

  void InstrumentMOVS(const MCInst &Inst, OperandVector &Operands,
                      MCContext &Ctx, const MCInstrInfo &MII, MCStreamer &Out);
  void InstrumentMOV(const MCInst &Inst, OperandVector &Operands,
                     MCContext &Ctx, const MCInstrInfo &MII, MCStreamer &Out);

protected:
  void EmitLabel(MCStreamer &Out, MCSymbol *Label) { Out.EmitLabel(Label); }

  // True when previous instruction was actually REP prefix.
  bool RepPrefix;
};

void X86AddressSanitizer::InstrumentMemOperand(
    X86Operand &Op, unsigned AccessSize, bool IsWrite,
    const RegisterContext &RegCtx, MCContext &Ctx, MCStreamer &Out) {
  assert(Op.isMem() && "Op should be a memory operand.");
  assert((AccessSize & (AccessSize - 1)) == 0 && AccessSize <= 16 &&
         "AccessSize should be a power of two, less or equal than 16.");
  // FIXME: take into account load/store alignment.
  if (IsSmallMemAccess(AccessSize))
    InstrumentMemOperandSmall(Op, AccessSize, IsWrite, RegCtx, Ctx, Out);
  else
    InstrumentMemOperandLarge(Op, AccessSize, IsWrite, RegCtx, Ctx, Out);
}

void X86AddressSanitizer::InstrumentMOVSBase(unsigned DstReg, unsigned SrcReg,
                                             unsigned CntReg,
                                             unsigned AccessSize,
                                             MCContext &Ctx, MCStreamer &Out) {
  // FIXME: check whole ranges [DstReg .. DstReg + AccessSize * (CntReg - 1)]
  // and [SrcReg .. SrcReg + AccessSize * (CntReg - 1)].
  RegisterContext RegCtx(X86::RDX /* AddressReg */, X86::RAX /* ShadowReg */,
                         IsSmallMemAccess(AccessSize)
                             ? X86::RBX
                             : X86::NoRegister /* ScratchReg */);

  InstrumentMemOperandPrologue(RegCtx, Ctx, Out);

  // Test (%SrcReg)
  {
    const MCExpr *Disp = MCConstantExpr::Create(0, Ctx);
    std::unique_ptr<X86Operand> Op(X86Operand::CreateMem(
        0, Disp, SrcReg, 0, AccessSize, SMLoc(), SMLoc()));
    InstrumentMemOperand(*Op, AccessSize, false /* IsWrite */, RegCtx, Ctx,
                         Out);
  }

  // Test -1(%SrcReg, %CntReg, AccessSize)
  {
    const MCExpr *Disp = MCConstantExpr::Create(-1, Ctx);
    std::unique_ptr<X86Operand> Op(X86Operand::CreateMem(
        0, Disp, SrcReg, CntReg, AccessSize, SMLoc(), SMLoc()));
    InstrumentMemOperand(*Op, AccessSize, false /* IsWrite */, RegCtx, Ctx,
                         Out);
  }

  // Test (%DstReg)
  {
    const MCExpr *Disp = MCConstantExpr::Create(0, Ctx);
    std::unique_ptr<X86Operand> Op(X86Operand::CreateMem(
        0, Disp, DstReg, 0, AccessSize, SMLoc(), SMLoc()));
    InstrumentMemOperand(*Op, AccessSize, true /* IsWrite */, RegCtx, Ctx, Out);
  }

  // Test -1(%DstReg, %CntReg, AccessSize)
  {
    const MCExpr *Disp = MCConstantExpr::Create(-1, Ctx);
    std::unique_ptr<X86Operand> Op(X86Operand::CreateMem(
        0, Disp, DstReg, CntReg, AccessSize, SMLoc(), SMLoc()));
    InstrumentMemOperand(*Op, AccessSize, true /* IsWrite */, RegCtx, Ctx, Out);
  }

  InstrumentMemOperandEpilogue(RegCtx, Ctx, Out);
}

void X86AddressSanitizer::InstrumentMOVS(const MCInst &Inst,
                                         OperandVector &Operands,
                                         MCContext &Ctx, const MCInstrInfo &MII,
                                         MCStreamer &Out) {
  // Access size in bytes.
  unsigned AccessSize = 0;

  switch (Inst.getOpcode()) {
  case X86::MOVSB:
    AccessSize = 1;
    break;
  case X86::MOVSW:
    AccessSize = 2;
    break;
  case X86::MOVSL:
    AccessSize = 4;
    break;
  case X86::MOVSQ:
    AccessSize = 8;
    break;
  default:
    return;
  }

  InstrumentMOVSImpl(AccessSize, Ctx, Out);
}

void X86AddressSanitizer::InstrumentMOV(const MCInst &Inst,
                                        OperandVector &Operands, MCContext &Ctx,
                                        const MCInstrInfo &MII,
                                        MCStreamer &Out) {
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
  RegisterContext RegCtx(X86::RDI /* AddressReg */, X86::RAX /* ShadowReg */,
                         IsSmallMemAccess(AccessSize)
                             ? X86::RCX
                             : X86::NoRegister /* ScratchReg */);

  for (unsigned Ix = 0; Ix < Operands.size(); ++Ix) {
    assert(Operands[Ix]);
    MCParsedAsmOperand &Op = *Operands[Ix];
    if (Op.isMem()) {
      X86Operand &MemOp = static_cast<X86Operand &>(Op);
      // FIXME: get rid of this limitation.
      if (IsStackReg(MemOp.getMemBaseReg()) ||
          IsStackReg(MemOp.getMemIndexReg())) {
        continue;
      }

      InstrumentMemOperandPrologue(RegCtx, Ctx, Out);
      InstrumentMemOperand(MemOp, AccessSize, IsWrite, RegCtx, Ctx, Out);
      InstrumentMemOperandEpilogue(RegCtx, Ctx, Out);
    }
  }
}

class X86AddressSanitizer32 : public X86AddressSanitizer {
public:
  static const long kShadowOffset = 0x20000000;

  X86AddressSanitizer32(const MCSubtargetInfo &STI)
      : X86AddressSanitizer(STI) {}

  virtual ~X86AddressSanitizer32() {}

  virtual void StoreFlags(MCStreamer &Out) override {
    EmitInstruction(Out, MCInstBuilder(X86::PUSHF32));
  }

  virtual void RestoreFlags(MCStreamer &Out) override {
    EmitInstruction(Out, MCInstBuilder(X86::POPF32));
  }

  virtual void InstrumentMemOperandPrologue(const RegisterContext &RegCtx,
                                            MCContext &Ctx,
                                            MCStreamer &Out) override {
    const MCRegisterInfo* MRI = Ctx.getRegisterInfo();
    if (MRI && FrameReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::PUSH32r).addReg(X86::EBP));
      if (FrameReg == X86::ESP) {
        Out.EmitCFIAdjustCfaOffset(4 /* byte size of the FrameReg */);
        Out.EmitCFIRelOffset(
            MRI->getDwarfRegNum(X86::EBP, true /* IsEH */), 0);
      }
      EmitInstruction(
          Out, MCInstBuilder(X86::MOV32rr).addReg(X86::EBP).addReg(FrameReg));
      Out.EmitCFIRememberState();
      Out.EmitCFIDefCfaRegister(
          MRI->getDwarfRegNum(X86::EBP, true /* IsEH */));
    }

    EmitInstruction(
        Out, MCInstBuilder(X86::PUSH32r).addReg(RegCtx.addressReg(MVT::i32)));
    EmitInstruction(
        Out, MCInstBuilder(X86::PUSH32r).addReg(RegCtx.shadowReg(MVT::i32)));
    if (RegCtx.ScratchReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::PUSH32r).addReg(RegCtx.scratchReg(MVT::i32)));
    }
    StoreFlags(Out);
  }

  virtual void InstrumentMemOperandEpilogue(const RegisterContext &RegCtx,
                                            MCContext &Ctx,
                                            MCStreamer &Out) override {
    RestoreFlags(Out);
    if (RegCtx.ScratchReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::POP32r).addReg(RegCtx.scratchReg(MVT::i32)));
    }
    EmitInstruction(
        Out, MCInstBuilder(X86::POP32r).addReg(RegCtx.shadowReg(MVT::i32)));
    EmitInstruction(
        Out, MCInstBuilder(X86::POP32r).addReg(RegCtx.addressReg(MVT::i32)));

    if (Ctx.getRegisterInfo() && FrameReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::POP32r).addReg(X86::EBP));
      Out.EmitCFIRestoreState();
      if (FrameReg == X86::ESP)
        Out.EmitCFIAdjustCfaOffset(-4 /* byte size of the FrameReg */);
    }
  }

  virtual void InstrumentMemOperandSmall(X86Operand &Op, unsigned AccessSize,
                                         bool IsWrite,
                                         const RegisterContext &RegCtx,
                                         MCContext &Ctx,
                                         MCStreamer &Out) override;
  virtual void InstrumentMemOperandLarge(X86Operand &Op, unsigned AccessSize,
                                         bool IsWrite,
                                         const RegisterContext &RegCtx,
                                         MCContext &Ctx,
                                         MCStreamer &Out) override;
  virtual void InstrumentMOVSImpl(unsigned AccessSize, MCContext &Ctx,
                                  MCStreamer &Out) override;

private:
  void EmitCallAsanReport(unsigned AccessSize, bool IsWrite, MCContext &Ctx,
                          MCStreamer &Out, const RegisterContext &RegCtx) {
    EmitInstruction(Out, MCInstBuilder(X86::CLD));
    EmitInstruction(Out, MCInstBuilder(X86::MMX_EMMS));

    EmitInstruction(Out, MCInstBuilder(X86::AND64ri8)
                             .addReg(X86::ESP)
                             .addReg(X86::ESP)
                             .addImm(-16));
    EmitInstruction(
        Out, MCInstBuilder(X86::PUSH32r).addReg(RegCtx.addressReg(MVT::i32)));

    const std::string &Fn = FuncName(AccessSize, IsWrite);
    MCSymbol *FnSym = Ctx.GetOrCreateSymbol(StringRef(Fn));
    const MCSymbolRefExpr *FnExpr =
        MCSymbolRefExpr::Create(FnSym, MCSymbolRefExpr::VK_PLT, Ctx);
    EmitInstruction(Out, MCInstBuilder(X86::CALLpcrel32).addExpr(FnExpr));
  }
};

void X86AddressSanitizer32::InstrumentMemOperandSmall(
    X86Operand &Op, unsigned AccessSize, bool IsWrite,
    const RegisterContext &RegCtx, MCContext &Ctx, MCStreamer &Out) {
  unsigned AddressRegI32 = RegCtx.addressReg(MVT::i32);
  unsigned ShadowRegI32 = RegCtx.shadowReg(MVT::i32);
  unsigned ShadowRegI8 = RegCtx.shadowReg(MVT::i8);

  assert(RegCtx.ScratchReg != X86::NoRegister);
  unsigned ScratchRegI32 = RegCtx.scratchReg(MVT::i32);

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(AddressRegI32));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(Out, MCInstBuilder(X86::MOV32rr).addReg(ShadowRegI32).addReg(
                           AddressRegI32));
  EmitInstruction(Out, MCInstBuilder(X86::SHR32ri)
                           .addReg(ShadowRegI32)
                           .addReg(ShadowRegI32)
                           .addImm(3));

  {
    MCInst Inst;
    Inst.setOpcode(X86::MOV8rm);
    Inst.addOperand(MCOperand::CreateReg(ShadowRegI8));
    const MCExpr *Disp = MCConstantExpr::Create(kShadowOffset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, ShadowRegI32, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(
      Out, MCInstBuilder(X86::TEST8rr).addReg(ShadowRegI8).addReg(ShadowRegI8));
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitInstruction(Out, MCInstBuilder(X86::MOV32rr).addReg(ScratchRegI32).addReg(
                           AddressRegI32));
  EmitInstruction(Out, MCInstBuilder(X86::AND32ri)
                           .addReg(ScratchRegI32)
                           .addReg(ScratchRegI32)
                           .addImm(7));

  switch (AccessSize) {
  case 1:
    break;
  case 2: {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(ScratchRegI32));

    const MCExpr *Disp = MCConstantExpr::Create(1, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, ScratchRegI32, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
    break;
  }
  case 4:
    EmitInstruction(Out, MCInstBuilder(X86::ADD32ri8)
                             .addReg(ScratchRegI32)
                             .addReg(ScratchRegI32)
                             .addImm(3));
    break;
  default:
    assert(false && "Incorrect access size");
    break;
  }

  EmitInstruction(
      Out,
      MCInstBuilder(X86::MOVSX32rr8).addReg(ShadowRegI32).addReg(ShadowRegI8));
  EmitInstruction(Out, MCInstBuilder(X86::CMP32rr).addReg(ScratchRegI32).addReg(
                           ShadowRegI32));
  EmitInstruction(Out, MCInstBuilder(X86::JL_4).addExpr(DoneExpr));

  EmitCallAsanReport(AccessSize, IsWrite, Ctx, Out, RegCtx);
  EmitLabel(Out, DoneSym);
}

void X86AddressSanitizer32::InstrumentMemOperandLarge(
    X86Operand &Op, unsigned AccessSize, bool IsWrite,
    const RegisterContext &RegCtx, MCContext &Ctx, MCStreamer &Out) {
  unsigned AddressRegI32 = RegCtx.addressReg(MVT::i32);
  unsigned ShadowRegI32 = RegCtx.shadowReg(MVT::i32);

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(AddressRegI32));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(Out, MCInstBuilder(X86::MOV32rr).addReg(ShadowRegI32).addReg(
                           AddressRegI32));
  EmitInstruction(Out, MCInstBuilder(X86::SHR32ri)
                           .addReg(ShadowRegI32)
                           .addReg(ShadowRegI32)
                           .addImm(3));
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
        X86Operand::CreateMem(0, Disp, ShadowRegI32, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    Inst.addOperand(MCOperand::CreateImm(0));
    EmitInstruction(Out, Inst);
  }
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitCallAsanReport(AccessSize, IsWrite, Ctx, Out, RegCtx);
  EmitLabel(Out, DoneSym);
}

void X86AddressSanitizer32::InstrumentMOVSImpl(unsigned AccessSize,
                                               MCContext &Ctx,
                                               MCStreamer &Out) {
  StoreFlags(Out);

  // No need to test when ECX is equals to zero.
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(
      Out, MCInstBuilder(X86::TEST32rr).addReg(X86::ECX).addReg(X86::ECX));
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  // Instrument first and last elements in src and dst range.
  InstrumentMOVSBase(X86::EDI /* DstReg */, X86::ESI /* SrcReg */,
                     X86::ECX /* CntReg */, AccessSize, Ctx, Out);

  EmitLabel(Out, DoneSym);
  RestoreFlags(Out);
}

class X86AddressSanitizer64 : public X86AddressSanitizer {
public:
  static const long kShadowOffset = 0x7fff8000;

  X86AddressSanitizer64(const MCSubtargetInfo &STI)
      : X86AddressSanitizer(STI) {}

  virtual ~X86AddressSanitizer64() {}

  virtual void StoreFlags(MCStreamer &Out) override {
    EmitInstruction(Out, MCInstBuilder(X86::PUSHF64));
  }

  virtual void RestoreFlags(MCStreamer &Out) override {
    EmitInstruction(Out, MCInstBuilder(X86::POPF64));
  }

  virtual void InstrumentMemOperandPrologue(const RegisterContext &RegCtx,
                                            MCContext &Ctx,
                                            MCStreamer &Out) override {
    const MCRegisterInfo *RegisterInfo = Ctx.getRegisterInfo();
    if (RegisterInfo && FrameReg != X86::NoRegister) {
      EmitInstruction(Out, MCInstBuilder(X86::PUSH64r).addReg(X86::RBP));
      if (FrameReg == X86::RSP) {
        Out.EmitCFIAdjustCfaOffset(8 /* byte size of the FrameReg */);
        Out.EmitCFIRelOffset(
            RegisterInfo->getDwarfRegNum(X86::RBP, true /* IsEH */), 0);
      }
      EmitInstruction(
          Out, MCInstBuilder(X86::MOV64rr).addReg(X86::RBP).addReg(FrameReg));
      Out.EmitCFIRememberState();
      Out.EmitCFIDefCfaRegister(
          RegisterInfo->getDwarfRegNum(X86::RBP, true /* IsEH */));
    }

    EmitAdjustRSP(Ctx, Out, -128);
    EmitInstruction(
        Out, MCInstBuilder(X86::PUSH64r).addReg(RegCtx.shadowReg(MVT::i64)));
    EmitInstruction(
        Out, MCInstBuilder(X86::PUSH64r).addReg(RegCtx.addressReg(MVT::i64)));
    if (RegCtx.ScratchReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::PUSH64r).addReg(RegCtx.scratchReg(MVT::i64)));
    }
    StoreFlags(Out);
  }

  virtual void InstrumentMemOperandEpilogue(const RegisterContext &RegCtx,
                                            MCContext &Ctx,
                                            MCStreamer &Out) override {
    RestoreFlags(Out);
    if (RegCtx.ScratchReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::POP64r).addReg(RegCtx.scratchReg(MVT::i64)));
    }
    EmitInstruction(
        Out, MCInstBuilder(X86::POP64r).addReg(RegCtx.addressReg(MVT::i64)));
    EmitInstruction(
        Out, MCInstBuilder(X86::POP64r).addReg(RegCtx.shadowReg(MVT::i64)));
    EmitAdjustRSP(Ctx, Out, 128);

    if (Ctx.getRegisterInfo() && FrameReg != X86::NoRegister) {
      EmitInstruction(
          Out, MCInstBuilder(X86::POP64r).addReg(X86::RBP));
      Out.EmitCFIRestoreState();
      if (FrameReg == X86::RSP)
        Out.EmitCFIAdjustCfaOffset(-8 /* byte size of the FrameReg */);
    }
  }

  virtual void InstrumentMemOperandSmall(X86Operand &Op, unsigned AccessSize,
                                         bool IsWrite,
                                         const RegisterContext &RegCtx,
                                         MCContext &Ctx,
                                         MCStreamer &Out) override;
  virtual void InstrumentMemOperandLarge(X86Operand &Op, unsigned AccessSize,
                                         bool IsWrite,
                                         const RegisterContext &RegCtx,
                                         MCContext &Ctx,
                                         MCStreamer &Out) override;
  virtual void InstrumentMOVSImpl(unsigned AccessSize, MCContext &Ctx,
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

  void EmitCallAsanReport(unsigned AccessSize, bool IsWrite, MCContext &Ctx,
                          MCStreamer &Out, const RegisterContext &RegCtx) {
    EmitInstruction(Out, MCInstBuilder(X86::CLD));
    EmitInstruction(Out, MCInstBuilder(X86::MMX_EMMS));

    EmitInstruction(Out, MCInstBuilder(X86::AND64ri8)
                             .addReg(X86::RSP)
                             .addReg(X86::RSP)
                             .addImm(-16));

    if (RegCtx.AddressReg != X86::RDI) {
      EmitInstruction(Out, MCInstBuilder(X86::MOV64rr).addReg(X86::RDI).addReg(
                               RegCtx.addressReg(MVT::i64)));
    }
    const std::string &Fn = FuncName(AccessSize, IsWrite);
    MCSymbol *FnSym = Ctx.GetOrCreateSymbol(StringRef(Fn));
    const MCSymbolRefExpr *FnExpr =
        MCSymbolRefExpr::Create(FnSym, MCSymbolRefExpr::VK_PLT, Ctx);
    EmitInstruction(Out, MCInstBuilder(X86::CALL64pcrel32).addExpr(FnExpr));
  }
};

void X86AddressSanitizer64::InstrumentMemOperandSmall(
    X86Operand &Op, unsigned AccessSize, bool IsWrite,
    const RegisterContext &RegCtx, MCContext &Ctx, MCStreamer &Out) {
  unsigned AddressRegI64 = RegCtx.addressReg(MVT::i64);
  unsigned AddressRegI32 = RegCtx.addressReg(MVT::i32);
  unsigned ShadowRegI64 = RegCtx.shadowReg(MVT::i64);
  unsigned ShadowRegI32 = RegCtx.shadowReg(MVT::i32);
  unsigned ShadowRegI8 = RegCtx.shadowReg(MVT::i8);

  assert(RegCtx.ScratchReg != X86::NoRegister);
  unsigned ScratchRegI32 = RegCtx.scratchReg(MVT::i32);

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA64r);
    Inst.addOperand(MCOperand::CreateReg(AddressRegI64));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  EmitInstruction(Out, MCInstBuilder(X86::MOV64rr).addReg(ShadowRegI64).addReg(
                           AddressRegI64));
  EmitInstruction(Out, MCInstBuilder(X86::SHR64ri)
                           .addReg(ShadowRegI64)
                           .addReg(ShadowRegI64)
                           .addImm(3));
  {
    MCInst Inst;
    Inst.setOpcode(X86::MOV8rm);
    Inst.addOperand(MCOperand::CreateReg(ShadowRegI8));
    const MCExpr *Disp = MCConstantExpr::Create(kShadowOffset, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, ShadowRegI64, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }

  EmitInstruction(
      Out, MCInstBuilder(X86::TEST8rr).addReg(ShadowRegI8).addReg(ShadowRegI8));
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitInstruction(Out, MCInstBuilder(X86::MOV32rr).addReg(ScratchRegI32).addReg(
                           AddressRegI32));
  EmitInstruction(Out, MCInstBuilder(X86::AND32ri)
                           .addReg(ScratchRegI32)
                           .addReg(ScratchRegI32)
                           .addImm(7));

  switch (AccessSize) {
  case 1:
    break;
  case 2: {
    MCInst Inst;
    Inst.setOpcode(X86::LEA32r);
    Inst.addOperand(MCOperand::CreateReg(ScratchRegI32));

    const MCExpr *Disp = MCConstantExpr::Create(1, Ctx);
    std::unique_ptr<X86Operand> Op(
        X86Operand::CreateMem(0, Disp, ScratchRegI32, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
    break;
  }
  case 4:
    EmitInstruction(Out, MCInstBuilder(X86::ADD32ri8)
                             .addReg(ScratchRegI32)
                             .addReg(ScratchRegI32)
                             .addImm(3));
    break;
  default:
    assert(false && "Incorrect access size");
    break;
  }

  EmitInstruction(
      Out,
      MCInstBuilder(X86::MOVSX32rr8).addReg(ShadowRegI32).addReg(ShadowRegI8));
  EmitInstruction(Out, MCInstBuilder(X86::CMP32rr).addReg(ScratchRegI32).addReg(
                           ShadowRegI32));
  EmitInstruction(Out, MCInstBuilder(X86::JL_4).addExpr(DoneExpr));

  EmitCallAsanReport(AccessSize, IsWrite, Ctx, Out, RegCtx);
  EmitLabel(Out, DoneSym);
}

void X86AddressSanitizer64::InstrumentMemOperandLarge(
    X86Operand &Op, unsigned AccessSize, bool IsWrite,
    const RegisterContext &RegCtx, MCContext &Ctx, MCStreamer &Out) {
  unsigned AddressRegI64 = RegCtx.addressReg(MVT::i64);
  unsigned ShadowRegI64 = RegCtx.shadowReg(MVT::i64);

  {
    MCInst Inst;
    Inst.setOpcode(X86::LEA64r);
    Inst.addOperand(MCOperand::CreateReg(AddressRegI64));
    Op.addMemOperands(Inst, 5);
    EmitInstruction(Out, Inst);
  }
  EmitInstruction(Out, MCInstBuilder(X86::MOV64rr).addReg(ShadowRegI64).addReg(
                           AddressRegI64));
  EmitInstruction(Out, MCInstBuilder(X86::SHR64ri)
                           .addReg(ShadowRegI64)
                           .addReg(ShadowRegI64)
                           .addImm(3));
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
        X86Operand::CreateMem(0, Disp, ShadowRegI64, 0, 1, SMLoc(), SMLoc()));
    Op->addMemOperands(Inst, 5);
    Inst.addOperand(MCOperand::CreateImm(0));
    EmitInstruction(Out, Inst);
  }

  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  EmitCallAsanReport(AccessSize, IsWrite, Ctx, Out, RegCtx);
  EmitLabel(Out, DoneSym);
}

void X86AddressSanitizer64::InstrumentMOVSImpl(unsigned AccessSize,
                                               MCContext &Ctx,
                                               MCStreamer &Out) {
  StoreFlags(Out);

  // No need to test when RCX is equals to zero.
  MCSymbol *DoneSym = Ctx.CreateTempSymbol();
  const MCExpr *DoneExpr = MCSymbolRefExpr::Create(DoneSym, Ctx);
  EmitInstruction(
      Out, MCInstBuilder(X86::TEST64rr).addReg(X86::RCX).addReg(X86::RCX));
  EmitInstruction(Out, MCInstBuilder(X86::JE_4).addExpr(DoneExpr));

  // Instrument first and last elements in src and dst range.
  InstrumentMOVSBase(X86::RDI /* DstReg */, X86::RSI /* SrcReg */,
                     X86::RCX /* CntReg */, AccessSize, Ctx, Out);

  EmitLabel(Out, DoneSym);
  RestoreFlags(Out);
}

} // End anonymous namespace

X86AsmInstrumentation::X86AsmInstrumentation(const MCSubtargetInfo &STI)
    : STI(STI), FrameReg(X86::NoRegister) {}

X86AsmInstrumentation::~X86AsmInstrumentation() {}

void X86AsmInstrumentation::InstrumentAndEmitInstruction(
    const MCInst &Inst, OperandVector &Operands, MCContext &Ctx,
    const MCInstrInfo &MII, MCStreamer &Out) {
  EmitInstruction(Out, Inst);
}

void X86AsmInstrumentation::EmitInstruction(MCStreamer &Out,
                                            const MCInst &Inst) {
  Out.EmitInstruction(Inst, STI);
}

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
  return new X86AsmInstrumentation(STI);
}

} // End llvm namespace
