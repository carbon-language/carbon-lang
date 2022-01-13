//===-- SystemZAsmPrinter.cpp - SystemZ LLVM assembly printer -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Streams SystemZ assembly language and associated data, in the form of
// MCInsts and MCExprs respectively.
//
//===----------------------------------------------------------------------===//

#include "SystemZAsmPrinter.h"
#include "MCTargetDesc/SystemZInstPrinter.h"
#include "SystemZConstantPoolValue.h"
#include "SystemZMCInstLower.h"
#include "TargetInfo/SystemZTargetInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

// Return an RI instruction like MI with opcode Opcode, but with the
// GR64 register operands turned into GR32s.
static MCInst lowerRILow(const MachineInstr *MI, unsigned Opcode) {
  if (MI->isCompare())
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(1).getImm());
  else
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(1).getReg()))
      .addImm(MI->getOperand(2).getImm());
}

// Return an RI instruction like MI with opcode Opcode, but with the
// GR64 register operands turned into GRH32s.
static MCInst lowerRIHigh(const MachineInstr *MI, unsigned Opcode) {
  if (MI->isCompare())
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(1).getImm());
  else
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(1).getReg()))
      .addImm(MI->getOperand(2).getImm());
}

// Return an RI instruction like MI with opcode Opcode, but with the
// R2 register turned into a GR64.
static MCInst lowerRIEfLow(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(MI->getOperand(0).getReg())
    .addReg(MI->getOperand(1).getReg())
    .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()))
    .addImm(MI->getOperand(3).getImm())
    .addImm(MI->getOperand(4).getImm())
    .addImm(MI->getOperand(5).getImm());
}

static const MCSymbolRefExpr *getTLSGetOffset(MCContext &Context) {
  StringRef Name = "__tls_get_offset";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 MCSymbolRefExpr::VK_PLT,
                                 Context);
}

static const MCSymbolRefExpr *getGlobalOffsetTable(MCContext &Context) {
  StringRef Name = "_GLOBAL_OFFSET_TABLE_";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 MCSymbolRefExpr::VK_None,
                                 Context);
}

// MI is an instruction that accepts an optional alignment hint,
// and which was already lowered to LoweredMI.  If the alignment
// of the original memory operand is known, update LoweredMI to
// an instruction with the corresponding hint set.
static void lowerAlignmentHint(const MachineInstr *MI, MCInst &LoweredMI,
                               unsigned Opcode) {
  if (!MI->hasOneMemOperand())
    return;
  const MachineMemOperand *MMO = *MI->memoperands_begin();
  unsigned AlignmentHint = 0;
  if (MMO->getAlign() >= Align(16))
    AlignmentHint = 4;
  else if (MMO->getAlign() >= Align(8))
    AlignmentHint = 3;
  if (AlignmentHint == 0)
    return;

  LoweredMI.setOpcode(Opcode);
  LoweredMI.addOperand(MCOperand::createImm(AlignmentHint));
}

// MI loads the high part of a vector from memory.  Return an instruction
// that uses replicating vector load Opcode to do the same thing.
static MCInst lowerSubvectorLoad(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
    .addReg(MI->getOperand(1).getReg())
    .addImm(MI->getOperand(2).getImm())
    .addReg(MI->getOperand(3).getReg());
}

// MI stores the high part of a vector to memory.  Return an instruction
// that uses elemental vector store Opcode to do the same thing.
static MCInst lowerSubvectorStore(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
    .addReg(MI->getOperand(1).getReg())
    .addImm(MI->getOperand(2).getImm())
    .addReg(MI->getOperand(3).getReg())
    .addImm(0);
}

void SystemZAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SystemZMCInstLower Lower(MF->getContext(), *this);
  const SystemZSubtarget *Subtarget = &MF->getSubtarget<SystemZSubtarget>();
  MCInst LoweredMI;
  switch (MI->getOpcode()) {
  case SystemZ::Return:
    if (Subtarget->isTargetXPLINK64())
      LoweredMI =
          MCInstBuilder(SystemZ::B).addReg(SystemZ::R7D).addImm(2).addReg(0);
    else
      LoweredMI = MCInstBuilder(SystemZ::BR).addReg(SystemZ::R14D);
    break;

  case SystemZ::CondReturn:
    LoweredMI = MCInstBuilder(SystemZ::BCR)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(SystemZ::R14D);
    break;

  case SystemZ::CRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CGRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CGIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLGRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLGIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CallBRASL_XPLINK64:
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::BRASL)
                       .addReg(SystemZ::R7D)
                       .addExpr(Lower.getExpr(MI->getOperand(0),
                                              MCSymbolRefExpr::VK_PLT)));
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::BCRAsm).addImm(0).addReg(SystemZ::R3D));
    return;

  case SystemZ::CallBASR_XPLINK64:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BASR)
                                     .addReg(SystemZ::R7D)
                                     .addReg(MI->getOperand(0).getReg()));
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::BCRAsm).addImm(0).addReg(SystemZ::R0D));
    return;

  case SystemZ::CallBRASL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBASR:
    LoweredMI = MCInstBuilder(SystemZ::BASR)
      .addReg(SystemZ::R14D)
      .addReg(MI->getOperand(0).getReg());
    break;

  case SystemZ::CallJG:
    LoweredMI = MCInstBuilder(SystemZ::JG)
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBRCL:
    LoweredMI = MCInstBuilder(SystemZ::BRCL)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addExpr(Lower.getExpr(MI->getOperand(2), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBR:
    LoweredMI = MCInstBuilder(SystemZ::BR)
      .addReg(MI->getOperand(0).getReg());
    break;

  case SystemZ::CallBCR:
    LoweredMI = MCInstBuilder(SystemZ::BCR)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(MI->getOperand(2).getReg());
    break;

  case SystemZ::CRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CGRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CGIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLGRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLGIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::TLS_GDCALL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(getTLSGetOffset(MF->getContext()))
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_TLSGD));
    break;

  case SystemZ::TLS_LDCALL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(getTLSGetOffset(MF->getContext()))
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_TLSLDM));
    break;

  case SystemZ::GOT:
    LoweredMI = MCInstBuilder(SystemZ::LARL)
      .addReg(MI->getOperand(0).getReg())
      .addExpr(getGlobalOffsetTable(MF->getContext()));
    break;

  case SystemZ::IILF64:
    LoweredMI = MCInstBuilder(SystemZ::IILF)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::IIHF64:
    LoweredMI = MCInstBuilder(SystemZ::IIHF)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::RISBHH:
  case SystemZ::RISBHL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBHG);
    break;

  case SystemZ::RISBLH:
  case SystemZ::RISBLL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBLG);
    break;

  case SystemZ::VLVGP32:
    LoweredMI = MCInstBuilder(SystemZ::VLVGP)
      .addReg(MI->getOperand(0).getReg())
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(1).getReg()))
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()));
    break;

  case SystemZ::VLR32:
  case SystemZ::VLR64:
    LoweredMI = MCInstBuilder(SystemZ::VLR)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()));
    break;

  case SystemZ::VL:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VLAlign);
    break;

  case SystemZ::VST:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VSTAlign);
    break;

  case SystemZ::VLM:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VLMAlign);
    break;

  case SystemZ::VSTM:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VSTMAlign);
    break;

  case SystemZ::VL32:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPF);
    break;

  case SystemZ::VL64:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPG);
    break;

  case SystemZ::VST32:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEF);
    break;

  case SystemZ::VST64:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEG);
    break;

  case SystemZ::LFER:
    LoweredMI = MCInstBuilder(SystemZ::VLGVF)
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()))
      .addReg(0).addImm(0);
    break;

  case SystemZ::LEFR:
    LoweredMI = MCInstBuilder(SystemZ::VLVGF)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(MI->getOperand(1).getReg())
      .addReg(0).addImm(0);
    break;

#define LOWER_LOW(NAME)                                                 \
  case SystemZ::NAME##64: LoweredMI = lowerRILow(MI, SystemZ::NAME); break

  LOWER_LOW(IILL);
  LOWER_LOW(IILH);
  LOWER_LOW(TMLL);
  LOWER_LOW(TMLH);
  LOWER_LOW(NILL);
  LOWER_LOW(NILH);
  LOWER_LOW(NILF);
  LOWER_LOW(OILL);
  LOWER_LOW(OILH);
  LOWER_LOW(OILF);
  LOWER_LOW(XILF);

#undef LOWER_LOW

#define LOWER_HIGH(NAME) \
  case SystemZ::NAME##64: LoweredMI = lowerRIHigh(MI, SystemZ::NAME); break

  LOWER_HIGH(IIHL);
  LOWER_HIGH(IIHH);
  LOWER_HIGH(TMHL);
  LOWER_HIGH(TMHH);
  LOWER_HIGH(NIHL);
  LOWER_HIGH(NIHH);
  LOWER_HIGH(NIHF);
  LOWER_HIGH(OIHL);
  LOWER_HIGH(OIHH);
  LOWER_HIGH(OIHF);
  LOWER_HIGH(XIHF);

#undef LOWER_HIGH

  case SystemZ::Serialize:
    if (MF->getSubtarget<SystemZSubtarget>().hasFastSerialization())
      LoweredMI = MCInstBuilder(SystemZ::BCRAsm)
        .addImm(14).addReg(SystemZ::R0D);
    else
      LoweredMI = MCInstBuilder(SystemZ::BCRAsm)
        .addImm(15).addReg(SystemZ::R0D);
    break;

  // Emit nothing here but a comment if we can.
  case SystemZ::MemBarrier:
    OutStreamer->emitRawComment("MEMBARRIER");
    return;

  // We want to emit "j .+2" for traps, jumping to the relative immediate field
  // of the jump instruction, which is an illegal instruction. We cannot emit a
  // "." symbol, so create and emit a temp label before the instruction and use
  // that instead.
  case SystemZ::Trap: {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCConstantExpr *ConstExpr = MCConstantExpr::create(2, OutContext);
    LoweredMI = MCInstBuilder(SystemZ::J)
      .addExpr(MCBinaryExpr::createAdd(Expr, ConstExpr, OutContext));
    }
    break;

  // Conditional traps will create a branch on condition instruction that jumps
  // to the relative immediate field of the jump instruction. (eg. "jo .+2")
  case SystemZ::CondTrap: {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCConstantExpr *ConstExpr = MCConstantExpr::create(2, OutContext);
    LoweredMI = MCInstBuilder(SystemZ::BRC)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addExpr(MCBinaryExpr::createAdd(Expr, ConstExpr, OutContext));
    }
    break;

  case TargetOpcode::FENTRY_CALL:
    LowerFENTRY_CALL(*MI, Lower);
    return;

  case TargetOpcode::STACKMAP:
    LowerSTACKMAP(*MI);
    return;

  case TargetOpcode::PATCHPOINT:
    LowerPATCHPOINT(*MI, Lower);
    return;

  case SystemZ::EXRL_Pseudo: {
    unsigned TargetInsOpc = MI->getOperand(0).getImm();
    Register LenMinus1Reg = MI->getOperand(1).getReg();
    Register DestReg = MI->getOperand(2).getReg();
    int64_t DestDisp = MI->getOperand(3).getImm();
    Register SrcReg = MI->getOperand(4).getReg();
    int64_t SrcDisp = MI->getOperand(5).getImm();

    MCSymbol *DotSym = nullptr;
    MCInst ET = MCInstBuilder(TargetInsOpc).addReg(DestReg)
      .addImm(DestDisp).addImm(1).addReg(SrcReg).addImm(SrcDisp);
    MCInstSTIPair ET_STI(ET, &MF->getSubtarget());
    EXRLT2SymMap::iterator I = EXRLTargets2Sym.find(ET_STI);
    if (I != EXRLTargets2Sym.end())
      DotSym = I->second;
    else
      EXRLTargets2Sym[ET_STI] = DotSym = OutContext.createTempSymbol();
    const MCSymbolRefExpr *Dot = MCSymbolRefExpr::create(DotSym, OutContext);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::EXRL).addReg(LenMinus1Reg).addExpr(Dot));
    return;
  }

  default:
    Lower.lower(MI, LoweredMI);
    break;
  }
  EmitToStreamer(*OutStreamer, LoweredMI);
}

// Emit the largest nop instruction smaller than or equal to NumBytes
// bytes.  Return the size of nop emitted.
static unsigned EmitNop(MCContext &OutContext, MCStreamer &OutStreamer,
                        unsigned NumBytes, const MCSubtargetInfo &STI) {
  if (NumBytes < 2) {
    llvm_unreachable("Zero nops?");
    return 0;
  }
  else if (NumBytes < 4) {
    OutStreamer.emitInstruction(
        MCInstBuilder(SystemZ::BCRAsm).addImm(0).addReg(SystemZ::R0D), STI);
    return 2;
  }
  else if (NumBytes < 6) {
    OutStreamer.emitInstruction(
        MCInstBuilder(SystemZ::BCAsm).addImm(0).addReg(0).addImm(0).addReg(0),
        STI);
    return 4;
  }
  else {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    const MCSymbolRefExpr *Dot = MCSymbolRefExpr::create(DotSym, OutContext);
    OutStreamer.emitLabel(DotSym);
    OutStreamer.emitInstruction(
        MCInstBuilder(SystemZ::BRCLAsm).addImm(0).addExpr(Dot), STI);
    return 6;
  }
}

void SystemZAsmPrinter::LowerFENTRY_CALL(const MachineInstr &MI,
                                         SystemZMCInstLower &Lower) {
  MCContext &Ctx = MF->getContext();
  if (MF->getFunction().hasFnAttribute("mrecord-mcount")) {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->PushSection();
    OutStreamer->SwitchSection(
        Ctx.getELFSection("__mcount_loc", ELF::SHT_PROGBITS, ELF::SHF_ALLOC));
    OutStreamer->emitSymbolValue(DotSym, 8);
    OutStreamer->PopSection();
    OutStreamer->emitLabel(DotSym);
  }

  if (MF->getFunction().hasFnAttribute("mnop-mcount")) {
    EmitNop(Ctx, *OutStreamer, 6, getSubtargetInfo());
    return;
  }

  MCSymbol *fentry = Ctx.getOrCreateSymbol("__fentry__");
  const MCSymbolRefExpr *Op =
      MCSymbolRefExpr::create(fentry, MCSymbolRefExpr::VK_PLT, Ctx);
  OutStreamer->emitInstruction(
      MCInstBuilder(SystemZ::BRASL).addReg(SystemZ::R0D).addExpr(Op),
      getSubtargetInfo());
}

void SystemZAsmPrinter::LowerSTACKMAP(const MachineInstr &MI) {
  const SystemZInstrInfo *TII =
    static_cast<const SystemZInstrInfo *>(MF->getSubtarget().getInstrInfo());

  unsigned NumNOPBytes = MI.getOperand(1).getImm();

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);
  
  SM.recordStackMap(*MILabel, MI);
  assert(NumNOPBytes % 2 == 0 && "Invalid number of NOP bytes requested!");

  // Scan ahead to trim the shadow.
  unsigned ShadowBytes = 0;
  const MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::const_iterator MII(MI);
  ++MII;
  while (ShadowBytes < NumNOPBytes) {
    if (MII == MBB.end() ||
        MII->getOpcode() == TargetOpcode::PATCHPOINT ||
        MII->getOpcode() == TargetOpcode::STACKMAP)
      break;
    ShadowBytes += TII->getInstSizeInBytes(*MII);
    if (MII->isCall())
      break;
    ++MII;
  }

  // Emit nops.
  while (ShadowBytes < NumNOPBytes)
    ShadowBytes += EmitNop(OutContext, *OutStreamer, NumNOPBytes - ShadowBytes,
                           getSubtargetInfo());
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>
void SystemZAsmPrinter::LowerPATCHPOINT(const MachineInstr &MI,
                                        SystemZMCInstLower &Lower) {
  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordPatchPoint(*MILabel, MI);
  PatchPointOpers Opers(&MI);

  unsigned EncodedBytes = 0;
  const MachineOperand &CalleeMO = Opers.getCallTarget();

  if (CalleeMO.isImm()) {
    uint64_t CallTarget = CalleeMO.getImm();
    if (CallTarget) {
      unsigned ScratchIdx = -1;
      unsigned ScratchReg = 0;
      do {
        ScratchIdx = Opers.getNextScratchIdx(ScratchIdx + 1);
        ScratchReg = MI.getOperand(ScratchIdx).getReg();
      } while (ScratchReg == SystemZ::R0D);

      // Materialize the call target address
      EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::LLILF)
                                      .addReg(ScratchReg)
                                      .addImm(CallTarget & 0xFFFFFFFF));
      EncodedBytes += 6;
      if (CallTarget >> 32) {
        EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::IIHF)
                                        .addReg(ScratchReg)
                                        .addImm(CallTarget >> 32));
        EncodedBytes += 6;
      }

      EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BASR)
                                     .addReg(SystemZ::R14D)
                                     .addReg(ScratchReg));
      EncodedBytes += 2;
    }
  } else if (CalleeMO.isGlobal()) {
    const MCExpr *Expr = Lower.getExpr(CalleeMO, MCSymbolRefExpr::VK_PLT);
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BRASL)
                                   .addReg(SystemZ::R14D)
                                   .addExpr(Expr));
    EncodedBytes += 6;
  }

  // Emit padding.
  unsigned NumBytes = Opers.getNumPatchBytes();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");
  assert((NumBytes - EncodedBytes) % 2 == 0 &&
         "Invalid number of NOP bytes requested!");
  while (EncodedBytes < NumBytes)
    EncodedBytes += EmitNop(OutContext, *OutStreamer, NumBytes - EncodedBytes,
                            getSubtargetInfo());
}

void SystemZAsmPrinter::emitEXRLTargetInstructions() {
  if (EXRLTargets2Sym.empty())
    return;
  // Switch to the .text section.
  OutStreamer->SwitchSection(getObjFileLowering().getTextSection());
  for (auto &I : EXRLTargets2Sym) {
    OutStreamer->emitLabel(I.second);
    const MCInstSTIPair &MCI_STI = I.first;
    OutStreamer->emitInstruction(MCI_STI.first, *MCI_STI.second);
  }
  EXRLTargets2Sym.clear();
}

// Convert a SystemZ-specific constant pool modifier into the associated
// MCSymbolRefExpr variant kind.
static MCSymbolRefExpr::VariantKind
getModifierVariantKind(SystemZCP::SystemZCPModifier Modifier) {
  switch (Modifier) {
  case SystemZCP::TLSGD: return MCSymbolRefExpr::VK_TLSGD;
  case SystemZCP::TLSLDM: return MCSymbolRefExpr::VK_TLSLDM;
  case SystemZCP::DTPOFF: return MCSymbolRefExpr::VK_DTPOFF;
  case SystemZCP::NTPOFF: return MCSymbolRefExpr::VK_NTPOFF;
  }
  llvm_unreachable("Invalid SystemCPModifier!");
}

void SystemZAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  auto *ZCPV = static_cast<SystemZConstantPoolValue*>(MCPV);

  const MCExpr *Expr =
    MCSymbolRefExpr::create(getSymbol(ZCPV->getGlobalValue()),
                            getModifierVariantKind(ZCPV->getModifier()),
                            OutContext);
  uint64_t Size = getDataLayout().getTypeAllocSize(ZCPV->getType());

  OutStreamer->emitValue(Expr, Size);
}

bool SystemZAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                        const char *ExtraCode,
                                        raw_ostream &OS) {
  const MCRegisterInfo &MRI = *TM.getMCRegisterInfo();
  const MachineOperand &MO = MI->getOperand(OpNo);
  MCOperand MCOp;
  if (ExtraCode) {
    if (ExtraCode[0] == 'N' && !ExtraCode[1] && MO.isReg() &&
        SystemZ::GR128BitRegClass.contains(MO.getReg()))
      MCOp =
          MCOperand::createReg(MRI.getSubReg(MO.getReg(), SystemZ::subreg_l64));
    else
      return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS);
  } else {
    SystemZMCInstLower Lower(MF->getContext(), *this);
    MCOp = Lower.lowerOperand(MO);
  }
  SystemZInstPrinter::printOperand(MCOp, MAI, OS);
  return false;
}

bool SystemZAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                              unsigned OpNo,
                                              const char *ExtraCode,
                                              raw_ostream &OS) {
  SystemZInstPrinter::printAddress(MAI, MI->getOperand(OpNo).getReg(),
                                   MI->getOperand(OpNo + 1).getImm(),
                                   MI->getOperand(OpNo + 2).getReg(), OS);
  return false;
}

void SystemZAsmPrinter::emitEndOfAsmFile(Module &M) {
  emitEXRLTargetInstructions();
  emitStackMaps(SM);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSystemZAsmPrinter() {
  RegisterAsmPrinter<SystemZAsmPrinter> X(getTheSystemZTarget());
}
