//===-- AArch64MCInstLower.cpp - Convert AArch64 MachineInstr to an MCInst -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower AArch64 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "AArch64AsmPrinter.h"
#include "AArch64TargetMachine.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

MCOperand
AArch64AsmPrinter::lowerSymbolOperand(const MachineOperand &MO,
                                      const MCSymbol *Sym) const {
  const MCExpr *Expr = 0;

  Expr = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, OutContext);

  switch (MO.getTargetFlags()) {
  case AArch64II::MO_GOT:
    Expr = AArch64MCExpr::CreateGOT(Expr, OutContext);
    break;
  case AArch64II::MO_GOT_LO12:
    Expr = AArch64MCExpr::CreateGOTLo12(Expr, OutContext);
    break;
  case AArch64II::MO_LO12:
    Expr = AArch64MCExpr::CreateLo12(Expr, OutContext);
    break;
  case AArch64II::MO_DTPREL_G1:
    Expr = AArch64MCExpr::CreateDTPREL_G1(Expr, OutContext);
    break;
  case AArch64II::MO_DTPREL_G0_NC:
    Expr = AArch64MCExpr::CreateDTPREL_G0_NC(Expr, OutContext);
    break;
  case AArch64II::MO_GOTTPREL:
    Expr = AArch64MCExpr::CreateGOTTPREL(Expr, OutContext);
    break;
  case AArch64II::MO_GOTTPREL_LO12:
    Expr = AArch64MCExpr::CreateGOTTPRELLo12(Expr, OutContext);
    break;
  case AArch64II::MO_TLSDESC:
    Expr = AArch64MCExpr::CreateTLSDesc(Expr, OutContext);
    break;
  case AArch64II::MO_TLSDESC_LO12:
    Expr = AArch64MCExpr::CreateTLSDescLo12(Expr, OutContext);
    break;
  case AArch64II::MO_TPREL_G1:
    Expr = AArch64MCExpr::CreateTPREL_G1(Expr, OutContext);
    break;
  case AArch64II::MO_TPREL_G0_NC:
    Expr = AArch64MCExpr::CreateTPREL_G0_NC(Expr, OutContext);
    break;
  case AArch64II::MO_ABS_G3:
    Expr = AArch64MCExpr::CreateABS_G3(Expr, OutContext);
    break;
  case AArch64II::MO_ABS_G2_NC:
    Expr = AArch64MCExpr::CreateABS_G2_NC(Expr, OutContext);
    break;
  case AArch64II::MO_ABS_G1_NC:
    Expr = AArch64MCExpr::CreateABS_G1_NC(Expr, OutContext);
    break;
  case AArch64II::MO_ABS_G0_NC:
    Expr = AArch64MCExpr::CreateABS_G0_NC(Expr, OutContext);
    break;
  case AArch64II::MO_NO_FLAG:
    // Expr is already correct
    break;
  default:
    llvm_unreachable("Unexpected MachineOperand flag");
  }

  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(),
                                                          OutContext),
                                   OutContext);

  return MCOperand::CreateExpr(Expr);
}

bool AArch64AsmPrinter::lowerOperand(const MachineOperand &MO,
                                     MCOperand &MCOp) const {
  switch (MO.getType()) {
  default: llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    if (MO.isImplicit())
      return false;
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    MCOp = MCOperand::CreateReg(MO.getReg());
    break;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::CreateImm(MO.getImm());
    break;
  case MachineOperand::MO_FPImmediate: {
    assert(MO.getFPImm()->isZero() && "Only fp imm 0.0 is supported");
    MCOp = MCOperand::CreateFPImm(0.0);
    break;
  }
  case MachineOperand::MO_BlockAddress:
    MCOp = lowerSymbolOperand(MO, GetBlockAddressSymbol(MO.getBlockAddress()));
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = lowerSymbolOperand(MO, GetExternalSymbolSymbol(MO.getSymbolName()));
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = lowerSymbolOperand(MO, getSymbol(MO.getGlobal()));
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                                   MO.getMBB()->getSymbol(), OutContext));
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = lowerSymbolOperand(MO, GetJTISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, GetCPISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_RegisterMask:
    // Ignore call clobbers
    return false;

  }

  return true;
}

void llvm::LowerAArch64MachineInstrToMCInst(const MachineInstr *MI,
                                            MCInst &OutMI,
                                            AArch64AsmPrinter &AP) {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    if (AP.lowerOperand(MO, MCOp))
      OutMI.addOperand(MCOp);
  }
}
