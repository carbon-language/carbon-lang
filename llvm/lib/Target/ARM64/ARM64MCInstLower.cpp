//===-- ARM64MCInstLower.cpp - Convert ARM64 MachineInstr to an MCInst---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower ARM64 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "ARM64MCInstLower.h"
#include "MCTargetDesc/ARM64MCExpr.h"
#include "Utils/ARM64BaseInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

ARM64MCInstLower::ARM64MCInstLower(MCContext &ctx, Mangler &mang,
                                   AsmPrinter &printer)
    : Ctx(ctx), Printer(printer), TargetTriple(printer.getTargetTriple()) {}

MCSymbol *
ARM64MCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  return Printer.getSymbol(MO.getGlobal());
}

MCSymbol *
ARM64MCInstLower::GetExternalSymbolSymbol(const MachineOperand &MO) const {
  return Printer.GetExternalSymbolSymbol(MO.getSymbolName());
}

MCOperand ARM64MCInstLower::lowerSymbolOperandDarwin(const MachineOperand &MO,
                                                     MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;
  if ((MO.getTargetFlags() & ARM64II::MO_GOT) != 0) {
    if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_PAGE)
      RefKind = MCSymbolRefExpr::VK_GOTPAGE;
    else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) ==
             ARM64II::MO_PAGEOFF)
      RefKind = MCSymbolRefExpr::VK_GOTPAGEOFF;
    else
      assert(0 && "Unexpected target flags with MO_GOT on GV operand");
  } else if ((MO.getTargetFlags() & ARM64II::MO_TLS) != 0) {
    if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_PAGE)
      RefKind = MCSymbolRefExpr::VK_TLVPPAGE;
    else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) ==
             ARM64II::MO_PAGEOFF)
      RefKind = MCSymbolRefExpr::VK_TLVPPAGEOFF;
    else
      llvm_unreachable("Unexpected target flags with MO_TLS on GV operand");
  } else {
    if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_PAGE)
      RefKind = MCSymbolRefExpr::VK_PAGE;
    else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) ==
             ARM64II::MO_PAGEOFF)
      RefKind = MCSymbolRefExpr::VK_PAGEOFF;
  }
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, RefKind, Ctx);
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(
        Expr, MCConstantExpr::Create(MO.getOffset(), Ctx), Ctx);
  return MCOperand::CreateExpr(Expr);
}

MCOperand ARM64MCInstLower::lowerSymbolOperandELF(const MachineOperand &MO,
                                                  MCSymbol *Sym) const {
  uint32_t RefFlags = 0;

  if (MO.getTargetFlags() & ARM64II::MO_GOT)
    RefFlags |= ARM64MCExpr::VK_GOT;
  else if (MO.getTargetFlags() & ARM64II::MO_TLS) {
    TLSModel::Model Model;
    if (MO.isGlobal()) {
      const GlobalValue *GV = MO.getGlobal();
      Model = Printer.TM.getTLSModel(GV);
    } else {
      assert(MO.isSymbol() &&
             StringRef(MO.getSymbolName()) == "_TLS_MODULE_BASE_" &&
             "unexpected external TLS symbol");
      Model = TLSModel::GeneralDynamic;
    }
    switch (Model) {
    case TLSModel::InitialExec:
      RefFlags |= ARM64MCExpr::VK_GOTTPREL;
      break;
    case TLSModel::LocalExec:
      RefFlags |= ARM64MCExpr::VK_TPREL;
      break;
    case TLSModel::LocalDynamic:
      RefFlags |= ARM64MCExpr::VK_DTPREL;
      break;
    case TLSModel::GeneralDynamic:
      RefFlags |= ARM64MCExpr::VK_TLSDESC;
      break;
    }
  } else {
    // No modifier means this is a generic reference, classified as absolute for
    // the cases where it matters (:abs_g0: etc).
    RefFlags |= ARM64MCExpr::VK_ABS;
  }

  if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_PAGE)
    RefFlags |= ARM64MCExpr::VK_PAGE;
  else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_PAGEOFF)
    RefFlags |= ARM64MCExpr::VK_PAGEOFF;
  else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_G3)
    RefFlags |= ARM64MCExpr::VK_G3;
  else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_G2)
    RefFlags |= ARM64MCExpr::VK_G2;
  else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_G1)
    RefFlags |= ARM64MCExpr::VK_G1;
  else if ((MO.getTargetFlags() & ARM64II::MO_FRAGMENT) == ARM64II::MO_G0)
    RefFlags |= ARM64MCExpr::VK_G0;

  if (MO.getTargetFlags() & ARM64II::MO_NC)
    RefFlags |= ARM64MCExpr::VK_NC;

  const MCExpr *Expr =
      MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, Ctx);
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(
        Expr, MCConstantExpr::Create(MO.getOffset(), Ctx), Ctx);

  ARM64MCExpr::VariantKind RefKind;
  RefKind = static_cast<ARM64MCExpr::VariantKind>(RefFlags);
  Expr = ARM64MCExpr::Create(Expr, RefKind, Ctx);

  return MCOperand::CreateExpr(Expr);
}

MCOperand ARM64MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                               MCSymbol *Sym) const {
  if (TargetTriple.isOSDarwin())
    return lowerSymbolOperandDarwin(MO, Sym);

  assert(TargetTriple.isOSBinFormatELF() && "Expect Darwin or ELF target");
  return lowerSymbolOperandELF(MO, Sym);
}

bool ARM64MCInstLower::lowerOperand(const MachineOperand &MO,
                                    MCOperand &MCOp) const {
  switch (MO.getType()) {
  default:
    assert(0 && "unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::CreateReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::CreateImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = MCOperand::CreateExpr(
        MCSymbolRefExpr::Create(MO.getMBB()->getSymbol(), Ctx));
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = LowerSymbolOperand(MO, GetGlobalAddressSymbol(MO));
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = LowerSymbolOperand(MO, GetExternalSymbolSymbol(MO));
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = LowerSymbolOperand(MO, Printer.GetJTISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = LowerSymbolOperand(MO, Printer.GetCPISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = LowerSymbolOperand(
        MO, Printer.GetBlockAddressSymbol(MO.getBlockAddress()));
    break;
  }
  return true;
}

void ARM64MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MCOperand MCOp;
    if (lowerOperand(MI->getOperand(i), MCOp))
      OutMI.addOperand(MCOp);
  }
}
