//===-- SystemZMCInstLower.cpp - Lower MachineInstr to MCInst -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZMCInstLower.h"
#include "SystemZAsmPrinter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/Mangler.h"

using namespace llvm;

// Return the VK_* enumeration for MachineOperand target flags Flags.
static MCSymbolRefExpr::VariantKind getVariantKind(unsigned Flags) {
  switch (Flags & SystemZII::MO_SYMBOL_MODIFIER) {
    case 0:
      return MCSymbolRefExpr::VK_None;
    case SystemZII::MO_GOT:
      return MCSymbolRefExpr::VK_GOT;
  }
  llvm_unreachable("Unrecognised MO_ACCESS_MODEL");
}

SystemZMCInstLower::SystemZMCInstLower(MCContext &ctx,
                                       SystemZAsmPrinter &asmprinter)
  : Ctx(ctx), AsmPrinter(asmprinter) {}

const MCExpr *
SystemZMCInstLower::getExpr(const MachineOperand &MO,
                            MCSymbolRefExpr::VariantKind Kind) const {
  const MCSymbol *Symbol;
  bool HasOffset = true;
  switch (MO.getType()) {
  case MachineOperand::MO_MachineBasicBlock:
    Symbol = MO.getMBB()->getSymbol();
    HasOffset = false;
    break;

  case MachineOperand::MO_GlobalAddress:
    Symbol = AsmPrinter.getSymbol(MO.getGlobal());
    break;

  case MachineOperand::MO_ExternalSymbol:
    Symbol = AsmPrinter.GetExternalSymbolSymbol(MO.getSymbolName());
    break;

  case MachineOperand::MO_JumpTableIndex:
    Symbol = AsmPrinter.GetJTISymbol(MO.getIndex());
    HasOffset = false;
    break;

  case MachineOperand::MO_ConstantPoolIndex:
    Symbol = AsmPrinter.GetCPISymbol(MO.getIndex());
    break;

  case MachineOperand::MO_BlockAddress:
    Symbol = AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress());
    break;

  default:
    llvm_unreachable("unknown operand type");
  }
  const MCExpr *Expr = MCSymbolRefExpr::Create(Symbol, Kind, Ctx);
  if (HasOffset)
    if (int64_t Offset = MO.getOffset()) {
      const MCExpr *OffsetExpr = MCConstantExpr::Create(Offset, Ctx);
      Expr = MCBinaryExpr::CreateAdd(Expr, OffsetExpr, Ctx);
    }
  return Expr;
}

MCOperand SystemZMCInstLower::lowerOperand(const MachineOperand &MO) const {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    return MCOperand::CreateReg(MO.getReg());

  case MachineOperand::MO_Immediate:
    return MCOperand::CreateImm(MO.getImm());

  default: {
    MCSymbolRefExpr::VariantKind Kind = getVariantKind(MO.getTargetFlags());
    return MCOperand::CreateExpr(getExpr(MO, Kind));
  }
  }
}

void SystemZMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    // Ignore all implicit register operands.
    if (!MO.isReg() || !MO.isImplicit())
      OutMI.addOperand(lowerOperand(MO));
  }
}
