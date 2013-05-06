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

// Where relaxable pairs of reloc-generating instructions exist,
// we tend to use the longest form by default, since that produces
// correct assembly in cases where no relaxation is performed.
// If Opcode is one such instruction, return the opcode for the
// shortest possible form instead, otherwise return Opcode itself.
static unsigned getShortenedInstr(unsigned Opcode) {
  switch (Opcode) {
  case SystemZ::BRCL:  return SystemZ::BRC;
  case SystemZ::JG:    return SystemZ::J;
  case SystemZ::BRASL: return SystemZ::BRAS;
  }
  return Opcode;
}

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

SystemZMCInstLower::SystemZMCInstLower(Mangler *mang, MCContext &ctx,
                                       SystemZAsmPrinter &asmprinter)
  : Mang(mang), Ctx(ctx), AsmPrinter(asmprinter) {}

MCOperand SystemZMCInstLower::lowerSymbolOperand(const MachineOperand &MO,
                                                 const MCSymbol *Symbol,
                                                 int64_t Offset) const {
  MCSymbolRefExpr::VariantKind Kind = getVariantKind(MO.getTargetFlags());
  const MCExpr *Expr = MCSymbolRefExpr::Create(Symbol, Kind, Ctx);
  if (Offset) {
    const MCExpr *OffsetExpr = MCConstantExpr::Create(Offset, Ctx);
    Expr = MCBinaryExpr::CreateAdd(Expr, OffsetExpr, Ctx);
  }
  return MCOperand::CreateExpr(Expr);
}

MCOperand SystemZMCInstLower::lowerOperand(const MachineOperand &MO) const {
  switch (MO.getType()) {
  default:
    llvm_unreachable("unknown operand type");

  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return MCOperand();
    return MCOperand::CreateReg(MO.getReg());

  case MachineOperand::MO_Immediate:
    return MCOperand::CreateImm(MO.getImm());

  case MachineOperand::MO_MachineBasicBlock:
    return lowerSymbolOperand(MO, MO.getMBB()->getSymbol(),
                              /* MO has no offset field */0);

  case MachineOperand::MO_GlobalAddress:
    return lowerSymbolOperand(MO, Mang->getSymbol(MO.getGlobal()),
                              MO.getOffset());

  case MachineOperand::MO_ExternalSymbol: {
    StringRef Name = MO.getSymbolName();
    return lowerSymbolOperand(MO, AsmPrinter.GetExternalSymbolSymbol(Name),
                              MO.getOffset());
  }

  case MachineOperand::MO_JumpTableIndex:
    return lowerSymbolOperand(MO, AsmPrinter.GetJTISymbol(MO.getIndex()),
                              /* MO has no offset field */0);

  case MachineOperand::MO_ConstantPoolIndex:
    return lowerSymbolOperand(MO, AsmPrinter.GetCPISymbol(MO.getIndex()),
                              MO.getOffset());

  case MachineOperand::MO_BlockAddress: {
    const BlockAddress *BA = MO.getBlockAddress();
    return lowerSymbolOperand(MO, AsmPrinter.GetBlockAddressSymbol(BA),
                              MO.getOffset());
  }
  }
}

void SystemZMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  unsigned Opcode = MI->getOpcode();
  // When emitting binary code, start with the shortest form of an instruction
  // and then relax it where necessary.
  if (!AsmPrinter.OutStreamer.hasRawTextSupport())
    Opcode = getShortenedInstr(Opcode);
  OutMI.setOpcode(Opcode);
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    MCOperand MCOp = lowerOperand(MO);
    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
