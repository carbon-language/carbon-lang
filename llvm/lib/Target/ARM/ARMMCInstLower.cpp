//===-- ARMMCInstLower.cpp - Convert ARM MachineInstr to an MCInst --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower ARM MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMMCInstLower.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Constants.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

MCSymbol *ARMMCInstLower::GetGlobalAddressSymbol(const GlobalValue *GV) const {
  return Printer.Mang->getSymbol(GV);
}

const MCSymbolRefExpr *ARMMCInstLower::
GetSymbolRef(const MachineOperand &MO) const {
  assert(MO.isGlobal() && "Isn't a global address reference?");

  const MCSymbolRefExpr *SymRef;
  const MCSymbol *Symbol = GetGlobalAddressSymbol(MO.getGlobal());

  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on GV operand");
  case 0:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, Ctx);
    break;
  case ARMII::MO_LO16:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_LO16, Ctx);
    break;
  case ARMII::MO_HI16:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_HI16, Ctx);
    break;
  case ARMII::MO_PLT:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_PLT, Ctx);
    break;
  }

  return SymRef;
}

const MCSymbolRefExpr *ARMMCInstLower::
GetExternalSymbolSymbol(const MachineOperand &MO) const {
  const MCSymbolRefExpr *SymRef;
  const MCSymbol *Symbol = Printer.GetExternalSymbolSymbol(MO.getSymbolName());

  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on external symbol operand");
  case 0:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, Ctx);
    break;
  case ARMII::MO_LO16:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_LO16, Ctx);
    break;
  case ARMII::MO_HI16:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_HI16, Ctx);
    break;
  case ARMII::MO_PLT:
    SymRef = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_PLT, Ctx);
    break;
  }

  return SymRef;
}



MCSymbol *ARMMCInstLower::
GetJumpTableSymbol(const MachineOperand &MO) const {
  SmallString<256> Name;
  raw_svector_ostream(Name) << Printer.MAI->getPrivateGlobalPrefix() << "JTI"
    << Printer.getFunctionNumber() << '_' << MO.getIndex();

  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on jump table operand");
  case 0: break;
  }

  // Create a symbol for the name.
  return Ctx.GetOrCreateSymbol(Name.str());
}

MCSymbol *ARMMCInstLower::
GetConstantPoolIndexSymbol(const MachineOperand &MO) const {
  SmallString<256> Name;
  raw_svector_ostream(Name) << Printer.MAI->getPrivateGlobalPrefix() << "CPI"
    << Printer.getFunctionNumber() << '_' << MO.getIndex();

  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on CPI operand");
  case 0: break;
  }

  // Create a symbol for the name.
  return Ctx.GetOrCreateSymbol(Name.str());
}

MCOperand ARMMCInstLower::
LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, Ctx);

  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on Symbol operand");
  case 0: break;
  }

  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(), Ctx),
                                   Ctx);
  return MCOperand::CreateExpr(Expr);
}

MCOperand ARMMCInstLower::
LowerSymbolRefOperand(const MachineOperand &MO,
                      const MCSymbolRefExpr *Sym) const {
  const MCExpr *Expr = Sym;
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(), Ctx),
                                   Ctx);
  return MCOperand::CreateExpr(Expr);
}


void ARMMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->dump();
      assert(0 && "unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all non-CPSR implicit register operands.
      if (MO.isImplicit() && MO.getReg() != ARM::CPSR) continue;
      assert(!MO.getSubReg() && "Subregs should be eliminated!");
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                       MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_GlobalAddress:
      MCOp = LowerSymbolRefOperand(MO, GetSymbolRef(MO));
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerSymbolRefOperand(MO, GetExternalSymbolSymbol(MO));
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = LowerSymbolOperand(MO, GetJumpTableSymbol(MO));
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = LowerSymbolOperand(MO, GetConstantPoolIndexSymbol(MO));
      break;
    case MachineOperand::MO_BlockAddress:
      MCOp = LowerSymbolOperand(MO, Printer.GetBlockAddressSymbol(
                                              MO.getBlockAddress()));
      break;
    case MachineOperand::MO_FPImmediate:
      APFloat Val = MO.getFPImm()->getValueAPF();
      bool ignored;
      Val.convert(APFloat::IEEEdouble, APFloat::rmTowardZero, &ignored);
      MCOp = MCOperand::CreateFPImm(Val.convertToDouble());
      break;
    }

    OutMI.addOperand(MCOp);
  }

}
