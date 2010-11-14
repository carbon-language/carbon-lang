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

#include "ARMMCInstLower.h"
#include "ARM.h"
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

MCOperand ARMMCInstLower::
GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol) const {
  const MCExpr *Expr;
  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on symbol operand");
  case 0:
    Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, Ctx);
    break;
  case ARMII::MO_LO16:
    Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_LO16, Ctx);
    break;
  case ARMII::MO_HI16:
    Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_HI16, Ctx);
    break;
  case ARMII::MO_PLT:
    Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_ARM_PLT, Ctx);
    break;
  }
  
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
      MCOp = GetSymbolRef(MO, GetGlobalAddressSymbol(MO.getGlobal()));
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = GetSymbolRef(MO, 
                          Printer.GetExternalSymbolSymbol(MO.getSymbolName()));
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = GetSymbolRef(MO, Printer.GetJTISymbol(MO.getIndex()));
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = GetSymbolRef(MO, Printer.GetCPISymbol(MO.getIndex()));
      break;
    case MachineOperand::MO_BlockAddress:
      MCOp = GetSymbolRef(MO,
                          Printer.GetBlockAddressSymbol(MO.getBlockAddress()));
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
