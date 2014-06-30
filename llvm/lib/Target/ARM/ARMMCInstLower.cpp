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
#include "ARMAsmPrinter.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;


MCOperand ARMAsmPrinter::GetSymbolRef(const MachineOperand &MO,
                                      const MCSymbol *Symbol) {
  const MCExpr *Expr;
  unsigned Option = MO.getTargetFlags() & ARMII::MO_OPTION_MASK;
  switch (Option) {
  default: {
    Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None,
                                   OutContext);
    switch (Option) {
    default: llvm_unreachable("Unknown target flag on symbol operand");
    case ARMII::MO_NO_FLAG:
      break;
    case ARMII::MO_LO16:
      Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None,
                                     OutContext);
      Expr = ARMMCExpr::CreateLower16(Expr, OutContext);
      break;
    case ARMII::MO_HI16:
      Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None,
                                     OutContext);
      Expr = ARMMCExpr::CreateUpper16(Expr, OutContext);
      break;
    }
    break;
  }

  case ARMII::MO_PLT:
    Expr = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_PLT,
                                   OutContext);
    break;
  }

  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(),
                                                          OutContext),
                                   OutContext);
  return MCOperand::CreateExpr(Expr);

}

bool ARMAsmPrinter::lowerOperand(const MachineOperand &MO,
                                 MCOperand &MCOp) {
  switch (MO.getType()) {
  default: llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all non-CPSR implicit register operands.
    if (MO.isImplicit() && MO.getReg() != ARM::CPSR)
      return false;
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    MCOp = MCOperand::CreateReg(MO.getReg());
    break;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::CreateImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
        MO.getMBB()->getSymbol(), OutContext));
    break;
  case MachineOperand::MO_GlobalAddress: {
    MCOp = GetSymbolRef(MO,
                        GetARMGVSymbol(MO.getGlobal(), MO.getTargetFlags()));
    break;
  }
  case MachineOperand::MO_ExternalSymbol:
   MCOp = GetSymbolRef(MO,
                        GetExternalSymbolSymbol(MO.getSymbolName()));
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = GetSymbolRef(MO, GetJTISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = GetSymbolRef(MO, GetCPISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = GetSymbolRef(MO, GetBlockAddressSymbol(MO.getBlockAddress()));
    break;
  case MachineOperand::MO_FPImmediate: {
    APFloat Val = MO.getFPImm()->getValueAPF();
    bool ignored;
    Val.convert(APFloat::IEEEdouble, APFloat::rmTowardZero, &ignored);
    MCOp = MCOperand::CreateFPImm(Val.convertToDouble());
    break;
  }
  case MachineOperand::MO_RegisterMask:
    // Ignore call clobbers.
    return false;
  }
  return true;
}

void llvm::LowerARMMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                        ARMAsmPrinter &AP) {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    if (AP.lowerOperand(MO, MCOp))
      OutMI.addOperand(MCOp);
  }
}
