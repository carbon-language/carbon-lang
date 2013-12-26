//===-- SparcMCInstLower.cpp - Convert Sparc MachineInstr to MCInst -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Sparc MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "MCTargetDesc/SparcBaseInfo.h"
#include "MCTargetDesc/SparcMCExpr.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;


static MCOperand LowerSymbolOperand(const MachineInstr *MI,
                                    const MachineOperand &MO,
                                    AsmPrinter &AP) {

  SparcMCExpr::VariantKind Kind;
  const MCSymbol *Symbol = 0;

  unsigned TF = MO.getTargetFlags();

  switch(TF) {
  default:      llvm_unreachable("Unknown target flags on operand");
  case SPII::MO_NO_FLAG:      Kind = SparcMCExpr::VK_Sparc_None; break;
  case SPII::MO_LO:           Kind = SparcMCExpr::VK_Sparc_LO; break;
  case SPII::MO_HI:           Kind = SparcMCExpr::VK_Sparc_HI; break;
  case SPII::MO_H44:          Kind = SparcMCExpr::VK_Sparc_H44; break;
  case SPII::MO_M44:          Kind = SparcMCExpr::VK_Sparc_M44; break;
  case SPII::MO_L44:          Kind = SparcMCExpr::VK_Sparc_L44; break;
  case SPII::MO_HH:           Kind = SparcMCExpr::VK_Sparc_HH; break;
  case SPII::MO_HM:           Kind = SparcMCExpr::VK_Sparc_HM; break;
  case SPII::MO_TLS_GD_HI22:  Kind = SparcMCExpr::VK_Sparc_TLS_GD_HI22; break;
  case SPII::MO_TLS_GD_LO10:  Kind = SparcMCExpr::VK_Sparc_TLS_GD_LO10; break;
  case SPII::MO_TLS_GD_ADD:   Kind = SparcMCExpr::VK_Sparc_TLS_GD_ADD; break;
  case SPII::MO_TLS_GD_CALL:  Kind = SparcMCExpr::VK_Sparc_TLS_GD_CALL; break;
  case SPII::MO_TLS_LDM_HI22: Kind = SparcMCExpr::VK_Sparc_TLS_LDM_HI22; break;
  case SPII::MO_TLS_LDM_LO10: Kind = SparcMCExpr::VK_Sparc_TLS_LDM_LO10; break;
  case SPII::MO_TLS_LDM_ADD:  Kind = SparcMCExpr::VK_Sparc_TLS_LDM_ADD; break;
  case SPII::MO_TLS_LDM_CALL: Kind = SparcMCExpr::VK_Sparc_TLS_LDM_CALL; break;
  case SPII::MO_TLS_LDO_HIX22:Kind = SparcMCExpr::VK_Sparc_TLS_LDO_HIX22; break;
  case SPII::MO_TLS_LDO_LOX10:Kind = SparcMCExpr::VK_Sparc_TLS_LDO_LOX10; break;
  case SPII::MO_TLS_LDO_ADD:  Kind = SparcMCExpr::VK_Sparc_TLS_LDO_ADD; break;
  case SPII::MO_TLS_IE_HI22:  Kind = SparcMCExpr::VK_Sparc_TLS_IE_HI22; break;
  case SPII::MO_TLS_IE_LO10:  Kind = SparcMCExpr::VK_Sparc_TLS_IE_LO10; break;
  case SPII::MO_TLS_IE_LD:    Kind = SparcMCExpr::VK_Sparc_TLS_IE_LD; break;
  case SPII::MO_TLS_IE_LDX:   Kind = SparcMCExpr::VK_Sparc_TLS_IE_LDX; break;
  case SPII::MO_TLS_IE_ADD:   Kind = SparcMCExpr::VK_Sparc_TLS_IE_ADD; break;
  case SPII::MO_TLS_LE_HIX22: Kind = SparcMCExpr::VK_Sparc_TLS_LE_HIX22; break;
  case SPII::MO_TLS_LE_LOX10: Kind = SparcMCExpr::VK_Sparc_TLS_LE_LOX10; break;
  }

  switch(MO.getType()) {
  default: llvm_unreachable("Unknown type in LowerSymbolOperand");
  case MachineOperand::MO_MachineBasicBlock:
    Symbol = MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_GlobalAddress:
    Symbol = AP.getSymbol(MO.getGlobal());
    break;

  case MachineOperand::MO_BlockAddress:
    Symbol = AP.GetBlockAddressSymbol(MO.getBlockAddress());
    break;

  case MachineOperand::MO_ExternalSymbol:
    Symbol = AP.GetExternalSymbolSymbol(MO.getSymbolName());
    break;

  case MachineOperand::MO_ConstantPoolIndex:
    Symbol = AP.GetCPISymbol(MO.getIndex());
    break;
  }

  const MCSymbolRefExpr *MCSym = MCSymbolRefExpr::Create(Symbol,
                                                         AP.OutContext);
  const SparcMCExpr *expr = SparcMCExpr::Create(Kind, MCSym,
                                                AP.OutContext);
  return MCOperand::CreateExpr(expr);
}

static MCOperand LowerOperand(const MachineInstr *MI,
                              const MachineOperand &MO,
                              AsmPrinter &AP) {
  switch(MO.getType()) {
  default: llvm_unreachable("unknown operand type"); break;
  case MachineOperand::MO_Register:
    if (MO.isImplicit())
      break;
    return MCOperand::CreateReg(MO.getReg());

  case MachineOperand::MO_Immediate:
    return MCOperand::CreateImm(MO.getImm());

  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_BlockAddress:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_ConstantPoolIndex:
    return LowerSymbolOperand(MI, MO, AP);

  case MachineOperand::MO_RegisterMask:   break;

  }
  return MCOperand();
}

void llvm::LowerSparcMachineInstrToMCInst(const MachineInstr *MI,
                                          MCInst &OutMI,
                                          AsmPrinter &AP)
{

  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = LowerOperand(MI, MO, AP);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
