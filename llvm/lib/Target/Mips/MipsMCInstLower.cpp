//===-- MipsMCInstLower.cpp - Convert Mips MachineInstr to MCInst ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Mips MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "MipsMCInstLower.h"
#include "MipsAsmPrinter.h"
#include "MipsInstrInfo.h"
#include "MipsMCSymbolRefExpr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"
using namespace llvm;

MipsMCInstLower::MipsMCInstLower(Mangler *mang, const MachineFunction &mf,
                                 MipsAsmPrinter &asmprinter)
  : Ctx(mf.getContext()), Mang(mang), AsmPrinter(asmprinter) {}

MCOperand MipsMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                              MachineOperandType MOTy,
                                              unsigned Offset) const {
  MipsMCSymbolRefExpr::VariantKind Kind;
  const MCSymbol *Symbol;

  switch(MO.getTargetFlags()) {
  default:                  assert(0 && "Invalid target flag!");
  case MipsII::MO_NO_FLAG:  Kind = MipsMCSymbolRefExpr::VK_Mips_None; break;
  case MipsII::MO_GPREL:    Kind = MipsMCSymbolRefExpr::VK_Mips_GPREL; break;
  case MipsII::MO_GOT_CALL: Kind = MipsMCSymbolRefExpr::VK_Mips_GOT_CALL; break;
  case MipsII::MO_GOT:      Kind = MipsMCSymbolRefExpr::VK_Mips_GOT; break;
  case MipsII::MO_ABS_HI:   Kind = MipsMCSymbolRefExpr::VK_Mips_ABS_HI; break;
  case MipsII::MO_ABS_LO:   Kind = MipsMCSymbolRefExpr::VK_Mips_ABS_LO; break;
  case MipsII::MO_TLSGD:    Kind = MipsMCSymbolRefExpr::VK_Mips_TLSGD; break;
  case MipsII::MO_GOTTPREL: Kind = MipsMCSymbolRefExpr::VK_Mips_GOTTPREL; break;
  case MipsII::MO_TPREL_HI: Kind = MipsMCSymbolRefExpr::VK_Mips_TPREL_HI; break;
  case MipsII::MO_TPREL_LO: Kind = MipsMCSymbolRefExpr::VK_Mips_TPREL_LO; break;
  }

  switch (MOTy) {
    case MachineOperand::MO_MachineBasicBlock:
      Symbol = MO.getMBB()->getSymbol();
      break;

    case MachineOperand::MO_GlobalAddress:
      Symbol = Mang->getSymbol(MO.getGlobal());
      break;

    case MachineOperand::MO_BlockAddress:
      Symbol = AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress());
      break;

    case MachineOperand::MO_ExternalSymbol:
      Symbol = AsmPrinter.GetExternalSymbolSymbol(MO.getSymbolName());
      break;

    case MachineOperand::MO_JumpTableIndex:
      Symbol = AsmPrinter.GetJTISymbol(MO.getIndex());
      break;

    case MachineOperand::MO_ConstantPoolIndex:
      Symbol = AsmPrinter.GetCPISymbol(MO.getIndex());
      if (MO.getOffset())
        Offset += MO.getOffset();  
      break;

    default:
      llvm_unreachable("<unknown operand type>");
  }
  
  return MCOperand::CreateExpr(MipsMCSymbolRefExpr::Create(Kind, Symbol, Offset,
                                                           Ctx));
}

// If target is Mips1, expand double precision load/store to two single
// precision loads/stores.
// 
//  ldc1 $f0, lo($CPI0_0)($5) gets expanded to the following two instructions:
//  (little endian)
//   lwc1 $f0, lo($CPI0_0)($5) and
//   lwc1 $f1, lo($CPI0_0+4)($5)
//  (big endian)
//   lwc1 $f1, lo($CPI0_0)($5) and
//   lwc1 $f0, lo($CPI0_0+4)($5)
void MipsMCInstLower::LowerMips1F64LoadStore(const MachineInstr *MI,
                                             unsigned Opc,
                                             SmallVector<MCInst, 4>& MCInsts,
                                             bool isLittle,
                                             const unsigned *SubReg) const {
  MCInst InstLo, InstHi, DelaySlot;
  unsigned SingleOpc = (Opc == Mips::LDC1 ? Mips::LWC1 : Mips::SWC1);
  unsigned RegLo = isLittle ? *SubReg : *(SubReg + 1);
  unsigned RegHi = isLittle ? *(SubReg + 1) : *SubReg;
  const MachineOperand &MO1 = MI->getOperand(1);
  const MachineOperand &MO2 = MI->getOperand(2);

  InstLo.setOpcode(SingleOpc);
  InstLo.addOperand(MCOperand::CreateReg(RegLo));
  InstLo.addOperand(LowerOperand(MO1));
  InstLo.addOperand(LowerOperand(MO2));
  MCInsts.push_back(InstLo);

  InstHi.setOpcode(SingleOpc);
  InstHi.addOperand(MCOperand::CreateReg(RegHi));
  InstHi.addOperand(LowerOperand(MO1));
  if (MO2.isImm())// The offset of addr operand is an immediate: e.g. 0($sp)
    InstHi.addOperand(MCOperand::CreateImm(MO2.getImm() + 4));
  else// Otherwise, the offset must be a symbol: e.g. lo($CPI0_0)($5)
    InstHi.addOperand(LowerSymbolOperand(MO2, MO2.getType(), 4));
  MCInsts.push_back(InstHi);

  // Need to insert a NOP in LWC1's delay slot.
  if (SingleOpc == Mips::LWC1) {
    DelaySlot.setOpcode(Mips::NOP);
    MCInsts.push_back(DelaySlot);
  }
}

MCOperand MipsMCInstLower::LowerOperand(const MachineOperand& MO) const {
  MachineOperandType MOTy = MO.getType();
  
  switch (MOTy) {
  default:
    assert(0 && "unknown operand type");
    break;
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit()) break;
    return MCOperand::CreateReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::CreateImm(MO.getImm());
  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_BlockAddress:
    return LowerSymbolOperand(MO, MOTy, 0);
 }

  return MCOperand();
}

void MipsMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = LowerOperand(MO);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
