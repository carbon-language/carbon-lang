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

#include "MipsAsmPrinter.h"
#include "MipsInstrInfo.h"
#include "MipsMCInstLower.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"

using namespace llvm;

MipsMCInstLower::MipsMCInstLower(Mangler *mang, const MachineFunction &mf,
                                 MipsAsmPrinter &asmprinter)
  : Ctx(mf.getContext()), Mang(mang), AsmPrinter(asmprinter) {}

MCOperand MipsMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                              MachineOperandType MOTy,
                                              unsigned Offset) const {
  MCSymbolRefExpr::VariantKind Kind;
  const MCSymbol *Symbol;

  switch(MO.getTargetFlags()) {
  default:                  assert(0 && "Invalid target flag!");
  case MipsII::MO_NO_FLAG:  Kind = MCSymbolRefExpr::VK_None; break;
  case MipsII::MO_GPREL:    Kind = MCSymbolRefExpr::VK_Mips_GPREL; break;
  case MipsII::MO_GOT_CALL: Kind = MCSymbolRefExpr::VK_Mips_GOT_CALL; break;
  case MipsII::MO_GOT16:    Kind = MCSymbolRefExpr::VK_Mips_GOT16; break;
  case MipsII::MO_GOT:      Kind = MCSymbolRefExpr::VK_Mips_GOT; break;
  case MipsII::MO_ABS_HI:   Kind = MCSymbolRefExpr::VK_Mips_ABS_HI; break;
  case MipsII::MO_ABS_LO:   Kind = MCSymbolRefExpr::VK_Mips_ABS_LO; break;
  case MipsII::MO_TLSGD:    Kind = MCSymbolRefExpr::VK_Mips_TLSGD; break;
  case MipsII::MO_GOTTPREL: Kind = MCSymbolRefExpr::VK_Mips_GOTTPREL; break;
  case MipsII::MO_TPREL_HI: Kind = MCSymbolRefExpr::VK_Mips_TPREL_HI; break;
  case MipsII::MO_TPREL_LO: Kind = MCSymbolRefExpr::VK_Mips_TPREL_LO; break;
  case MipsII::MO_GPOFF_HI: Kind = MCSymbolRefExpr::VK_Mips_GPOFF_HI; break;
  case MipsII::MO_GPOFF_LO: Kind = MCSymbolRefExpr::VK_Mips_GPOFF_LO; break;
  case MipsII::MO_GOT_DISP: Kind = MCSymbolRefExpr::VK_Mips_GOT_DISP; break;
  case MipsII::MO_GOT_PAGE: Kind = MCSymbolRefExpr::VK_Mips_GOT_PAGE; break;
  case MipsII::MO_GOT_OFST: Kind = MCSymbolRefExpr::VK_Mips_GOT_OFST; break;
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
  
  const MCSymbolRefExpr *MCSym = MCSymbolRefExpr::Create(Symbol, Kind, Ctx);

  if (!Offset)
    return MCOperand::CreateExpr(MCSym);

  // Assume offset is never negative.
  assert(Offset > 0);
   
  const MCConstantExpr *OffsetExpr =  MCConstantExpr::Create(Offset, Ctx);
  const MCBinaryExpr *AddExpr = MCBinaryExpr::CreateAdd(MCSym, OffsetExpr, Ctx);
  return MCOperand::CreateExpr(AddExpr);
}

// Lower ".cpload $reg" to
//  "lui   $gp, %hi(_gp_disp)"
//  "addiu $gp, $gp, %lo(_gp_disp)"
//  "addu  $gp. $gp, $reg"
void MipsMCInstLower::LowerCPLOAD(const MachineInstr *MI,
                                  SmallVector<MCInst, 4>& MCInsts) {
  MCInst Lui, Addiu, Addu;
  StringRef SymName("_gp_disp");
  const MCSymbol *Symbol = Ctx.GetOrCreateSymbol(SymName);
  const MCSymbolRefExpr *MCSym;

  // lui   $gp, %hi(_gp_disp)
  Lui.setOpcode(Mips::LUi);
  Lui.addOperand(MCOperand::CreateReg(Mips::GP));
  MCSym = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_Mips_ABS_HI, Ctx);
  Lui.addOperand(MCOperand::CreateExpr(MCSym));
  MCInsts.push_back(Lui);

  // addiu $gp, $gp, %lo(_gp_disp)
  Addiu.setOpcode(Mips::ADDiu);
  Addiu.addOperand(MCOperand::CreateReg(Mips::GP));
  Addiu.addOperand(MCOperand::CreateReg(Mips::GP));
  MCSym = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_Mips_ABS_LO, Ctx);
  Addiu.addOperand(MCOperand::CreateExpr(MCSym));
  MCInsts.push_back(Addiu);

  // addu  $gp. $gp, $reg
  Addu.setOpcode(Mips::ADDu);
  Addu.addOperand(MCOperand::CreateReg(Mips::GP));
  Addu.addOperand(MCOperand::CreateReg(Mips::GP));
  const MachineOperand &MO = MI->getOperand(0);
  assert(MO.isReg() && "CPLOAD's operand must be a register.");
  Addu.addOperand(MCOperand::CreateReg(MO.getReg()));
  MCInsts.push_back(Addu);
}

// Lower ".cprestore offset" to "sw $gp, offset($sp)".
void MipsMCInstLower::LowerCPRESTORE(const MachineInstr *MI,
                                     SmallVector<MCInst, 4>& MCInsts) {
  const MachineOperand &MO = MI->getOperand(0);
  assert(MO.isImm() && "CPRESTORE's operand must be an immediate.");
  unsigned Offset = MO.getImm(), Reg = Mips::SP;
  MCInst Sw;

  if (Offset >= 0x8000) {
    unsigned Hi = (Offset >> 16) + ((Offset & 0x8000) != 0); 
    Offset &= 0xffff;
    Reg = Mips::AT;

    // lui   at,hi
    // addu  at,at,sp
    MCInsts.resize(2);
    MCInsts[0].setOpcode(Mips::LUi);
    MCInsts[0].addOperand(MCOperand::CreateReg(Mips::AT));
    MCInsts[0].addOperand(MCOperand::CreateImm(Hi));
    MCInsts[1].setOpcode(Mips::ADDu);
    MCInsts[1].addOperand(MCOperand::CreateReg(Mips::AT));
    MCInsts[1].addOperand(MCOperand::CreateReg(Mips::AT));
    MCInsts[1].addOperand(MCOperand::CreateReg(Mips::SP));
  }
  
  Sw.setOpcode(Mips::SW);
  Sw.addOperand(MCOperand::CreateReg(Mips::GP));
  Sw.addOperand(MCOperand::CreateReg(Reg));
  Sw.addOperand(MCOperand::CreateImm(Offset));
  MCInsts.push_back(Sw);
}

MCOperand MipsMCInstLower::LowerOperand(const MachineOperand& MO,
		                                    unsigned offset) const {
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
    return MCOperand::CreateImm(MO.getImm() + offset);
  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_BlockAddress:
    return LowerSymbolOperand(MO, MOTy, offset);
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

void MipsMCInstLower::LowerUnalignedLoadStore(const MachineInstr *MI,
		                                          SmallVector<MCInst,
		                                          4>& MCInsts) {
  unsigned Opc = MI->getOpcode();
  MCInst instr1, instr2, instr3, move;

  bool two_instructions = false;

  assert(MI->getNumOperands() == 3);
  assert(MI->getOperand(0).isReg());
  assert(MI->getOperand(1).isReg());

  MCOperand target = LowerOperand(MI->getOperand(0));
  MCOperand base = LowerOperand(MI->getOperand(1));
  MCOperand atReg = MCOperand::CreateReg(Mips::AT);
  MCOperand zeroReg = MCOperand::CreateReg(Mips::ZERO);

  MachineOperand unloweredName = MI->getOperand(2);
  MCOperand name = LowerOperand(unloweredName);

  move.setOpcode(Mips::ADDu);
  move.addOperand(target);
  move.addOperand(atReg);
  move.addOperand(zeroReg);

  switch (Opc) {
  case Mips::ULW: {
    // FIXME: only works for little endian right now
    MCOperand adj_name = LowerOperand(unloweredName, 3);
    if (base.getReg() == (target.getReg())) {
      instr1.setOpcode(Mips::LWL);
      instr1.addOperand(atReg);
      instr1.addOperand(base);
      instr1.addOperand(adj_name);
      instr2.setOpcode(Mips::LWR);
      instr2.addOperand(atReg);
      instr2.addOperand(base);
      instr2.addOperand(name);
      instr3 = move;
    } else {
      two_instructions = true;
      instr1.setOpcode(Mips::LWL);
      instr1.addOperand(target);
      instr1.addOperand(base);
      instr1.addOperand(adj_name);
      instr2.setOpcode(Mips::LWR);
      instr2.addOperand(target);
      instr2.addOperand(base);
      instr2.addOperand(name);
    }
    break;
  }
  case Mips::ULHu: {
    // FIXME: only works for little endian right now
    MCOperand adj_name = LowerOperand(unloweredName, 1);
    instr1.setOpcode(Mips::LBu);
    instr1.addOperand(atReg);
    instr1.addOperand(base);
    instr1.addOperand(adj_name);
    instr2.setOpcode(Mips::LBu);
    instr2.addOperand(target);
    instr2.addOperand(base);
    instr2.addOperand(name);
    instr3.setOpcode(Mips::INS);
    instr3.addOperand(target);
    instr3.addOperand(atReg);
    instr3.addOperand(MCOperand::CreateImm(0x8));
    instr3.addOperand(MCOperand::CreateImm(0x18));
    break;
  }

  case Mips::USW: {
    // FIXME: only works for little endian right now
    assert (base.getReg() != target.getReg());
    two_instructions = true;
    MCOperand adj_name = LowerOperand(unloweredName, 3);
    instr1.setOpcode(Mips::SWL);
    instr1.addOperand(target);
    instr1.addOperand(base);
    instr1.addOperand(adj_name);
    instr2.setOpcode(Mips::SWR);
    instr2.addOperand(target);
    instr2.addOperand(base);
    instr2.addOperand(name);
    break;
  }
  case Mips::USH: {
    MCOperand adj_name = LowerOperand(unloweredName, 1);
    instr1.setOpcode(Mips::SB);
    instr1.addOperand(target);
    instr1.addOperand(base);
    instr1.addOperand(name);
    instr2.setOpcode(Mips::SRL);
    instr2.addOperand(atReg);
    instr2.addOperand(target);
    instr2.addOperand(MCOperand::CreateImm(8));
    instr3.setOpcode(Mips::SB);
    instr3.addOperand(atReg);
    instr3.addOperand(base);
    instr3.addOperand(adj_name);
    break;
  }
  default:
    // FIXME: need to add others
    assert(0 && "unaligned instruction not processed");
  }

  MCInsts.push_back(instr1);
  MCInsts.push_back(instr2);
  if (!two_instructions) MCInsts.push_back(instr3);
}

