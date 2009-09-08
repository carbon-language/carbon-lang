//===-- X86MCInstLower.cpp - Convert X86 MachineInstr to an MCInst --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower X86 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//


#include "X86ATTAsmPrinter.h"
#include "X86MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

MCSymbol *X86ATTAsmPrinter::GetPICBaseSymbol() {
  // FIXME: the actual label generated doesn't matter here!  Just mangle in
  // something unique (the function number) with Private prefix.
  SmallString<60> Name;
  
  if (Subtarget->isTargetDarwin()) {
    raw_svector_ostream(Name) << 'L' << getFunctionNumber() << "$pb";
  } else {
    assert(Subtarget->isTargetELF() && "Don't know how to print PIC label!");
    raw_svector_ostream(Name) << ".Lllvm$" << getFunctionNumber()<<".$piclabel";
  }
  return OutContext.GetOrCreateSymbol(Name.str());
}


static void lower_subreg32(MCInst *MI, unsigned OpNo) {
  // Convert registers in the addr mode according to subreg32.
  unsigned Reg = MI->getOperand(OpNo).getReg();
  if (Reg != 0)
    MI->getOperand(OpNo).setReg(getX86SubSuperRegister(Reg, MVT::i32));
}


static void lower_lea64_32mem(MCInst *MI, unsigned OpNo) {
  // Convert registers in the addr mode according to subreg64.
  for (unsigned i = 0; i != 4; ++i) {
    if (!MI->getOperand(OpNo+i).isReg()) continue;
    
    unsigned Reg = MI->getOperand(OpNo+i).getReg();
    if (Reg == 0) continue;
    
    MI->getOperand(OpNo+i).setReg(getX86SubSuperRegister(Reg, MVT::i64));
  }
}

/// LowerGlobalAddressOperand - Lower an MO_GlobalAddress operand to an
/// MCOperand.
MCSymbol *X86ATTAsmPrinter::GetGlobalAddressSymbol(const MachineOperand &MO) {
  const GlobalValue *GV = MO.getGlobal();
  
  const char *Suffix = "";
  if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB)
    Suffix = "$stub";
  else if (MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
           MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE ||
           MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
    Suffix = "$non_lazy_ptr";
  
  std::string Name = Mang->getMangledName(GV, Suffix, Suffix[0] != '\0');
  if (Subtarget->isTargetCygMing())
    DecorateCygMingName(Name, GV);
  
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:                // No flag.
  case X86II::MO_GOT_ABSOLUTE_ADDRESS:   // Doesn't modify symbol name.
  case X86II::MO_PIC_BASE_OFFSET:        // Doesn't modify symbol name.
    break;
  case X86II::MO_DLLIMPORT:
    // Handle dllimport linkage.
    Name = "__imp_" + Name;
    break;
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
    GVStubs[Name] = Mang->getMangledName(GV);
    break;
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    HiddenGVStubs[Name] = Mang->getMangledName(GV);
    break;
  case X86II::MO_DARWIN_STUB:
    FnStubs[Name] = Mang->getMangledName(GV);
    break;
  // FIXME: These probably should be a modifier on the symbol or something??
  case X86II::MO_TLSGD:     Name += "@TLSGD";     break;
  case X86II::MO_GOTTPOFF:  Name += "@GOTTPOFF";  break;
  case X86II::MO_INDNTPOFF: Name += "@INDNTPOFF"; break;
  case X86II::MO_TPOFF:     Name += "@TPOFF";     break;
  case X86II::MO_NTPOFF:    Name += "@NTPOFF";    break;
  case X86II::MO_GOTPCREL:  Name += "@GOTPCREL";  break;
  case X86II::MO_GOT:       Name += "@GOT";       break;
  case X86II::MO_GOTOFF:    Name += "@GOTOFF";    break;
  case X86II::MO_PLT:       Name += "@PLT";       break;
  }
  
  return OutContext.GetOrCreateSymbol(Name);
}

MCSymbol *X86ATTAsmPrinter::GetExternalSymbolSymbol(const MachineOperand &MO) {
  std::string Name = Mang->makeNameProper(MO.getSymbolName());
  if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
    FnStubs[Name+"$stub"] = Name;
    Name += "$stub";
  }
  
  return OutContext.GetOrCreateSymbol(Name);
}

MCSymbol *X86ATTAsmPrinter::GetJumpTableSymbol(const MachineOperand &MO) {
  SmallString<256> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "JTI"
    << getFunctionNumber() << '_' << MO.getIndex();
  
  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    break;
    // FIXME: These probably should be a modifier on the symbol or something??
  case X86II::MO_TLSGD:     Name += "@TLSGD";     break;
  case X86II::MO_GOTTPOFF:  Name += "@GOTTPOFF";  break;
  case X86II::MO_INDNTPOFF: Name += "@INDNTPOFF"; break;
  case X86II::MO_TPOFF:     Name += "@TPOFF";     break;
  case X86II::MO_NTPOFF:    Name += "@NTPOFF";    break;
  case X86II::MO_GOTPCREL:  Name += "@GOTPCREL";  break;
  case X86II::MO_GOT:       Name += "@GOT";       break;
  case X86II::MO_GOTOFF:    Name += "@GOTOFF";    break;
  case X86II::MO_PLT:       Name += "@PLT";       break;
  }
  
  // Create a symbol for the name.
  return OutContext.GetOrCreateSymbol(Name.str());
}


MCSymbol *X86ATTAsmPrinter::
GetConstantPoolIndexSymbol(const MachineOperand &MO) {
  SmallString<256> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "CPI"
  << getFunctionNumber() << '_' << MO.getIndex();
  
  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    break;
    // FIXME: These probably should be a modifier on the symbol or something??
  case X86II::MO_TLSGD:     Name += "@TLSGD";     break;
  case X86II::MO_GOTTPOFF:  Name += "@GOTTPOFF";  break;
  case X86II::MO_INDNTPOFF: Name += "@INDNTPOFF"; break;
  case X86II::MO_TPOFF:     Name += "@TPOFF";     break;
  case X86II::MO_NTPOFF:    Name += "@NTPOFF";    break;
  case X86II::MO_GOTPCREL:  Name += "@GOTPCREL";  break;
  case X86II::MO_GOT:       Name += "@GOT";       break;
  case X86II::MO_GOTOFF:    Name += "@GOTOFF";    break;
  case X86II::MO_PLT:       Name += "@PLT";       break;
  }
  
  // Create a symbol for the name.
  return OutContext.GetOrCreateSymbol(Name.str());
}

MCOperand X86ATTAsmPrinter::LowerSymbolOperand(const MachineOperand &MO,
                                               MCSymbol *Sym) {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, OutContext);
  
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
      
  // These affect the name of the symbol, not any suffix.
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DLLIMPORT:
  case X86II::MO_DARWIN_STUB:
  case X86II::MO_TLSGD:
  case X86II::MO_GOTTPOFF:
  case X86II::MO_INDNTPOFF:
  case X86II::MO_TPOFF:
  case X86II::MO_NTPOFF:
  case X86II::MO_GOTPCREL:
  case X86II::MO_GOT:
  case X86II::MO_GOTOFF:
  case X86II::MO_PLT:
    break;
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    // Subtract the pic base.
    Expr = MCBinaryExpr::CreateSub(Expr, 
                                   MCSymbolRefExpr::Create(GetPICBaseSymbol(),
                                                           OutContext),
                                   OutContext);
    break;
  case X86II::MO_GOT_ABSOLUTE_ADDRESS: {
    // For this, we want to print something like:
    //   MYSYMBOL + (. - PICBASE)
    // However, we can't generate a ".", so just emit a new label here and refer
    // to it.  We know that this operand flag occurs at most once per function.
    SmallString<64> Name;
    raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "picbaseref"
      << getFunctionNumber();
    MCSymbol *DotSym = OutContext.GetOrCreateSymbol(Name.str());
    OutStreamer.EmitLabel(DotSym);

    const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, OutContext);
    const MCExpr *PICBase = MCSymbolRefExpr::Create(GetPICBaseSymbol(),
                                                    OutContext);
    DotExpr = MCBinaryExpr::CreateSub(DotExpr, PICBase, OutContext);
    Expr = MCBinaryExpr::CreateAdd(Expr, DotExpr, OutContext);
    break;      
  }
  }
  
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr, MCConstantExpr::Create(MO.getOffset(),
                                                                OutContext),
                                   OutContext);
  return MCOperand::CreateExpr(Expr);
}

void X86ATTAsmPrinter::
printInstructionThroughMCStreamer(const MachineInstr *MI) {
  
  MCInst TmpInst;
  
  switch (MI->getOpcode()) {
  case TargetInstrInfo::DBG_LABEL:
  case TargetInstrInfo::EH_LABEL:
  case TargetInstrInfo::GC_LABEL:
    printLabel(MI);
    return;
  case TargetInstrInfo::INLINEASM:
    O << '\t';
    printInlineAsm(MI);
    return;
  case TargetInstrInfo::IMPLICIT_DEF:
    printImplicitDef(MI);
    return;
  case X86::MOVPC32r: {
    // This is a pseudo op for a two instruction sequence with a label, which
    // looks like:
    //     call "L1$pb"
    // "L1$pb":
    //     popl %esi
    
    // Emit the call.
    MCSymbol *PICBase = GetPICBaseSymbol();
    TmpInst.setOpcode(X86::CALLpcrel32);
    // FIXME: We would like an efficient form for this, so we don't have to do a
    // lot of extra uniquing.
    TmpInst.addOperand(MCOperand::CreateExpr(MCSymbolRefExpr::Create(PICBase,
                                                                 OutContext)));
    printInstruction(&TmpInst);
    
    // Emit the label.
    OutStreamer.EmitLabel(PICBase);
    
    // popl $reg
    TmpInst.setOpcode(X86::POP32r);
    TmpInst.getOperand(0) = MCOperand::CreateReg(MI->getOperand(0).getReg());
    printInstruction(&TmpInst);
    return;
    }
  }
  
  TmpInst.setOpcode(MI->getOpcode());
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    
    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      O.flush();
      errs() << "Cannot lower operand #" << i << " of :" << *MI;
      llvm_unreachable("Unimp");
    case MachineOperand::MO_Register:
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::CreateMBBLabel(getFunctionNumber(), 
                                       MO.getMBB()->getNumber());
      break;
    case MachineOperand::MO_GlobalAddress:
      MCOp = LowerSymbolOperand(MO, GetGlobalAddressSymbol(MO));
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerSymbolOperand(MO, GetExternalSymbolSymbol(MO));
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = LowerSymbolOperand(MO, GetJumpTableSymbol(MO));
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = LowerSymbolOperand(MO, GetConstantPoolIndexSymbol(MO));
      break;
    }
    
    TmpInst.addOperand(MCOp);
  }
  
  switch (TmpInst.getOpcode()) {
  case X86::LEA64_32r:
    // Handle the 'subreg rewriting' for the lea64_32mem operand.
    lower_lea64_32mem(&TmpInst, 1);
    break;
  case X86::MOV16r0:
    TmpInst.setOpcode(X86::MOV32r0);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX16rr8:
    TmpInst.setOpcode(X86::MOVZX32rr8);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX16rm8:
    TmpInst.setOpcode(X86::MOVZX32rm8);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVSX16rr8:
    TmpInst.setOpcode(X86::MOVSX32rr8);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVSX16rm8:
    TmpInst.setOpcode(X86::MOVSX32rm8);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX64rr32:
    TmpInst.setOpcode(X86::MOV32rr);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX64rm32:
    TmpInst.setOpcode(X86::MOV32rm);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOV64ri64i32:
    TmpInst.setOpcode(X86::MOV32ri);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX64rr8:
    TmpInst.setOpcode(X86::MOVZX32rr8);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX64rm8:
    TmpInst.setOpcode(X86::MOVZX32rm8);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX64rr16:
    TmpInst.setOpcode(X86::MOVZX32rr16);
    lower_subreg32(&TmpInst, 0);
    break;
  case X86::MOVZX64rm16:
    TmpInst.setOpcode(X86::MOVZX32rm16);
    lower_subreg32(&TmpInst, 0);
    break;
  }
  
  printInstruction(&TmpInst);
}
