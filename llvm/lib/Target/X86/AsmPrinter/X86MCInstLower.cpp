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

#include "X86MCInstLower.h"
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


const X86Subtarget &X86MCInstLower::getSubtarget() const {
  return AsmPrinter.getSubtarget();
}


MCSymbol *X86MCInstLower::GetPICBaseSymbol() const {
  // FIXME: the actual label generated doesn't matter here!  Just mangle in
  // something unique (the function number) with Private prefix.
  SmallString<60> Name;
  
  if (getSubtarget().isTargetDarwin()) {
    raw_svector_ostream(Name) << 'L' << AsmPrinter.getFunctionNumber() << "$pb";
  } else {
    assert(getSubtarget().isTargetELF() && "Don't know how to print PIC label!");
    raw_svector_ostream(Name) << ".Lllvm$" << AsmPrinter.getFunctionNumber()
       << ".$piclabel";
  }
  return Ctx.GetOrCreateSymbol(Name.str());
}


/// LowerGlobalAddressOperand - Lower an MO_GlobalAddress operand to an
/// MCOperand.
MCSymbol *X86MCInstLower::
GetGlobalAddressSymbol(const MachineOperand &MO) const {
  const GlobalValue *GV = MO.getGlobal();
  
  bool isImplicitlyPrivate = false;
  if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB ||
      MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
      MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE ||
      MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
    isImplicitlyPrivate = true;
  
  SmallString<128> Name;
  Mang->getNameWithPrefix(Name, GV, isImplicitlyPrivate);
  
  if (getSubtarget().isTargetCygMing())
    AsmPrinter.DecorateCygMingName(Name, GV);
  
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:                // No flag.
  case X86II::MO_GOT_ABSOLUTE_ADDRESS:   // Doesn't modify symbol name.
  case X86II::MO_PIC_BASE_OFFSET:        // Doesn't modify symbol name.
    break;
  case X86II::MO_DLLIMPORT: {
    // Handle dllimport linkage.
    const char *Prefix = "__imp_";
    Name.insert(Name.begin(), Prefix, Prefix+strlen(Prefix));
    break;
  }
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE: {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MCSymbol *&StubSym = AsmPrinter.GVStubs[Sym];
    if (StubSym == 0) {
      Name.clear();
      Mang->getNameWithPrefix(Name, GV, false);
      StubSym = Ctx.GetOrCreateSymbol(Name.str());
    }
    return Sym;
  }
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MCSymbol *&StubSym = AsmPrinter.HiddenGVStubs[Sym];
    if (StubSym == 0) {
      Name.clear();
      Mang->getNameWithPrefix(Name, GV, false);
      StubSym = Ctx.GetOrCreateSymbol(Name.str());
    }
    return Sym;
  }
  case X86II::MO_DARWIN_STUB: {
    Name += "$stub";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MCSymbol *&StubSym = AsmPrinter.FnStubs[Sym];
    if (StubSym == 0) {
      Name.clear();
      Mang->getNameWithPrefix(Name, GV, false);
      StubSym = Ctx.GetOrCreateSymbol(Name.str());
    }
    return Sym;
  }
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
  
  return Ctx.GetOrCreateSymbol(Name.str());
}

MCSymbol *X86MCInstLower::
GetExternalSymbolSymbol(const MachineOperand &MO) const {
  SmallString<128> Name;
  Name += AsmPrinter.MAI->getGlobalPrefix();
  Name += MO.getSymbolName();
  
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:                // No flag.
  case X86II::MO_GOT_ABSOLUTE_ADDRESS:   // Doesn't modify symbol name.
  case X86II::MO_PIC_BASE_OFFSET:        // Doesn't modify symbol name.
    break;
  case X86II::MO_DLLIMPORT: {
    // Handle dllimport linkage.
    const char *Prefix = "__imp_";
    Name.insert(Name.begin(), Prefix, Prefix+strlen(Prefix));
    break;
  }
  case X86II::MO_DARWIN_STUB: {
    Name += "$stub";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MCSymbol *&StubSym = AsmPrinter.FnStubs[Sym];
    if (StubSym == 0) {
      Name.erase(Name.end()-5, Name.end());
      StubSym = Ctx.GetOrCreateSymbol(Name.str());
    }
    return Sym;
  }
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
  
  return Ctx.GetOrCreateSymbol(Name.str());
}

MCSymbol *X86MCInstLower::GetJumpTableSymbol(const MachineOperand &MO) const {
  SmallString<256> Name;
  raw_svector_ostream(Name) << AsmPrinter.MAI->getPrivateGlobalPrefix() << "JTI"
    << AsmPrinter.getFunctionNumber() << '_' << MO.getIndex();
  
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
  return Ctx.GetOrCreateSymbol(Name.str());
}


MCSymbol *X86MCInstLower::
GetConstantPoolIndexSymbol(const MachineOperand &MO) const {
  SmallString<256> Name;
  raw_svector_ostream(Name) << AsmPrinter.MAI->getPrivateGlobalPrefix() << "CPI"
    << AsmPrinter.getFunctionNumber() << '_' << MO.getIndex();
  
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
  return Ctx.GetOrCreateSymbol(Name.str());
}

MCOperand X86MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                             MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, Ctx);
  
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
                                                           Ctx),
                                   Ctx);
    break;
  case X86II::MO_GOT_ABSOLUTE_ADDRESS: {
    // For this, we want to print something like:
    //   MYSYMBOL + (. - PICBASE)
    // However, we can't generate a ".", so just emit a new label here and refer
    // to it.  We know that this operand flag occurs at most once per function.
    SmallString<64> Name;
    raw_svector_ostream(Name) << AsmPrinter.MAI->getPrivateGlobalPrefix()
      << "picbaseref" << AsmPrinter.getFunctionNumber();
    MCSymbol *DotSym = Ctx.GetOrCreateSymbol(Name.str());
// FIXME: This instruction should be lowered before we get here...    
    AsmPrinter.OutStreamer.EmitLabel(DotSym);

    const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, Ctx);
    const MCExpr *PICBase = MCSymbolRefExpr::Create(GetPICBaseSymbol(),
                                                    Ctx);
    DotExpr = MCBinaryExpr::CreateSub(DotExpr, PICBase, Ctx);
    Expr = MCBinaryExpr::CreateAdd(Expr, DotExpr, Ctx);
    break;      
  }
  }
  
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(), Ctx),
                                   Ctx);
  return MCOperand::CreateExpr(Expr);
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



void X86MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    
    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->dump();
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
// FIXME: Kill MBBLabel operand type!
      MCOp = MCOperand::CreateMBBLabel(AsmPrinter.getFunctionNumber(), 
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
    
    OutMI.addOperand(MCOp);
  }
  
  // Handle a few special cases to eliminate operand modifiers.
  switch (OutMI.getOpcode()) {
  case X86::LEA64_32r: // Handle 'subreg rewriting' for the lea64_32mem operand.
    lower_lea64_32mem(&OutMI, 1);
    break;
  case X86::MOV16r0:
    OutMI.setOpcode(X86::MOV32r0);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX16rr8:
    OutMI.setOpcode(X86::MOVZX32rr8);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX16rm8:
    OutMI.setOpcode(X86::MOVZX32rm8);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVSX16rr8:
    OutMI.setOpcode(X86::MOVSX32rr8);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVSX16rm8:
    OutMI.setOpcode(X86::MOVSX32rm8);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX64rr32:
    OutMI.setOpcode(X86::MOV32rr);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX64rm32:
    OutMI.setOpcode(X86::MOV32rm);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOV64ri64i32:
    OutMI.setOpcode(X86::MOV32ri);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX64rr8:
    OutMI.setOpcode(X86::MOVZX32rr8);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX64rm8:
    OutMI.setOpcode(X86::MOVZX32rm8);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX64rr16:
    OutMI.setOpcode(X86::MOVZX32rr16);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOVZX64rm16:
    OutMI.setOpcode(X86::MOVZX32rm16);
    lower_subreg32(&OutMI, 0);
    break;
  }
}



void X86ATTAsmPrinter::
printInstructionThroughMCStreamer(const MachineInstr *MI) {
  X86MCInstLower MCInstLowering(OutContext, Mang, *this);
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
    MCInst TmpInst;
    // This is a pseudo op for a two instruction sequence with a label, which
    // looks like:
    //     call "L1$pb"
    // "L1$pb":
    //     popl %esi
    
    // Emit the call.
    MCSymbol *PICBase = MCInstLowering.GetPICBaseSymbol();
    TmpInst.setOpcode(X86::CALLpcrel32);
    // FIXME: We would like an efficient form for this, so we don't have to do a
    // lot of extra uniquing.
    TmpInst.addOperand(MCOperand::CreateExpr(MCSymbolRefExpr::Create(PICBase,
                                                                 OutContext)));
    printInstruction(&TmpInst);
    O << '\n';
    
    // Emit the label.
    OutStreamer.EmitLabel(PICBase);
    
    // popl $reg
    TmpInst.setOpcode(X86::POP32r);
    TmpInst.getOperand(0) = MCOperand::CreateReg(MI->getOperand(0).getReg());
    printInstruction(&TmpInst);
    return;
    }
  }
  
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  
  
  printInstruction(&TmpInst);
}
