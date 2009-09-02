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
#include "llvm/ADT/StringExtras.h"  // fixme, kill utostr.

using namespace llvm;

MCSymbol *X86ATTAsmPrinter::GetPICBaseSymbol() {
  // FIXME: the actual label generated doesn't matter here!  Just mangle in
  // something unique (the function number) with Private prefix.
  std::string Name;
  
  if (Subtarget->isTargetDarwin()) {
    Name = "L" + utostr(getFunctionNumber())+"$pb";
  } else {
    assert(Subtarget->isTargetELF() && "Don't know how to print PIC label!");
    Name = ".Lllvm$" + utostr(getFunctionNumber())+".$piclabel";
  }     
  return OutContext.GetOrCreateSymbol(Name);
}



static void lower_lea64_32mem(MCInst *MI, unsigned OpNo) {
  // Convert registers in the addr mode according to subreg64.
  for (unsigned i = 0; i != 4; ++i) {
    if (!MI->getOperand(i).isReg()) continue;
    
    unsigned Reg = MI->getOperand(i).getReg();
    if (Reg == 0) continue;
    
    MI->getOperand(i).setReg(getX86SubSuperRegister(Reg, MVT::i64));
  }
}

/// LowerGlobalAddressOperand - Lower an MO_GlobalAddress operand to an
/// MCOperand.
MCOperand X86ATTAsmPrinter::LowerGlobalAddressOperand(const MachineOperand &MO){
  const GlobalValue *GV = MO.getGlobal();
  
  const char *Suffix = "";
  if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB)
    Suffix = "$stub";
  else if (MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
           MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE ||
           MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY ||
           MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
    Suffix = "$non_lazy_ptr";
  
  std::string Name = Mang->getMangledName(GV, Suffix, Suffix[0] != '\0');
  if (Subtarget->isTargetCygMing())
    DecorateCygMingName(Name, GV);
  
  // Handle dllimport linkage.
  if (MO.getTargetFlags() == X86II::MO_DLLIMPORT)
    Name = "__imp_" + Name;
  
  if (MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
      MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE)
    GVStubs[Name] = Mang->getMangledName(GV);
  else if (MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY ||
           MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
    HiddenGVStubs[Name] = Mang->getMangledName(GV);
  else if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB)
    FnStubs[Name] = Mang->getMangledName(GV);
  
  
  // Handle target operand flags.
  // FIXME: This should be common between external symbols, constant pool etc.
  MCSymbol *NegatedSymbol = 0;
  
  switch (MO.getTargetFlags()) {
    default:
      llvm_unreachable("Unknown target flag on GV operand");
    case X86II::MO_NO_FLAG:    // No flag.
      break;
    case X86II::MO_DARWIN_NONLAZY:
    case X86II::MO_DARWIN_HIDDEN_NONLAZY:
    case X86II::MO_DLLIMPORT:
    case X86II::MO_DARWIN_STUB:
      // These affect the name of the symbol, not any suffix.
      break;
    case X86II::MO_GOT_ABSOLUTE_ADDRESS:
      assert(0 && "Reloc mode unimp!");
      //O << " + [.-";
      //PrintPICBaseSymbol();
      //O << ']';
      break;      
    case X86II::MO_PIC_BASE_OFFSET:
    case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
    case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
      // Subtract the pic base.
      NegatedSymbol = GetPICBaseSymbol();
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
  MCSymbol *Sym = OutContext.GetOrCreateSymbol(Name);
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, OutContext);
  if (NegatedSymbol)
    Expr = MCBinaryExpr::CreateSub(Expr, MCSymbolRefExpr::Create(NegatedSymbol,
                                                                 OutContext),
                                   OutContext);
  if (MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr, MCConstantExpr::Create(MO.getOffset(),
                                                                OutContext),
                                   OutContext);
  return MCOperand::CreateExpr(Expr);
}

MCOperand X86ATTAsmPrinter::
LowerExternalSymbolOperand(const MachineOperand &MO) {
  std::string Name = Mang->makeNameProper(MO.getSymbolName());
  if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
    FnStubs[Name+"$stub"] = Name;
    Name += "$stub";
  }
  
  MCSymbol *Sym = OutContext.GetOrCreateSymbol(Name);
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, OutContext);
  if (MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(),
                                                          OutContext),
                                   OutContext);
  return MCOperand::CreateExpr(Expr);
}

MCOperand X86ATTAsmPrinter::LowerJumpTableOperand(const MachineOperand &MO) {
  SmallString<256> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "JTI"
  << getFunctionNumber() << '_' << MO.getIndex();
  
  MCSymbol *NegatedSymbol = 0;
  switch (MO.getTargetFlags()) {
    default:
      llvm_unreachable("Unknown target flag on GV operand");
    case X86II::MO_PIC_BASE_OFFSET:
    case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
    case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
      // Subtract the pic base.
      NegatedSymbol = GetPICBaseSymbol();
      break;
  }
  
  // Create a symbol for the name.
  MCSymbol *Sym = OutContext.GetOrCreateSymbol(Name.str());
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, OutContext);
  if (NegatedSymbol)
    Expr = MCBinaryExpr::CreateSub(Expr, MCSymbolRefExpr::Create(NegatedSymbol,
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
    case MachineOperand::MO_JumpTableIndex:
      MCOp = LowerJumpTableOperand(MO);
      break;
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
      MCOp = LowerGlobalAddressOperand(MO);
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerExternalSymbolOperand(MO);
      break;
    }
    
    TmpInst.addOperand(MCOp);
  }
  
  switch (TmpInst.getOpcode()) {
  case X86::LEA64_32r:
    // Handle the 'subreg rewriting' for the lea64_32mem operand.
    lower_lea64_32mem(&TmpInst, 1);
    break;
  }
  
  printInstruction(&TmpInst);
}
