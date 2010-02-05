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
#include "X86AsmPrinter.h"
#include "X86MCAsmInfo.h"
#include "X86COFFMachineModuleInfo.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Type.h"
using namespace llvm;


const X86Subtarget &X86MCInstLower::getSubtarget() const {
  return AsmPrinter.getSubtarget();
}

MachineModuleInfoMachO &X86MCInstLower::getMachOMMI() const {
  assert(getSubtarget().isTargetDarwin() &&"Can only get MachO info on darwin");
  return AsmPrinter.MMI->getObjFileInfo<MachineModuleInfoMachO>(); 
}


MCSymbol *X86MCInstLower::GetPICBaseSymbol() const {
  const TargetLowering *TLI = AsmPrinter.TM.getTargetLowering();
  return static_cast<const X86TargetLowering*>(TLI)->
    getPICBaseSymbol(AsmPrinter.MF, Ctx);
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
  
  if (getSubtarget().isTargetCygMing()) {
    X86COFFMachineModuleInfo &COFFMMI = 
      AsmPrinter.MMI->getObjFileInfo<X86COFFMachineModuleInfo>();
    COFFMMI.DecorateCygMingName(Name, GV, *AsmPrinter.TM.getTargetData());
  }
  
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:                // No flag.
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

    MCSymbol *&StubSym = getMachOMMI().getGVStubEntry(Sym);
    if (StubSym == 0)
      StubSym = AsmPrinter.GetGlobalValueSymbol(GV);
    return Sym;
  }
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MCSymbol *&StubSym = getMachOMMI().getHiddenGVStubEntry(Sym);
    if (StubSym == 0)
      StubSym = AsmPrinter.GetGlobalValueSymbol(GV);
    return Sym;
  }
  case X86II::MO_DARWIN_STUB: {
    Name += "$stub";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MCSymbol *&StubSym = getMachOMMI().getFnStubEntry(Sym);
    if (StubSym == 0)
      StubSym = AsmPrinter.GetGlobalValueSymbol(GV);
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
    MCSymbol *&StubSym = getMachOMMI().getFnStubEntry(Sym);

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
  // FIXME: Use AsmPrinter.GetJTISymbol.  @TLSGD shouldn't be part of the symbol
  // name!
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
  // FIXME: USe AsmPrinter.GetCPISymbol.  @TLSGD shouldn't be part of the symbol
  // name!
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

MCSymbol *X86MCInstLower::
GetBlockAddressSymbol(const MachineOperand &MO) const {
  const char *Suffix = "";
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on BA operand");
  case X86II::MO_NO_FLAG:         break; // No flag.
  case X86II::MO_PIC_BASE_OFFSET: break; // Doesn't modify symbol name.
  case X86II::MO_GOTOFF: Suffix = "@GOTOFF"; break;
  }

  return AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress(), Suffix);
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
                               MCSymbolRefExpr::Create(GetPICBaseSymbol(), Ctx),
                                   Ctx);
    break;
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

/// LowerMOVxX32 - Things like MOVZX16rr8 -> MOVZX32rr8.
static void LowerMOVxX32(MCInst &OutMI, unsigned NewOpc) {
  OutMI.setOpcode(NewOpc);
  lower_subreg32(&OutMI, 0);
}
/// LowerSETB - R = setb   -> R = sbb R, R
static void LowerSETB(MCInst &OutMI, unsigned NewOpc) {
  OutMI.setOpcode(NewOpc);
  OutMI.addOperand(OutMI.getOperand(0));
  OutMI.addOperand(OutMI.getOperand(0));
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
      // Ignore all implicit register operands.
      if (MO.isImplicit()) continue;
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                       MO.getMBB()->getSymbol(Ctx), Ctx));
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
    case MachineOperand::MO_BlockAddress:
      MCOp = LowerSymbolOperand(MO, GetBlockAddressSymbol(MO));
      break;
    }
    
    OutMI.addOperand(MCOp);
  }
  
  // Handle a few special cases to eliminate operand modifiers.
  switch (OutMI.getOpcode()) {
  case X86::LEA64_32r: // Handle 'subreg rewriting' for the lea64_32mem operand.
    lower_lea64_32mem(&OutMI, 1);
    break;
  case X86::MOVZX16rr8:   LowerMOVxX32(OutMI, X86::MOVZX32rr8); break;
  case X86::MOVZX16rm8:   LowerMOVxX32(OutMI, X86::MOVZX32rm8); break;
  case X86::MOVSX16rr8:   LowerMOVxX32(OutMI, X86::MOVSX32rr8); break;
  case X86::MOVSX16rm8:   LowerMOVxX32(OutMI, X86::MOVSX32rm8); break;
  case X86::MOVZX64rr32:  LowerMOVxX32(OutMI, X86::MOV32rr); break;
  case X86::MOVZX64rm32:  LowerMOVxX32(OutMI, X86::MOV32rm); break;
  case X86::MOV64ri64i32: LowerMOVxX32(OutMI, X86::MOV32ri); break;
  case X86::MOVZX64rr8:   LowerMOVxX32(OutMI, X86::MOVZX32rr8); break;
  case X86::MOVZX64rm8:   LowerMOVxX32(OutMI, X86::MOVZX32rm8); break;
  case X86::MOVZX64rr16:  LowerMOVxX32(OutMI, X86::MOVZX32rr16); break;
  case X86::MOVZX64rm16:  LowerMOVxX32(OutMI, X86::MOVZX32rm16); break;
  case X86::MOV16r0:
    OutMI.setOpcode(X86::MOV32r0);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::MOV64r0:
    OutMI.setOpcode(X86::MOV32r0);
    lower_subreg32(&OutMI, 0);
    break;
  case X86::SETB_C8r:  LowerSETB(OutMI, X86::SBB8rr); break;
  case X86::SETB_C16r: LowerSETB(OutMI, X86::SBB16rr); break;
  case X86::SETB_C32r: LowerSETB(OutMI, X86::SBB32rr); break;
  case X86::SETB_C64r: LowerSETB(OutMI, X86::SBB64rr); break;
  }
}



void X86AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  X86MCInstLower MCInstLowering(OutContext, Mang, *this);
  switch (MI->getOpcode()) {
  case TargetInstrInfo::DEBUG_VALUE: {
    // FIXME: if this is implemented for another target before it goes
    // away completely, the common part should be moved into AsmPrinter.
    if (!VerboseAsm)
      return;
    O << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
    unsigned NOps = MI->getNumOperands();
    // cast away const; DIetc do not take const operands for some reason.
    DIVariable V((MDNode*)(MI->getOperand(NOps-1).getMetadata()));
    O << V.getName();
    O << " <- ";
    if (NOps==3) {
      // Register or immediate value. Register 0 means undef.
      assert(MI->getOperand(0).getType()==MachineOperand::MO_Register ||
             MI->getOperand(0).getType()==MachineOperand::MO_Immediate ||
             MI->getOperand(0).getType()==MachineOperand::MO_FPImmediate);
      if (MI->getOperand(0).getType()==MachineOperand::MO_Register &&
          MI->getOperand(0).getReg()==0) {
        // Suppress offset in this case, it is not meaningful.
        O << "undef\n";
        return;
      } else if (MI->getOperand(0).getType()==MachineOperand::MO_FPImmediate) {
        // This is more naturally done in printOperand, but since the only use
        // of such an operand is in this comment and that is temporary (and it's
        // ugly), we prefer to keep this localized.
        // The include of Type.h may be removable when this code is.
        if (MI->getOperand(0).getFPImm()->getType()->isFloatTy() ||
            MI->getOperand(0).getFPImm()->getType()->isDoubleTy())
          MI->getOperand(0).print(O, &TM);
        else {
          // There is no good way to print long double.  Convert a copy to
          // double.  Ah well, it's only a comment.
          bool ignored;
          APFloat APF = APFloat(MI->getOperand(0).getFPImm()->getValueAPF());
          APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                      &ignored);
          O << "(long double) " << APF.convertToDouble();
        }
      } else
        printOperand(MI, 0);
    } else {
      // Frame address.  Currently handles register +- offset only.
      assert(MI->getOperand(0).getType()==MachineOperand::MO_Register);
      assert(MI->getOperand(3).getType()==MachineOperand::MO_Immediate);
      O << '['; printOperand(MI, 0); O << '+'; printOperand(MI, 3); O << ']';
    }
    O << "+";
    printOperand(MI, NOps-2);
    O << '\n';
    return;
  }
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
    OutStreamer.EmitInstruction(TmpInst);
    
    // Emit the label.
    OutStreamer.EmitLabel(PICBase);
    
    // popl $reg
    TmpInst.setOpcode(X86::POP32r);
    TmpInst.getOperand(0) = MCOperand::CreateReg(MI->getOperand(0).getReg());
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
      
  case X86::ADD32ri: {
    // Lower the MO_GOT_ABSOLUTE_ADDRESS form of ADD32ri.
    if (MI->getOperand(2).getTargetFlags() != X86II::MO_GOT_ABSOLUTE_ADDRESS)
      break;
    
    // Okay, we have something like:
    //  EAX = ADD32ri EAX, MO_GOT_ABSOLUTE_ADDRESS(@MYGLOBAL)
    
    // For this, we want to print something like:
    //   MYGLOBAL + (. - PICBASE)
    // However, we can't generate a ".", so just emit a new label here and refer
    // to it.  We know that this operand flag occurs at most once per function.
    const char *Prefix = MAI->getPrivateGlobalPrefix();
    MCSymbol *DotSym = OutContext.GetOrCreateSymbol(Twine(Prefix)+"picbaseref"+
                                                    Twine(getFunctionNumber()));
    OutStreamer.EmitLabel(DotSym);
    
    // Now that we have emitted the label, lower the complex operand expression.
    MCSymbol *OpSym = MCInstLowering.GetExternalSymbolSymbol(MI->getOperand(2));
    
    const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, OutContext);
    const MCExpr *PICBase =
      MCSymbolRefExpr::Create(MCInstLowering.GetPICBaseSymbol(), OutContext);
    DotExpr = MCBinaryExpr::CreateSub(DotExpr, PICBase, OutContext);
    
    DotExpr = MCBinaryExpr::CreateAdd(MCSymbolRefExpr::Create(OpSym,OutContext), 
                                      DotExpr, OutContext);
    
    MCInst TmpInst;
    TmpInst.setOpcode(X86::ADD32ri);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    TmpInst.addOperand(MCOperand::CreateExpr(DotExpr));
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  }
  
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  
  
  OutStreamer.EmitInstruction(TmpInst);
}

