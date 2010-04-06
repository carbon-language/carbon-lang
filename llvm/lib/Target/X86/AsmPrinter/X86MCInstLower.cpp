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
#include "X86COFFMachineModuleInfo.h"
#include "X86MCAsmInfo.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
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

/// GetSymbolFromOperand - Lower an MO_GlobalAddress or MO_ExternalSymbol
/// operand to an MCSymbol.
MCSymbol *X86MCInstLower::
GetSymbolFromOperand(const MachineOperand &MO) const {
  assert((MO.isGlobal() || MO.isSymbol()) && "Isn't a symbol reference");

  SmallString<128> Name;
  
  if (!MO.isGlobal()) {
    assert(MO.isSymbol());
    Name += AsmPrinter.MAI->getGlobalPrefix();
    Name += MO.getSymbolName();
  } else {    
    const GlobalValue *GV = MO.getGlobal();
    bool isImplicitlyPrivate = false;
    if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB ||
        MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
        MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE ||
        MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
      isImplicitlyPrivate = true;
    
    Mang->getNameWithPrefix(Name, GV, isImplicitlyPrivate);
  }

  // If the target flags on the operand changes the name of the symbol, do that
  // before we return the symbol.
  switch (MO.getTargetFlags()) {
  default: break;
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

    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI().getGVStubEntry(Sym);
    if (StubSym.getPointer() == 0) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(AsmPrinter.Mang->getSymbol(MO.getGlobal()),
                    !MO.getGlobal()->hasInternalLinkage());
    }
    return Sym;
  }
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI().getHiddenGVStubEntry(Sym);
    if (StubSym.getPointer() == 0) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(AsmPrinter.Mang->getSymbol(MO.getGlobal()),
                    !MO.getGlobal()->hasInternalLinkage());
    }
    return Sym;
  }
  case X86II::MO_DARWIN_STUB: {
    Name += "$stub";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI().getFnStubEntry(Sym);
    if (StubSym.getPointer())
      return Sym;
    
    if (MO.isGlobal()) {
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(AsmPrinter.Mang->getSymbol(MO.getGlobal()),
                    !MO.getGlobal()->hasInternalLinkage());
    } else {
      Name.erase(Name.end()-5, Name.end());
      StubSym =
        MachineModuleInfoImpl::
        StubValueTy(Ctx.GetOrCreateSymbol(Name.str()), false);
    }
    return Sym;
  }
  }

  return Ctx.GetOrCreateSymbol(Name.str());
}

MCOperand X86MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                             MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  const MCExpr *Expr = 0;
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;
  
  switch (MO.getTargetFlags()) {
  default: llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
  // These affect the name of the symbol, not any suffix.
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DLLIMPORT:
  case X86II::MO_DARWIN_STUB:
    break;
      
  case X86II::MO_TLSGD:     RefKind = MCSymbolRefExpr::VK_TLSGD; break;
  case X86II::MO_GOTTPOFF:  RefKind = MCSymbolRefExpr::VK_GOTTPOFF; break;
  case X86II::MO_INDNTPOFF: RefKind = MCSymbolRefExpr::VK_INDNTPOFF; break;
  case X86II::MO_TPOFF:     RefKind = MCSymbolRefExpr::VK_TPOFF; break;
  case X86II::MO_NTPOFF:    RefKind = MCSymbolRefExpr::VK_NTPOFF; break;
  case X86II::MO_GOTPCREL:  RefKind = MCSymbolRefExpr::VK_GOTPCREL; break;
  case X86II::MO_GOT:       RefKind = MCSymbolRefExpr::VK_GOT; break;
  case X86II::MO_GOTOFF:    RefKind = MCSymbolRefExpr::VK_GOTOFF; break;
  case X86II::MO_PLT:       RefKind = MCSymbolRefExpr::VK_PLT; break;
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    Expr = MCSymbolRefExpr::Create(Sym, Ctx);
    // Subtract the pic base.
    Expr = MCBinaryExpr::CreateSub(Expr, 
                               MCSymbolRefExpr::Create(GetPICBaseSymbol(), Ctx),
                                   Ctx);
    break;
  }
  
  if (Expr == 0)
    Expr = MCSymbolRefExpr::Create(Sym, RefKind, Ctx);
  
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

/// LowerSubReg32_Op0 - Things like MOVZX16rr8 -> MOVZX32rr8.
static void LowerSubReg32_Op0(MCInst &OutMI, unsigned NewOpc) {
  OutMI.setOpcode(NewOpc);
  lower_subreg32(&OutMI, 0);
}
/// LowerUnaryToTwoAddr - R = setb   -> R = sbb R, R
static void LowerUnaryToTwoAddr(MCInst &OutMI, unsigned NewOpc) {
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
                       MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_GlobalAddress:
      MCOp = LowerSymbolOperand(MO, GetSymbolFromOperand(MO));
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerSymbolOperand(MO, GetSymbolFromOperand(MO));
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = LowerSymbolOperand(MO, AsmPrinter.GetJTISymbol(MO.getIndex()));
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = LowerSymbolOperand(MO, AsmPrinter.GetCPISymbol(MO.getIndex()));
      break;
    case MachineOperand::MO_BlockAddress:
      MCOp = LowerSymbolOperand(MO,
                        AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress()));
      break;
    }
    
    OutMI.addOperand(MCOp);
  }
  
  // Handle a few special cases to eliminate operand modifiers.
  switch (OutMI.getOpcode()) {
  case X86::LEA64_32r: // Handle 'subreg rewriting' for the lea64_32mem operand.
    lower_lea64_32mem(&OutMI, 1);
    break;
  case X86::MOVZX16rr8:   LowerSubReg32_Op0(OutMI, X86::MOVZX32rr8); break;
  case X86::MOVZX16rm8:   LowerSubReg32_Op0(OutMI, X86::MOVZX32rm8); break;
  case X86::MOVSX16rr8:   LowerSubReg32_Op0(OutMI, X86::MOVSX32rr8); break;
  case X86::MOVSX16rm8:   LowerSubReg32_Op0(OutMI, X86::MOVSX32rm8); break;
  case X86::MOVZX64rr32:  LowerSubReg32_Op0(OutMI, X86::MOV32rr); break;
  case X86::MOVZX64rm32:  LowerSubReg32_Op0(OutMI, X86::MOV32rm); break;
  case X86::MOV64ri64i32: LowerSubReg32_Op0(OutMI, X86::MOV32ri); break;
  case X86::MOVZX64rr8:   LowerSubReg32_Op0(OutMI, X86::MOVZX32rr8); break;
  case X86::MOVZX64rm8:   LowerSubReg32_Op0(OutMI, X86::MOVZX32rm8); break;
  case X86::MOVZX64rr16:  LowerSubReg32_Op0(OutMI, X86::MOVZX32rr16); break;
  case X86::MOVZX64rm16:  LowerSubReg32_Op0(OutMI, X86::MOVZX32rm16); break;
  case X86::SETB_C8r:     LowerUnaryToTwoAddr(OutMI, X86::SBB8rr); break;
  case X86::SETB_C16r:    LowerUnaryToTwoAddr(OutMI, X86::SBB16rr); break;
  case X86::SETB_C32r:    LowerUnaryToTwoAddr(OutMI, X86::SBB32rr); break;
  case X86::SETB_C64r:    LowerUnaryToTwoAddr(OutMI, X86::SBB64rr); break;
  case X86::MOV8r0:       LowerUnaryToTwoAddr(OutMI, X86::XOR8rr); break;
  case X86::MOV32r0:      LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); break;
  case X86::MMX_V_SET0:   LowerUnaryToTwoAddr(OutMI, X86::MMX_PXORrr); break;
  case X86::MMX_V_SETALLONES:
    LowerUnaryToTwoAddr(OutMI, X86::MMX_PCMPEQDrr); break;
  case X86::FsFLD0SS:     LowerUnaryToTwoAddr(OutMI, X86::PXORrr); break;
  case X86::FsFLD0SD:     LowerUnaryToTwoAddr(OutMI, X86::PXORrr); break;
  case X86::V_SET0PS:     LowerUnaryToTwoAddr(OutMI, X86::XORPSrr); break;
  case X86::V_SET0PD:     LowerUnaryToTwoAddr(OutMI, X86::XORPDrr); break;
  case X86::V_SET0PI:     LowerUnaryToTwoAddr(OutMI, X86::PXORrr); break;
  case X86::V_SETALLONES: LowerUnaryToTwoAddr(OutMI, X86::PCMPEQDrr); break;

  case X86::MOV16r0:
    LowerSubReg32_Op0(OutMI, X86::MOV32r0);   // MOV16r0 -> MOV32r0
    LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); // MOV32r0 -> XOR32rr
    break;
  case X86::MOV64r0:
    LowerSubReg32_Op0(OutMI, X86::MOV32r0);   // MOV64r0 -> MOV32r0
    LowerUnaryToTwoAddr(OutMI, X86::XOR32rr); // MOV32r0 -> XOR32rr
    break;
      
      
  // The assembler backend wants to see branches in their small form and relax
  // them to their large form.  The JIT can only handle the large form because
  // it does not do relaxation.  For now, translate the large form to the
  // small one here.
  case X86::JMP_4: OutMI.setOpcode(X86::JMP_1); break;
  case X86::JO_4:  OutMI.setOpcode(X86::JO_1); break;
  case X86::JNO_4: OutMI.setOpcode(X86::JNO_1); break;
  case X86::JB_4:  OutMI.setOpcode(X86::JB_1); break;
  case X86::JAE_4: OutMI.setOpcode(X86::JAE_1); break;
  case X86::JE_4:  OutMI.setOpcode(X86::JE_1); break;
  case X86::JNE_4: OutMI.setOpcode(X86::JNE_1); break;
  case X86::JBE_4: OutMI.setOpcode(X86::JBE_1); break;
  case X86::JA_4:  OutMI.setOpcode(X86::JA_1); break;
  case X86::JS_4:  OutMI.setOpcode(X86::JS_1); break;
  case X86::JNS_4: OutMI.setOpcode(X86::JNS_1); break;
  case X86::JP_4:  OutMI.setOpcode(X86::JP_1); break;
  case X86::JNP_4: OutMI.setOpcode(X86::JNP_1); break;
  case X86::JL_4:  OutMI.setOpcode(X86::JL_1); break;
  case X86::JGE_4: OutMI.setOpcode(X86::JGE_1); break;
  case X86::JLE_4: OutMI.setOpcode(X86::JLE_1); break;
  case X86::JG_4:  OutMI.setOpcode(X86::JG_1); break;
  }
}

void X86AsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                           raw_ostream &O) {
  // FIXME: if this is implemented for another target before it goes
  // away completely, the common part should be moved into AsmPrinter.
  O << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
  unsigned NOps = MI->getNumOperands();
  // cast away const; DIetc do not take const operands for some reason.
  DIVariable V((MDNode*)(MI->getOperand(NOps-1).getMetadata()));
  O << V.getName();
  O << " <- ";
  if (NOps==3) {
    // Register or immediate value. Register 0 means undef.
    assert(MI->getOperand(0).isReg() ||
           MI->getOperand(0).isImm() ||
           MI->getOperand(0).isFPImm());
    if (MI->getOperand(0).isReg() && MI->getOperand(0).getReg() == 0) {
      // Suppress offset in this case, it is not meaningful.
      O << "undef";
      OutStreamer.AddBlankLine();
      return;
    }
    
    if (MI->getOperand(0).isFPImm()) {
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
      printOperand(MI, 0, O);
  } else {
    if (MI->getOperand(0).isReg() && MI->getOperand(0).getReg() == 0) {
      // Suppress offset in this case, it is not meaningful.
      O << "undef";
      OutStreamer.AddBlankLine();
      return;
    }
    // Frame address.  Currently handles register +- offset only.
    assert(MI->getOperand(0).isReg() && MI->getOperand(3).isImm());
    O << '['; printOperand(MI, 0, O); O << '+'; printOperand(MI, 3, O);
    O << ']';
  }
  O << "+";
  printOperand(MI, NOps-2, O);
}


void X86AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  X86MCInstLower MCInstLowering(OutContext, Mang, *this);
  switch (MI->getOpcode()) {
  case TargetOpcode::DBG_VALUE:
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      std::string TmpStr;
      raw_string_ostream OS(TmpStr);
      PrintDebugValueComment(MI, OS);
      OutStreamer.EmitRawText(StringRef(OS.str()));
    }
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
    // to it.
    MCSymbol *DotSym = OutContext.CreateTempSymbol();
    OutStreamer.EmitLabel(DotSym);
    
    // Now that we have emitted the label, lower the complex operand expression.
    MCSymbol *OpSym = MCInstLowering.GetSymbolFromOperand(MI->getOperand(2));
    
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

