//===-- X86ATTInstPrinter.cpp - AT&T assembly instruction printing --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file includes code for rendering MCInst instances as AT&T-style
// assembly.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/MC/MCInst.h"
#include "X86ATTAsmPrinter.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define NO_ASM_WRITER_BOILERPLATE
#include "X86GenAsmWriter.inc"
#undef MachineInstr

void X86ATTAsmPrinter::printSSECC(const MCInst *MI, unsigned Op) {
  switch (MI->getOperand(Op).getImm()) {
  default: assert(0 && "Invalid ssecc argument!");
  case 0: O << "eq"; break;
  case 1: O << "lt"; break;
  case 2: O << "le"; break;
  case 3: O << "unord"; break;
  case 4: O << "neq"; break;
  case 5: O << "nlt"; break;
  case 6: O << "nle"; break;
  case 7: O << "ord"; break;
  }
}


void X86ATTAsmPrinter::printPICLabel(const MCInst *MI, unsigned Op) {
  assert(0 &&
         "This is only used for MOVPC32r, should lower before asm printing!");
}


/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.  These print slightly differently, for
/// example, a $ is not emitted.
void X86ATTAsmPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo) {
  const MCOperand &Op = MI->getOperand(OpNo);
  
  if (Op.isImm())
    O << Op.getImm();
  else if (Op.isMBBLabel())
    // FIXME: Keep in sync with printBasicBlockLabel.  printBasicBlockLabel
    // should eventually call into this code, not the other way around.
    O << TAI->getPrivateGlobalPrefix() << "BB" << Op.getMBBLabelFunction()
      << '_' << Op.getMBBLabelBlock();
  else
    assert(0 && "Unknown pcrel immediate operand");
}


void X86ATTAsmPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    const char *Modifier, bool NotRIPRel) {
  assert(Modifier == 0 && "Modifiers should not be used");
  
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << '%';
    unsigned Reg = Op.getReg();
#if 0
    if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
      MVT VT = (strcmp(Modifier+6,"64") == 0) ?
      MVT::i64 : ((strcmp(Modifier+6, "32") == 0) ? MVT::i32 :
                  ((strcmp(Modifier+6,"16") == 0) ? MVT::i16 : MVT::i8));
      Reg = getX86SubSuperRegister(Reg, VT);
    }
#endif
    O << TRI->getAsmName(Reg);
    return;
  } else if (Op.isImm()) {
    //if (!Modifier || (strcmp(Modifier, "debug") && strcmp(Modifier, "mem") &&
    // strcmp(Modifier, "call")))
    O << '$';
    O << Op.getImm();
    return;
  } else if (Op.isMBBLabel()) {
    assert(0 && "labels should only be used as pc-relative values");
    // FIXME: Keep in sync with printBasicBlockLabel.  printBasicBlockLabel
    // should eventually call into this code, not the other way around.
    
    O << TAI->getPrivateGlobalPrefix() << "BB" << Op.getMBBLabelFunction()
      << '_' << Op.getMBBLabelBlock();
    
    // FIXME: with verbose asm print llvm bb name, add to operand.
    return;
  }
  
  O << "<<UNKNOWN OPERAND KIND>>";
  
  
#if 0
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
    case MachineOperand::MO_Register: {
      assert(TargetRegisterInfo::isPhysicalRegister(MO.getReg()) &&
             "Virtual registers should not make it this far!");
      O << '%';
      unsigned Reg = MO.getReg();
      if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
        MVT VT = (strcmp(Modifier+6,"64") == 0) ?
        MVT::i64 : ((strcmp(Modifier+6, "32") == 0) ? MVT::i32 :
                    ((strcmp(Modifier+6,"16") == 0) ? MVT::i16 : MVT::i8));
        Reg = getX86SubSuperRegister(Reg, VT);
      }
      O << TRI->getAsmName(Reg);
      return;
    }
      
    case MachineOperand::MO_Immediate:
      if (!Modifier || (strcmp(Modifier, "debug") &&
                        strcmp(Modifier, "mem") &&
                        strcmp(Modifier, "call")))
        O << '$';
      O << MO.getImm();
      return;
    case MachineOperand::MO_JumpTableIndex: {
      bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
      if (!isMemOp) O << '$';
      O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << '_'
      << MO.getIndex();
      
      if (TM.getRelocationModel() == Reloc::PIC_) {
        if (Subtarget->isPICStyleStub())
          O << "-\"" << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
          << "$pb\"";
        else if (Subtarget->isPICStyleGOT())
          O << "@GOTOFF";
      }
      
      if (isMemOp && Subtarget->isPICStyleRIPRel() && !NotRIPRel)
        O << "(%rip)";
      return;
    }
    case MachineOperand::MO_ConstantPoolIndex: {
      bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
      if (!isMemOp) O << '$';
      O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << MO.getIndex();
      
      if (TM.getRelocationModel() == Reloc::PIC_) {
        if (Subtarget->isPICStyleStub())
          O << "-\"" << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
          << "$pb\"";
        else if (Subtarget->isPICStyleGOT())
          O << "@GOTOFF";
      }
      
      printOffset(MO.getOffset());
      
      if (isMemOp && Subtarget->isPICStyleRIPRel() && !NotRIPRel)
        O << "(%rip)";
      return;
    }
    case MachineOperand::MO_GlobalAddress: {
      bool isCallOp = Modifier && !strcmp(Modifier, "call");
      bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
      bool needCloseParen = false;
      
      const GlobalValue *GV = MO.getGlobal();
      const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
      if (!GVar) {
        // If GV is an alias then use the aliasee for determining
        // thread-localness.
        if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
          GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal(false));
      }
      
      bool isThreadLocal = GVar && GVar->isThreadLocal();
      
      std::string Name = Mang->getValueName(GV);
      decorateName(Name, GV);
      
      if (!isMemOp && !isCallOp)
        O << '$';
      else if (Name[0] == '$') {
        // The name begins with a dollar-sign. In order to avoid having it look
        // like an integer immediate to the assembler, enclose it in parens.
        O << '(';
        needCloseParen = true;
      }
      
      if (shouldPrintStub(TM, Subtarget)) {
        // Link-once, declaration, or Weakly-linked global variables need
        // non-lazily-resolved stubs
        if (GV->isDeclaration() || GV->isWeakForLinker()) {
          // Dynamically-resolved functions need a stub for the function.
          if (isCallOp && isa<Function>(GV)) {
            // Function stubs are no longer needed for Mac OS X 10.5 and up.
            if (Subtarget->isTargetDarwin() && Subtarget->getDarwinVers() >= 9) {
              O << Name;
            } else {
              FnStubs.insert(Name);
              printSuffixedName(Name, "$stub");
            }
          } else if (GV->hasHiddenVisibility()) {
            if (!GV->isDeclaration() && !GV->hasCommonLinkage())
              // Definition is not definitely in the current translation unit.
              O << Name;
            else {
              HiddenGVStubs.insert(Name);
              printSuffixedName(Name, "$non_lazy_ptr");
            }
          } else {
            GVStubs.insert(Name);
            printSuffixedName(Name, "$non_lazy_ptr");
          }
        } else {
          if (GV->hasDLLImportLinkage())
            O << "__imp_";
          O << Name;
        }
        
        if (!isCallOp && TM.getRelocationModel() == Reloc::PIC_)
          O << '-' << getPICLabelString(getFunctionNumber(), TAI, Subtarget);
      } else {
        if (GV->hasDLLImportLinkage()) {
          O << "__imp_";
        }
        O << Name;
        
        if (isCallOp) {
          if (shouldPrintPLT(TM, Subtarget)) {
            // Assemble call via PLT for externally visible symbols
            if (!GV->hasHiddenVisibility() && !GV->hasProtectedVisibility() &&
                !GV->hasLocalLinkage())
              O << "@PLT";
          }
          if (Subtarget->isTargetCygMing() && GV->isDeclaration())
            // Save function name for later type emission
            FnStubs.insert(Name);
        }
      }
      
      if (GV->hasExternalWeakLinkage())
        ExtWeakSymbols.insert(GV);
      
      printOffset(MO.getOffset());
      
      if (isThreadLocal) {
        TLSModel::Model model = getTLSModel(GVar, TM.getRelocationModel());
        switch (model) {
          case TLSModel::GeneralDynamic:
            O << "@TLSGD";
            break;
          case TLSModel::LocalDynamic:
            // O << "@TLSLD"; // local dynamic not implemented
            O << "@TLSGD";
            break;
          case TLSModel::InitialExec:
            if (Subtarget->is64Bit()) {
              assert (!NotRIPRel);
              O << "@GOTTPOFF(%rip)";
            } else {
              O << "@INDNTPOFF";
            }
            break;
          case TLSModel::LocalExec:
            if (Subtarget->is64Bit())
              O << "@TPOFF";
            else
              O << "@NTPOFF";
            break;
          default:
            assert (0 && "Unknown TLS model");
        }
      } else if (isMemOp) {
        if (shouldPrintGOT(TM, Subtarget)) {
          if (Subtarget->GVRequiresExtraLoad(GV, TM, false))
            O << "@GOT";
          else
            O << "@GOTOFF";
        } else if (Subtarget->isPICStyleRIPRel() && !NotRIPRel) {
          if (TM.getRelocationModel() != Reloc::Static) {
            if (Subtarget->GVRequiresExtraLoad(GV, TM, false))
              O << "@GOTPCREL";
            
            if (needCloseParen) {
              needCloseParen = false;
              O << ')';
            }
          }
          
          // Use rip when possible to reduce code size, except when
          // index or base register are also part of the address. e.g.
          // foo(%rip)(%rcx,%rax,4) is not legal
          O << "(%rip)";
        }
      }
      
      if (needCloseParen)
        O << ')';
      
      return;
    }
    case MachineOperand::MO_ExternalSymbol: {
      bool isCallOp = Modifier && !strcmp(Modifier, "call");
      bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
      bool needCloseParen = false;
      std::string Name(TAI->getGlobalPrefix());
      Name += MO.getSymbolName();
      // Print function stub suffix unless it's Mac OS X 10.5 and up.
      if (isCallOp && shouldPrintStub(TM, Subtarget) && 
          !(Subtarget->isTargetDarwin() && Subtarget->getDarwinVers() >= 9)) {
        FnStubs.insert(Name);
        printSuffixedName(Name, "$stub");
        return;
      }
      if (!isMemOp && !isCallOp)
        O << '$';
      else if (Name[0] == '$') {
        // The name begins with a dollar-sign. In order to avoid having it look
        // like an integer immediate to the assembler, enclose it in parens.
        O << '(';
        needCloseParen = true;
      }
      
      O << Name;
      
      if (shouldPrintPLT(TM, Subtarget)) {
        std::string GOTName(TAI->getGlobalPrefix());
        GOTName+="_GLOBAL_OFFSET_TABLE_";
        if (Name == GOTName)
          // HACK! Emit extra offset to PC during printing GOT offset to
          // compensate for the size of popl instruction. The resulting code
          // should look like:
          //   call .piclabel
          // piclabel:
          //   popl %some_register
          //   addl $_GLOBAL_ADDRESS_TABLE_ + [.-piclabel], %some_register
          O << " + [.-"
          << getPICLabelString(getFunctionNumber(), TAI, Subtarget) << ']';
        
        if (isCallOp)
          O << "@PLT";
      }
      
      if (needCloseParen)
        O << ')';
      
      if (!isCallOp && Subtarget->isPICStyleRIPRel())
        O << "(%rip)";
      
      return;
    }
    default:
      O << "<unknown operand type>"; return;
  }
#endif
}

void X86ATTAsmPrinter::printLeaMemReference(const MCInst *MI, unsigned Op) {
  bool NotRIPRel = false;

  const MCOperand &BaseReg  = MI->getOperand(Op);
  const MCOperand &IndexReg = MI->getOperand(Op+2);
  const MCOperand &DispSpec = MI->getOperand(Op+3);
  
  NotRIPRel |= IndexReg.getReg() || BaseReg.getReg();
  if (DispSpec.isImm()) {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  } else {
    abort();
    //assert(DispSpec.isGlobal() || DispSpec.isCPI() ||
    //       DispSpec.isJTI() || DispSpec.isSymbol());
    //printOperand(MI, Op+3, "mem", NotRIPRel);
  }
  
  if (IndexReg.getReg() || BaseReg.getReg()) {
    // There are cases where we can end up with ESP/RSP in the indexreg slot.
    // If this happens, swap the base/index register to support assemblers that
    // don't work when the index is *SP.
    // FIXME: REMOVE THIS.
    assert(IndexReg.getReg() != X86::ESP && IndexReg.getReg() != X86::RSP);
    
    O << '(';
    if (BaseReg.getReg())
      printOperand(MI, Op);
    
    if (IndexReg.getReg()) {
      O << ',';
      printOperand(MI, Op+2);
      unsigned ScaleVal = MI->getOperand(Op+1).getImm();
      if (ScaleVal != 1)
        O << ',' << ScaleVal;
    }
    O << ')';
  }
}

void X86ATTAsmPrinter::printMemReference(const MCInst *MI, unsigned Op) {
  const MCOperand &Segment = MI->getOperand(Op+4);
  if (Segment.getReg()) {
    printOperand(MI, Op+4);
    O << ':';
  }
  printLeaMemReference(MI, Op);
}
