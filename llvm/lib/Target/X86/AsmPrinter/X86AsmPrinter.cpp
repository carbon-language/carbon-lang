//===-- X86AsmPrinter.cpp - Convert X86 LLVM code to AT&T assembly --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to AT&T format assembly
// language. This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "X86AsmPrinter.h"
#include "X86ATTInstPrinter.h"
#include "X86IntelInstPrinter.h"
#include "X86MCInstLower.h"
#include "X86.h"
#include "X86COFF.h"
#include "X86COFFMachineModuleInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

//===----------------------------------------------------------------------===//
// Primitive Helper Functions.
//===----------------------------------------------------------------------===//

void X86AsmPrinter::printMCInst(const MCInst *MI) {
  if (MAI->getAssemblerDialect() == 0)
    X86ATTInstPrinter(O, *MAI).printInstruction(MI);
  else
    X86IntelInstPrinter(O, *MAI).printInstruction(MI);
}

void X86AsmPrinter::PrintPICBaseSymbol() const {
  // FIXME: Gross const cast hack.
  X86AsmPrinter *AP = const_cast<X86AsmPrinter*>(this);
  O << *X86MCInstLower(OutContext, 0, *AP).GetPICBaseSymbol();
}

void X86AsmPrinter::emitFunctionHeader(const MachineFunction &MF) {
  unsigned FnAlign = MF.getAlignment();
  const Function *F = MF.getFunction();

  if (Subtarget->isTargetCygMing()) {
    X86COFFMachineModuleInfo &COFFMMI = 
      MMI->getObjFileInfo<X86COFFMachineModuleInfo>();
    COFFMMI.DecorateCygMingName(CurrentFnSym, OutContext, F,
                                *TM.getTargetData());
  }

  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang, TM));
  EmitAlignment(FnAlign, F);

  switch (F->getLinkage()) {
  default: llvm_unreachable("Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
  case Function::PrivateLinkage:
    break;
  case Function::DLLExportLinkage:
  case Function::ExternalLinkage:
    OutStreamer.EmitSymbolAttribute(CurrentFnSym, MCStreamer::Global);
    break;
  case Function::LinkerPrivateLinkage:
  case Function::LinkOnceAnyLinkage:
  case Function::LinkOnceODRLinkage:
  case Function::WeakAnyLinkage:
  case Function::WeakODRLinkage:
    if (Subtarget->isTargetDarwin()) {
      OutStreamer.EmitSymbolAttribute(CurrentFnSym, MCStreamer::Global);
      O << MAI->getWeakDefDirective() << *CurrentFnSym << '\n';
    } else if (Subtarget->isTargetCygMing()) {
      OutStreamer.EmitSymbolAttribute(CurrentFnSym, MCStreamer::Global);
      O << "\t.linkonce discard\n";
    } else {
      O << "\t.weak\t" << *CurrentFnSym << '\n';
    }
    break;
  }

  printVisibility(CurrentFnSym, F->getVisibility());

  if (Subtarget->isTargetELF()) {
    O << "\t.type\t" << *CurrentFnSym << ",@function\n";
  } else if (Subtarget->isTargetCygMing()) {
    O << "\t.def\t " << *CurrentFnSym;
    O << ";\t.scl\t" <<
      (F->hasInternalLinkage() ? COFF::C_STAT : COFF::C_EXT)
      << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
      << ";\t.endef\n";
  }

  O << *CurrentFnSym << ':';
  if (VerboseAsm) {
    O.PadToColumn(MAI->getCommentColumn());
    O << MAI->getCommentString() << ' ';
    WriteAsOperand(O, F, /*PrintType=*/false, F->getParent());
  }
  O << '\n';

  // Add some workaround for linkonce linkage on Cygwin\MinGW
  if (Subtarget->isTargetCygMing() &&
      (F->hasLinkOnceLinkage() || F->hasWeakLinkage()))
    O << "Lllvm$workaround$fake$stub$" << *CurrentFnSym << ":\n";
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  this->MF = &MF;
  CallingConv::ID CC = F->getCallingConv();

  SetupMachineFunction(MF);
  O << "\n\n";

  if (Subtarget->isTargetCOFF()) {
    X86COFFMachineModuleInfo &COFFMMI = 
    MMI->getObjFileInfo<X86COFFMachineModuleInfo>();

    // Populate function information map.  Don't want to populate
    // non-stdcall or non-fastcall functions' information right now.
    if (CC == CallingConv::X86_StdCall || CC == CallingConv::X86_FastCall)
      COFFMMI.AddFunctionInfo(F, *MF.getInfo<X86MachineFunctionInfo>());
  }

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print the 'header' of function
  emitFunctionHeader(MF);

  // Emit pre-function debug and/or EH information.
  if (MAI->doesSupportDebugInformation() || MAI->doesSupportExceptionHandling())
    DW->BeginFunction(&MF);

  // Print out code for the function.
  bool hasAnyRealCode = false;
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    EmitBasicBlockStart(I);
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      // Print the assembly for the instruction.
      if (!II->isLabel())
        hasAnyRealCode = true;
      printMachineInstruction(II);
    }
  }

  if (Subtarget->isTargetDarwin() && !hasAnyRealCode) {
    // If the function is empty, then we need to emit *something*. Otherwise,
    // the function's label might be associated with something that it wasn't
    // meant to be associated with. We emit a noop in this situation.
    // We are assuming inline asms are code.
    O << "\tnop\n";
  }

  if (MAI->hasDotTypeDotSizeDirective())
    O << "\t.size\t" << *CurrentFnSym << ", .-" << *CurrentFnSym << '\n';

  // Emit post-function debug information.
  if (MAI->doesSupportDebugInformation() || MAI->doesSupportExceptionHandling())
    DW->EndFunction(&MF);

  // Print out jump tables referenced by the function.
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);

  // We didn't modify anything.
  return false;
}

/// printSymbolOperand - Print a raw symbol reference operand.  This handles
/// jump tables, constant pools, global address and external symbols, all of
/// which print to a label with various suffixes for relocation types etc.
void X86AsmPrinter::printSymbolOperand(const MachineOperand &MO) {
  switch (MO.getType()) {
  default: llvm_unreachable("unknown symbol type!");
  case MachineOperand::MO_JumpTableIndex:
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << '_'
      << MO.getIndex();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << MO.getIndex();
    printOffset(MO.getOffset());
    break;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    
    MCSymbol *GVSym;
    if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB)
      GVSym = GetSymbolWithGlobalValueBase(GV, "$stub");
    else if (MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
             MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE ||
             MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE)
      GVSym = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
    else
      GVSym = GetGlobalValueSymbol(GV);

    if (Subtarget->isTargetCygMing()) {
      X86COFFMachineModuleInfo &COFFMMI =
        MMI->getObjFileInfo<X86COFFMachineModuleInfo>();
      COFFMMI.DecorateCygMingName(GVSym, OutContext, GV, *TM.getTargetData());
    }
    
    // Handle dllimport linkage.
    if (MO.getTargetFlags() == X86II::MO_DLLIMPORT)
      GVSym = OutContext.GetOrCreateSymbol(Twine("__imp_") + GVSym->getName());
    
    if (MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
        MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE) {
      MCSymbol *Sym = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
      
      const MCSymbol *&StubSym = 
        MMI->getObjFileInfo<MachineModuleInfoMachO>().getGVStubEntry(Sym);
      if (StubSym == 0)
        StubSym = GetGlobalValueSymbol(GV);
      
    } else if (MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE){
      MCSymbol *Sym = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
      const MCSymbol *&StubSym =
        MMI->getObjFileInfo<MachineModuleInfoMachO>().getHiddenGVStubEntry(Sym);
      if (StubSym == 0)
        StubSym = GetGlobalValueSymbol(GV);
    } else if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
      MCSymbol *Sym = GetSymbolWithGlobalValueBase(GV, "$stub");
      const MCSymbol *&StubSym =
        MMI->getObjFileInfo<MachineModuleInfoMachO>().getFnStubEntry(Sym);
      if (StubSym == 0)
        StubSym = GetGlobalValueSymbol(GV);
    }
    
    // If the name begins with a dollar-sign, enclose it in parens.  We do this
    // to avoid having it look like an integer immediate to the assembler.
    if (GVSym->getName()[0] != '$')
      O << *GVSym;
    else
      O << '(' << *GVSym << ')';
    printOffset(MO.getOffset());
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    const MCSymbol *SymToPrint;
    if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
      SmallString<128> TempNameStr;
      TempNameStr += StringRef(MO.getSymbolName());
      TempNameStr += StringRef("$stub");
      
      const MCSymbol *Sym = GetExternalSymbolSymbol(TempNameStr.str());
      const MCSymbol *&StubSym =
        MMI->getObjFileInfo<MachineModuleInfoMachO>().getFnStubEntry(Sym);
      if (StubSym == 0) {
        TempNameStr.erase(TempNameStr.end()-5, TempNameStr.end());
        StubSym = OutContext.GetOrCreateSymbol(TempNameStr.str());
      }
      SymToPrint = StubSym;
    } else {
      SymToPrint = GetExternalSymbolSymbol(MO.getSymbolName());
    }
    
    // If the name begins with a dollar-sign, enclose it in parens.  We do this
    // to avoid having it look like an integer immediate to the assembler.
    if (SymToPrint->getName()[0] != '$') 
      O << *SymToPrint;
    else
      O << '(' << *SymToPrint << '(';
    break;
  }
  }
  
  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
    break;
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DLLIMPORT:
  case X86II::MO_DARWIN_STUB:
    // These affect the name of the symbol, not any suffix.
    break;
  case X86II::MO_GOT_ABSOLUTE_ADDRESS:
    O << " + [.-";
    PrintPICBaseSymbol();
    O << ']';
    break;      
  case X86II::MO_PIC_BASE_OFFSET:
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE:
    O << '-';
    PrintPICBaseSymbol();
    break;
  case X86II::MO_TLSGD:     O << "@TLSGD";     break;
  case X86II::MO_GOTTPOFF:  O << "@GOTTPOFF";  break;
  case X86II::MO_INDNTPOFF: O << "@INDNTPOFF"; break;
  case X86II::MO_TPOFF:     O << "@TPOFF";     break;
  case X86II::MO_NTPOFF:    O << "@NTPOFF";    break;
  case X86II::MO_GOTPCREL:  O << "@GOTPCREL";  break;
  case X86II::MO_GOT:       O << "@GOT";       break;
  case X86II::MO_GOTOFF:    O << "@GOTOFF";    break;
  case X86II::MO_PLT:       O << "@PLT";       break;
  }
}

/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.  These print slightly differently, for
/// example, a $ is not emitted.
void X86AsmPrinter::print_pcrel_imm(const MachineInstr *MI, unsigned OpNo) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  default: llvm_unreachable("Unknown pcrel immediate operand");
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    O << *GetMBBSymbol(MO.getMBB()->getNumber());
    return;
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ExternalSymbol:
    printSymbolOperand(MO);
    return;
  }
}


void X86AsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                 const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  default: llvm_unreachable("unknown operand type!");
  case MachineOperand::MO_Register: {
    O << '%';
    unsigned Reg = MO.getReg();
    if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
      EVT VT = (strcmp(Modifier+6,"64") == 0) ?
        MVT::i64 : ((strcmp(Modifier+6, "32") == 0) ? MVT::i32 :
                    ((strcmp(Modifier+6,"16") == 0) ? MVT::i16 : MVT::i8));
      Reg = getX86SubSuperRegister(Reg, VT);
    }
    O << X86ATTInstPrinter::getRegisterName(Reg);
    return;
  }

  case MachineOperand::MO_Immediate:
    O << '$' << MO.getImm();
    return;

  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_GlobalAddress: 
  case MachineOperand::MO_ExternalSymbol: {
    O << '$';
    printSymbolOperand(MO);
    break;
  }
  }
}

void X86AsmPrinter::printSSECC(const MachineInstr *MI, unsigned Op) {
  unsigned char value = MI->getOperand(Op).getImm();
  assert(value <= 7 && "Invalid ssecc argument!");
  switch (value) {
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

void X86AsmPrinter::printLeaMemReference(const MachineInstr *MI, unsigned Op,
                                         const char *Modifier) {
  const MachineOperand &BaseReg  = MI->getOperand(Op);
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  // If we really don't want to print out (rip), don't.
  bool HasBaseReg = BaseReg.getReg() != 0;
  if (HasBaseReg && Modifier && !strcmp(Modifier, "no-rip") &&
      BaseReg.getReg() == X86::RIP)
    HasBaseReg = false;
  
  // HasParenPart - True if we will print out the () part of the mem ref.
  bool HasParenPart = IndexReg.getReg() || HasBaseReg;
  
  if (DispSpec.isImm()) {
    int DispVal = DispSpec.getImm();
    if (DispVal || !HasParenPart)
      O << DispVal;
  } else {
    assert(DispSpec.isGlobal() || DispSpec.isCPI() ||
           DispSpec.isJTI() || DispSpec.isSymbol());
    printSymbolOperand(MI->getOperand(Op+3));
  }

  if (HasParenPart) {
    assert(IndexReg.getReg() != X86::ESP &&
           "X86 doesn't allow scaling by ESP");

    O << '(';
    if (HasBaseReg)
      printOperand(MI, Op, Modifier);

    if (IndexReg.getReg()) {
      O << ',';
      printOperand(MI, Op+2, Modifier);
      unsigned ScaleVal = MI->getOperand(Op+1).getImm();
      if (ScaleVal != 1)
        O << ',' << ScaleVal;
    }
    O << ')';
  }
}

void X86AsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op,
                                      const char *Modifier) {
  assert(isMem(MI, Op) && "Invalid memory reference!");
  const MachineOperand &Segment = MI->getOperand(Op+4);
  if (Segment.getReg()) {
    printOperand(MI, Op+4, Modifier);
    O << ':';
  }
  printLeaMemReference(MI, Op, Modifier);
}

void X86AsmPrinter::printPICJumpTableSetLabel(unsigned uid,
                                           const MachineBasicBlock *MBB) const {
  if (!MAI->getSetDirective())
    return;

  // We don't need .set machinery if we have GOT-style relocations
  if (Subtarget->isPICStyleGOT())
    return;

  O << MAI->getSetDirective() << ' ' << MAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << "_set_" << MBB->getNumber() << ',';
  
  O << *GetMBBSymbol(MBB->getNumber());
  
  if (Subtarget->isPICStyleRIPRel())
    O << '-' << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << uid << '\n';
  else {
    O << '-';
    PrintPICBaseSymbol();
    O << '\n';
  }
}


void X86AsmPrinter::printPICLabel(const MachineInstr *MI, unsigned Op) {
  PrintPICBaseSymbol();
  O << '\n';
  PrintPICBaseSymbol();
  O << ':';
}

void X86AsmPrinter::printPICJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                           const MachineBasicBlock *MBB,
                                           unsigned uid) const {
  const char *JTEntryDirective = MJTI->getEntrySize() == 4 ?
    MAI->getData32bitsDirective() : MAI->getData64bitsDirective();

  O << JTEntryDirective << ' ';

  if (Subtarget->isPICStyleRIPRel() || Subtarget->isPICStyleStubPIC()) {
    O << MAI->getPrivateGlobalPrefix() << getFunctionNumber()
      << '_' << uid << "_set_" << MBB->getNumber();
  } else if (Subtarget->isPICStyleGOT())
    O << *GetMBBSymbol(MBB->getNumber()) << "@GOTOFF";
  else
    O << *GetMBBSymbol(MBB->getNumber());
}

bool X86AsmPrinter::printAsmMRegister(const MachineOperand &MO, char Mode) {
  unsigned Reg = MO.getReg();
  switch (Mode) {
  default: return true;  // Unknown mode.
  case 'b': // Print QImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i8);
    break;
  case 'h': // Print QImode high register
    Reg = getX86SubSuperRegister(Reg, MVT::i8, true);
    break;
  case 'w': // Print HImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i16);
    break;
  case 'k': // Print SImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i32);
    break;
  case 'q': // Print DImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i64);
    break;
  }

  O << '%' << X86ATTInstPrinter::getRegisterName(Reg);
  return false;
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool X86AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    unsigned AsmVariant,
                                    const char *ExtraCode) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    const MachineOperand &MO = MI->getOperand(OpNo);
    
    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'a': // This is an address.  Currently only 'i' and 'r' are expected.
      if (MO.isImm()) {
        O << MO.getImm();
        return false;
      } 
      if (MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isSymbol()) {
        printSymbolOperand(MO);
        return false;
      }
      if (MO.isReg()) {
        O << '(';
        printOperand(MI, OpNo);
        O << ')';
        return false;
      }
      return true;

    case 'c': // Don't print "$" before a global var name or constant.
      if (MO.isImm())
        O << MO.getImm();
      else if (MO.isGlobal() || MO.isCPI() || MO.isJTI() || MO.isSymbol())
        printSymbolOperand(MO);
      else
        printOperand(MI, OpNo);
      return false;

    case 'A': // Print '*' before a register (it must be a register)
      if (MO.isReg()) {
        O << '*';
        printOperand(MI, OpNo);
        return false;
      }
      return true;

    case 'b': // Print QImode register
    case 'h': // Print QImode high register
    case 'w': // Print HImode register
    case 'k': // Print SImode register
    case 'q': // Print DImode register
      if (MO.isReg())
        return printAsmMRegister(MO, ExtraCode[0]);
      printOperand(MI, OpNo);
      return false;

    case 'P': // This is the operand of a call, treat specially.
      print_pcrel_imm(MI, OpNo);
      return false;

    case 'n':  // Negate the immediate or print a '-' before the operand.
      // Note: this is a temporary solution. It should be handled target
      // independently as part of the 'MC' work.
      if (MO.isImm()) {
        O << -MO.getImm();
        return false;
      }
      O << '-';
    }
  }

  printOperand(MI, OpNo);
  return false;
}

bool X86AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNo, unsigned AsmVariant,
                                          const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'b': // Print QImode register
    case 'h': // Print QImode high register
    case 'w': // Print HImode register
    case 'k': // Print SImode register
    case 'q': // Print SImode register
      // These only apply to registers, ignore on mem.
      break;
    case 'P': // Don't print @PLT, but do print as memory.
      printMemReference(MI, OpNo, "no-rip");
      return false;
    }
  }
  printMemReference(MI, OpNo);
  return false;
}



/// printMachineInstruction -- Print out a single X86 LLVM instruction MI in
/// AT&T syntax to the current output stream.
///
void X86AsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  processDebugLoc(MI, true);
  
  printInstructionThroughMCStreamer(MI);
  
  if (VerboseAsm)
    EmitComments(*MI);
  O << '\n';

  processDebugLoc(MI, false);
}

void X86AsmPrinter::PrintGlobalVariable(const GlobalVariable* GVar) {

  MCSymbol *GVarSym = GetGlobalValueSymbol(GVar);
  Constant *C = GVar->getInitializer();
  const Type *Type = C->getType();
  
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(Type);
  unsigned Align = TD->getPreferredAlignmentLog(GVar);

  printVisibility(GVarSym, GVar->getVisibility());

  if (MAI->hasDotTypeDotSizeDirective())
    O << "\t.type\t" << *GVarSym << ",@object\n";
  
  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GVar, TM);

  // Handle normal common symbols.
  if (GVKind.isCommon()) {
    if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
    
    O << ".comm " << *GVarSym << ',' << Size;
    if (MAI->getCOMMDirectiveTakesAlignment())
      O << ',' << (MAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
    
    if (VerboseAsm) {
      O << "\t\t" << MAI->getCommentString() << " '";
      WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
      O << '\'';
    }
    O << '\n';
    return;
  }
  
  if (GVKind.isBSSLocal()) {
    if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
    
    if (const char *LComm = MAI->getLCOMMDirective()) {
      if (GVar->hasLocalLinkage()) {
        O << LComm << *GVarSym << ',' << Size;
        if (MAI->getLCOMMDirectiveTakesAlignment())
          O << ',' << Align;
      }
    } else {
      if (!Subtarget->isTargetCygMing())
        O << "\t.local\t" << *GVarSym << '\n';
      O << MAI->getCOMMDirective() << *GVarSym << ',' << Size;
      if (MAI->getCOMMDirectiveTakesAlignment())
        O << ',' << (MAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
    }
    if (VerboseAsm) {
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << ' ';
      WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
    }
    O << '\n';
    return;
  }
  
  const MCSection *TheSection =
    getObjFileLowering().SectionForGlobal(GVar, GVKind, Mang, TM);

  // Handle the zerofill directive on darwin, which is a special form of BSS
  // emission.
  if (GVKind.isBSSExtern() && MAI->hasMachoZeroFillDirective()) {
    // .globl _foo
    OutStreamer.EmitSymbolAttribute(GVarSym, MCStreamer::Global);
    // .zerofill __DATA, __common, _foo, 400, 5
    OutStreamer.EmitZerofill(TheSection, GVarSym, Size, 1 << Align);
    return;
  }

  OutStreamer.SwitchSection(TheSection);

  switch (GVar->getLinkage()) {
  case GlobalValue::CommonLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::LinkerPrivateLinkage:
    if (Subtarget->isTargetDarwin()) {
      OutStreamer.EmitSymbolAttribute(GVarSym, MCStreamer::Global);
      O << MAI->getWeakDefDirective() << *GVarSym << '\n';
    } else if (Subtarget->isTargetCygMing()) {
      OutStreamer.EmitSymbolAttribute(GVarSym, MCStreamer::Global);
      O << "\t.linkonce same_size\n";
    } else
      O << "\t.weak\t" << *GVarSym << '\n';
    break;
  case GlobalValue::DLLExportLinkage:
  case GlobalValue::AppendingLinkage:
    // FIXME: appending linkage variables should go into a section of
    // their name or something.  For now, just emit them as external.
  case GlobalValue::ExternalLinkage:
    // If external or appending, declare as a global symbol
    OutStreamer.EmitSymbolAttribute(GVarSym, MCStreamer::Global);
    break;
  case GlobalValue::PrivateLinkage:
  case GlobalValue::InternalLinkage:
     break;
  default:
    llvm_unreachable("Unknown linkage type!");
  }

  EmitAlignment(Align, GVar);
  O << *GVarSym << ":";
  if (VerboseAsm){
    O.PadToColumn(MAI->getCommentColumn());
    O << MAI->getCommentString() << ' ';
    WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
  }
  O << '\n';

  EmitGlobalConstant(C);

  if (MAI->hasDotTypeDotSizeDirective())
    O << "\t.size\t" << *GVarSym << ", " << Size << '\n';
}

void X86AsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    // All darwin targets use mach-o.
    TargetLoweringObjectFileMachO &TLOFMacho = 
      static_cast<TargetLoweringObjectFileMachO &>(getObjFileLowering());
    
    MachineModuleInfoMachO &MMIMacho =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();
    
    // Output stubs for dynamically-linked functions.
    MachineModuleInfoMachO::SymbolListTy Stubs;

    Stubs = MMIMacho.GetFnStubList();
    if (!Stubs.empty()) {
      const MCSection *TheSection = 
        TLOFMacho.getMachOSection("__IMPORT", "__jump_table",
                                  MCSectionMachO::S_SYMBOL_STUBS |
                                  MCSectionMachO::S_ATTR_SELF_MODIFYING_CODE |
                                  MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                  5, SectionKind::getMetadata());
      OutStreamer.SwitchSection(TheSection);

      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        O << *Stubs[i].first << ":\n";
        // Get the MCSymbol without the $stub suffix.
        O << "\t.indirect_symbol " << *Stubs[i].second;
        O << "\n\thlt ; hlt ; hlt ; hlt ; hlt\n";
      }
      O << '\n';
      
      Stubs.clear();
    }

    // Output stubs for external and common global variables.
    Stubs = MMIMacho.GetGVStubList();
    if (!Stubs.empty()) {
      const MCSection *TheSection = 
        TLOFMacho.getMachOSection("__IMPORT", "__pointers",
                                  MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS,
                                  SectionKind::getMetadata());
      OutStreamer.SwitchSection(TheSection);

      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        O << *Stubs[i].first << ":\n\t.indirect_symbol " << *Stubs[i].second;
        O << "\n\t.long\t0\n";
      }
      Stubs.clear();
    }

    Stubs = MMIMacho.GetHiddenGVStubList();
    if (!Stubs.empty()) {
      OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
      EmitAlignment(2);

      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        O << *Stubs[i].first << ":\n" << MAI->getData32bitsDirective();
        O << *Stubs[i].second << '\n';
      }
      Stubs.clear();
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    OutStreamer.EmitAssemblerFlag(MCStreamer::SubsectionsViaSymbols);
  }

  if (Subtarget->isTargetCOFF()) {
    X86COFFMachineModuleInfo &COFFMMI =
      MMI->getObjFileInfo<X86COFFMachineModuleInfo>();

    // Emit type information for external functions
    for (X86COFFMachineModuleInfo::stub_iterator I = COFFMMI.stub_begin(),
           E = COFFMMI.stub_end(); I != E; ++I) {
      O << "\t.def\t " << I->getKeyData()
        << ";\t.scl\t" << COFF::C_EXT
        << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
        << ";\t.endef\n";
    }

    if (Subtarget->isTargetCygMing()) {
      // Necessary for dllexport support
      std::vector<const MCSymbol*> DLLExportedFns, DLLExportedGlobals;

      TargetLoweringObjectFileCOFF &TLOFCOFF =
        static_cast<TargetLoweringObjectFileCOFF&>(getObjFileLowering());

      for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I)
        if (I->hasDLLExportLinkage()) {
          MCSymbol *Sym = GetGlobalValueSymbol(I);
          COFFMMI.DecorateCygMingName(Sym, OutContext, I, *TM.getTargetData());
          DLLExportedFns.push_back(Sym);
        }

      for (Module::const_global_iterator I = M.global_begin(),
             E = M.global_end(); I != E; ++I)
        if (I->hasDLLExportLinkage())
          DLLExportedGlobals.push_back(GetGlobalValueSymbol(I));

      // Output linker support code for dllexported globals on windows.
      if (!DLLExportedGlobals.empty() || !DLLExportedFns.empty()) {
        OutStreamer.SwitchSection(TLOFCOFF.getCOFFSection(".section .drectve",
                                                          true,
                                                   SectionKind::getMetadata()));
        for (unsigned i = 0, e = DLLExportedGlobals.size(); i != e; ++i)
          O << "\t.ascii \" -export:" << *DLLExportedGlobals[i] << ",data\"\n";

        for (unsigned i = 0, e = DLLExportedFns.size(); i != e; ++i)
          O << "\t.ascii \" -export:" << *DLLExportedFns[i] << "\"\n";
      }
    }
  }
}


//===----------------------------------------------------------------------===//
// Target Registry Stuff
//===----------------------------------------------------------------------===//

static MCInstPrinter *createX86MCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             raw_ostream &O) {
  if (SyntaxVariant == 0)
    return new X86ATTInstPrinter(O, MAI);
  if (SyntaxVariant == 1)
    return new X86IntelInstPrinter(O, MAI);
  return 0;
}

// Force static initialization.
extern "C" void LLVMInitializeX86AsmPrinter() { 
  RegisterAsmPrinter<X86AsmPrinter> X(TheX86_32Target);
  RegisterAsmPrinter<X86AsmPrinter> Y(TheX86_64Target);
  
  TargetRegistry::RegisterMCInstPrinter(TheX86_32Target,createX86MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheX86_64Target,createX86MCInstPrinter);
}
