//===-- X86ATTAsmPrinter.cpp - Convert X86 LLVM code to AT&T assembly -----===//
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
#include "X86ATTAsmPrinter.h"
#include "X86.h"
#include "X86COFF.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "X86TargetAsmInfo.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

static cl::opt<bool> NewAsmPrinter("experimental-asm-printer",
                                   cl::Hidden);

//===----------------------------------------------------------------------===//
// Primitive Helper Functions.
//===----------------------------------------------------------------------===//

void X86ATTAsmPrinter::PrintPICBaseSymbol() const {
  // FIXME: the actual label generated doesn't matter here!  Just mangle in
  // something unique (the function number) with Private prefix.
  if (Subtarget->isTargetDarwin())
    O << "\"L" << getFunctionNumber() << "$pb\"";
  else {
    assert(Subtarget->isTargetELF() && "Don't know how to print PIC label!");
    O << ".Lllvm$" << getFunctionNumber() << ".$piclabel";
  }
}

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

static X86MachineFunctionInfo calculateFunctionInfo(const Function *F,
                                                    const TargetData *TD) {
  X86MachineFunctionInfo Info;
  uint64_t Size = 0;

  switch (F->getCallingConv()) {
  case CallingConv::X86_StdCall:
    Info.setDecorationStyle(StdCall);
    break;
  case CallingConv::X86_FastCall:
    Info.setDecorationStyle(FastCall);
    break;
  default:
    return Info;
  }

  unsigned argNum = 1;
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI, ++argNum) {
    const Type* Ty = AI->getType();

    // 'Dereference' type in case of byval parameter attribute
    if (F->paramHasAttr(argNum, Attribute::ByVal))
      Ty = cast<PointerType>(Ty)->getElementType();

    // Size should be aligned to DWORD boundary
    Size += ((TD->getTypeAllocSize(Ty) + 3)/4)*4;
  }

  // We're not supporting tooooo huge arguments :)
  Info.setBytesToPopOnReturn((unsigned int)Size);
  return Info;
}

/// DecorateCygMingName - Query FunctionInfoMap and use this information for
/// various name decorations for Cygwin and MingW.
void X86ATTAsmPrinter::DecorateCygMingName(std::string &Name,
                                           const GlobalValue *GV) {
  assert(Subtarget->isTargetCygMing() && "This is only for cygwin and mingw");
  
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

  // Save function name for later type emission.
  if (F->isDeclaration())
    CygMingStubs.insert(Name);
  
  // We don't want to decorate non-stdcall or non-fastcall functions right now
  unsigned CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;


  const X86MachineFunctionInfo *Info;
  
  FMFInfoMap::const_iterator info_item = FunctionInfoMap.find(F);
  if (info_item == FunctionInfoMap.end()) {
    // Calculate apropriate function info and populate map
    FunctionInfoMap[F] = calculateFunctionInfo(F, TM.getTargetData());
    Info = &FunctionInfoMap[F];
  } else {
    Info = &info_item->second;
  }

  const FunctionType *FT = F->getFunctionType();
  switch (Info->getDecorationStyle()) {
  case None:
    break;
  case StdCall:
    // "Pure" variadic functions do not receive @0 suffix.
    if (!FT->isVarArg() || (FT->getNumParams() == 0) ||
        (FT->getNumParams() == 1 && F->hasStructRetAttr()))
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());
    break;
  case FastCall:
    // "Pure" variadic functions do not receive @0 suffix.
    if (!FT->isVarArg() || (FT->getNumParams() == 0) ||
        (FT->getNumParams() == 1 && F->hasStructRetAttr()))
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());

    if (Name[0] == '_') {
      Name[0] = '@';
    } else {
      Name = '@' + Name;
    }
    break;
  default:
    llvm_unreachable("Unsupported DecorationStyle");
  }
}

void X86ATTAsmPrinter::emitFunctionHeader(const MachineFunction &MF) {
  unsigned FnAlign = MF.getAlignment();
  const Function *F = MF.getFunction();

  if (Subtarget->isTargetCygMing())
    DecorateCygMingName(CurrentFnName, F);

  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang, TM));
  EmitAlignment(FnAlign, F);

  switch (F->getLinkage()) {
  default: llvm_unreachable("Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
  case Function::PrivateLinkage:
    break;
  case Function::DLLExportLinkage:
  case Function::ExternalLinkage:
    O << "\t.globl\t" << CurrentFnName << '\n';
    break;
  case Function::LinkerPrivateLinkage:
  case Function::LinkOnceAnyLinkage:
  case Function::LinkOnceODRLinkage:
  case Function::WeakAnyLinkage:
  case Function::WeakODRLinkage:
    if (Subtarget->isTargetDarwin()) {
      O << "\t.globl\t" << CurrentFnName << '\n';
      O << TAI->getWeakDefDirective() << CurrentFnName << '\n';
    } else if (Subtarget->isTargetCygMing()) {
      O << "\t.globl\t" << CurrentFnName << "\n"
           "\t.linkonce discard\n";
    } else {
      O << "\t.weak\t" << CurrentFnName << '\n';
    }
    break;
  }

  printVisibility(CurrentFnName, F->getVisibility());

  if (Subtarget->isTargetELF())
    O << "\t.type\t" << CurrentFnName << ",@function\n";
  else if (Subtarget->isTargetCygMing()) {
    O << "\t.def\t " << CurrentFnName
      << ";\t.scl\t" <<
      (F->hasInternalLinkage() ? COFF::C_STAT : COFF::C_EXT)
      << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
      << ";\t.endef\n";
  }

  O << CurrentFnName << ':';
  if (VerboseAsm) {
    O.PadToColumn(TAI->getCommentColumn());
    O << TAI->getCommentString() << ' ';
    WriteAsOperand(O, F, /*PrintType=*/false, F->getParent());
  }
  O << '\n';

  // Add some workaround for linkonce linkage on Cygwin\MinGW
  if (Subtarget->isTargetCygMing() &&
      (F->hasLinkOnceLinkage() || F->hasWeakLinkage()))
    O << "Lllvm$workaround$fake$stub$" << CurrentFnName << ":\n";
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86ATTAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  this->MF = &MF;
  unsigned CC = F->getCallingConv();

  SetupMachineFunction(MF);
  O << "\n\n";

  // Populate function information map.  Actually, We don't want to populate
  // non-stdcall or non-fastcall functions' information right now.
  if (CC == CallingConv::X86_StdCall || CC == CallingConv::X86_FastCall)
    FunctionInfoMap[F] = *MF.getInfo<X86MachineFunctionInfo>();

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  if (F->hasDLLExportLinkage())
    DLLExportedFns.insert(Mang->getMangledName(F));

  // Print the 'header' of function
  emitFunctionHeader(MF);

  // Emit pre-function debug and/or EH information.
  if (TAI->doesSupportDebugInformation() || TAI->doesSupportExceptionHandling())
    DW->BeginFunction(&MF);

  // Print out code for the function.
  bool hasAnyRealCode = false;
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (!VerboseAsm && (I->pred_empty() || I->isOnlyReachableByFallthrough())) {
      // This is an entry block or a block that's only reachable via a
      // fallthrough edge. In non-VerboseAsm mode, don't print the label.
    } else {
      printBasicBlockLabel(I, true, true, VerboseAsm);
      O << '\n';
    }
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

  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size\t" << CurrentFnName << ", .-" << CurrentFnName << '\n';

  // Emit post-function debug information.
  if (TAI->doesSupportDebugInformation() || TAI->doesSupportExceptionHandling())
    DW->EndFunction(&MF);

  // Print out jump tables referenced by the function.
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);

  // We didn't modify anything.
  return false;
}

/// printSymbolOperand - Print a raw symbol reference operand.  This handles
/// jump tables, constant pools, global address and external symbols, all of
/// which print to a label with various suffixes for relocation types etc.
void X86ATTAsmPrinter::printSymbolOperand(const MachineOperand &MO) {
  switch (MO.getType()) {
  default: llvm_unreachable("unknown symbol type!");
  case MachineOperand::MO_JumpTableIndex:
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << '_'
      << MO.getIndex();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << MO.getIndex();
    printOffset(MO.getOffset());
    break;
  case MachineOperand::MO_GlobalAddress: {
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
    
    // If the name begins with a dollar-sign, enclose it in parens.  We do this
    // to avoid having it look like an integer immediate to the assembler.
    if (Name[0] == '$') 
      O << '(' << Name << ')';
    else
      O << Name;
    
    printOffset(MO.getOffset());
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    std::string Name = Mang->makeNameProper(MO.getSymbolName());
    if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
      FnStubs[Name+"$stub"] = Name;
      Name += "$stub";
    }
    
    // If the name begins with a dollar-sign, enclose it in parens.  We do this
    // to avoid having it look like an integer immediate to the assembler.
    if (Name[0] == '$') 
      O << '(' << Name << ')';
    else
      O << Name;
    break;
  }
  }
  
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
void X86ATTAsmPrinter::print_pcrel_imm(const MachineInstr *MI, unsigned OpNo) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  default: llvm_unreachable("Unknown pcrel immediate operand");
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB(), false, false, false);
    return;
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ExternalSymbol:
    printSymbolOperand(MO);
    return;
  }
}



void X86ATTAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  default: llvm_unreachable("unknown operand type!");
  case MachineOperand::MO_Register: {
    assert(TargetRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Virtual registers should not make it this far!");
    O << '%';
    unsigned Reg = MO.getReg();
    if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
      EVT VT = (strcmp(Modifier+6,"64") == 0) ?
        MVT::i64 : ((strcmp(Modifier+6, "32") == 0) ? MVT::i32 :
                    ((strcmp(Modifier+6,"16") == 0) ? MVT::i16 : MVT::i8));
      Reg = getX86SubSuperRegister(Reg, VT);
    }
    O << TRI->getAsmName(Reg);
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

void X86ATTAsmPrinter::printSSECC(const MachineInstr *MI, unsigned Op) {
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

void X86ATTAsmPrinter::printLeaMemReference(const MachineInstr *MI, unsigned Op,
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

void X86ATTAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op,
                                         const char *Modifier) {
  assert(isMem(MI, Op) && "Invalid memory reference!");
  const MachineOperand &Segment = MI->getOperand(Op+4);
  if (Segment.getReg()) {
    printOperand(MI, Op+4, Modifier);
    O << ':';
  }
  printLeaMemReference(MI, Op, Modifier);
}

void X86ATTAsmPrinter::printPICJumpTableSetLabel(unsigned uid,
                                           const MachineBasicBlock *MBB) const {
  if (!TAI->getSetDirective())
    return;

  // We don't need .set machinery if we have GOT-style relocations
  if (Subtarget->isPICStyleGOT())
    return;

  O << TAI->getSetDirective() << ' ' << TAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << "_set_" << MBB->getNumber() << ',';
  printBasicBlockLabel(MBB, false, false, false);
  if (Subtarget->isPICStyleRIPRel())
    O << '-' << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << uid << '\n';
  else {
    O << '-';
    PrintPICBaseSymbol();
    O << '\n';
  }
}


void X86ATTAsmPrinter::printPICLabel(const MachineInstr *MI, unsigned Op) {
  PrintPICBaseSymbol();
  O << '\n';
  PrintPICBaseSymbol();
  O << ':';
}


void X86ATTAsmPrinter::printPICJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                              const MachineBasicBlock *MBB,
                                              unsigned uid) const {
  const char *JTEntryDirective = MJTI->getEntrySize() == 4 ?
    TAI->getData32bitsDirective() : TAI->getData64bitsDirective();

  O << JTEntryDirective << ' ';

  if (Subtarget->isPICStyleRIPRel() || Subtarget->isPICStyleStubPIC()) {
    O << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
      << '_' << uid << "_set_" << MBB->getNumber();
  } else if (Subtarget->isPICStyleGOT()) {
    printBasicBlockLabel(MBB, false, false, false);
    O << "@GOTOFF";
  } else
    printBasicBlockLabel(MBB, false, false, false);
}

bool X86ATTAsmPrinter::printAsmMRegister(const MachineOperand &MO, char Mode) {
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

  O << '%'<< TRI->getAsmName(Reg);
  return false;
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool X86ATTAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    const MachineOperand &MO = MI->getOperand(OpNo);
    
    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
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

bool X86ATTAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                             unsigned OpNo,
                                             unsigned AsmVariant,
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
  return MCOperand::CreateMCValue(MCValue::get(Sym, NegatedSymbol,
                                               MO.getOffset()));
}

MCOperand X86ATTAsmPrinter::
LowerExternalSymbolOperand(const MachineOperand &MO){
  std::string Name = Mang->makeNameProper(MO.getSymbolName());
  if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
    FnStubs[Name+"$stub"] = Name;
    Name += "$stub";
  }

  MCSymbol *Sym = OutContext.GetOrCreateSymbol(Name);
  return MCOperand::CreateMCValue(MCValue::get(Sym, 0, MO.getOffset()));
}


/// printMachineInstruction -- Print out a single X86 LLVM instruction MI in
/// AT&T syntax to the current output stream.
///
void X86ATTAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  if (!NewAsmPrinter) {
    // Call the autogenerated instruction printer routines.
    printInstruction(MI);
    return;
  }
  
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
  case TargetInstrInfo::DECLARE:
    printDeclare(MI);
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
    TmpInst.addOperand(MCOperand::CreateMCValue(MCValue::get(PICBase)));
    printInstruction(&TmpInst);

    // Emit the label.
    OutStreamer.EmitLabel(PICBase);
    
    // popl $reg
    TmpInst.setOpcode(X86::POP32r);
    TmpInst.getOperand(0) = MCOperand::CreateReg(MI->getOperand(0).getReg());
    printInstruction(&TmpInst);
    O << "OLD: ";
    // Call the autogenerated instruction printer routines.
    printInstruction(MI);
    return;
  }
  }
  
  O << "NEW: ";
  
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
  
  // FIXME: Convert TmpInst.
  printInstruction(&TmpInst);
  O << "OLD: ";
  
  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

void X86ATTAsmPrinter::PrintGlobalVariable(const GlobalVariable* GVar) {
  const TargetData *TD = TM.getTargetData();

  if (!GVar->hasInitializer())
    return;   // External global require no code

  // Check to see if this is a special global used by LLVM, if so, emit it.
  if (EmitSpecialLLVMGlobal(GVar)) {
    if (Subtarget->isTargetDarwin() &&
        TM.getRelocationModel() == Reloc::Static) {
      if (GVar->getName() == "llvm.global_ctors")
        O << ".reference .constructors_used\n";
      else if (GVar->getName() == "llvm.global_dtors")
        O << ".reference .destructors_used\n";
    }
    return;
  }

  std::string name = Mang->getMangledName(GVar);
  Constant *C = GVar->getInitializer();
  const Type *Type = C->getType();
  unsigned Size = TD->getTypeAllocSize(Type);
  unsigned Align = TD->getPreferredAlignmentLog(GVar);

  printVisibility(name, GVar->getVisibility());

  if (Subtarget->isTargetELF())
    O << "\t.type\t" << name << ",@object\n";

  
  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GVar, TM);
  const MCSection *TheSection =
    getObjFileLowering().SectionForGlobal(GVar, GVKind, Mang, TM);
  OutStreamer.SwitchSection(TheSection);

  // FIXME: get this stuff from section kind flags.
  if (C->isNullValue() && !GVar->hasSection() &&
      // Don't put things that should go in the cstring section into "comm".
      !TheSection->getKind().isMergeableCString()) {
    if (GVar->hasExternalLinkage()) {
      if (const char *Directive = TAI->getZeroFillDirective()) {
        O << "\t.globl " << name << '\n';
        O << Directive << "__DATA, __common, " << name << ", "
          << Size << ", " << Align << '\n';
        return;
      }
    }

    if (!GVar->isThreadLocal() &&
        (GVar->hasLocalLinkage() || GVar->isWeakForLinker())) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.

      if (TAI->getLCOMMDirective() != NULL) {
        if (GVar->hasLocalLinkage()) {
          O << TAI->getLCOMMDirective() << name << ',' << Size;
          if (Subtarget->isTargetDarwin())
            O << ',' << Align;
        } else if (Subtarget->isTargetDarwin() && !GVar->hasCommonLinkage()) {
          O << "\t.globl " << name << '\n'
            << TAI->getWeakDefDirective() << name << '\n';
          EmitAlignment(Align, GVar);
          O << name << ":";
          if (VerboseAsm) {
            O.PadToColumn(TAI->getCommentColumn());
            O << TAI->getCommentString() << ' ';
            WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
          }
          O << '\n';
          EmitGlobalConstant(C);
          return;
        } else {
          O << TAI->getCOMMDirective()  << name << ',' << Size;
          if (TAI->getCOMMDirectiveTakesAlignment())
            O << ',' << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
        }
      } else {
        if (!Subtarget->isTargetCygMing()) {
          if (GVar->hasLocalLinkage())
            O << "\t.local\t" << name << '\n';
        }
        O << TAI->getCOMMDirective()  << name << ',' << Size;
        if (TAI->getCOMMDirectiveTakesAlignment())
          O << ',' << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
      }
      if (VerboseAsm) {
        O.PadToColumn(TAI->getCommentColumn());
        O << TAI->getCommentString() << ' ';
        WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
      }
      O << '\n';
      return;
    }
  }

  switch (GVar->getLinkage()) {
  case GlobalValue::CommonLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::LinkerPrivateLinkage:
    if (Subtarget->isTargetDarwin()) {
      O << "\t.globl " << name << '\n'
        << TAI->getWeakDefDirective() << name << '\n';
    } else if (Subtarget->isTargetCygMing()) {
      O << "\t.globl\t" << name << "\n"
           "\t.linkonce same_size\n";
    } else {
      O << "\t.weak\t" << name << '\n';
    }
    break;
  case GlobalValue::DLLExportLinkage:
  case GlobalValue::AppendingLinkage:
    // FIXME: appending linkage variables should go into a section of
    // their name or something.  For now, just emit them as external.
  case GlobalValue::ExternalLinkage:
    // If external or appending, declare as a global symbol
    O << "\t.globl " << name << '\n';
    // FALL THROUGH
  case GlobalValue::PrivateLinkage:
  case GlobalValue::InternalLinkage:
     break;
  default:
    llvm_unreachable("Unknown linkage type!");
  }

  EmitAlignment(Align, GVar);
  O << name << ":";
  if (VerboseAsm){
    O.PadToColumn(TAI->getCommentColumn());
    O << TAI->getCommentString() << ' ';
    WriteAsOperand(O, GVar, /*PrintType=*/false, GVar->getParent());
  }
  O << '\n';

  EmitGlobalConstant(C);

  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size\t" << name << ", " << Size << '\n';
}

bool X86ATTAsmPrinter::doFinalization(Module &M) {
  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->hasDLLExportLinkage())
      DLLExportedGVs.insert(Mang->getMangledName(I));
  }

  if (Subtarget->isTargetDarwin()) {
    // All darwin targets use mach-o.
    TargetLoweringObjectFileMachO &TLOFMacho = 
      static_cast<TargetLoweringObjectFileMachO &>(getObjFileLowering());
    
    // Add the (possibly multiple) personalities to the set of global value
    // stubs.  Only referenced functions get into the Personalities list.
    if (TAI->doesSupportExceptionHandling() && MMI && !Subtarget->is64Bit()) {
      const std::vector<Function*> &Personalities = MMI->getPersonalities();
      for (unsigned i = 0, e = Personalities.size(); i != e; ++i) {
        if (Personalities[i])
          GVStubs[Mang->getMangledName(Personalities[i], "$non_lazy_ptr",
                                       true /*private label*/)] = 
            Mang->getMangledName(Personalities[i]);
      }
    }

    // Output stubs for dynamically-linked functions
    if (!FnStubs.empty()) {
      const MCSection *TheSection = 
        TLOFMacho.getMachOSection("__IMPORT", "__jump_table",
                                  MCSectionMachO::S_SYMBOL_STUBS |
                                  MCSectionMachO::S_ATTR_SELF_MODIFYING_CODE |
                                  MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                  5, SectionKind::getMetadata());
      OutStreamer.SwitchSection(TheSection);
      for (StringMap<std::string>::iterator I = FnStubs.begin(),
           E = FnStubs.end(); I != E; ++I)
        O << I->getKeyData() << ":\n" << "\t.indirect_symbol " << I->second
          << "\n\thlt ; hlt ; hlt ; hlt ; hlt\n";
      O << '\n';
    }

    // Output stubs for external and common global variables.
    if (!GVStubs.empty()) {
      const MCSection *TheSection = 
        TLOFMacho.getMachOSection("__IMPORT", "__pointers",
                                  MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS,
                                  SectionKind::getMetadata());
      OutStreamer.SwitchSection(TheSection);
      for (StringMap<std::string>::iterator I = GVStubs.begin(),
           E = GVStubs.end(); I != E; ++I)
        O << I->getKeyData() << ":\n\t.indirect_symbol "
          << I->second << "\n\t.long\t0\n";
    }

    if (!HiddenGVStubs.empty()) {
      OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
      EmitAlignment(2);
      for (StringMap<std::string>::iterator I = HiddenGVStubs.begin(),
           E = HiddenGVStubs.end(); I != E; ++I)
        O << I->getKeyData() << ":\n" << TAI->getData32bitsDirective()
          << I->second << '\n';
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  } else if (Subtarget->isTargetCygMing()) {
    // Emit type information for external functions
    for (StringSet<>::iterator i = CygMingStubs.begin(), e = CygMingStubs.end();
         i != e; ++i) {
      O << "\t.def\t " << i->getKeyData()
        << ";\t.scl\t" << COFF::C_EXT
        << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
        << ";\t.endef\n";
    }
  }
  
  
  // Output linker support code for dllexported globals on windows.
  if (!DLLExportedGVs.empty() || !DLLExportedFns.empty()) {
    // dllexport symbols only exist on coff targets.
    TargetLoweringObjectFileCOFF &TLOFMacho = 
      static_cast<TargetLoweringObjectFileCOFF&>(getObjFileLowering());
    
    OutStreamer.SwitchSection(TLOFMacho.getCOFFSection(".section .drectve",true,
                                                 SectionKind::getMetadata()));
  
    for (StringSet<>::iterator i = DLLExportedGVs.begin(),
         e = DLLExportedGVs.end(); i != e; ++i)
      O << "\t.ascii \" -export:" << i->getKeyData() << ",data\"\n";
  
    for (StringSet<>::iterator i = DLLExportedFns.begin(),
         e = DLLExportedFns.end();
         i != e; ++i)
      O << "\t.ascii \" -export:" << i->getKeyData() << "\"\n";
  }
  
  // Do common shutdown.
  return AsmPrinter::doFinalization(M);
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"
