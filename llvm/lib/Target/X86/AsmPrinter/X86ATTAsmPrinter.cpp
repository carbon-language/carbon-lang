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
#include "llvm/MDNode.h"
#include "llvm/Type.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

static cl::opt<bool> NewAsmPrinter("experimental-asm-printer",
                                   cl::Hidden);

//===----------------------------------------------------------------------===//
// Primitive Helper Functions.
//===----------------------------------------------------------------------===//

void X86ATTAsmPrinter::PrintPICBaseSymbol() const {
  if (Subtarget->isTargetDarwin())
    O << "\"L" << getFunctionNumber() << "$pb\"";
  else if (Subtarget->isTargetELF())
    O << ".Lllvm$" << getFunctionNumber() << ".$piclabel";
  else
    assert(0 && "Don't know how to print PIC label!\n");
}

/// PrintUnmangledNameSafely - Print out the printable characters in the name.
/// Don't print things like \\n or \\0.
static void PrintUnmangledNameSafely(const Value *V, raw_ostream &OS) {
  for (const char *Name = V->getNameStart(), *E = Name+V->getNameLen();
       Name != E; ++Name)
    if (isprint(*Name))
      OS << *Name;
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

/// decorateName - Query FunctionInfoMap and use this information for various
/// name decoration.
void X86ATTAsmPrinter::decorateName(std::string &Name,
                                    const GlobalValue *GV) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

  // Save function name for later type emission.
  if (Subtarget->isTargetCygMing() && F->isDeclaration())
    CygMingStubs.insert(Name);
  
  // We don't want to decorate non-stdcall or non-fastcall functions right now
  unsigned CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;

  // Decorate names only when we're targeting Cygwin/Mingw32 targets
  if (!Subtarget->isTargetCygMing())
    return;

  FMFInfoMap::const_iterator info_item = FunctionInfoMap.find(F);

  const X86MachineFunctionInfo *Info;
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
    assert(0 && "Unsupported DecorationStyle");
  }
}

void X86ATTAsmPrinter::emitFunctionHeader(const MachineFunction &MF) {
  unsigned FnAlign = MF.getAlignment();
  const Function *F = MF.getFunction();

  decorateName(CurrentFnName, F);

  SwitchToSection(TAI->SectionForGlobal(F));
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
  case Function::PrivateLinkage:
    EmitAlignment(FnAlign, F);
    break;
  case Function::DLLExportLinkage:
  case Function::ExternalLinkage:
    EmitAlignment(FnAlign, F);
    O << "\t.globl\t" << CurrentFnName << '\n';
    break;
  case Function::LinkOnceAnyLinkage:
  case Function::LinkOnceODRLinkage:
  case Function::WeakAnyLinkage:
  case Function::WeakODRLinkage:
    EmitAlignment(FnAlign, F);
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

  O << CurrentFnName << ":\n";
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
    DLLExportedFns.insert(Mang->makeNameProper(F->getName(), ""));

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

  O.flush();

  // We didn't modify anything.
  return false;
}

/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.  These print slightly differently, for
/// example, a $ is not emitted.
void X86ATTAsmPrinter::print_pcrel_imm(const MachineInstr *MI, unsigned OpNo) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  default: assert(0 && "Unknown pcrel immediate operand");
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB(), false, false, VerboseAsm);
    return;
      
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);
    decorateName(Name, GV);
    
    bool needCloseParen = false;
    if (Name[0] == '$') {
      // The name begins with a dollar-sign. In order to avoid having it look
      // like an integer immediate to the assembler, enclose it in parens.
      O << '(';
      needCloseParen = true;
    }
    
    // Handle dllimport linkage.
    if (MO.getTargetFlags() == X86II::MO_DLLIMPORT)
      O << "__imp_" << Name;
    else if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
      FnStubs.insert(Name);
      printSuffixedName(Name, "$stub");
    } else {
      O << Name;
    }
    
    if (needCloseParen)
      O << ')';

    // Assemble call via PLT for externally visible symbols.
    if (MO.getTargetFlags() == X86II::MO_PLT)
      O << "@PLT";
    
    printOffset(MO.getOffset());
    
    return;
  }
      
  case MachineOperand::MO_ExternalSymbol: {
    bool needCloseParen = false;
    std::string Name(TAI->getGlobalPrefix());
    Name += MO.getSymbolName();
     
    if (Name[0] == '$') {
      // The name begins with a dollar-sign. In order to avoid having it look
      // like an integer immediate to the assembler, enclose it in parens.
      O << '(';
      needCloseParen = true;
    }
    
    if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
      FnStubs.insert(Name);
      printSuffixedName(Name, "$stub");
    } else {
      O << Name;
    }
    
    if (MO.getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS) {
      O << " + [.-";
      PrintPICBaseSymbol();
      O << ']';
    }
    
    if (MO.getTargetFlags() == X86II::MO_PLT)
      O << "@PLT";
    
    if (needCloseParen)
      O << ')';
    
    return;
  }
  }
}

void X86ATTAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  switch (MO.getType()) {
  default: assert(0 && "unknown operand type!");
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
    if (!Modifier || (strcmp(Modifier, "debug") && strcmp(Modifier, "mem")))
      O << '$';
    O << MO.getImm();
    return;
  case MachineOperand::MO_JumpTableIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << '$';
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << '_'
      << MO.getIndex();
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << '$';
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << MO.getIndex();

    printOffset(MO.getOffset());
    break;
  }
  case MachineOperand::MO_GlobalAddress: {
    bool isMemOp = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp)
      O << '$';
    
    const GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);
    decorateName(Name, GV);

    bool needCloseParen = false;
    if (Name[0] == '$') {
      // The name begins with a dollar-sign. In order to avoid having it look
      // like an integer immediate to the assembler, enclose it in parens.
      O << '(';
      needCloseParen = true;
    }

    // Handle dllimport linkage.
    if (MO.getTargetFlags() == X86II::MO_DLLIMPORT) {
      O << "__imp_" << Name;
    } else if (MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY ||
               MO.getTargetFlags() == X86II::MO_DARWIN_NONLAZY_PIC_BASE) {
      GVStubs.insert(Name);
      printSuffixedName(Name, "$non_lazy_ptr");
    } else if (MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY ||
               MO.getTargetFlags() == X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE){
      HiddenGVStubs.insert(Name);
      printSuffixedName(Name, "$non_lazy_ptr");
    } else if (MO.getTargetFlags() == X86II::MO_DARWIN_STUB) {
      FnStubs.insert(Name);
      printSuffixedName(Name, "$stub");
    } else {
      O << Name;
    }

    if (needCloseParen)
      O << ')';
    
    // Assemble call via PLT for externally visible symbols.
    if (MO.getTargetFlags() == X86II::MO_PLT)
      O << "@PLT";
    
    printOffset(MO.getOffset());
    break;
  }
  case MachineOperand::MO_ExternalSymbol:
    /// NOTE: MO_ExternalSymbol in a non-pcrel_imm context is *only* generated
    /// by _GLOBAL_OFFSET_TABLE_ on X86-32.  All others are call operands, which
    /// are pcrel_imm's.
    assert(!Subtarget->is64Bit());
    // These are never used as memory operands.
    assert(Modifier == 0 || strcmp(Modifier, "mem"));
    O << '$';
    O << TAI->getGlobalPrefix();
    O << MO.getSymbolName();
    break;
  }
  
  switch (MO.getTargetFlags()) {
  default:
    assert(0 && "Unknown target flag on GV operand");
  case X86II::MO_NO_FLAG:    // No flag.
    break;
  case X86II::MO_DARWIN_NONLAZY:
  case X86II::MO_DARWIN_HIDDEN_NONLAZY:
  case X86II::MO_DLLIMPORT:
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
    printOperand(MI, Op+3, "mem");
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

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'c': // Don't print "$" before a global var name or constant.
      printOperand(MI, OpNo, "mem");
      return false;

    case 'A': // Print '*' before a register (it must be a register)
      if (MI->getOperand(OpNo).isReg()) {
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
      if (MI->getOperand(OpNo).isReg())
        return printAsmMRegister(MI->getOperand(OpNo), ExtraCode[0]);
      printOperand(MI, OpNo);
      return false;

    case 'P': // This is the operand of a call, treat specially.
      print_pcrel_imm(MI, OpNo);
      return false;

    case 'n': { // Negate the immediate or print a '-' before the operand.
      // Note: this is a temporary solution. It should be handled target
      // independently as part of the 'MC' work.
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.isImm()) {
        O << -MO.getImm();
        return false;
      }
      O << '-';
    }
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

/// printMachineInstruction -- Print out a single X86 LLVM instruction MI in
/// AT&T syntax to the current output stream.
///
void X86ATTAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  if (NewAsmPrinter) {
    if (MI->getOpcode() == TargetInstrInfo::INLINEASM) {
      O << "\t";
      printInlineAsm(MI);
      return;
    } else if (MI->isLabel()) {
      printLabel(MI);
      return;
    } else if (MI->getOpcode() == TargetInstrInfo::DECLARE) {
      printDeclare(MI);
      return;
    } else if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
      printImplicitDef(MI);
      return;
    }
    
    O << "NEW: ";
    MCInst TmpInst;
    
    TmpInst.setOpcode(MI->getOpcode());
    
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      
      MCOperand MCOp;
      if (MO.isReg()) {
        MCOp.MakeReg(MO.getReg());
      } else if (MO.isImm()) {
        MCOp.MakeImm(MO.getImm());
      } else if (MO.isMBB()) {
        MCOp.MakeMBBLabel(getFunctionNumber(), MO.getMBB()->getNumber());
      } else {
        assert(0 && "Unimp");
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
  }
  
  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

/// doInitialization
bool X86ATTAsmPrinter::doInitialization(Module &M) {
  if (NewAsmPrinter) {
    Context = new MCContext();
    // FIXME: Send this to "O" instead of outs().  For now, we force it to
    // stdout to make it easy to compare.
    Streamer = createAsmStreamer(*Context, outs());
  }
  
  return AsmPrinter::doInitialization(M);
}

void X86ATTAsmPrinter::printModuleLevelGV(const GlobalVariable* GVar) {
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

  std::string name = Mang->getValueName(GVar);
  Constant *C = GVar->getInitializer();
  if (isa<MDNode>(C) || isa<MDString>(C))
    return;
  const Type *Type = C->getType();
  unsigned Size = TD->getTypeAllocSize(Type);
  unsigned Align = TD->getPreferredAlignmentLog(GVar);

  printVisibility(name, GVar->getVisibility());

  if (Subtarget->isTargetELF())
    O << "\t.type\t" << name << ",@object\n";

  SwitchToSection(TAI->SectionForGlobal(GVar));

  if (C->isNullValue() && !GVar->hasSection() &&
      !(Subtarget->isTargetDarwin() &&
        TAI->SectionKindForGlobal(GVar) == SectionKind::RODataMergeStr)) {
    // FIXME: This seems to be pretty darwin-specific
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
            O << "\t\t\t\t" << TAI->getCommentString() << ' ';
            PrintUnmangledNameSafely(GVar, O);
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
        O << "\t\t" << TAI->getCommentString() << ' ';
        PrintUnmangledNameSafely(GVar, O);
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
    assert(0 && "Unknown linkage type!");
  }

  EmitAlignment(Align, GVar);
  O << name << ":";
  if (VerboseAsm){
    O << "\t\t\t\t" << TAI->getCommentString() << ' ';
    PrintUnmangledNameSafely(GVar, O);
  }
  O << '\n';
  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size\t" << name << ", " << Size << '\n';

  EmitGlobalConstant(C);
}

bool X86ATTAsmPrinter::doFinalization(Module &M) {
  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    printModuleLevelGV(I);

    if (I->hasDLLExportLinkage())
      DLLExportedGVs.insert(Mang->makeNameProper(I->getName(),""));
  }

  if (Subtarget->isTargetDarwin()) {
    SwitchToDataSection("");
    
    // Add the (possibly multiple) personalities to the set of global value
    // stubs.  Only referenced functions get into the Personalities list.
    if (TAI->doesSupportExceptionHandling() && MMI && !Subtarget->is64Bit()) {
      const std::vector<Function*> &Personalities = MMI->getPersonalities();
      for (unsigned i = 0, e = Personalities.size(); i != e; ++i) {
        if (Personalities[i] == 0)
          continue;
        std::string Name = Mang->getValueName(Personalities[i]);
        decorateName(Name, Personalities[i]);
        GVStubs.insert(Name);
      }
    }

    // Output stubs for dynamically-linked functions
    if (!FnStubs.empty()) {
      for (StringSet<>::iterator I = FnStubs.begin(), E = FnStubs.end();
           I != E; ++I) {
        SwitchToDataSection("\t.section __IMPORT,__jump_table,symbol_stubs,"
                            "self_modifying_code+pure_instructions,5", 0);
        const char *Name = I->getKeyData();
        printSuffixedName(Name, "$stub");
        O << ":\n"
             "\t.indirect_symbol " << Name << "\n"
             "\thlt ; hlt ; hlt ; hlt ; hlt\n";
      }
      O << '\n';
    }

    // Output stubs for external and common global variables.
    if (!GVStubs.empty()) {
      SwitchToDataSection(
                    "\t.section __IMPORT,__pointers,non_lazy_symbol_pointers");
      for (StringSet<>::iterator I = GVStubs.begin(), E = GVStubs.end();
           I != E; ++I) {
        const char *Name = I->getKeyData();
        printSuffixedName(Name, "$non_lazy_ptr");
        O << ":\n\t.indirect_symbol " << Name << "\n\t.long\t0\n";
      }
    }

    if (!HiddenGVStubs.empty()) {
      SwitchToSection(TAI->getDataSection());
      EmitAlignment(2);
      for (StringSet<>::iterator I = HiddenGVStubs.begin(),
           E = HiddenGVStubs.end(); I != E; ++I) {
        const char *Name = I->getKeyData();
        printSuffixedName(Name, "$non_lazy_ptr");
        O << ":\n" << TAI->getData32bitsDirective() << Name << '\n';
      }
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
  if (!DLLExportedGVs.empty()) {
    SwitchToDataSection(".section .drectve");
  
    for (StringSet<>::iterator i = DLLExportedGVs.begin(),
         e = DLLExportedGVs.end(); i != e; ++i)
      O << "\t.ascii \" -export:" << i->getKeyData() << ",data\"\n";
  }
  
  if (!DLLExportedFns.empty()) {
    SwitchToDataSection(".section .drectve");
  
    for (StringSet<>::iterator i = DLLExportedFns.begin(),
         e = DLLExportedFns.end();
         i != e; ++i)
      O << "\t.ascii \" -export:" << i->getKeyData() << "\"\n";
  }
  
  // Do common shutdown.
  bool Changed = AsmPrinter::doFinalization(M);
  
  if (NewAsmPrinter) {
    Streamer->Finish();
    
    delete Streamer;
    delete Context;
    Streamer = 0;
    Context = 0;
  }
  
  return Changed;
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"
