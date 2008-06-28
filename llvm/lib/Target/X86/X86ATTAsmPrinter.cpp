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
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

static std::string getPICLabelString(unsigned FnNum,
                                     const TargetAsmInfo *TAI,
                                     const X86Subtarget* Subtarget) {
  std::string label;
  if (Subtarget->isTargetDarwin())
    label =  "\"L" + utostr_32(FnNum) + "$pb\"";
  else if (Subtarget->isTargetELF())
    label = ".Lllvm$" + utostr_32(FnNum) + "." + "$piclabel";
  else
    assert(0 && "Don't know how to print PIC label!\n");

  return label;
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
    if (F->paramHasAttr(argNum, ParamAttr::ByVal))
      Ty = cast<PointerType>(Ty)->getElementType();

    // Size should be aligned to DWORD boundary
    Size += ((TD->getABITypeSize(Ty) + 3)/4)*4;
  }

  // We're not supporting tooooo huge arguments :)
  Info.setBytesToPopOnReturn((unsigned int)Size);
  return Info;
}

/// PrintUnmangledNameSafely - Print out the printable characters in the name.
/// Don't print things like \n or \0.
static void PrintUnmangledNameSafely(const Value *V, std::ostream &OS) {
  for (const char *Name = V->getNameStart(), *E = Name+V->getNameLen();
       Name != E; ++Name)
    if (isprint(*Name))
      OS << *Name;
}

/// decorateName - Query FunctionInfoMap and use this information for various
/// name decoration.
void X86ATTAsmPrinter::decorateName(std::string &Name,
                                    const GlobalValue *GV) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

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

/// getSectionForFunction - Return the section that we should emit the
/// specified function body into.
std::string X86ATTAsmPrinter::getSectionForFunction(const Function &F) const {
  switch (F.getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:
  case Function::DLLExportLinkage:
  case Function::ExternalLinkage:
    return TAI->getTextSection();
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    if (Subtarget->isTargetDarwin()) {
      return ".section __TEXT,__textcoal_nt,coalesced,pure_instructions";
    } else if (Subtarget->isTargetCygMing()) {
      return "\t.section\t.text$linkonce." + CurrentFnName + ",\"ax\"";
    } else {
      return "\t.section\t.llvm.linkonce.t." + CurrentFnName +
             ",\"ax\",@progbits";
    }
  }
}

void X86ATTAsmPrinter::emitFunctionHeader(const MachineFunction &MF) {
  const Function *F = MF.getFunction();

  decorateName(CurrentFnName, F);

  SwitchToTextSection(getSectionForFunction(*F).c_str(), F);

  unsigned FnAlign = OptimizeForSize ? 1 : 4;
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
    EmitAlignment(FnAlign, F);
    break;
  case Function::DLLExportLinkage:
  case Function::ExternalLinkage:
    EmitAlignment(FnAlign, F);
    O << "\t.globl\t" << CurrentFnName << "\n";
    break;
  case Function::LinkOnceLinkage:
  case Function::WeakLinkage:
    EmitAlignment(FnAlign, F);
    if (Subtarget->isTargetDarwin()) {
      O << "\t.globl\t" << CurrentFnName << "\n";
      O << TAI->getWeakDefDirective() << CurrentFnName << "\n";
    } else if (Subtarget->isTargetCygMing()) {
      O << "\t.globl\t" << CurrentFnName << "\n";
      O << "\t.linkonce discard\n";
    } else {
      O << "\t.weak\t" << CurrentFnName << "\n";
    }
    break;
  }
  if (F->hasHiddenVisibility()) {
    if (const char *Directive = TAI->getHiddenDirective())
      O << Directive << CurrentFnName << "\n";
  } else if (F->hasProtectedVisibility()) {
    if (const char *Directive = TAI->getProtectedDirective())
      O << Directive << CurrentFnName << "\n";
  }

  if (Subtarget->isTargetELF())
    O << "\t.type\t" << CurrentFnName << ",@function\n";
  else if (Subtarget->isTargetCygMing()) {
    O << "\t.def\t " << CurrentFnName
      << ";\t.scl\t" <<
      (F->getLinkage() == Function::InternalLinkage ? COFF::C_STAT : COFF::C_EXT)
      << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
      << ";\t.endef\n";
  }

  O << CurrentFnName << ":\n";
  // Add some workaround for linkonce linkage on Cygwin\MinGW
  if (Subtarget->isTargetCygMing() &&
      (F->getLinkage() == Function::LinkOnceLinkage ||
       F->getLinkage() == Function::WeakLinkage))
    O << "Lllvm$workaround$fake$stub$" << CurrentFnName << ":\n";
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86ATTAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  unsigned CC = F->getCallingConv();

  if (TAI->doesSupportDebugInformation()) {
    // Let PassManager know we need debug information and relay
    // the MachineModuleInfo address on to DwarfWriter.
    MMI = &getAnalysis<MachineModuleInfo>();
    DW.SetModuleInfo(MMI);
  }

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
    DW.BeginFunction(&MF);

  // Print out code for the function.
  bool hasAnyRealCode = false;
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (!I->pred_empty()) {
      printBasicBlockLabel(I, true, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      // Print the assembly for the instruction.
      if (II->getOpcode() != X86::LABEL)
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
    O << "\t.size\t" << CurrentFnName << ", .-" << CurrentFnName << "\n";

  // Emit post-function debug information.
  if (TAI->doesSupportDebugInformation())
    DW.EndFunction();

  // Print out jump tables referenced by the function.
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);

  // We didn't modify anything.
  return false;
}

static inline bool shouldPrintGOT(TargetMachine &TM, const X86Subtarget* ST) {
  return ST->isPICStyleGOT() && TM.getRelocationModel() == Reloc::PIC_;
}

static inline bool shouldPrintPLT(TargetMachine &TM, const X86Subtarget* ST) {
  return ST->isTargetELF() && TM.getRelocationModel() == Reloc::PIC_ &&
      (ST->isPICStyleRIPRel() || ST->isPICStyleGOT());
}

static inline bool shouldPrintStub(TargetMachine &TM, const X86Subtarget* ST) {
  return ST->isPICStyleStub() && TM.getRelocationModel() != Reloc::Static;
}

void X86ATTAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *Modifier, bool NotRIPRel) {
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
    for (const char *Name = TRI->getAsmName(Reg); *Name; ++Name)
      O << (char)tolower(*Name);
    return;
  }

  case MachineOperand::MO_Immediate:
    if (!Modifier ||
        (strcmp(Modifier, "debug") && strcmp(Modifier, "mem")))
      O << '$';
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB());
    return;
  case MachineOperand::MO_JumpTableIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << '$';
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << "_"
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
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << "_"
      << MO.getIndex();

    if (TM.getRelocationModel() == Reloc::PIC_) {
      if (Subtarget->isPICStyleStub())
        O << "-\"" << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
          << "$pb\"";
      else if (Subtarget->isPICStyleGOT())
        O << "@GOTOFF";
    }

    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;

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
        GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal());
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
      if (GV->isDeclaration() ||
          GV->hasWeakLinkage() ||
          GV->hasLinkOnceLinkage() ||
          GV->hasCommonLinkage()) {
        // Dynamically-resolved functions need a stub for the function.
        if (isCallOp && isa<Function>(GV)) {
          FnStubs.insert(Name);
          printSuffixedName(Name, "$stub");
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
              !GV->hasInternalLinkage())
            O << "@PLT";
        }
        if (Subtarget->isTargetCygMing() && GV->isDeclaration())
          // Save function name for later type emission
          FnStubs.insert(Name);
      }
    }

    if (GV->hasExternalWeakLinkage())
      ExtWeakSymbols.insert(GV);

    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;

    if (isThreadLocal) {
      if (TM.getRelocationModel() == Reloc::PIC_ || Subtarget->is64Bit())
        O << "@TLSGD"; // general dynamic TLS model
      else
        if (GV->isDeclaration())
          O << "@INDNTPOFF"; // initial exec TLS model
        else
          O << "@NTPOFF"; // local exec TLS model
    } else if (isMemOp) {
      if (shouldPrintGOT(TM, Subtarget)) {
        if (Subtarget->GVRequiresExtraLoad(GV, TM, false))
          O << "@GOT";
        else
          O << "@GOTOFF";
      } else if (Subtarget->isPICStyleRIPRel() && !NotRIPRel &&
                 TM.getRelocationModel() != Reloc::Static) {
        if (Subtarget->GVRequiresExtraLoad(GV, TM, false))
          O << "@GOTPCREL";

        if (needCloseParen) {
          needCloseParen = false;
          O << ')';
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
    bool needCloseParen = false;
    std::string Name(TAI->getGlobalPrefix());
    Name += MO.getSymbolName();
    if (isCallOp && shouldPrintStub(TM, Subtarget)) {
      FnStubs.insert(Name);
      printSuffixedName(Name, "$stub");
      return;
    }
    if (!isCallOp)
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
          << getPICLabelString(getFunctionNumber(), TAI, Subtarget) << "]";

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

void X86ATTAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op,
                                         const char *Modifier){
  assert(isMem(MI, Op) && "Invalid memory reference!");
  MachineOperand BaseReg  = MI->getOperand(Op);
  MachineOperand IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  bool NotRIPRel = IndexReg.getReg() || BaseReg.getReg();
  if (DispSpec.isGlobalAddress() ||
      DispSpec.isConstantPoolIndex() ||
      DispSpec.isJumpTableIndex()) {
    printOperand(MI, Op+3, "mem", NotRIPRel);
  } else {
    int DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  }

  if (IndexReg.getReg() || BaseReg.getReg()) {
    unsigned ScaleVal = MI->getOperand(Op+1).getImm();
    unsigned BaseRegOperand = 0, IndexRegOperand = 2;

    // There are cases where we can end up with ESP/RSP in the indexreg slot.
    // If this happens, swap the base/index register to support assemblers that
    // don't work when the index is *SP.
    if (IndexReg.getReg() == X86::ESP || IndexReg.getReg() == X86::RSP) {
      assert(ScaleVal == 1 && "Scale not supported for stack pointer!");
      std::swap(BaseReg, IndexReg);
      std::swap(BaseRegOperand, IndexRegOperand);
    }

    O << "(";
    if (BaseReg.getReg())
      printOperand(MI, Op+BaseRegOperand, Modifier);

    if (IndexReg.getReg()) {
      O << ",";
      printOperand(MI, Op+IndexRegOperand, Modifier);
      if (ScaleVal != 1)
        O << "," << ScaleVal;
    }
    O << ")";
  }
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
  else
    O << '-' << getPICLabelString(getFunctionNumber(), TAI, Subtarget) << '\n';
}

void X86ATTAsmPrinter::printPICLabel(const MachineInstr *MI, unsigned Op) {
  std::string label = getPICLabelString(getFunctionNumber(), TAI, Subtarget);
  O << label << "\n" << label << ":";
}


void X86ATTAsmPrinter::printPICJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                              const MachineBasicBlock *MBB,
                                              unsigned uid) const
{
  const char *JTEntryDirective = MJTI->getEntrySize() == 4 ?
    TAI->getData32bitsDirective() : TAI->getData64bitsDirective();

  O << JTEntryDirective << ' ';

  if (TM.getRelocationModel() == Reloc::PIC_) {
    if (Subtarget->isPICStyleRIPRel() || Subtarget->isPICStyleStub()) {
      O << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
        << '_' << uid << "_set_" << MBB->getNumber();
    } else if (Subtarget->isPICStyleGOT()) {
      printBasicBlockLabel(MBB, false, false, false);
      O << "@GOTOFF";
    } else
      assert(0 && "Don't know how to print MBB label for this PIC mode");
  } else
    printBasicBlockLabel(MBB, false, false, false);
}

bool X86ATTAsmPrinter::printAsmMRegister(const MachineOperand &MO,
                                         const char Mode) {
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

  O << '%';
  for (const char *Name = TRI->getAsmName(Reg); *Name; ++Name)
    O << (char)tolower(*Name);
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
    case 'b': // Print QImode register
    case 'h': // Print QImode high register
    case 'w': // Print HImode register
    case 'k': // Print SImode register
    case 'q': // Print DImode register
      if (MI->getOperand(OpNo).isRegister())
        return printAsmMRegister(MI->getOperand(OpNo), ExtraCode[0]);
      printOperand(MI, OpNo);
      return false;

    case 'P': // Don't print @PLT, but do print as memory.
      printOperand(MI, OpNo, "mem");
      return false;
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
    }
  }
  printMemReference(MI, OpNo);
  return false;
}

/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in AT&T syntax to the current output stream.
///
void X86ATTAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

/// doInitialization
bool X86ATTAsmPrinter::doInitialization(Module &M) {
  if (TAI->doesSupportDebugInformation()) {
    // Emit initial debug information.
    DW.BeginModule(&M);
  }

  bool Result = AsmPrinter::doInitialization(M);

  // Darwin wants symbols to be quoted if they have complex names.
  if (Subtarget->isTargetDarwin())
    Mang->setUseQuotes(true);

  return Result;
}


bool X86ATTAsmPrinter::doFinalization(Module &M) {
  // Note: this code is not shared by the Intel printer as it is too different
  // from how MASM does things.  When making changes here don't forget to look
  // at X86IntelAsmPrinter::doFinalization().
  const TargetData *TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer())
      continue;   // External global require no code

    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I)) {
      if (Subtarget->isTargetDarwin() &&
          TM.getRelocationModel() == Reloc::Static) {
        if (I->getName() == "llvm.global_ctors")
          O << ".reference .constructors_used\n";
        else if (I->getName() == "llvm.global_dtors")
          O << ".reference .destructors_used\n";
      }
      continue;
    }

    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    const Type *Type = C->getType();
    unsigned Size = TD->getABITypeSize(Type);
    unsigned Align = TD->getPreferredAlignmentLog(I);

    if (I->hasHiddenVisibility()) {
      if (const char *Directive = TAI->getHiddenDirective())
        O << Directive << name << "\n";
    } else if (I->hasProtectedVisibility()) {
      if (const char *Directive = TAI->getProtectedDirective())
        O << Directive << name << "\n";
    }

    if (Subtarget->isTargetELF())
      O << "\t.type\t" << name << ",@object\n";

    if (C->isNullValue() && !I->hasSection()) {
      if (I->hasExternalLinkage()) {
        if (const char *Directive = TAI->getZeroFillDirective()) {
          O << "\t.globl " << name << "\n";
          O << Directive << "__DATA, __common, " << name << ", "
            << Size << ", " << Align << "\n";
          continue;
        }
      }

      if (!I->isThreadLocal() &&
          (I->hasInternalLinkage() || I->hasWeakLinkage() ||
           I->hasLinkOnceLinkage() || I->hasCommonLinkage())) {
        if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
        if (!NoZerosInBSS && TAI->getBSSSection())
          SwitchToDataSection(TAI->getBSSSection(), I);
        else
          SwitchToDataSection(TAI->getDataSection(), I);
        if (TAI->getLCOMMDirective() != NULL) {
          if (I->hasInternalLinkage()) {
            O << TAI->getLCOMMDirective() << name << "," << Size;
            if (Subtarget->isTargetDarwin())
              O << "," << Align;
          } else if (Subtarget->isTargetDarwin() && !I->hasCommonLinkage()) {
            O << "\t.globl " << name << "\n"
              << TAI->getWeakDefDirective() << name << "\n";
            SwitchToDataSection("\t.section __DATA,__datacoal_nt,coalesced", I);
            EmitAlignment(Align, I);
            O << name << ":\t\t\t\t" << TAI->getCommentString() << " ";
            PrintUnmangledNameSafely(I, O);
            O << "\n";
            EmitGlobalConstant(C);
            continue;
          } else {
            O << TAI->getCOMMDirective()  << name << "," << Size;

            // Leopard and above support aligned common symbols.
            if (Subtarget->getDarwinVers() >= 9)
              O << "," << Align;
          }
        } else {
          if (!Subtarget->isTargetCygMing()) {
            if (I->hasInternalLinkage())
              O << "\t.local\t" << name << "\n";
          }
          O << TAI->getCOMMDirective()  << name << "," << Size;
          if (TAI->getCOMMDirectiveTakesAlignment())
            O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
        }
        O << "\t\t" << TAI->getCommentString() << " ";
        PrintUnmangledNameSafely(I, O);
        O << "\n";
        continue;
      }
    }

    switch (I->getLinkage()) {
    case GlobalValue::CommonLinkage:
    case GlobalValue::LinkOnceLinkage:
    case GlobalValue::WeakLinkage:
      if (Subtarget->isTargetDarwin()) {
        O << "\t.globl " << name << "\n"
          << TAI->getWeakDefDirective() << name << "\n";
        if (!I->isConstant())
          SwitchToDataSection("\t.section __DATA,__datacoal_nt,coalesced", I);
        else {
          const ArrayType *AT = dyn_cast<ArrayType>(Type);
          if (AT && AT->getElementType()==Type::Int8Ty)
            SwitchToDataSection("\t.section __TEXT,__const_coal,coalesced", I);
          else
            SwitchToDataSection("\t.section __DATA,__const_coal,coalesced", I);
        }
      } else if (Subtarget->isTargetCygMing()) {
        std::string SectionName(".section\t.data$linkonce." +
                                name +
                                ",\"aw\"");
        SwitchToDataSection(SectionName.c_str(), I);
        O << "\t.globl\t" << name << "\n"
          << "\t.linkonce same_size\n";
      } else {
        std::string SectionName("\t.section\t.llvm.linkonce.d." +
                                name +
                                ",\"aw\",@progbits");
        SwitchToDataSection(SectionName.c_str(), I);
        O << "\t.weak\t" << name << "\n";
      }
      break;
    case GlobalValue::DLLExportLinkage:
      DLLExportedGVs.insert(Mang->makeNameProper(I->getName(),""));
      // FALL THROUGH
    case GlobalValue::AppendingLinkage:
      // FIXME: appending linkage variables should go into a section of
      // their name or something.  For now, just emit them as external.
    case GlobalValue::ExternalLinkage:
      // If external or appending, declare as a global symbol
      O << "\t.globl " << name << "\n";
      // FALL THROUGH
    case GlobalValue::InternalLinkage: {
      if (I->isConstant()) {
        const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
        if (TAI->getCStringSection() && CVA && CVA->isCString()) {
          SwitchToDataSection(TAI->getCStringSection(), I);
          break;
        }
      }
      // FIXME: special handling for ".ctors" & ".dtors" sections
      if (I->hasSection() &&
          (I->getSection() == ".ctors" ||
           I->getSection() == ".dtors")) {
        std::string SectionName = ".section " + I->getSection();

        if (Subtarget->isTargetCygMing()) {
          SectionName += ",\"aw\"";
        } else {
          assert(!Subtarget->isTargetDarwin());
          SectionName += ",\"aw\",@progbits";
        }
        SwitchToDataSection(SectionName.c_str());
      } else if (I->hasSection() && Subtarget->isTargetDarwin()) {
        // Honor all section names on Darwin; ObjC uses this
        std::string SectionName = ".section " + I->getSection();
        SwitchToDataSection(SectionName.c_str());
      } else {
        if (C->isNullValue() && !NoZerosInBSS && TAI->getBSSSection())
          SwitchToDataSection(I->isThreadLocal() ? TAI->getTLSBSSSection() :
                              TAI->getBSSSection(), I);
        else if (!I->isConstant())
          SwitchToDataSection(I->isThreadLocal() ? TAI->getTLSDataSection() :
                              TAI->getDataSection(), I);
        else if (I->isThreadLocal())
          SwitchToDataSection(TAI->getTLSDataSection());
        else {
          // Read-only data.
          bool HasReloc = C->ContainsRelocations();
          if (HasReloc &&
              Subtarget->isTargetDarwin() &&
              TM.getRelocationModel() != Reloc::Static)
            SwitchToDataSection("\t.const_data\n");
          else if (!HasReloc && Size == 4 &&
                   TAI->getFourByteConstantSection())
            SwitchToDataSection(TAI->getFourByteConstantSection(), I);
          else if (!HasReloc && Size == 8 &&
                   TAI->getEightByteConstantSection())
            SwitchToDataSection(TAI->getEightByteConstantSection(), I);
          else if (!HasReloc && Size == 16 &&
                   TAI->getSixteenByteConstantSection())
            SwitchToDataSection(TAI->getSixteenByteConstantSection(), I);
          else if (TAI->getReadOnlySection())
            SwitchToDataSection(TAI->getReadOnlySection(), I);
          else
            SwitchToDataSection(TAI->getDataSection(), I);
        }
      }

      break;
    }
    default:
      assert(0 && "Unknown linkage type!");
    }

    EmitAlignment(Align, I);
    O << name << ":\t\t\t\t" << TAI->getCommentString() << " ";
    PrintUnmangledNameSafely(I, O);
    O << "\n";
    if (TAI->hasDotTypeDotSizeDirective())
      O << "\t.size\t" << name << ", " << Size << "\n";
    // If the initializer is a extern weak symbol, remember to emit the weak
    // reference!
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
      if (GV->hasExternalWeakLinkage())
        ExtWeakSymbols.insert(GV);

    EmitGlobalConstant(C);
  }

  // Output linker support code for dllexported globals
  if (!DLLExportedGVs.empty())
    SwitchToDataSection(".section .drectve");

  for (StringSet<>::iterator i = DLLExportedGVs.begin(),
         e = DLLExportedGVs.end();
         i != e; ++i)
    O << "\t.ascii \" -export:" << i->getKeyData() << ",data\"\n";

  if (!DLLExportedFns.empty()) {
    SwitchToDataSection(".section .drectve");
  }

  for (StringSet<>::iterator i = DLLExportedFns.begin(),
         e = DLLExportedFns.end();
         i != e; ++i)
    O << "\t.ascii \" -export:" << i->getKeyData() << "\"\n";

  if (Subtarget->isTargetDarwin()) {
    SwitchToDataSection("");

    // Output stubs for dynamically-linked functions
    unsigned j = 1;
    for (StringSet<>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i, ++j) {
      SwitchToDataSection("\t.section __IMPORT,__jump_table,symbol_stubs,"
                          "self_modifying_code+pure_instructions,5", 0);
      std::string p = i->getKeyData();
      printSuffixedName(p, "$stub");
      O << ":\n";
      O << "\t.indirect_symbol " << p << "\n";
      O << "\thlt ; hlt ; hlt ; hlt ; hlt\n";
    }

    O << "\n";

    if (TAI->doesSupportExceptionHandling() && MMI && !Subtarget->is64Bit()) {
      // Add the (possibly multiple) personalities to the set of global values.
      // Only referenced functions get into the Personalities list.
      const std::vector<Function *>& Personalities = MMI->getPersonalities();

      for (std::vector<Function *>::const_iterator I = Personalities.begin(),
             E = Personalities.end(); I != E; ++I)
        if (*I) GVStubs.insert("_" + (*I)->getName());
    }

    // Output stubs for external and common global variables.
    if (!GVStubs.empty())
      SwitchToDataSection(
                    "\t.section __IMPORT,__pointers,non_lazy_symbol_pointers");
    for (StringSet<>::iterator i = GVStubs.begin(), e = GVStubs.end();
         i != e; ++i) {
      std::string p = i->getKeyData();
      printSuffixedName(p, "$non_lazy_ptr");
      O << ":\n";
      O << "\t.indirect_symbol " << p << "\n";
      O << "\t.long\t0\n";
    }

    // Emit final debug information.
    DW.EndModule();

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  } else if (Subtarget->isTargetCygMing()) {
    // Emit type information for external functions
    for (StringSet<>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      O << "\t.def\t " << i->getKeyData()
        << ";\t.scl\t" << COFF::C_EXT
        << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
        << ";\t.endef\n";
    }

    // Emit final debug information.
    DW.EndModule();
  } else if (Subtarget->isTargetELF()) {
    // Emit final debug information.
    DW.EndModule();
  }

  return AsmPrinter::doFinalization(M);
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"
