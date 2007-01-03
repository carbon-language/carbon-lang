//===-- X86ATTAsmPrinter.cpp - Convert X86 LLVM code to Intel assembly ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "X86TargetAsmInfo.h"
#include "llvm/CallingConv.h"
#include "llvm/Module.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

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
      return "\t.section\t.text$linkonce." + CurrentFnName + ",\"ax\"\n";
    } else {
      return "\t.section\t.llvm.linkonce.t." + CurrentFnName +
             ",\"ax\",@progbits\n";
    }
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86ATTAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  if (Subtarget->isTargetDarwin() ||
      Subtarget->isTargetELF() ||
      Subtarget->isTargetCygMing()) {
    // Let PassManager know we need debug information and relay
    // the MachineDebugInfo address on to DwarfWriter.
    DW.SetDebugInfo(&getAnalysis<MachineDebugInfo>());
  }

  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  const Function *F = MF.getFunction();
  unsigned CC = F->getCallingConv();

  // Populate function information map.  Actually, We don't want to populate
  // non-stdcall or non-fastcall functions' information right now.
  if (CC == CallingConv::X86_StdCall || CC == CallingConv::X86_FastCall)
    FunctionInfoMap[F] = *MF.getInfo<X86FunctionInfo>();

  X86SharedAsmPrinter::decorateName(CurrentFnName, F);

  SwitchToTextSection(getSectionForFunction(*F).c_str(), F);
  
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
    EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
    break;
  case Function::DLLExportLinkage:
    DLLExportedFns.insert(Mang->makeNameProper(F->getName(), ""));
    //FALLS THROUGH
  case Function::ExternalLinkage:
    EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
    O << "\t.globl\t" << CurrentFnName << "\n";    
    break;
  case Function::LinkOnceLinkage:
  case Function::WeakLinkage:
    if (Subtarget->isTargetDarwin()) {
      O << "\t.globl\t" << CurrentFnName << "\n";
      O << "\t.weak_definition\t" << CurrentFnName << "\n";
    } else if (Subtarget->isTargetCygMing()) {
      EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
      O << "\t.linkonce discard\n";
      O << "\t.globl " << CurrentFnName << "\n";
    } else {
      EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
      O << "\t.weak " << CurrentFnName << "\n";
    }
    break;
  }
  O << CurrentFnName << ":\n";
  // Add some workaround for linkonce linkage on Cygwin\MinGW
  if (Subtarget->isTargetCygMing() &&
      (F->getLinkage() == Function::LinkOnceLinkage ||
       F->getLinkage() == Function::WeakLinkage))
    O << "_llvm$workaround$fake$stub_" << CurrentFnName << ":\n";

  if (Subtarget->isTargetDarwin() ||
      Subtarget->isTargetELF() ||
      Subtarget->isTargetCygMing()) {
    // Emit pre-function debug information.
    DW.BeginFunction(&MF);
  }

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I->pred_begin() != I->pred_end()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // Print out jump tables referenced by the function.
  
  // Mac OS X requires that the jump table follow the function, so that the jump
  // table is part of the same atom that the function is in.
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);
  
  if (TAI->hasDotTypeDotSizeDirective())
    O << "\t.size " << CurrentFnName << ", .-" << CurrentFnName << "\n";

  if (Subtarget->isTargetDarwin() ||
      Subtarget->isTargetELF() ||
      Subtarget->isTargetCygMing()) {
    // Emit post-function debug information.
    DW.EndFunction();
  }

  // We didn't modify anything.
  return false;
}

void X86ATTAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *Modifier, bool NotRIPRel) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Virtual registers should not make it this far!");
    O << '%';
    unsigned Reg = MO.getReg();
    if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
      MVT::ValueType VT = (strcmp(Modifier+6,"64") == 0) ?
        MVT::i64 : ((strcmp(Modifier+6, "32") == 0) ? MVT::i32 :
                    ((strcmp(Modifier+6,"16") == 0) ? MVT::i16 : MVT::i8));
      Reg = getX86SubSuperRegister(Reg, VT);
    }
    for (const char *Name = RI.get(Reg).Name; *Name; ++Name)
      O << (char)tolower(*Name);
    return;
  }

  case MachineOperand::MO_Immediate:
    if (!Modifier || strcmp(Modifier, "debug") != 0)
      O << '$';
    O << MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMachineBasicBlock());
    return;
  case MachineOperand::MO_JumpTableIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << '$';
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << "_"
      << MO.getJumpTableIndex();
    if (X86PICStyle == PICStyle::Stub &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "-\"L" << getFunctionNumber() << "$pb\"";
    if (isMemOp && Subtarget->is64Bit() && !NotRIPRel)
      O << "(%rip)";
    return;
  }
  case MachineOperand::MO_ConstantPoolIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << '$';
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << "_"
      << MO.getConstantPoolIndex();
    if (X86PICStyle == PICStyle::Stub &&
        TM.getRelocationModel() == Reloc::PIC_)
      O << "-\"L" << getFunctionNumber() << "$pb\"";
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;

    if (isMemOp && Subtarget->is64Bit() && !NotRIPRel)
      O << "(%rip)";
    return;
  }
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp && !isCallOp) O << '$';

    GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);
    
    bool isExt = (GV->isExternal() || GV->hasWeakLinkage() ||
                  GV->hasLinkOnceLinkage());
    
    X86SharedAsmPrinter::decorateName(Name, GV);
    
    if (X86PICStyle == PICStyle::Stub &&
        TM.getRelocationModel() != Reloc::Static) {
      // Link-once, External, or Weakly-linked global variables need
      // non-lazily-resolved stubs
      if (isExt) {
        // Dynamically-resolved functions need a stub for the function.
        if (isCallOp && isa<Function>(GV)) {
          FnStubs.insert(Name);
          O << "L" << Name << "$stub";
        } else {
          GVStubs.insert(Name);
          O << "L" << Name << "$non_lazy_ptr";
        }
      } else {
        if (GV->hasDLLImportLinkage()) {
          O << "__imp_";          
        } 
        O << Name;
      }
      
      if (!isCallOp && TM.getRelocationModel() == Reloc::PIC_)
        O << "-\"L" << getFunctionNumber() << "$pb\"";
    } else {
      if (GV->hasDLLImportLinkage()) {
        O << "__imp_";          
      }       
      O << Name;
    }

    if (GV->hasExternalWeakLinkage())
      ExtWeakSymbols.insert(GV);
    
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;

    if (isMemOp && Subtarget->is64Bit()) {
      if (isExt && TM.getRelocationModel() != Reloc::Static)
        O << "@GOTPCREL(%rip)";
      else if (!NotRIPRel)
        // Use rip when possible to reduce code size, except when index or
        // base register are also part of the address. e.g.
        // foo(%rip)(%rcx,%rax,4) is not legal
        O << "(%rip)";        
    }

    return;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    if (isCallOp && 
        X86PICStyle == PICStyle::Stub &&
        TM.getRelocationModel() != Reloc::Static) {
      std::string Name(TAI->getGlobalPrefix());
      Name += MO.getSymbolName();
      FnStubs.insert(Name);
      O << "L" << Name << "$stub";
      return;
    }
    if (!isCallOp) O << '$';
    O << TAI->getGlobalPrefix() << MO.getSymbolName();

    if (!isCallOp && Subtarget->is64Bit())
      O << "(%rip)";

    return;
  }
  default:
    O << "<unknown operand type>"; return;
  }
}

void X86ATTAsmPrinter::printSSECC(const MachineInstr *MI, unsigned Op) {
  unsigned char value = MI->getOperand(Op).getImmedValue();
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

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImmedValue();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  if (BaseReg.isFrameIndex()) {
    O << "[frame slot #" << BaseReg.getFrameIndex();
    if (DispSpec.getImmedValue())
      O << " + " << DispSpec.getImmedValue();
    O << "]";
    return;
  }

  bool NotRIPRel = IndexReg.getReg() || BaseReg.getReg();
  if (DispSpec.isGlobalAddress() ||
      DispSpec.isConstantPoolIndex() ||
      DispSpec.isJumpTableIndex()) {
    printOperand(MI, Op+3, "mem", NotRIPRel);
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  }

  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << "(";
    if (BaseReg.getReg()) {
      printOperand(MI, Op, Modifier);
    }

    if (IndexReg.getReg()) {
      O << ",";
      printOperand(MI, Op+2, Modifier);
      if (ScaleVal != 1)
        O << "," << ScaleVal;
    }

    O << ")";
  }
}

void X86ATTAsmPrinter::printPICLabel(const MachineInstr *MI, unsigned Op) {
  O << "\"L" << getFunctionNumber() << "$pb\"\n";
  O << "\"L" << getFunctionNumber() << "$pb\":";
}


bool X86ATTAsmPrinter::printAsmMRegister(const MachineOperand &MO,
                                         const char Mode) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
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
  }

  O << '%';
  for (const char *Name = RI.get(Reg).Name; *Name; ++Name)
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
    case 'c': // Don't print "$" before a global var name.
      printOperand(MI, OpNo, "mem");
      return false;
    case 'b': // Print QImode register
    case 'h': // Print QImode high register
    case 'w': // Print HImode register
    case 'k': // Print SImode register
      return printAsmMRegister(MI->getOperand(OpNo), ExtraCode[0]);
    }
  }
  
  printOperand(MI, OpNo);
  return false;
}

bool X86ATTAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                             unsigned OpNo,
                                             unsigned AsmVariant, 
                                             const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.
  printMemReference(MI, OpNo);
  return false;
}

/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86ATTAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  // See if a truncate instruction can be turned into a nop.
  switch (MI->getOpcode()) {
  default: break;
  case X86::TRUNC_64to32:
  case X86::TRUNC_64to16:
  case X86::TRUNC_32to16:
  case X86::TRUNC_32to8:
  case X86::TRUNC_16to8:
  case X86::TRUNC_32_to8:
  case X86::TRUNC_16_to8: {
    const MachineOperand &MO0 = MI->getOperand(0);
    const MachineOperand &MO1 = MI->getOperand(1);
    unsigned Reg0 = MO0.getReg();
    unsigned Reg1 = MO1.getReg();
    unsigned Opc = MI->getOpcode();
    if (Opc == X86::TRUNC_64to32)
      Reg1 = getX86SubSuperRegister(Reg1, MVT::i32);
    else if (Opc == X86::TRUNC_32to16 || Opc == X86::TRUNC_64to16)
      Reg1 = getX86SubSuperRegister(Reg1, MVT::i16);
    else
      Reg1 = getX86SubSuperRegister(Reg1, MVT::i8);
    O << TAI->getCommentString() << " TRUNCATE ";
    if (Reg0 != Reg1)
      O << "\n\t";
    break;
  }
  case X86::PsMOVZX64rr32:
    O << TAI->getCommentString() << " ZERO-EXTEND " << "\n\t";
    break;
  }

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"

