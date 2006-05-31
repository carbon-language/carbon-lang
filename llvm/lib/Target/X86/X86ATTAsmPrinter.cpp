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

#include "X86ATTAsmPrinter.h"
#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/Module.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>
using namespace llvm;

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86ATTAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  // Let PassManager know we need debug information and relay
  // the MachineDebugInfo address on to DwarfWriter.
  DW.SetDebugInfo(&getAnalysis<MachineDebugInfo>());

  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out jump tables referenced by the function
  EmitJumpTableInfo(MF.getJumpTableInfo());
  
  // Print out labels for the function.
  const Function *F = MF.getFunction();
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
    SwitchToTextSection(DefaultTextSection, F);
    EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
    break;
  case Function::ExternalLinkage:
    SwitchToTextSection(DefaultTextSection, F);
    EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
    O << "\t.globl\t" << CurrentFnName << "\n";
    break;
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    if (Subtarget->TargetType == X86Subtarget::isDarwin) {
      SwitchToTextSection(
                ".section __TEXT,__textcoal_nt,coalesced,pure_instructions", F);
      O << "\t.globl\t" << CurrentFnName << "\n";
      O << "\t.weak_definition\t" << CurrentFnName << "\n";
    } else if (Subtarget->TargetType == X86Subtarget::isCygwin) {
      EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
      O << "\t.section\t.llvm.linkonce.t." << CurrentFnName
        << ",\"ax\"\n";
      SwitchToTextSection("", F);
      O << "\t.weak " << CurrentFnName << "\n";
    } else {
      EmitAlignment(4, F);     // FIXME: This should be parameterized somewhere.
      O << "\t.section\t.llvm.linkonce.t." << CurrentFnName
        << ",\"ax\",@progbits\n";
      SwitchToTextSection("", F);
      O << "\t.weak " << CurrentFnName << "\n";
    }
    break;
  }
  O << CurrentFnName << ":\n";

  if (Subtarget->TargetType == X86Subtarget::isDarwin) {
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
  if (HasDotTypeDotSizeDirective)
    O << "\t.size " << CurrentFnName << ", .-" << CurrentFnName << "\n";

  if (Subtarget->TargetType == X86Subtarget::isDarwin) {
    // Emit post-function debug information.
    DW.EndFunction();
  }

  // We didn't modify anything.
  return false;
}

void X86ATTAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Virtual registers should not make it this far!");
    O << '%';
    unsigned Reg = MO.getReg();
    if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
      MVT::ValueType VT = (strcmp(Modifier,"subreg16") == 0)
        ? MVT::i16 : MVT::i8;
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
    O << PrivateGlobalPrefix << "JTI" << getFunctionNumber() << "_"
      << MO.getJumpTableIndex();
    // FIXME: PIC relocation model
    return;
  }
  case MachineOperand::MO_ConstantPoolIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << '$';
    O << PrivateGlobalPrefix << "CPI" << getFunctionNumber() << "_"
      << MO.getConstantPoolIndex();
    if (Subtarget->TargetType == X86Subtarget::isDarwin && 
        TM.getRelocationModel() == Reloc::PIC)
      O << "-\"L" << getFunctionNumber() << "$pb\"";
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;
    return;
  }
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp && !isCallOp) O << '$';
    // Darwin block shameless ripped from PPCAsmPrinter.cpp
    if (Subtarget->TargetType == X86Subtarget::isDarwin && 
        TM.getRelocationModel() != Reloc::Static) {
      GlobalValue *GV = MO.getGlobal();
      std::string Name = Mang->getValueName(GV);
      // Link-once, External, or Weakly-linked global variables need
      // non-lazily-resolved stubs
      if (GV->isExternal() || GV->hasWeakLinkage() ||
          GV->hasLinkOnceLinkage()) {
        // Dynamically-resolved functions need a stub for the function.
        if (isCallOp && isa<Function>(GV) && cast<Function>(GV)->isExternal()) {
          FnStubs.insert(Name);
          O << "L" << Name << "$stub";
        } else {
          GVStubs.insert(Name);
          O << "L" << Name << "$non_lazy_ptr";
        }
      } else {
        O << Mang->getValueName(GV);
      } 
      if (!isCallOp && TM.getRelocationModel() == Reloc::PIC)
        O << "-\"L" << getFunctionNumber() << "$pb\"";
   } else
      O << Mang->getValueName(MO.getGlobal());
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    if (isCallOp && 
        Subtarget->TargetType == X86Subtarget::isDarwin && 
        TM.getRelocationModel() != Reloc::Static) {
      std::string Name(GlobalPrefix);
      Name += MO.getSymbolName();
      FnStubs.insert(Name);
      O << "L" << Name << "$stub";
      return;
    }
    if (!isCallOp) O << '$';
    O << GlobalPrefix << MO.getSymbolName();
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

void X86ATTAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op){
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

  if (DispSpec.isGlobalAddress() || DispSpec.isConstantPoolIndex()) {
    printOperand(MI, Op+3, "mem");
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  }

  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << "(";
    if (BaseReg.getReg())
      printOperand(MI, Op);

    if (IndexReg.getReg()) {
      O << ",";
      printOperand(MI, Op+2);
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
  // This works around some Darwin assembler bugs.
  if (Subtarget->TargetType == X86Subtarget::isDarwin) {
    switch (MI->getOpcode()) {
    case X86::REP_MOVSB:
      O << "rep/movsb (%esi),(%edi)\n";
      return;
    case X86::REP_MOVSD:
      O << "rep/movsl (%esi),(%edi)\n";
      return;
    case X86::REP_MOVSW:
      O << "rep/movsw (%esi),(%edi)\n";
      return;
    case X86::REP_STOSB:
      O << "rep/stosb\n";
      return;
    case X86::REP_STOSD:
      O << "rep/stosl\n";
      return;
    case X86::REP_STOSW:
      O << "rep/stosw\n";
      return;
    default:
      break;
    }
  }

  // See if a truncate instruction can be turned into a nop.
  switch (MI->getOpcode()) {
  default: break;
  case X86::TRUNC_GR32_GR16:
  case X86::TRUNC_GR32_GR8:
  case X86::TRUNC_GR16_GR8: {
    const MachineOperand &MO0 = MI->getOperand(0);
    const MachineOperand &MO1 = MI->getOperand(1);
    unsigned Reg0 = MO0.getReg();
    unsigned Reg1 = MO1.getReg();
    if (MI->getOpcode() == X86::TRUNC_GR32_GR16)
      Reg1 = getX86SubSuperRegister(Reg1, MVT::i16);
    else
      Reg1 = getX86SubSuperRegister(Reg1, MVT::i8);
    O << CommentString << " TRUNCATE ";
    if (Reg0 != Reg1)
      O << "\n\t";
    break;
  }
  }

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"

