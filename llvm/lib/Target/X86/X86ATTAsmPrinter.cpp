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
using namespace llvm;
using namespace x86;

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86ATTAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  setupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n";
  emitAlignment(4);
  O << "\t.globl\t" << CurrentFnName << "\n";
  if (!forCygwin && !forDarwin)
    O << "\t.type\t" << CurrentFnName << ", @function\n";
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I->pred_begin() != I->pred_end())
      O << ".LBB" << CurrentFnName << "_" << I->getNumber() << ":\t"
        << CommentString << " " << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // We didn't modify anything.
  return false;
}

void X86ATTAsmPrinter::printOp(const MachineOperand &MO, bool isCallOp) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
  case MachineOperand::MO_MachineRegister:
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Virtual registers should not make it this far!");
    O << '%';
    for (const char *Name = RI.get(MO.getReg()).Name; *Name; ++Name)
      O << (char)tolower(*Name);
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << '$' << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber () << "\t# "
      << MBBOp->getBasicBlock ()->getName ();
    return;
  }
  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building X86 MachineInstrs";
    abort ();
    return;
  case MachineOperand::MO_GlobalAddress: {
    // Darwin block shameless ripped from PowerPCAsmPrinter.cpp
    if (forDarwin) {
      if (!isCallOp) O << '$';
      GlobalValue *GV = MO.getGlobal();
      std::string Name = Mang->getValueName(GV);

      // Dynamically-resolved functions need a stub for the function.  Be
      // wary however not to output $stub for external functions whose addresses
      // are taken.  Those should be emitted as $non_lazy_ptr below.
      Function *F = dyn_cast<Function>(GV);
      if (F && isCallOp && F->isExternal()) {
        FnStubs.insert(Name);
        O << "L" << Name << "$stub";
        return;
      }

      // Link-once, External, or Weakly-linked global variables need 
      // non-lazily-resolved stubs
      if (GV->hasLinkOnceLinkage()) {
        LinkOnceStubs.insert(Name);
        O << "L" << Name << "$non_lazy_ptr";
        return;
      }
      if (GV->isExternal() || GV->hasWeakLinkage()) {
        GVStubs.insert(Name);
        O << "L" << Name << "$non_lazy_ptr";
        return;
      }
      O << Mang->getValueName(GV);
      return;
    }
    if (!isCallOp) O << '$';
    O << Mang->getValueName(MO.getGlobal());
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol:
    if (isCallOp && forDarwin) {
      std::string Name(GlobalPrefix); Name += MO.getSymbolName();
      FnStubs.insert(Name);
      O << "L" << Name << "$stub";
      return;
    }
    if (!isCallOp) O << '$';
    O << GlobalPrefix << MO.getSymbolName();
    return;
  default:
    O << "<unknown operand type>"; return;
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
  } else if (BaseReg.isConstantPoolIndex()) {
    O << ".CPI" << CurrentFnName << "_"
      << BaseReg.getConstantPoolIndex();
    if (DispSpec.getImmedValue())
      O << "+" << DispSpec.getImmedValue();
    if (IndexReg.getReg()) {
      O << "(,";
      printOp(IndexReg);
      if (ScaleVal != 1)
        O << "," << ScaleVal;
      O << ")";
    }
    return;
  }

  if (DispSpec.isGlobalAddress()) {
    printOp(DispSpec, true);
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  }

  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << "(";
    if (BaseReg.getReg())
      printOp(BaseReg);

    if (IndexReg.getReg()) {
      O << ",";
      printOp(IndexReg);
      if (ScaleVal != 1)
        O << "," << ScaleVal;
    }

    O << ")";
  }
}

/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86ATTAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;
  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"

