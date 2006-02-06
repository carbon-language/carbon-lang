//===-- X86IntelAsmPrinter.cpp - Convert X86 LLVM code to Intel assembly --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to Intel format assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86IntelAsmPrinter.h"
#include "X86.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
using namespace llvm;
using namespace x86;

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86IntelAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  SwitchSection("\t.text\n", MF.getFunction());
  EmitAlignment(4);
  O << "\t.globl\t" << CurrentFnName << "\n";
  if (HasDotTypeDotSizeDirective)
    O << "\t.type\t" << CurrentFnName << ", @function\n";
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block if there are any predecessors.
    if (I->pred_begin() != I->pred_end())
      O << PrivateGlobalPrefix << "BB" << CurrentFnName << "_" << I->getNumber()
        << ":\t"
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

void X86IntelAsmPrinter::printSSECC(const MachineInstr *MI, unsigned Op) {
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

void X86IntelAsmPrinter::printOp(const MachineOperand &MO, 
                                 const char *Modifier) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
    if (MRegisterInfo::isPhysicalRegister(MO.getReg()))
      // Bug Workaround: See note in Printer::doInitialization about %.
      O << "%" << RI.get(MO.getReg()).Name;
    else
      O << "%reg" << MO.getReg();
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << PrivateGlobalPrefix << "BB"
      << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber () << "\t# "
      << MBBOp->getBasicBlock ()->getName ();
    return;
  }
  case MachineOperand::MO_PCRelativeDisp:
    assert(0 && "Shouldn't use addPCDisp() when building X86 MachineInstrs");
    abort ();
    return;
  case MachineOperand::MO_GlobalAddress: {
    if (!Modifier || strcmp(Modifier, "call"))
      O << "OFFSET ";
    O << Mang->getValueName(MO.getGlobal());
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << " + " << Offset;
    else if (Offset < 0)
      O << Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol:
    O << GlobalPrefix << MO.getSymbolName();
    return;
  default:
    O << "<unknown operand type>"; return;
  }
}

void X86IntelAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op){
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
    O << "[" << PrivateGlobalPrefix << "CPI" << getFunctionNumber() << "_"
      << BaseReg.getConstantPoolIndex();

    if (IndexReg.getReg()) {
      O << " + ";
      if (ScaleVal != 1)
        O << ScaleVal << "*";
      printOp(IndexReg);
    }

    if (DispSpec.getImmedValue())
      O << " + " << DispSpec.getImmedValue();
    O << "]";
    return;
  }

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(BaseReg, "call");
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << "*";
    printOp(IndexReg);
    NeedPlus = true;
  }

  if (DispSpec.isGlobalAddress()) {
    if (NeedPlus)
      O << " + ";
    printOp(DispSpec, "call");
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!BaseReg.getReg() && !IndexReg.getReg())) {
      if (NeedPlus)
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      O << DispVal;
    }
  }
  O << "]";
}


/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86IntelAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool X86IntelAsmPrinter::doInitialization(Module &M) {
  X86SharedAsmPrinter::doInitialization(M);
  // Tell gas we are outputting Intel syntax (not AT&T syntax) assembly.
  //
  // Bug: gas in `intel_syntax noprefix' mode interprets the symbol `Sp' in an
  // instruction as a reference to the register named sp, and if you try to
  // reference a symbol `Sp' (e.g. `mov ECX, OFFSET Sp') then it gets lowercased
  // before being looked up in the symbol table. This creates spurious
  // `undefined symbol' errors when linking. Workaround: Do not use `noprefix'
  // mode, and decorate all register names with percent signs.
  O << "\t.intel_syntax\n";
  return false;
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter1.inc"
