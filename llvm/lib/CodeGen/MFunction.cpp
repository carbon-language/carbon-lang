//===-- MFunction.cpp - Implementation code for the MFunction class -------===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MFunction.h"
#include "llvm/Target/MInstructionInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include <iostream>

static void printMRegister(unsigned RegNo, const MRegisterInfo &MRI,
                           std::ostream &OS) {
  if (RegNo < MRegisterInfo::FirstVirtualRegister) {
    OS << "%" << MRI[RegNo].Name;  // Hard registers are prefixed with %
  } else {
    OS << "reg" << RegNo;  // SSA registers are printed with 'reg' prefix
  }
}

static void printMInstruction(const MInstruction &MI, std::ostream &OS,
                              const MInstructionInfo &MII) {
  const MRegisterInfo &MRI = MII.getRegisterInfo();
  OS << "\t";
  if (MI.getDestinationReg() != MRegisterInfo::NoRegister) {// Produces a value?
    printMRegister(MI.getDestinationReg(), MRI, OS);
    OS << " = ";
  }

  OS << MII[MI.getOpcode()].Name << " ";

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    if (i != 0) OS << ", ";
    switch (MI.getOperandInterpretation(i)) {
    case MOperand::Register:
      printMRegister(MI.getRegisterOperand(i), MRI, OS);
      break;
    case MOperand::SignExtImmediate:
      OS << MI.getSignExtOperand(i) << "s";
      break;
    case MOperand::ZeroExtImmediate:
      OS << MI.getZeroExtOperand(i) << "z";
      break;
    case MOperand::PCRelativeDisp:
      if (MI.getPCRelativeOperand(i) >= 0)
        OS << "pc+" << MI.getPCRelativeOperand(i);
      else
        OS << "pc" << MI.getPCRelativeOperand(i);
      break;
    default:
      OS << "*UNKNOWN OPERAND INTERPRETATION*";
      break;
    }
  }
  OS << "\n";
}

/// print - Provide a way to get a simple debugging dump.  This dumps the
/// machine code in a simple "assembly" language that is not really suitable
/// for an assembler, but is useful for debugging.  This is completely target
/// independant.
///
void MFunction::print(std::ostream &OS, const MInstructionInfo &MII) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    for (MBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II)
      printMInstruction(*II, OS, MII);
    OS << "\n";  // blank line between basic blocks...
  }
}

void MFunction::dump(const MInstructionInfo &MII) const {
  print(std::cerr, MII);
}

