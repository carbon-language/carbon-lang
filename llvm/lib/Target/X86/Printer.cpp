//===-- X86/Printer.cpp - Convert X86 code to human readable rep. ---------===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace {
  struct Printer : public FunctionPass {
    TargetMachine &TM;
    std::ostream &O;

    Printer(TargetMachine &tm, std::ostream &o) : TM(tm), O(o) {}

    bool runOnFunction(Function &F);
  };
}

/// createX86CodePrinterPass - Print out the specified machine code function to
/// the specified stream.  This function should work regardless of whether or
/// not the function is in SSA form or not.
///
Pass *createX86CodePrinterPass(TargetMachine &TM, std::ostream &O) {
  return new Printer(TM, O);
}


/// runOnFunction - This uses the X86InstructionInfo::print method
/// to print assembly for each instruction.
bool Printer::runOnFunction (Function & F)
{
  static unsigned bbnumber = 0;
  MachineFunction & MF = MachineFunction::get (&F);
  const MachineInstrInfo & MII = TM.getInstrInfo ();

  O << "; x86 printing only sorta implemented so far!\n";

  // Print out labels for the function.
  O << "\t.globl\t" << F.getName () << "\n";
  O << "\t.type\t" << F.getName () << ", @function\n";
  O << F.getName () << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator bb_i = MF.begin (), bb_e = MF.end ();
       bb_i != bb_e; ++bb_i)
    {
      // Print a label for the basic block.
      O << ".BB" << bbnumber++ << ":\n";
      for (MachineBasicBlock::const_iterator i_i = bb_i->begin (), i_e =
	   bb_i->end (); i_i != i_e; ++i_i)
	{
	  // Print the assembly for the instruction.
	  O << "\t";
          MII.print(*i_i, O, TM);
	}
    }

  // We didn't modify anything.
  return false;
}

static void printOp(std::ostream &O, const MachineOperand &MO,
                    const MRegisterInfo &RI) {
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
  case MachineOperand::MO_MachineRegister:
    if (MO.getReg() < MRegisterInfo::FirstVirtualRegister)
      O << RI.get(MO.getReg()).Name;
    else
      O << "%reg" << MO.getReg();
    return;
    
  default:
    O << "<unknown op ty>"; return;    
  }
}

static inline void toHexDigit(std::ostream &O, unsigned char V) {
  if (V >= 10)
    O << (char)('A'+V-10);
  else
    O << (char)('0'+V);
}

static std::ostream &toHex(std::ostream &O, unsigned char V) {
  toHexDigit(O, V >> 4);
  toHexDigit(O, V & 0xF);
  return O;
}


// print - Print out an x86 instruction in intel syntax
void X86InstrInfo::print(const MachineInstr *MI, std::ostream &O,
                         const TargetMachine &TM) const {
  unsigned Opcode = MI->getOpcode();
  const MachineInstrDescriptor &Desc = get(Opcode);

  if (Desc.TSFlags & X86II::TB)
    O << "0F ";

  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::OtherFrm:
    O << "\t";
    O << "-"; MI->print(O, TM);
    break;
  case X86II::RawFrm:
    toHex(O, getBaseOpcodeFor(Opcode)) << "\t";
    O << getName(MI->getOpCode()) << " ";

    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      if (i) O << ", ";
      printOp(O, MI->getOperand(i), RI);
    }
    O << "\n";
    return;


  case X86II::AddRegFrm:
    O << "\t-"; MI->print(O, TM); break;

  case X86II::MRMDestReg:
    // There are two acceptable forms of MRMDestReg instructions, those with 3
    // and 2 operands:
    //
    // 3 Operands: in this form, the first two registers (the destination, and
    // the first operand) should be the same, post register allocation.  The 3rd
    // operand is an additional input.  This should be for things like add
    // instructions.
    //
    // 2 Operands: this is for things like mov that do not read a second input
    //
    assert(((MI->getNumOperands() == 3 && 
             (MI->getOperand(0).getType()==MachineOperand::MO_VirtualRegister||
              MI->getOperand(0).getType()==MachineOperand::MO_MachineRegister)
             &&
             (MI->getOperand(1).getType()==MachineOperand::MO_VirtualRegister||
              MI->getOperand(1).getType()==MachineOperand::MO_MachineRegister))
            ||
            (MI->getNumOperands() == 2 && 
             (MI->getOperand(0).getType()==MachineOperand::MO_VirtualRegister||
              MI->getOperand(0).getType()==MachineOperand::MO_MachineRegister)
             && (MI->getOperand(MI->getNumOperands()-1).getType() ==
                 MachineOperand::MO_VirtualRegister||
                 MI->getOperand(MI->getNumOperands()-1).getType() ==
                 MachineOperand::MO_MachineRegister)))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 3 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << "\t";
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    O << "\n";
    return;
  case X86II::MRMDestMem:
  case X86II::MRMSrcReg:
  case X86II::MRMSrcMem:
  default:
    O << "\t-"; MI->print(O, TM); break;
  }
}
