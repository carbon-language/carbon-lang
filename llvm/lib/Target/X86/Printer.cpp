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
#include "Support/Statistic.h"

namespace {
  struct Printer : public FunctionPass {
    TargetMachine &TM;
    std::ostream &O;

    Printer(TargetMachine &tm, std::ostream &o) : TM(tm), O(o) {}

    virtual const char *getPassName() const {
      return "X86 Assembly Printer";
    }

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

static bool isScale(const MachineOperand &MO) {
  return MO.isImmediate() &&
           (MO.getImmedValue() == 1 || MO.getImmedValue() == 2 ||
            MO.getImmedValue() == 4 || MO.getImmedValue() == 8);
}

static bool isMem(const MachineInstr *MI, unsigned Op) {
  return Op+4 <= MI->getNumOperands() &&
         MI->getOperand(Op  ).isRegister() &&isScale(MI->getOperand(Op+1)) &&
         MI->getOperand(Op+2).isRegister() &&MI->getOperand(Op+3).isImmediate();
}

static void printOp(std::ostream &O, const MachineOperand &MO,
                    const MRegisterInfo &RI) {
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
  case MachineOperand::MO_MachineRegister:
    if (MO.getReg() < MRegisterInfo::FirstVirtualRegister)
      O << RI.get(MO.getReg()).Name;
    else
      O << "%reg" << MO.getReg();
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_PCRelativeDisp:
    O << "<" << MO.getVRegValue()->getName() << ">";
    return;
  default:
    O << "<unknown op ty>"; return;    
  }
}

static const std::string sizePtr (const MachineInstrDescriptor &Desc) {
  switch (Desc.TSFlags & X86II::ArgMask) {
    case X86II::Arg8:   return "BYTE PTR"; 
    case X86II::Arg16:  return "WORD PTR"; 
    case X86II::Arg32:  return "DWORD PTR"; 
    case X86II::Arg64:  return "QWORD PTR"; 
    case X86II::Arg80:  return "XWORD PTR"; 
    case X86II::Arg128: return "128BIT PTR";  // dunno what the real one is
    default: return "<SIZE?> PTR"; // crack being smoked
  }
}

static void printMemReference(std::ostream &O, const MachineInstr *MI,
                              unsigned Op, const MRegisterInfo &RI) {
  assert(isMem(MI, Op) && "Invalid memory reference!");
  const MachineOperand &BaseReg  = MI->getOperand(Op);
  const MachineOperand &Scale    = MI->getOperand(Op+1);
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &Disp     = MI->getOperand(Op+3);

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(O, BaseReg, RI);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (Scale.getImmedValue() != 1)
      O << Scale.getImmedValue() << "*";
    printOp(O, IndexReg, RI);
    NeedPlus = true;
  }

  if (Disp.getImmedValue()) {
    if (NeedPlus) O << " + ";
    printOp(O, Disp, RI);
  }
  O << "]";
}

// print - Print out an x86 instruction in intel syntax
void X86InstrInfo::print(const MachineInstr *MI, std::ostream &O,
                         const TargetMachine &TM) const {
  unsigned Opcode = MI->getOpcode();
  const MachineInstrDescriptor &Desc = get(Opcode);

  if (Opcode == X86::PHI) {
    printOp(O, MI->getOperand(0), RI);
    O << " = phi ";
    for (unsigned i = 1, e = MI->getNumOperands(); i != e; i+=2) {
      if (i != 1) O << ", ";
      O << "[";
      printOp(O, MI->getOperand(i), RI);
      O << ", ";
      printOp(O, MI->getOperand(i+1), RI);
      O << "]";
    }
    O << "\n";
    return;
  }


  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::RawFrm:
    // The accepted forms of Raw instructions are:
    //   1. nop     - No operand required
    //   2. jmp foo - PC relative displacement operand
    //
    assert(MI->getNumOperands() == 0 ||
           (MI->getNumOperands() == 1 && MI->getOperand(0).isPCRelativeDisp())&&
           "Illegal raw instruction!");
    O << getName(MI->getOpCode()) << " ";

    if (MI->getNumOperands() == 1) {
      printOp(O, MI->getOperand(0), RI);
    }
    O << "\n";
    return;

  case X86II::AddRegFrm: {
    // There are currently two forms of acceptable AddRegFrm instructions.
    // Either the instruction JUST takes a single register (like inc, dec, etc),
    // or it takes a register and an immediate of the same size as the register
    // (move immediate f.e.).  Note that this immediate value might be stored as
    // an LLVM value, to represent, for example, loading the address of a global
    // into a register.
    //
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 1 || 
            (MI->getNumOperands() == 2 &&
             (MI->getOperand(1).getVRegValueOrNull() ||
              MI->getOperand(1).isImmediate()))) &&
           "Illegal form for AddRegFrm instruction!");

    unsigned Reg = MI->getOperand(0).getReg();
    
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getNumOperands() == 2) {
      O << ", ";
      printOp(O, MI->getOperand(1), RI);
    }
    O << "\n";
    return;
  }
  case X86II::MRMDestReg: {
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
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 2 || 
            (MI->getNumOperands() == 3 && MI->getOperand(1).isRegister())) &&
           MI->getOperand(MI->getNumOperands()-1).isRegister()
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 3 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    O << "\n";
    return;
  }

  case X86II::MRMDestMem: {
    // These instructions are the same as MRMDestReg, but instead of having a
    // register reference for the mod/rm field, it's a memory reference.
    //
    assert(isMem(MI, 0) && MI->getNumOperands() == 4+1 &&
           MI->getOperand(4).isRegister() && "Bad format for MRMDestMem!");

    O << getName(MI->getOpCode()) << " " << sizePtr (Desc) << " ";
    printMemReference(O, MI, 0, RI);
    O << ", ";
    printOp(O, MI->getOperand(4), RI);
    O << "\n";
    return;
  }

  case X86II::MRMSrcReg: {
    // There is a two forms that are acceptable for MRMSrcReg instructions,
    // those with 3 and 2 operands:
    //
    // 3 Operands: in this form, the last register (the second input) is the
    // ModR/M input.  The first two operands should be the same, post register
    // allocation.  This is for things like: add r32, r/m32
    //
    // 2 Operands: this is for things like mov that do not read a second input
    //
    assert(MI->getOperand(0).isRegister() &&
           MI->getOperand(1).isRegister() &&
           (MI->getNumOperands() == 2 || 
            (MI->getNumOperands() == 3 && MI->getOperand(2).isRegister()))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 3 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    O << "\n";
    return;
  }

  case X86II::MRMSrcMem: {
    // These instructions are the same as MRMSrcReg, but instead of having a
    // register reference for the mod/rm field, it's a memory reference.
    //
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 1+4 && isMem(MI, 1)) || 
           (MI->getNumOperands() == 2+4 && MI->getOperand(1).isRegister() && 
            isMem(MI, 2))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 2+4 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", " << sizePtr (Desc) << " ";
    printMemReference(O, MI, MI->getNumOperands()-4, RI);
    O << "\n";
    return;
  }

  case X86II::MRMS0r: case X86II::MRMS1r:
  case X86II::MRMS2r: case X86II::MRMS3r:
  case X86II::MRMS4r: case X86II::MRMS5r:
  case X86II::MRMS6r: case X86II::MRMS7r: {
    // In this form, the following are valid formats:
    //  1. sete r
    //  2. cmp reg, immediate
    //  2. shl rdest, rinput  <implicit CL or 1>
    //  3. sbb rdest, rinput, immediate   [rdest = rinput]
    //    
    assert(MI->getNumOperands() > 0 && MI->getNumOperands() < 4 &&
           MI->getOperand(0).isRegister() && "Bad MRMSxR format!");
    assert((MI->getNumOperands() != 2 ||
            MI->getOperand(1).isRegister() || MI->getOperand(1).isImmediate())&&
           "Bad MRMSxR format!");
    assert((MI->getNumOperands() < 3 ||
        (MI->getOperand(1).isRegister() && MI->getOperand(2).isImmediate())) &&
           "Bad MRMSxR format!");

    if (MI->getNumOperands() > 1 && MI->getOperand(1).isRegister() && 
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getOperand(MI->getNumOperands()-1).isImmediate()) {
      O << ", ";
      printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    }
    O << "\n";

    return;
  }

  default:
    O << "\t\t\t-"; MI->print(O, TM); break;
  }
}
