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

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
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

static std::ostream &emitConstant(std::ostream &O, unsigned Val, unsigned Size){
  // Output the constant in little endian byte order...
  for (unsigned i = 0; i != Size; ++i) {
    toHex(O, Val) << " ";
    Val >>= 8;
  }
  return O;
}


static bool isReg(const MachineOperand &MO) {
  return MO.getType() == MachineOperand::MO_VirtualRegister ||
         MO.getType() == MachineOperand::MO_MachineRegister;
}

static bool isImmediate(const MachineOperand &MO) {
  return MO.getType() == MachineOperand::MO_SignExtendedImmed ||
         MO.getType() == MachineOperand::MO_UnextendedImmed;
}


// getX86RegNum - This function maps LLVM register identifiers to their X86
// specific numbering, which is used in various places encoding instructions.
//
static unsigned getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::EAX: case X86::AX: case X86::AL: return 0;
  case X86::ECX: case X86::CX: case X86::CL: return 1;
  case X86::EDX: case X86::DX: case X86::DL: return 2;
  case X86::EBX: case X86::BX: case X86::BL: return 3;
  case X86::ESP: case X86::SP: case X86::AH: return 4;
  case X86::EBP: case X86::BP: case X86::CH: return 5;
  case X86::ESI: case X86::SI: case X86::DH: return 6;
  case X86::EDI: case X86::DI: case X86::BH: return 7;
  default:
    assert(RegNo >= MRegisterInfo::FirstVirtualRegister &&
           "Unknown physical register!");
    DEBUG(std::cerr << "Register allocator hasn't allocated " << RegNo
                    << " correctly yet!\n");
    return 0;
  }
}

inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                      unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

static unsigned char regModRMByte(unsigned ModRMReg, unsigned RegOpcodeField) {
  return ModRMByte(3, RegOpcodeField, getX86RegNum(ModRMReg));
}


// print - Print out an x86 instruction in intel syntax
void X86InstrInfo::print(const MachineInstr *MI, std::ostream &O,
                         const TargetMachine &TM) const {
  unsigned Opcode = MI->getOpcode();
  const MachineInstrDescriptor &Desc = get(Opcode);

  // Print instruction prefixes if neccesary
  
  if (Desc.TSFlags & X86II::OpSize) O << "66 "; // Operand size...
  if (Desc.TSFlags & X86II::TB) O << "0F ";     // Two-byte opcode prefix

  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::OtherFrm:
    O << "\t\t\t";
    O << "-"; MI->print(O, TM);
    break;
  case X86II::RawFrm:
    toHex(O, getBaseOpcodeFor(Opcode));
    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " ";

    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      if (i) O << ", ";
      printOp(O, MI->getOperand(i), RI);
    }
    O << "\n";
    return;


  case X86II::AddRegFrm: {
    // There are currently two forms of acceptable AddRegFrm instructions.
    // Either the instruction JUST takes a single register (like inc, dec, etc),
    // or it takes a register and an immediate of the same size as the register
    // (move immediate f.e.).
    //
    assert(isReg(MI->getOperand(0)) &&
           (MI->getNumOperands() == 1 || 
            (MI->getNumOperands() == 2 && isImmediate(MI->getOperand(1)))) &&
           "Illegal form for AddRegFrm instruction!");

    unsigned Reg = MI->getOperand(0).getReg();
    toHex(O, getBaseOpcodeFor(Opcode) + getX86RegNum(Reg)) << " ";

    if (MI->getNumOperands() == 2) {
      unsigned Size = 4;
      emitConstant(O, MI->getOperand(1).getImmedValue(), Size);
    }
    
    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getNumOperands() == 2) {
      O << ", ";
      printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
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
    assert(isReg(MI->getOperand(0)) &&
           (MI->getNumOperands() == 2 || 
            (MI->getNumOperands() == 3 && isReg(MI->getOperand(1)))) &&
           isReg(MI->getOperand(MI->getNumOperands()-1))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 3 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    toHex(O, getBaseOpcodeFor(Opcode)) << " ";
    unsigned ModRMReg = MI->getOperand(0).getReg();
    unsigned ExtraReg = MI->getOperand(MI->getNumOperands()-1).getReg();
    toHex(O, regModRMByte(ModRMReg, getX86RegNum(ExtraReg)));

    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
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
    assert(isReg(MI->getOperand(0)) &&
           isReg(MI->getOperand(1)) &&
           (MI->getNumOperands() == 2 || 
            (MI->getNumOperands() == 3 && isReg(MI->getOperand(2))))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 3 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    toHex(O, getBaseOpcodeFor(Opcode)) << " ";
    unsigned ModRMReg = MI->getOperand(MI->getNumOperands()-1).getReg();
    unsigned ExtraReg = MI->getOperand(0).getReg();
    toHex(O, regModRMByte(ModRMReg, getX86RegNum(ExtraReg)));

    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    O << "\n";
    return;
  }
  case X86II::MRMDestMem:
  case X86II::MRMSrcMem:
  default:
    O << "\t\t\t-"; MI->print(O, TM); break;
  }
}
