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

static bool isReg(const MachineOperand &MO) {
  return MO.getType() == MachineOperand::MO_VirtualRegister ||
         MO.getType() == MachineOperand::MO_MachineRegister;
}

static bool isImmediate(const MachineOperand &MO) {
  return MO.getType() == MachineOperand::MO_SignExtendedImmed ||
         MO.getType() == MachineOperand::MO_UnextendedImmed;
}

static bool isScale(const MachineOperand &MO) {
  return isImmediate(MO) &&
           (MO.getImmedValue() == 1 || MO.getImmedValue() == 2 ||
            MO.getImmedValue() == 4 || MO.getImmedValue() == 8);
}

static bool isMem(const MachineInstr *MI, unsigned Op) {
  return Op+4 <= MI->getNumOperands() &&
         isReg(MI->getOperand(Op  )) && isScale(MI->getOperand(Op+1)) &&
         isReg(MI->getOperand(Op+2)) && isImmediate(MI->getOperand(Op+3));
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
    if (IndexReg.getImmedValue() != 1)
      O << IndexReg.getImmedValue() << "*";
    printOp(O, IndexReg, RI);
    NeedPlus = true;
  }

  if (Disp.getImmedValue()) {
    if (NeedPlus) O << " + ";
    printOp(O, Disp, RI);
  }
  O << "]";
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

namespace N86 {  // Native X86 Register numbers...
  enum {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7
  };
}


// getX86RegNum - This function maps LLVM register identifiers to their X86
// specific numbering, which is used in various places encoding instructions.
//
static unsigned getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::ESP: case X86::SP: case X86::AH: return N86::ESP;
  case X86::EBP: case X86::BP: case X86::CH: return N86::EBP;
  case X86::ESI: case X86::SI: case X86::DH: return N86::ESI;
  case X86::EDI: case X86::DI: case X86::BH: return N86::EDI;
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

static void emitRegModRMByte(std::ostream &O, unsigned ModRMReg,
                             unsigned RegOpcodeField) {
  toHex(O, ModRMByte(3, RegOpcodeField, getX86RegNum(ModRMReg)));
}

inline static void emitSIBByte(std::ostream &O, unsigned SS, unsigned Index,
                               unsigned Base) {
  // SIB byte is in the same format as the ModRMByte...
  toHex(O, ModRMByte(SS, Index, Base));
}

static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

static void emitMemModRMByte(std::ostream &O, const MachineInstr *MI,
                             unsigned Op, unsigned RegOpcodeField) {
  assert(isMem(MI, Op) && "Invalid memory reference!");
  const MachineOperand &BaseReg  = MI->getOperand(Op);
  const MachineOperand &Scale    = MI->getOperand(Op+1);
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &Disp     = MI->getOperand(Op+3);

  // Is a SIB byte needed?
  if (IndexReg.getReg() == 0 && BaseReg.getReg() != X86::ESP) {
    if (BaseReg.getReg() == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      toHex(O, ModRMByte(0, RegOpcodeField, 5));
      emitConstant(O, Disp.getImmedValue(), 4);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg.getReg());
      if (Disp.getImmedValue() == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        toHex(O, ModRMByte(0, RegOpcodeField, BaseRegNo));
      } else if (isDisp8(Disp.getImmedValue())) {
        // Emit the disp8 encoding... [REG+disp8]
        toHex(O, ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(O, Disp.getImmedValue(), 1);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        toHex(O, ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(O, Disp.getImmedValue(), 4);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    if (BaseReg.getReg() == 0) {
      // If there is no base register, we emit the special case SIB byte with
      // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
      toHex(O, ModRMByte(0, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (Disp.getImmedValue() == 0) {
      // Emit no displacement ModR/M byte
      toHex(O, ModRMByte(0, RegOpcodeField, 4));
    } else if (isDisp8(Disp.getImmedValue())) {
      // Emit the disp8 encoding...
      toHex(O, ModRMByte(1, RegOpcodeField, 4));
    } else {
      // Emit the normal disp32 encoding...
      toHex(O, ModRMByte(2, RegOpcodeField, 4));
    }

    // Calculate what the SS field value should be...
    static const unsigned SSTable[] = { ~0, 0, 1, ~0, 2, ~0, ~0, ~0, 3 };
    unsigned SS = SSTable[Scale.getImmedValue()];

    if (BaseReg.getReg() == 0) {
      // Handle the SIB byte for the case where there is no base.  The
      // displacement has already been output.
      assert(IndexReg.getReg() && "Index register must be specified!");
      emitSIBByte(O, SS, getX86RegNum(IndexReg.getReg()), 5);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg.getReg());
      unsigned IndexRegNo = getX86RegNum(IndexReg.getReg());
      emitSIBByte(O, SS, IndexRegNo, BaseRegNo);
    }

    // Do we need to output a displacement?
    if (Disp.getImmedValue() != 0 || ForceDisp32) {
      if (!ForceDisp32 && isDisp8(Disp.getImmedValue()))
        emitConstant(O, Disp.getImmedValue(), 1);
      else
        emitConstant(O, Disp.getImmedValue(), 4);
    }
  }
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
    emitRegModRMByte(O, ModRMReg, getX86RegNum(ExtraReg));

    O << "\n\t\t\t\t";
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
           isReg(MI->getOperand(4)) && "Bad format for MRMDestMem!");
    toHex(O, getBaseOpcodeFor(Opcode)) << " ";
    emitMemModRMByte(O, MI, 0, getX86RegNum(MI->getOperand(4).getReg()));

    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " <SIZE> PTR ";
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
    emitRegModRMByte(O, ModRMReg, getX86RegNum(ExtraReg));

    O << "\n\t\t\t\t";
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
    assert(isReg(MI->getOperand(0)) &&
           (MI->getNumOperands() == 1+4 && isMem(MI, 1)) || 
           (MI->getNumOperands() == 2+4 && isReg(MI->getOperand(1)) && 
            isMem(MI, 2))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 2+4 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    toHex(O, getBaseOpcodeFor(Opcode)) << " ";
    unsigned ExtraReg = MI->getOperand(0).getReg();
    emitMemModRMByte(O, MI, MI->getNumOperands()-4, getX86RegNum(ExtraReg));

    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", <SIZE> PTR ";
    printMemReference(O, MI, MI->getNumOperands()-4, RI);
    O << "\n";
    return;
  }

  case X86II::MRMS0r: case X86II::MRMS1r:
  case X86II::MRMS2r: case X86II::MRMS3r:
  case X86II::MRMS4r: case X86II::MRMS5r:
  case X86II::MRMS6r: case X86II::MRMS7r: {
    unsigned ExtraField = (Desc.TSFlags & X86II::FormMask)-X86II::MRMS0r;

    // In this form, the following are valid formats:
    //  1. sete r
    //  2. shl rdest, rinput  <implicit CL or 1>
    //  3. sbb rdest, rinput, immediate   [rdest = rinput]
    //    
    assert(MI->getNumOperands() > 0 && MI->getNumOperands() < 4 &&
           isReg(MI->getOperand(0)) && "Bad MRMSxR format!");
    assert((MI->getNumOperands() < 2 || isReg(MI->getOperand(1))) &&
           "Bad MRMSxR format!");
    assert((MI->getNumOperands() < 3 || isImmediate(MI->getOperand(2))) &&
           "Bad MRMSxR format!");

    if (MI->getNumOperands() > 1 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    toHex(O, getBaseOpcodeFor(Opcode)) << " ";
    emitRegModRMByte(O, MI->getOperand(0).getReg(), ExtraField);

    if (MI->getNumOperands() == 3) {
      unsigned Size = 4;
      emitConstant(O, MI->getOperand(1).getImmedValue(), Size);
    }

    O << "\n\t\t\t\t";
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getNumOperands() == 3) {
      O << ", ";
      printOp(O, MI->getOperand(2), RI);
    }
    O << "\n";

    return;
  }

  default:
    O << "\t\t\t-"; MI->print(O, TM); break;
  }
}
