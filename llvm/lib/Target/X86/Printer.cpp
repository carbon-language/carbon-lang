//===-- X86/Printer.cpp - Convert X86 code to human readable rep. ---------===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Function.h"
#include "llvm/Constant.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "Support/Statistic.h"

namespace {
  struct Printer : public MachineFunctionPass {
    std::ostream &O;
    unsigned ConstIdx;
    Printer(std::ostream &o) : O(o), ConstIdx(0) {}

    virtual const char *getPassName() const {
      return "X86 Assembly Printer";
    }

    void printConstantPool(MachineConstantPool *MCP, const TargetData &TD);
    bool runOnMachineFunction(MachineFunction &F);
  };
}

/// createX86CodePrinterPass - Print out the specified machine code function to
/// the specified stream.  This function should work regardless of whether or
/// not the function is in SSA form or not.
///
Pass *createX86CodePrinterPass(std::ostream &O) {
  return new Printer(O);
}


// printConstantPool - Print out any constants which have been spilled to
// memory...
void Printer::printConstantPool(MachineConstantPool *MCP, const TargetData &TD){
  const std::vector<Constant*> &CP = MCP->getConstants();
  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.section .rodata\n";
    O << "\t.align " << (unsigned)TD.getTypeAlignment(CP[i]->getType()) << "\n";
    O << ".CPI" << i+ConstIdx << ":\t\t\t\t\t;" << *CP[i] << "\n";
    O << "\t*Constant output not implemented yet!*\n\n";
  }
  ConstIdx += CP.size();  // Don't recycle constant pool index numbers
}

/// runOnFunction - This uses the X86InstructionInfo::print method
/// to print assembly for each instruction.
bool Printer::runOnMachineFunction(MachineFunction &MF) {
  static unsigned BBNumber = 0;
  const TargetMachine &TM = MF.getTarget();
  const MachineInstrInfo &MII = TM.getInstrInfo();

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool(), TM.getTargetData());

  // Print out labels for the function.
  O << "\t.text\n";
  O << "\t.align 16\n";
  O << "\t.globl\t" << MF.getFunction()->getName() << "\n";
  O << "\t.type\t" << MF.getFunction()->getName() << ", @function\n";
  O << MF.getFunction()->getName() << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << ".BB" << BBNumber++ << ":\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
	 II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      MII.print(*II, O, TM);
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
  if (MI->getOperand(Op).isFrameIndex()) return true;
  if (MI->getOperand(Op).isConstantPoolIndex()) return true;
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
    // FALLTHROUGH
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
  case MachineOperand::MO_GlobalAddress:
    O << "<" << MO.getGlobal()->getName() << ">";
    return;
  case MachineOperand::MO_ExternalSymbol:
    O << "<" << MO.getSymbolName() << ">";
    return;
  default:
    O << "<unknown op ty>"; return;    
  }
}

static const std::string sizePtr(const MachineInstrDescriptor &Desc) {
  switch (Desc.TSFlags & X86II::ArgMask) {
    default: assert(0 && "Unknown arg size!");
    case X86II::Arg8:   return "BYTE PTR"; 
    case X86II::Arg16:  return "WORD PTR"; 
    case X86II::Arg32:  return "DWORD PTR"; 
    case X86II::Arg64:  return "QWORD PTR"; 
    case X86II::ArgF32:  return "DWORD PTR"; 
    case X86II::ArgF64:  return "QWORD PTR"; 
    case X86II::ArgF80:  return "XWORD PTR"; 
  }
}

static void printMemReference(std::ostream &O, const MachineInstr *MI,
                              unsigned Op, const MRegisterInfo &RI) {
  assert(isMem(MI, Op) && "Invalid memory reference!");

  if (MI->getOperand(Op).isFrameIndex()) {
    O << "[frame slot #" << MI->getOperand(Op).getFrameIndex();
    if (MI->getOperand(Op+3).getImmedValue())
      O << " + " << MI->getOperand(Op+3).getImmedValue();
    O << "]";
    return;
  } else if (MI->getOperand(Op).isConstantPoolIndex()) {
    O << "[.CPI" << MI->getOperand(Op).getConstantPoolIndex();
    if (MI->getOperand(Op+3).getImmedValue())
      O << " + " << MI->getOperand(Op+3).getImmedValue();
    O << "]";
    return;
  }

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImmedValue();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  int DispVal                    = MI->getOperand(Op+3).getImmedValue();

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(O, BaseReg, RI);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << "*";
    printOp(O, IndexReg, RI);
    NeedPlus = true;
  }

  if (DispVal) {
    if (NeedPlus)
      if (DispVal > 0)
	O << " + ";
      else {
	O << " - ";
	DispVal = -DispVal;
      }
    O << DispVal;
  }
  O << "]";
}

// print - Print out an x86 instruction in intel syntax
void X86InstrInfo::print(const MachineInstr *MI, std::ostream &O,
                         const TargetMachine &TM) const {
  unsigned Opcode = MI->getOpcode();
  const MachineInstrDescriptor &Desc = get(Opcode);

  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::Pseudo:
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
    } else {
      unsigned i = 0;
      if (MI->getNumOperands() && MI->getOperand(0).opIsDef()) {
	printOp(O, MI->getOperand(0), RI);
	O << " = ";
	++i;
      }
      O << getName(MI->getOpcode());

      for (unsigned e = MI->getNumOperands(); i != e; ++i) {
	O << " ";
	if (MI->getOperand(i).opIsDef()) O << "*";
	printOp(O, MI->getOperand(i), RI);
	if (MI->getOperand(i).opIsDef()) O << "*";
      }
    }
    O << "\n";
    return;

  case X86II::RawFrm:
    // The accepted forms of Raw instructions are:
    //   1. nop     - No operand required
    //   2. jmp foo - PC relative displacement operand
    //   3. call bar - GlobalAddress Operand or External Symbol Operand
    //
    assert(MI->getNumOperands() == 0 ||
           (MI->getNumOperands() == 1 &&
	    (MI->getOperand(0).isPCRelativeDisp() ||
	     MI->getOperand(0).isGlobalAddress() ||
	     MI->getOperand(0).isExternalSymbol())) &&
           "Illegal raw instruction!");
    O << getName(MI->getOpcode()) << " ";

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
    // into a register.  The initial register might be duplicated if this is a
    // M_2_ADDR_REG instruction
    //
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 1 || 
            (MI->getNumOperands() == 2 &&
             (MI->getOperand(1).getVRegValueOrNull() ||
              MI->getOperand(1).isImmediate() ||
	      MI->getOperand(1).isRegister() ||
	      MI->getOperand(1).isGlobalAddress() ||
	      MI->getOperand(1).isExternalSymbol()))) &&
           "Illegal form for AddRegFrm instruction!");

    unsigned Reg = MI->getOperand(0).getReg();
    
    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getNumOperands() == 2 &&
	(!MI->getOperand(1).isRegister() ||
	 MI->getOperand(1).getVRegValueOrNull() ||
	 MI->getOperand(1).isGlobalAddress() ||
	 MI->getOperand(1).isExternalSymbol())) {
      O << ", ";
      printOp(O, MI->getOperand(1), RI);
    }
    O << "\n";
    return;
  }
  case X86II::MRMDestReg: {
    // There are two acceptable forms of MRMDestReg instructions, those with 2,
    // 3 and 4 operands:
    //
    // 2 Operands: this is for things like mov that do not read a second input
    //
    // 3 Operands: in this form, the first two registers (the destination, and
    // the first operand) should be the same, post register allocation.  The 3rd
    // operand is an additional input.  This should be for things like add
    // instructions.
    //
    // 4 Operands: This form is for instructions which are 3 operands forms, but
    // have a constant argument as well.
    //
    bool isTwoAddr = isTwoAddrInstr(Opcode);
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 2 ||
	    (isTwoAddr && MI->getOperand(1).isRegister() &&
	     MI->getOperand(0).getReg() == MI->getOperand(1).getReg() &&
	     (MI->getNumOperands() == 3 ||
	      (MI->getNumOperands() == 4 && MI->getOperand(3).isImmediate()))))
           && "Bad format for MRMDestReg!");

    O << getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(1+isTwoAddr), RI);
    if (MI->getNumOperands() == 4) {
      O << ", ";
      printOp(O, MI->getOperand(3), RI);
    }
    O << "\n";
    return;
  }

  case X86II::MRMDestMem: {
    // These instructions are the same as MRMDestReg, but instead of having a
    // register reference for the mod/rm field, it's a memory reference.
    //
    assert(isMem(MI, 0) && MI->getNumOperands() == 4+1 &&
           MI->getOperand(4).isRegister() && "Bad format for MRMDestMem!");

    O << getName(MI->getOpCode()) << " " << sizePtr(Desc) << " ";
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
           && "Bad format for MRMSrcReg!");
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
    O << ", " << sizePtr(Desc) << " ";
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

  case X86II::MRMS0m: case X86II::MRMS1m:
  case X86II::MRMS2m: case X86II::MRMS3m:
  case X86II::MRMS4m: case X86II::MRMS5m:
  case X86II::MRMS6m: case X86II::MRMS7m: {
    // In this form, the following are valid formats:
    //  1. sete [m]
    //  2. cmp [m], immediate
    //  2. shl [m], rinput  <implicit CL or 1>
    //  3. sbb [m], immediate
    //    
    assert(MI->getNumOperands() >= 4 && MI->getNumOperands() <= 5 &&
           isMem(MI, 0) && "Bad MRMSxM format!");
    assert((MI->getNumOperands() != 5 || MI->getOperand(4).isImmediate()) &&
           "Bad MRMSxM format!");

    O << getName(MI->getOpCode()) << " ";
    O << sizePtr(Desc) << " ";
    printMemReference(O, MI, 0, RI);
    if (MI->getNumOperands() == 5) {
      O << ", ";
      printOp(O, MI->getOperand(4), RI);
    }
    O << "\n";
    return;
  }

  default:
    O << "\tUNKNOWN FORM:\t\t-"; MI->print(O, TM); break;
  }
}
