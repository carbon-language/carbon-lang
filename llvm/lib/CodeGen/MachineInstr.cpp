//===-- MachineInstr.cpp --------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Methods common to all machine instructions.
//
// FIXME: Now that MachineInstrs have parent pointers, they should always
// print themselves using their MachineFunction's TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Value.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include "Support/LeakDetector.h"
using namespace llvm;

// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class TargetInstrInfo.
// 
// FIXME: This should be a property of the target so that more than one target
// at a time can be active...
//
namespace llvm {
  extern const TargetInstrDescriptor *TargetInstrDescriptors;
}

// Constructor for instructions with variable #operands
MachineInstr::MachineInstr(short opcode, unsigned numOperands)
  : Opcode(opcode),
    numImplicitRefs(0),
    operands(numOperands, MachineOperand()),
    parent(0) {
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

/// MachineInstr ctor - This constructor only does a _reserve_ of the operands,
/// not a resize for them.  It is expected that if you use this that you call
/// add* methods below to fill up the operands, instead of the Set methods.
/// Eventually, the "resizing" ctors will be phased out.
///
MachineInstr::MachineInstr(short opcode, unsigned numOperands, bool XX, bool YY)
  : Opcode(opcode), numImplicitRefs(0), parent(0) {
  operands.reserve(numOperands);
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

/// MachineInstr ctor - Work exactly the same as the ctor above, except that the
/// MachineInstr is created and added to the end of the specified basic block.
///
MachineInstr::MachineInstr(MachineBasicBlock *MBB, short opcode,
                           unsigned numOperands)
  : Opcode(opcode), numImplicitRefs(0), parent(0) {
  assert(MBB && "Cannot use inserting ctor with null basic block!");
  operands.reserve(numOperands);
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
  MBB->push_back(this);  // Add instruction to end of basic block!
}

///MachineInstr ctor - Copies MachineInstr arg exactly
MachineInstr::MachineInstr(const MachineInstr &MI) {
  Opcode = MI.getOpcode();
  numImplicitRefs = MI.getNumImplicitRefs();
  operands.reserve(MI.getNumOperands());

  //Add operands
  for(unsigned i=0; i < MI.getNumOperands(); ++i)
    operands.push_back(MachineOperand(MI.getOperand(i)));

  //Set parent, next, and prev to null
  parent = 0;
  prev = 0;
  next = 0;
  
}


MachineInstr::~MachineInstr()
{
  LeakDetector::removeGarbageObject(this);
}

///clone - Create a copy of 'this' instruction that is identical in
///all ways except the following: The instruction has no parent The
///instruction has no name
MachineInstr* MachineInstr::clone() const {
  return new MachineInstr(*this);
}

/// OperandComplete - Return true if it's illegal to add a new operand
///
bool MachineInstr::OperandsComplete() const {
  int NumOperands = TargetInstrDescriptors[Opcode].numOperands;
  if (NumOperands >= 0 && getNumOperands() >= (unsigned)NumOperands)
    return true;  // Broken: we have all the operands of this instruction!
  return false;
}

/// replace - Support for replacing opcode and operands of a MachineInstr in
/// place. This only resets the size of the operand vector and initializes it.
/// The new operands must be set explicitly later.
/// 
void MachineInstr::replace(short opcode, unsigned numOperands) {
  assert(getNumImplicitRefs() == 0 &&
         "This is probably broken because implicit refs are going to be lost.");
  Opcode = opcode;
  operands.clear();
  operands.resize(numOperands, MachineOperand());

}

void MachineInstr::SetMachineOperandVal(unsigned i,
                                        MachineOperand::MachineOperandType opTy,
                                        Value* V) {
  assert(i < operands.size());          // may be explicit or implicit op
  operands[i].opType = opTy;
  operands[i].contents.value = V;
  operands[i].regNum = -1;
}

void
MachineInstr::SetMachineOperandConst(unsigned i,
                                     MachineOperand::MachineOperandType opTy,
                                     int intValue) {
  assert(i < getNumOperands());          // must be explicit op
  assert(TargetInstrDescriptors[Opcode].resultPos != (int) i &&
         "immed. constant cannot be defined");

  operands[i].opType = opTy;
  operands[i].contents.value = NULL;
  operands[i].contents.immedVal = intValue;
  operands[i].regNum = -1;
  operands[i].flags = 0;
}

void MachineInstr::SetMachineOperandReg(unsigned i, int regNum) {
  assert(i < getNumOperands());          // must be explicit op

  operands[i].opType = MachineOperand::MO_MachineRegister;
  operands[i].contents.value = NULL;
  operands[i].regNum = regNum;
}

// Used only by the SPARC back-end.
void MachineInstr::SetRegForOperand(unsigned i, int regNum) {
  assert(i < getNumOperands());          // must be explicit op
  operands[i].setRegForValue(regNum);
}

// Used only by the SPARC back-end.
void MachineInstr::SetRegForImplicitRef(unsigned i, int regNum) {
  getImplicitOp(i).setRegForValue(regNum);
}

/// substituteValue - Substitute all occurrences of Value* oldVal with newVal
/// in all operands and all implicit refs. If defsOnly == true, substitute defs
/// only.
///
/// FIXME: Fold this into its single caller, at SparcInstrSelection.cpp:2865,
/// or make it a static function in that file.
///
unsigned
MachineInstr::substituteValue(const Value* oldVal, Value* newVal,
                              bool defsOnly, bool notDefsAndUses,
                              bool& someArgsWereIgnored)
{
  assert((!defsOnly || !notDefsAndUses) &&
         "notDefsAndUses is irrelevant if defsOnly == true.");
  
  unsigned numSubst = 0;

  // Substitute operands
  for (MachineInstr::val_op_iterator O = begin(), E = end(); O != E; ++O)
    if (*O == oldVal)
      if (!defsOnly ||
          notDefsAndUses && (O.isDef() && !O.isUse()) ||
          !notDefsAndUses && O.isDef())
        {
          O.getMachineOperand().contents.value = newVal;
          ++numSubst;
        }
      else
        someArgsWereIgnored = true;

  // Substitute implicit refs
  for (unsigned i=0, N=getNumImplicitRefs(); i < N; ++i)
    if (getImplicitRef(i) == oldVal)
      if (!defsOnly ||
          notDefsAndUses && (getImplicitOp(i).isDef() && !getImplicitOp(i).isUse()) ||
          !notDefsAndUses && getImplicitOp(i).isDef())
        {
          getImplicitOp(i).contents.value = newVal;
          ++numSubst;
        }
      else
        someArgsWereIgnored = true;

  return numSubst;
}

void MachineInstr::dump() const {
  std::cerr << "  " << *this;
}

static inline std::ostream& OutputValue(std::ostream &os, const Value* val) {
  os << "(val ";
  os << (void*) val;                    // print address always
  if (val && val->hasName())
    os << " " << val->getName(); // print name also, if available
  os << ")";
  return os;
}

static inline void OutputReg(std::ostream &os, unsigned RegNo,
                             const MRegisterInfo *MRI = 0) {
  if (!RegNo || MRegisterInfo::isPhysicalRegister(RegNo)) {
    if (MRI)
      os << "%" << MRI->get(RegNo).Name;
    else
      os << "%mreg(" << RegNo << ")";
  } else
    os << "%reg" << RegNo;
}

static void print(const MachineOperand &MO, std::ostream &OS,
                  const TargetMachine &TM) {
  const MRegisterInfo *MRI = TM.getRegisterInfo();
  bool CloseParen = true;
  if (MO.isHiBits32())
    OS << "%lm(";
  else if (MO.isLoBits32())
    OS << "%lo(";
  else if (MO.isHiBits64())
    OS << "%hh(";
  else if (MO.isLoBits64())
    OS << "%hm(";
  else
    CloseParen = false;
  
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (MO.getVRegValue()) {
      OS << "%reg";
      OutputValue(OS, MO.getVRegValue());
      if (MO.hasAllocatedReg())
        OS << "==";
    }
    if (MO.hasAllocatedReg())
      OutputReg(OS, MO.getReg(), MRI);
    break;
  case MachineOperand::MO_CCRegister:
    OS << "%ccreg";
    OutputValue(OS, MO.getVRegValue());
    if (MO.hasAllocatedReg()) {
      OS << "==";
      OutputReg(OS, MO.getReg(), MRI);
    }
    break;
  case MachineOperand::MO_MachineRegister:
    OutputReg(OS, MO.getMachineRegNum(), MRI);
    break;
  case MachineOperand::MO_SignExtendedImmed:
    OS << (long)MO.getImmedValue();
    break;
  case MachineOperand::MO_UnextendedImmed:
    OS << (long)MO.getImmedValue();
    break;
  case MachineOperand::MO_PCRelativeDisp: {
    const Value* opVal = MO.getVRegValue();
    bool isLabel = isa<Function>(opVal) || isa<BasicBlock>(opVal);
    OS << "%disp(" << (isLabel? "label " : "addr-of-val ");
    if (opVal->hasName())
      OS << opVal->getName();
    else
      OS << (const void*) opVal;
    OS << ")";
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    OS << "bb<"
       << ((Value*)MO.getMachineBasicBlock()->getBasicBlock())->getName()
       << "," << (void*)MO.getMachineBasicBlock()->getBasicBlock() << ">";
    break;
  case MachineOperand::MO_FrameIndex:
    OS << "<fi#" << MO.getFrameIndex() << ">";
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "<cp#" << MO.getConstantPoolIndex() << ">";
    break;
  case MachineOperand::MO_GlobalAddress:
    OS << "<ga:" << ((Value*)MO.getGlobal())->getName() << ">";
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << "<es:" << MO.getSymbolName() << ">";
    break;
  default:
    assert(0 && "Unrecognized operand type");
  }

  if (CloseParen)
    OS << ")";
}

void MachineInstr::print(std::ostream &OS, const TargetMachine &TM) const {
  unsigned StartOp = 0;

   // Specialize printing if op#0 is definition
  if (getNumOperands() && getOperand(0).isDef() && !getOperand(0).isUse()) {
    ::print(getOperand(0), OS, TM);
    OS << " = ";
    ++StartOp;   // Don't print this operand again!
  }
  OS << TM.getInstrInfo().getName(getOpcode());
  
  for (unsigned i = StartOp, e = getNumOperands(); i != e; ++i) {
    const MachineOperand& mop = getOperand(i);
    if (i != StartOp)
      OS << ",";
    OS << " ";
    ::print(mop, OS, TM);
    
    if (mop.isDef())
      if (mop.isUse())
        OS << "<def&use>";
      else
        OS << "<def>";
  }
    
  // code for printing implicit references
  if (getNumImplicitRefs()) {
    OS << "\tImplicitRefs: ";
    for(unsigned i = 0, e = getNumImplicitRefs(); i != e; ++i) {
      OS << "\t";
      OutputValue(OS, getImplicitRef(i));
      if (getImplicitOp(i).isDef())
          if (getImplicitOp(i).isUse())
            OS << "<def&use>";
          else
            OS << "<def>";
    }
  }
  
  OS << "\n";
}

namespace llvm {
std::ostream &operator<<(std::ostream &os, const MachineInstr &MI) {
  // If the instruction is embedded into a basic block, we can find the target
  // info for the instruction.
  if (const MachineBasicBlock *MBB = MI.getParent()) {
    const MachineFunction *MF = MBB->getParent();
    MI.print(os, MF->getTarget());
    return os;
  }

  // Otherwise, print it out in the "raw" format without symbolic register names
  // and such.
  os << TargetInstrDescriptors[MI.getOpcode()].Name;
  
  for (unsigned i=0, N=MI.getNumOperands(); i < N; i++) {
    os << "\t" << MI.getOperand(i);
    if (MI.getOperand(i).isDef())
      if (MI.getOperand(i).isUse())
        os << "<d&u>";
      else
        os << "<d>";
  }
  
  // code for printing implicit references
  unsigned NumOfImpRefs = MI.getNumImplicitRefs();
  if (NumOfImpRefs > 0) {
    os << "\tImplicit: ";
    for (unsigned z=0; z < NumOfImpRefs; z++) {
      OutputValue(os, MI.getImplicitRef(z)); 
      if (MI.getImplicitOp(z).isDef())
          if (MI.getImplicitOp(z).isUse())
            os << "<d&u>";
          else
            os << "<d>";
      os << "\t";
    }
  }
  
  return os << "\n";
}

std::ostream &operator<<(std::ostream &OS, const MachineOperand &MO) {
  if (MO.isHiBits32())
    OS << "%lm(";
  else if (MO.isLoBits32())
    OS << "%lo(";
  else if (MO.isHiBits64())
    OS << "%hh(";
  else if (MO.isLoBits64())
    OS << "%hm(";
  
  switch (MO.getType())
    {
    case MachineOperand::MO_VirtualRegister:
      if (MO.hasAllocatedReg())
        OutputReg(OS, MO.getReg());

      if (MO.getVRegValue()) {
	if (MO.hasAllocatedReg()) OS << "==";
	OS << "%vreg";
	OutputValue(OS, MO.getVRegValue());
      }
      break;
    case MachineOperand::MO_CCRegister:
      OS << "%ccreg";
      OutputValue(OS, MO.getVRegValue());
      if (MO.hasAllocatedReg()) {
        OS << "==";
        OutputReg(OS, MO.getReg());
      }
      break;
    case MachineOperand::MO_MachineRegister:
      OutputReg(OS, MO.getMachineRegNum());
      break;
    case MachineOperand::MO_SignExtendedImmed:
      OS << (long)MO.getImmedValue();
      break;
    case MachineOperand::MO_UnextendedImmed:
      OS << (long)MO.getImmedValue();
      break;
    case MachineOperand::MO_PCRelativeDisp:
      {
        const Value* opVal = MO.getVRegValue();
        bool isLabel = isa<Function>(opVal) || isa<BasicBlock>(opVal);
        OS << "%disp(" << (isLabel? "label " : "addr-of-val ");
        if (opVal->hasName())
          OS << opVal->getName();
        else
          OS << (const void*) opVal;
        OS << ")";
        break;
      }
    case MachineOperand::MO_MachineBasicBlock:
      OS << "bb<"
         << ((Value*)MO.getMachineBasicBlock()->getBasicBlock())->getName()
         << "," << (void*)MO.getMachineBasicBlock()->getBasicBlock() << ">";
      break;
    case MachineOperand::MO_FrameIndex:
      OS << "<fi#" << MO.getFrameIndex() << ">";
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      OS << "<cp#" << MO.getConstantPoolIndex() << ">";
      break;
    case MachineOperand::MO_GlobalAddress:
      OS << "<ga:" << ((Value*)MO.getGlobal())->getName() << ">";
      break;
    case MachineOperand::MO_ExternalSymbol:
      OS << "<es:" << MO.getSymbolName() << ">";
      break;
    default:
      assert(0 && "Unrecognized operand type");
      break;
    }
  
  if (MO.isHiBits32() || MO.isLoBits32() || MO.isHiBits64() || MO.isLoBits64())
    OS << ")";
  
  return OS;
}

}
