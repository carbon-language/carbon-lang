//===-- MachineInstr.cpp --------------------------------------------------===//
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Value.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/MRegisterInfo.h"
using std::cerr;


// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class TargetInstrInfo.
// 
// FIXME: This should be a property of the target so that more than one target
// at a time can be active...
//
extern const TargetInstrDescriptor *TargetInstrDescriptors;

// Constructor for instructions with variable #operands
MachineInstr::MachineInstr(MachineOpCode OpCode, unsigned  numOperands)
  : opCode(OpCode),
    opCodeFlags(0),
    operands(numOperands, MachineOperand()),
    numImplicitRefs(0)
{
}

/// MachineInstr ctor - This constructor only does a _reserve_ of the operands,
/// not a resize for them.  It is expected that if you use this that you call
/// add* methods below to fill up the operands, instead of the Set methods.
/// Eventually, the "resizing" ctors will be phased out.
///
MachineInstr::MachineInstr(MachineOpCode Opcode, unsigned numOperands,
                           bool XX, bool YY)
  : opCode(Opcode),
    opCodeFlags(0),
    numImplicitRefs(0)
{
  operands.reserve(numOperands);
}

/// MachineInstr ctor - Work exactly the same as the ctor above, except that the
/// MachineInstr is created and added to the end of the specified basic block.
///
MachineInstr::MachineInstr(MachineBasicBlock *MBB, MachineOpCode Opcode,
                           unsigned numOperands)
  : opCode(Opcode),
    opCodeFlags(0),
    numImplicitRefs(0)
{
  assert(MBB && "Cannot use inserting ctor with null basic block!");
  operands.reserve(numOperands);
  MBB->push_back(this);  // Add instruction to end of basic block!
}


// OperandComplete - Return true if it's illegal to add a new operand
bool MachineInstr::OperandsComplete() const
{
  int NumOperands = TargetInstrDescriptors[opCode].numOperands;
  if (NumOperands >= 0 && getNumOperands() >= (unsigned)NumOperands)
    return true;  // Broken: we have all the operands of this instruction!
  return false;
}


// 
// Support for replacing opcode and operands of a MachineInstr in place.
// This only resets the size of the operand vector and initializes it.
// The new operands must be set explicitly later.
// 
void MachineInstr::replace(MachineOpCode Opcode, unsigned numOperands)
{
  assert(getNumImplicitRefs() == 0 &&
         "This is probably broken because implicit refs are going to be lost.");
  opCode = Opcode;
  operands.clear();
  operands.resize(numOperands, MachineOperand());
}

void
MachineInstr::SetMachineOperandVal(unsigned i,
                                   MachineOperand::MachineOperandType opType,
                                   Value* V,
                                   bool isdef,
                                   bool isDefAndUse)
{
  assert(i < operands.size());          // may be explicit or implicit op
  operands[i].opType = opType;
  operands[i].value = V;
  operands[i].regNum = -1;

  if (isDefAndUse)
    operands[i].flags = MachineOperand::DEFUSEFLAG;
  else if (isdef || TargetInstrDescriptors[opCode].resultPos == (int) i)
    operands[i].flags = MachineOperand::DEFONLYFLAG;
  else
    operands[i].flags = 0;
}

void
MachineInstr::SetMachineOperandConst(unsigned i,
				MachineOperand::MachineOperandType operandType,
                                     int64_t intValue)
{
  assert(i < getNumOperands());          // must be explicit op
  assert(TargetInstrDescriptors[opCode].resultPos != (int) i &&
         "immed. constant cannot be defined");

  operands[i].opType = operandType;
  operands[i].value = NULL;
  operands[i].immedVal = intValue;
  operands[i].regNum = -1;
  operands[i].flags = 0;
}

void
MachineInstr::SetMachineOperandReg(unsigned i,
                                   int regNum,
                                   bool isdef) {
  assert(i < getNumOperands());          // must be explicit op

  operands[i].opType = MachineOperand::MO_MachineRegister;
  operands[i].value = NULL;
  operands[i].regNum = regNum;

  if (isdef || TargetInstrDescriptors[opCode].resultPos == (int) i)
    operands[i].flags = MachineOperand::DEFONLYFLAG;
  else
    operands[i].flags = 0;

  insertUsedReg(regNum);
}

void
MachineInstr::SetRegForOperand(unsigned i, int regNum)
{
  assert(i < getNumOperands());          // must be explicit op
  operands[i].setRegForValue(regNum);
  insertUsedReg(regNum);
}

void
MachineInstr::SetRegForImplicitRef(unsigned i, int regNum)
{
  getImplicitOp(i).setRegForValue(regNum);
  insertUsedReg(regNum);
}


// Subsitute all occurrences of Value* oldVal with newVal in all operands
// and all implicit refs.  If defsOnly == true, substitute defs only.
unsigned
MachineInstr::substituteValue(const Value* oldVal, Value* newVal, bool defsOnly)
{
  unsigned numSubst = 0;

  // Subsitute operands
  for (MachineInstr::val_op_iterator O = begin(), E = end(); O != E; ++O)
    if (*O == oldVal)
      if (!defsOnly || !O.isUseOnly())
        {
          O.getMachineOperand().value = newVal;
          ++numSubst;
        }

  // Subsitute implicit refs
  for (unsigned i=0, N=getNumImplicitRefs(); i < N; ++i)
    if (getImplicitRef(i) == oldVal)
      if (!defsOnly || !getImplicitOp(i).opIsUse())
        {
          getImplicitOp(i).value = newVal;
          ++numSubst;
        }

  return numSubst;
}


void
MachineInstr::dump() const 
{
  cerr << "  " << *this;
}

static inline std::ostream&
OutputValue(std::ostream &os, const Value* val)
{
  os << "(val ";
  if (val && val->hasName())
    return os << val->getName() << ")";
  else
    return os << (void*) val << ")";              // print address only
}

static inline void OutputReg(std::ostream &os, unsigned RegNo,
                             const MRegisterInfo *MRI = 0) {
  if (MRI) {
    if (RegNo < MRegisterInfo::FirstVirtualRegister)
      os << "%" << MRI->get(RegNo).Name;
    else
      os << "%reg" << RegNo;
  } else
    os << "%mreg(" << RegNo << ")";
}

static void print(const MachineOperand &MO, std::ostream &OS,
                  const TargetMachine &TM) {
  const MRegisterInfo *MRI = TM.getRegisterInfo();
  bool CloseParen = true;
  if (MO.opHiBits32())
    OS << "%lm(";
  else if (MO.opLoBits32())
    OS << "%lo(";
  else if (MO.opHiBits64())
    OS << "%hh(";
  else if (MO.opLoBits64())
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
      OutputReg(OS, MO.getAllocatedRegNum(), MRI);
    break;
  case MachineOperand::MO_CCRegister:
    OS << "%ccreg";
    OutputValue(OS, MO.getVRegValue());
    if (MO.hasAllocatedReg()) {
      OS << "==";
      OutputReg(OS, MO.getAllocatedRegNum(), MRI);
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
  if (getNumOperands() &&
      (getOperand(0).opIsDefOnly() || getOperand(0).opIsDefAndUse())) {
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
    
    if (mop.opIsDefAndUse())
      OS << "<def&use>";
    else if (mop.opIsDefOnly())
      OS << "<def>";
  }
    
  // code for printing implict references
  if (getNumImplicitRefs()) {
    OS << "\tImplicitRefs: ";
    for(unsigned i = 0, e = getNumImplicitRefs(); i != e; ++i) {
      OS << "\t";
      OutputValue(OS, getImplicitRef(i));
      if (getImplicitOp(i).opIsDefAndUse())
        OS << "<def&use>";
      else if (getImplicitOp(i).opIsDefOnly())
        OS << "<def>";
    }
  }
  
  OS << "\n";
}


std::ostream &operator<<(std::ostream& os, const MachineInstr& MI)
{
  os << TargetInstrDescriptors[MI.opCode].Name;
  
  for (unsigned i=0, N=MI.getNumOperands(); i < N; i++) {
    os << "\t" << MI.getOperand(i);
    if (MI.getOperand(i).opIsDefOnly())
      os << "<d>";
    if (MI.getOperand(i).opIsDefAndUse())
      os << "<d&u>";
  }
  
  // code for printing implict references
  unsigned NumOfImpRefs = MI.getNumImplicitRefs();
  if (NumOfImpRefs > 0) {
    os << "\tImplicit: ";
    for (unsigned z=0; z < NumOfImpRefs; z++) {
      OutputValue(os, MI.getImplicitRef(z)); 
      if (MI.getImplicitOp(z).opIsDefOnly()) os << "<d>";
      if (MI.getImplicitOp(z).opIsDefAndUse()) os << "<d&u>";
      os << "\t";
    }
  }
  
  return os << "\n";
}

std::ostream &operator<<(std::ostream &OS, const MachineOperand &MO)
{
  if (MO.opHiBits32())
    OS << "%lm(";
  else if (MO.opLoBits32())
    OS << "%lo(";
  else if (MO.opHiBits64())
    OS << "%hh(";
  else if (MO.opLoBits64())
    OS << "%hm(";
  
  switch (MO.getType())
    {
    case MachineOperand::MO_VirtualRegister:
      if (MO.hasAllocatedReg())
        OutputReg(OS, MO.getAllocatedRegNum());

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
        OutputReg(OS, MO.getAllocatedRegNum());
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
  
  if (MO.flags &
      (MachineOperand::HIFLAG32 | MachineOperand::LOFLAG32 | 
       MachineOperand::HIFLAG64 | MachineOperand::LOFLAG64))
    OS << ")";
  
  return OS;
}
