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
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/Streams.h"
#include <ostream>
using namespace llvm;

/// MachineInstr ctor - This constructor creates a dummy MachineInstr with
/// TID NULL and no operands.
MachineInstr::MachineInstr()
  : TID(0), NumImplicitOps(0), parent(0) {
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

void MachineInstr::addImplicitDefUseOperands() {
  if (TID->ImplicitDefs)
    for (const unsigned *ImpDefs = TID->ImplicitDefs; *ImpDefs; ++ImpDefs) {
      MachineOperand Op;
      Op.opType = MachineOperand::MO_Register;
      Op.IsDef = true;
      Op.IsImp = true;
      Op.IsKill = false;
      Op.IsDead = false;
      Op.contents.RegNo = *ImpDefs;
      Operands.push_back(Op);
    }
  if (TID->ImplicitUses)
    for (const unsigned *ImpUses = TID->ImplicitUses; *ImpUses; ++ImpUses) {
      MachineOperand Op;
      Op.opType = MachineOperand::MO_Register;
      Op.IsDef = false;
      Op.IsImp = true;
      Op.IsKill = false;
      Op.IsDead = false;
      Op.contents.RegNo = *ImpUses;
      Operands.push_back(Op);
    }
}

/// MachineInstr ctor - This constructor create a MachineInstr and add the
/// implicit operands. It reserves space for number of operands specified by
/// TargetInstrDescriptor or the numOperands if it is not zero. (for
/// instructions with variable number of operands).
MachineInstr::MachineInstr(const TargetInstrDescriptor &tid)
  : TID(&tid), NumImplicitOps(0), parent(0) {
  if (TID->ImplicitDefs)
    for (const unsigned *ImpDefs = TID->ImplicitDefs; *ImpDefs; ++ImpDefs)
      NumImplicitOps++;
  if (TID->ImplicitUses)
    for (const unsigned *ImpUses = TID->ImplicitUses; *ImpUses; ++ImpUses)
      NumImplicitOps++;
  Operands.reserve(NumImplicitOps + TID->numOperands);
  addImplicitDefUseOperands();
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

/// MachineInstr ctor - Work exactly the same as the ctor above, except that the
/// MachineInstr is created and added to the end of the specified basic block.
///
MachineInstr::MachineInstr(MachineBasicBlock *MBB,
                           const TargetInstrDescriptor &tid)
  : TID(&tid), NumImplicitOps(0), parent(0) {
  assert(MBB && "Cannot use inserting ctor with null basic block!");
  if (TID->ImplicitDefs)
    for (const unsigned *ImpDefs = TID->ImplicitDefs; *ImpDefs; ++ImpDefs)
      NumImplicitOps++;
  if (TID->ImplicitUses)
    for (const unsigned *ImpUses = TID->ImplicitUses; *ImpUses; ++ImpUses)
      NumImplicitOps++;
  Operands.reserve(NumImplicitOps + TID->numOperands);
  addImplicitDefUseOperands();
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
  MBB->push_back(this);  // Add instruction to end of basic block!
}

/// MachineInstr ctor - Copies MachineInstr arg exactly
///
MachineInstr::MachineInstr(const MachineInstr &MI) {
  TID = MI.getInstrDescriptor();
  NumImplicitOps = MI.NumImplicitOps;
  Operands.reserve(MI.getNumOperands());

  // Add operands
  for (unsigned i = 0; i != MI.getNumOperands(); ++i)
    Operands.push_back(MI.getOperand(i));

  // Set parent, next, and prev to null
  parent = 0;
  prev = 0;
  next = 0;
}


MachineInstr::~MachineInstr() {
  LeakDetector::removeGarbageObject(this);
}

/// getOpcode - Returns the opcode of this MachineInstr.
///
const int MachineInstr::getOpcode() const {
  return TID->Opcode;
}

/// removeFromParent - This method unlinks 'this' from the containing basic
/// block, and returns it, but does not delete it.
MachineInstr *MachineInstr::removeFromParent() {
  assert(getParent() && "Not embedded in a basic block!");
  getParent()->remove(this);
  return this;
}


/// OperandComplete - Return true if it's illegal to add a new operand
///
bool MachineInstr::OperandsComplete() const {
  unsigned short NumOperands = TID->numOperands;
  if ((TID->Flags & M_VARIABLE_OPS) == 0 &&
      getNumOperands()-NumImplicitOps >= NumOperands)
    return true;  // Broken: we have all the operands of this instruction!
  return false;
}

/// getNumExplicitOperands - Returns the number of non-implicit operands.
///
unsigned MachineInstr::getNumExplicitOperands() const {
  unsigned NumOperands = TID->numOperands;
  if ((TID->Flags & M_VARIABLE_OPS) == 0)
    return NumOperands;

  for (unsigned e = getNumOperands(); NumOperands != e; ++NumOperands) {
    const MachineOperand &MO = getOperand(NumOperands);
    if (!MO.isRegister() || !MO.isImplicit())
      NumOperands++;
  }
  return NumOperands;
}

/// isIdenticalTo - Return true if this operand is identical to the specified
/// operand.
bool MachineOperand::isIdenticalTo(const MachineOperand &Other) const {
  if (getType() != Other.getType()) return false;
  
  switch (getType()) {
  default: assert(0 && "Unrecognized operand type");
  case MachineOperand::MO_Register:
    return getReg() == Other.getReg() && isDef() == Other.isDef();
  case MachineOperand::MO_Immediate:
    return getImm() == Other.getImm();
  case MachineOperand::MO_MachineBasicBlock:
    return getMBB() == Other.getMBB();
  case MachineOperand::MO_FrameIndex:
    return getFrameIndex() == Other.getFrameIndex();
  case MachineOperand::MO_ConstantPoolIndex:
    return getConstantPoolIndex() == Other.getConstantPoolIndex() &&
           getOffset() == Other.getOffset();
  case MachineOperand::MO_JumpTableIndex:
    return getJumpTableIndex() == Other.getJumpTableIndex();
  case MachineOperand::MO_GlobalAddress:
    return getGlobal() == Other.getGlobal() && getOffset() == Other.getOffset();
  case MachineOperand::MO_ExternalSymbol:
    return !strcmp(getSymbolName(), Other.getSymbolName()) &&
           getOffset() == Other.getOffset();
  }
}

/// findRegisterUseOperandIdx() - Returns the MachineOperand that is a use of
/// the specific register or -1 if it is not found. It further tightening
/// the search criteria to a use that kills the register if isKill is true.
int MachineInstr::findRegisterUseOperandIdx(unsigned Reg, bool isKill) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = getOperand(i);
    if (MO.isReg() && MO.isUse() && MO.getReg() == Reg)
      if (!isKill || MO.isKill())
        return i;
  }
  return -1;
}
  
/// findRegisterDefOperand() - Returns the MachineOperand that is a def of
/// the specific register or NULL if it is not found.
MachineOperand *MachineInstr::findRegisterDefOperand(unsigned Reg) {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    MachineOperand &MO = getOperand(i);
    if (MO.isReg() && MO.isDef() && MO.getReg() == Reg)
      return &MO;
  }
  return NULL;
}

/// findFirstPredOperandIdx() - Find the index of the first operand in the
/// operand list that is used to represent the predicate. It returns -1 if
/// none is found.
int MachineInstr::findFirstPredOperandIdx() const {
  const TargetInstrDescriptor *TID = getInstrDescriptor();
  if (TID->Flags & M_PREDICABLE) {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if ((TID->OpInfo[i].Flags & M_PREDICATE_OPERAND))
        return i;
  }

  return -1;
}
  
/// copyKillDeadInfo - Copies kill / dead operand properties from MI.
///
void MachineInstr::copyKillDeadInfo(const MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || (!MO.isKill() && !MO.isDead()))
      continue;
    for (unsigned j = 0, ee = getNumOperands(); j != ee; ++j) {
      MachineOperand &MOp = getOperand(j);
      if (!MOp.isIdenticalTo(MO))
        continue;
      if (MO.isKill())
        MOp.setIsKill();
      else
        MOp.setIsDead();
      break;
    }
  }
}

/// copyPredicates - Copies predicate operand(s) from MI.
void MachineInstr::copyPredicates(const MachineInstr *MI) {
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  if (TID->Flags & M_PREDICABLE) {
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      if ((TID->OpInfo[i].Flags & M_PREDICATE_OPERAND)) {
        const MachineOperand &MO = MI->getOperand(i);
        // Predicated operands must be last operands.
        if (MO.isReg())
          addRegOperand(MO.getReg(), false);
        else {
          addImmOperand(MO.getImm());
        }
      }
    }
  }
}

void MachineInstr::dump() const {
  cerr << "  " << *this;
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
                  const TargetMachine *TM) {
  const MRegisterInfo *MRI = 0;

  if (TM) MRI = TM->getRegisterInfo();

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    OutputReg(OS, MO.getReg(), MRI);
    break;
  case MachineOperand::MO_Immediate:
    OS << MO.getImmedValue();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    OS << "mbb<"
       << ((Value*)MO.getMachineBasicBlock()->getBasicBlock())->getName()
       << "," << (void*)MO.getMachineBasicBlock() << ">";
    break;
  case MachineOperand::MO_FrameIndex:
    OS << "<fi#" << MO.getFrameIndex() << ">";
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "<cp#" << MO.getConstantPoolIndex() << ">";
    break;
  case MachineOperand::MO_JumpTableIndex:
    OS << "<jt#" << MO.getJumpTableIndex() << ">";
    break;
  case MachineOperand::MO_GlobalAddress:
    OS << "<ga:" << ((Value*)MO.getGlobal())->getName();
    if (MO.getOffset()) OS << "+" << MO.getOffset();
    OS << ">";
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << "<es:" << MO.getSymbolName();
    if (MO.getOffset()) OS << "+" << MO.getOffset();
    OS << ">";
    break;
  default:
    assert(0 && "Unrecognized operand type");
  }
}

void MachineInstr::print(std::ostream &OS, const TargetMachine *TM) const {
  unsigned StartOp = 0;

   // Specialize printing if op#0 is definition
  if (getNumOperands() && getOperand(0).isReg() && getOperand(0).isDef()) {
    ::print(getOperand(0), OS, TM);
    if (getOperand(0).isDead())
      OS << "<dead>";
    OS << " = ";
    ++StartOp;   // Don't print this operand again!
  }

  if (TID)
    OS << TID->Name;

  for (unsigned i = StartOp, e = getNumOperands(); i != e; ++i) {
    const MachineOperand& mop = getOperand(i);
    if (i != StartOp)
      OS << ",";
    OS << " ";
    ::print(mop, OS, TM);

    if (mop.isReg()) {
      if (mop.isDef() || mop.isKill() || mop.isDead() || mop.isImplicit()) {
        OS << "<";
        bool NeedComma = false;
        if (mop.isImplicit()) {
          OS << (mop.isDef() ? "imp-def" : "imp-use");
          NeedComma = true;
        } else if (mop.isDef()) {
          OS << "def";
          NeedComma = true;
        }
        if (mop.isKill() || mop.isDead()) {
          if (NeedComma)
            OS << ",";
          if (mop.isKill())
            OS << "kill";
          if (mop.isDead())
            OS << "dead";
        }
        OS << ">";
      }
    }
  }

  OS << "\n";
}

void MachineInstr::print(std::ostream &os) const {
  // If the instruction is embedded into a basic block, we can find the target
  // info for the instruction.
  if (const MachineBasicBlock *MBB = getParent()) {
    const MachineFunction *MF = MBB->getParent();
    if (MF)
      print(os, &MF->getTarget());
    else
      print(os, 0);
  }

  // Otherwise, print it out in the "raw" format without symbolic register names
  // and such.
  os << getInstrDescriptor()->Name;

  for (unsigned i = 0, N = getNumOperands(); i < N; i++) {
    os << "\t" << getOperand(i);
    if (getOperand(i).isReg() && getOperand(i).isDef())
      os << "<d>";
  }

  os << "\n";
}

void MachineOperand::print(std::ostream &OS) const {
  switch (getType()) {
  case MO_Register:
    OutputReg(OS, getReg());
    break;
  case MO_Immediate:
    OS << (long)getImmedValue();
    break;
  case MO_MachineBasicBlock:
    OS << "<mbb:"
       << ((Value*)getMachineBasicBlock()->getBasicBlock())->getName()
       << "@" << (void*)getMachineBasicBlock() << ">";
    break;
  case MO_FrameIndex:
    OS << "<fi#" << getFrameIndex() << ">";
    break;
  case MO_ConstantPoolIndex:
    OS << "<cp#" << getConstantPoolIndex() << ">";
    break;
  case MO_JumpTableIndex:
    OS << "<jt#" << getJumpTableIndex() << ">";
    break;
  case MO_GlobalAddress:
    OS << "<ga:" << ((Value*)getGlobal())->getName() << ">";
    break;
  case MO_ExternalSymbol:
    OS << "<es:" << getSymbolName() << ">";
    break;
  default:
    assert(0 && "Unrecognized operand type");
    break;
  }
}

