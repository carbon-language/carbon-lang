//===-- Instruction.cpp - Implement the Instruction class -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instruction.h"
#include "llvm/Type.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

Instruction::Instruction(Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         Instruction *InsertBefore)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(0) {
  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // If requested, insert this instruction into a basic block...
  if (InsertBefore) {
    assert(InsertBefore->getParent() &&
           "Instruction to insert before is not in a basic block!");
    InsertBefore->getParent()->getInstList().insert(InsertBefore, this);
  }
}

Instruction::Instruction(Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         BasicBlock *InsertAtEnd)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(0) {
  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // append this instruction into the basic block
  assert(InsertAtEnd && "Basic block to append to may not be NULL!");
  InsertAtEnd->getInstList().push_back(this);
}


// Out of line virtual method, so the vtable, etc has a home.
Instruction::~Instruction() {
  assert(Parent == 0 && "Instruction still linked in the program!");
  if (hasMetadataHashEntry())
    clearMetadataHashEntries();
}


void Instruction::setParent(BasicBlock *P) {
  if (getParent()) {
    if (!P) LeakDetector::addGarbageObject(this);
  } else {
    if (P) LeakDetector::removeGarbageObject(this);
  }

  Parent = P;
}

void Instruction::removeFromParent() {
  getParent()->getInstList().remove(this);
}

void Instruction::eraseFromParent() {
  getParent()->getInstList().erase(this);
}

/// insertBefore - Insert an unlinked instructions into a basic block
/// immediately before the specified instruction.
void Instruction::insertBefore(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insert(InsertPos, this);
}

/// insertAfter - Insert an unlinked instructions into a basic block
/// immediately after the specified instruction.
void Instruction::insertAfter(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insertAfter(InsertPos, this);
}

/// moveBefore - Unlink this instruction from its current basic block and
/// insert it into the basic block that MovePos lives in, right before
/// MovePos.
void Instruction::moveBefore(Instruction *MovePos) {
  MovePos->getParent()->getInstList().splice(MovePos,getParent()->getInstList(),
                                             this);
}


const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case IndirectBr: return "indirectbr";
  case Invoke: return "invoke";
  case Resume: return "resume";
  case Unwind: return "unwind";
  case Unreachable: return "unreachable";

  // Standard binary operators...
  case Add: return "add";
  case FAdd: return "fadd";
  case Sub: return "sub";
  case FSub: return "fsub";
  case Mul: return "mul";
  case FMul: return "fmul";
  case UDiv: return "udiv";
  case SDiv: return "sdiv";
  case FDiv: return "fdiv";
  case URem: return "urem";
  case SRem: return "srem";
  case FRem: return "frem";

  // Logical operators...
  case And: return "and";
  case Or : return "or";
  case Xor: return "xor";

  // Memory instructions...
  case Alloca:        return "alloca";
  case Load:          return "load";
  case Store:         return "store";
  case AtomicCmpXchg: return "cmpxchg";
  case AtomicRMW:     return "atomicrmw";
  case Fence:         return "fence";
  case GetElementPtr: return "getelementptr";

  // Convert instructions...
  case Trunc:     return "trunc";
  case ZExt:      return "zext";
  case SExt:      return "sext";
  case FPTrunc:   return "fptrunc";
  case FPExt:     return "fpext";
  case FPToUI:    return "fptoui";
  case FPToSI:    return "fptosi";
  case UIToFP:    return "uitofp";
  case SIToFP:    return "sitofp";
  case IntToPtr:  return "inttoptr";
  case PtrToInt:  return "ptrtoint";
  case BitCast:   return "bitcast";

  // Other instructions...
  case ICmp:           return "icmp";
  case FCmp:           return "fcmp";
  case PHI:            return "phi";
  case Select:         return "select";
  case Call:           return "call";
  case Shl:            return "shl";
  case LShr:           return "lshr";
  case AShr:           return "ashr";
  case VAArg:          return "va_arg";
  case ExtractElement: return "extractelement";
  case InsertElement:  return "insertelement";
  case ShuffleVector:  return "shufflevector";
  case ExtractValue:   return "extractvalue";
  case InsertValue:    return "insertvalue";
  case LandingPad:     return "landingpad";

  default: return "<Invalid operator> ";
  }

  return 0;
}

/// isIdenticalTo - Return true if the specified instruction is exactly
/// identical to the current one.  This means that all operands match and any
/// extra information (e.g. load is volatile) agree.
bool Instruction::isIdenticalTo(const Instruction *I) const {
  return isIdenticalToWhenDefined(I) &&
         SubclassOptionalData == I->SubclassOptionalData;
}

/// isIdenticalToWhenDefined - This is like isIdenticalTo, except that it
/// ignores the SubclassOptionalData flags, which specify conditions
/// under which the instruction's result is undefined.
bool Instruction::isIdenticalToWhenDefined(const Instruction *I) const {
  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      getType() != I->getType())
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i) != I->getOperand(i))
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(this))
    return LI->isVolatile() == cast<LoadInst>(I)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I)->getAlignment() &&
           LI->getOrdering() == cast<LoadInst>(I)->getOrdering() &&
           LI->getSynchScope() == cast<LoadInst>(I)->getSynchScope();
  if (const StoreInst *SI = dyn_cast<StoreInst>(this))
    return SI->isVolatile() == cast<StoreInst>(I)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I)->getAlignment() &&
           SI->getOrdering() == cast<StoreInst>(I)->getOrdering() &&
           SI->getSynchScope() == cast<StoreInst>(I)->getSynchScope();
  if (const CmpInst *CI = dyn_cast<CmpInst>(this))
    return CI->getPredicate() == cast<CmpInst>(I)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return CI->isTailCall() == cast<CallInst>(I)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I)->getCallingConv() &&
           CI->getAttributes() == cast<CallInst>(I)->getAttributes();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(this))
    return CI->getCallingConv() == cast<InvokeInst>(I)->getCallingConv() &&
           CI->getAttributes() == cast<InvokeInst>(I)->getAttributes();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(this))
    return IVI->getIndices() == cast<InsertValueInst>(I)->getIndices();
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(this))
    return EVI->getIndices() == cast<ExtractValueInst>(I)->getIndices();
  if (const FenceInst *FI = dyn_cast<FenceInst>(this))
    return FI->getOrdering() == cast<FenceInst>(FI)->getOrdering() &&
           FI->getSynchScope() == cast<FenceInst>(FI)->getSynchScope();
  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(this))
    return CXI->isVolatile() == cast<AtomicCmpXchgInst>(I)->isVolatile() &&
           CXI->getOrdering() == cast<AtomicCmpXchgInst>(I)->getOrdering() &&
           CXI->getSynchScope() == cast<AtomicCmpXchgInst>(I)->getSynchScope();
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(this))
    return RMWI->getOperation() == cast<AtomicRMWInst>(I)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I)->getOrdering() &&
           RMWI->getSynchScope() == cast<AtomicRMWInst>(I)->getSynchScope();

  return true;
}

// isSameOperationAs
// This should be kept in sync with isEquivalentOperation in
// lib/Transforms/IPO/MergeFunctions.cpp.
bool Instruction::isSameOperationAs(const Instruction *I) const {
  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      getType() != I->getType())
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i)->getType() != I->getOperand(i)->getType())
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(this))
    return LI->isVolatile() == cast<LoadInst>(I)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I)->getAlignment() &&
           LI->getOrdering() == cast<LoadInst>(I)->getOrdering() &&
           LI->getSynchScope() == cast<LoadInst>(I)->getSynchScope();
  if (const StoreInst *SI = dyn_cast<StoreInst>(this))
    return SI->isVolatile() == cast<StoreInst>(I)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I)->getAlignment() &&
           SI->getOrdering() == cast<StoreInst>(I)->getOrdering() &&
           SI->getSynchScope() == cast<StoreInst>(I)->getSynchScope();
  if (const CmpInst *CI = dyn_cast<CmpInst>(this))
    return CI->getPredicate() == cast<CmpInst>(I)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return CI->isTailCall() == cast<CallInst>(I)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I)->getCallingConv() &&
           CI->getAttributes() == cast<CallInst>(I)->getAttributes();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(this))
    return CI->getCallingConv() == cast<InvokeInst>(I)->getCallingConv() &&
           CI->getAttributes() ==
             cast<InvokeInst>(I)->getAttributes();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(this))
    return IVI->getIndices() == cast<InsertValueInst>(I)->getIndices();
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(this))
    return EVI->getIndices() == cast<ExtractValueInst>(I)->getIndices();
  if (const FenceInst *FI = dyn_cast<FenceInst>(this))
    return FI->getOrdering() == cast<FenceInst>(I)->getOrdering() &&
           FI->getSynchScope() == cast<FenceInst>(I)->getSynchScope();
  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(this))
    return CXI->isVolatile() == cast<AtomicCmpXchgInst>(I)->isVolatile() &&
           CXI->getOrdering() == cast<AtomicCmpXchgInst>(I)->getOrdering() &&
           CXI->getSynchScope() == cast<AtomicCmpXchgInst>(I)->getSynchScope();
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(this))
    return RMWI->getOperation() == cast<AtomicRMWInst>(I)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I)->getOrdering() &&
           RMWI->getSynchScope() == cast<AtomicRMWInst>(I)->getSynchScope();

  return true;
}

/// isUsedOutsideOfBlock - Return true if there are any uses of I outside of the
/// specified block.  Note that PHI nodes are considered to evaluate their
/// operands in the corresponding predecessor block.
bool Instruction::isUsedOutsideOfBlock(const BasicBlock *BB) const {
  for (const_use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    // PHI nodes uses values in the corresponding predecessor block.  For other
    // instructions, just check to see whether the parent of the use matches up.
    const User *U = *UI;
    const PHINode *PN = dyn_cast<PHINode>(U);
    if (PN == 0) {
      if (cast<Instruction>(U)->getParent() != BB)
        return true;
      continue;
    }

    if (PN->getIncomingBlock(UI) != BB)
      return true;
  }
  return false;
}

/// mayReadFromMemory - Return true if this instruction may read memory.
///
bool Instruction::mayReadFromMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::VAArg:
  case Instruction::Load:
  case Instruction::Fence: // FIXME: refine definition of mayReadFromMemory
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
    return true;
  case Instruction::Call:
    return !cast<CallInst>(this)->doesNotAccessMemory();
  case Instruction::Invoke:
    return !cast<InvokeInst>(this)->doesNotAccessMemory();
  case Instruction::Store:
    return !cast<StoreInst>(this)->isUnordered();
  }
}

/// mayWriteToMemory - Return true if this instruction may modify memory.
///
bool Instruction::mayWriteToMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::Fence: // FIXME: refine definition of mayWriteToMemory
  case Instruction::Store:
  case Instruction::VAArg:
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
    return true;
  case Instruction::Call:
    return !cast<CallInst>(this)->onlyReadsMemory();
  case Instruction::Invoke:
    return !cast<InvokeInst>(this)->onlyReadsMemory();
  case Instruction::Load:
    return !cast<LoadInst>(this)->isUnordered();
  }
}

/// mayThrow - Return true if this instruction may throw an exception.
///
bool Instruction::mayThrow() const {
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return !CI->doesNotThrow();
  return isa<ResumeInst>(this);
}

/// isAssociative - Return true if the instruction is associative:
///
///   Associative operators satisfy:  x op (y op z) === (x op y) op z
///
/// In LLVM, the Add, Mul, And, Or, and Xor operators are associative.
///
bool Instruction::isAssociative(unsigned Opcode) {
  return Opcode == And || Opcode == Or || Opcode == Xor ||
         Opcode == Add || Opcode == Mul;
}

/// isCommutative - Return true if the instruction is commutative:
///
///   Commutative operators satisfy: (x op y) === (y op x)
///
/// In LLVM, these are the associative operators, plus SetEQ and SetNE, when
/// applied to any type.
///
bool Instruction::isCommutative(unsigned op) {
  switch (op) {
  case Add:
  case FAdd:
  case Mul:
  case FMul:
  case And:
  case Or:
  case Xor:
    return true;
  default:
    return false;
  }
}

bool Instruction::isSafeToSpeculativelyExecute() const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (Constant *C = dyn_cast<Constant>(getOperand(i)))
      if (C->canTrap())
        return false;

  switch (getOpcode()) {
  default:
    return true;
  case UDiv:
  case URem: {
    // x / y is undefined if y == 0, but calcuations like x / 3 are safe.
    ConstantInt *Op = dyn_cast<ConstantInt>(getOperand(1));
    return Op && !Op->isNullValue();
  }
  case SDiv:
  case SRem: {
    // x / y is undefined if y == 0, and might be undefined if y == -1,
    // but calcuations like x / 3 are safe.
    ConstantInt *Op = dyn_cast<ConstantInt>(getOperand(1));
    return Op && !Op->isNullValue() && !Op->isAllOnesValue();
  }
  case Load: {
    const LoadInst *LI = cast<LoadInst>(this);
    if (!LI->isUnordered())
      return false;
    return LI->getPointerOperand()->isDereferenceablePointer();
  }
  case Call:
    return false; // The called function could have undefined behavior or
                  // side-effects.
                  // FIXME: We should special-case some intrinsics (bswap,
                  // overflow-checking arithmetic, etc.)
  case VAArg:
  case Alloca:
  case Invoke:
  case PHI:
  case Store:
  case Ret:
  case Br:
  case IndirectBr:
  case Switch:
  case Unwind:
  case Unreachable:
  case Fence:
    return false; // Misc instructions which have effects
  }
}

Instruction *Instruction::clone() const {
  Instruction *New = clone_impl();
  New->SubclassOptionalData = SubclassOptionalData;
  if (!hasMetadata())
    return New;
  
  // Otherwise, enumerate and copy over metadata from the old instruction to the
  // new one.
  SmallVector<std::pair<unsigned, MDNode*>, 4> TheMDs;
  getAllMetadataOtherThanDebugLoc(TheMDs);
  for (unsigned i = 0, e = TheMDs.size(); i != e; ++i)
    New->setMetadata(TheMDs[i].first, TheMDs[i].second);
  
  New->setDebugLoc(getDebugLoc());
  return New;
}
