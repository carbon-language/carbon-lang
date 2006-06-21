//===-- Instructions.cpp - Implement the LLVM instructions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements all of the non-inline methods for the LLVM instruction
// classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CallSite.h"
using namespace llvm;

unsigned CallSite::getCallingConv() const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->getCallingConv();
  else
    return cast<InvokeInst>(I)->getCallingConv();
}
void CallSite::setCallingConv(unsigned CC) {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    CI->setCallingConv(CC);
  else
    cast<InvokeInst>(I)->setCallingConv(CC);
}




//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

TerminatorInst::TerminatorInst(Instruction::TermOps iType,
                               Use *Ops, unsigned NumOps, Instruction *IB)
  : Instruction(Type::VoidTy, iType, Ops, NumOps, "", IB) {
}

TerminatorInst::TerminatorInst(Instruction::TermOps iType,
                               Use *Ops, unsigned NumOps, BasicBlock *IAE)
  : Instruction(Type::VoidTy, iType, Ops, NumOps, "", IAE) {
}

// Out of line virtual method, so the vtable, etc has a home.
TerminatorInst::~TerminatorInst() {
}

// Out of line virtual method, so the vtable, etc has a home.
UnaryInstruction::~UnaryInstruction() {
}


//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

PHINode::PHINode(const PHINode &PN)
  : Instruction(PN.getType(), Instruction::PHI,
                new Use[PN.getNumOperands()], PN.getNumOperands()),
    ReservedSpace(PN.getNumOperands()) {
  Use *OL = OperandList;
  for (unsigned i = 0, e = PN.getNumOperands(); i != e; i+=2) {
    OL[i].init(PN.getOperand(i), this);
    OL[i+1].init(PN.getOperand(i+1), this);
  }
}

PHINode::~PHINode() {
  delete [] OperandList;
}

// removeIncomingValue - Remove an incoming value.  This is useful if a
// predecessor basic block is deleted.
Value *PHINode::removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty) {
  unsigned NumOps = getNumOperands();
  Use *OL = OperandList;
  assert(Idx*2 < NumOps && "BB not in PHI node!");
  Value *Removed = OL[Idx*2];

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // client might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  for (unsigned i = (Idx+1)*2; i != NumOps; i += 2) {
    OL[i-2] = OL[i];
    OL[i-2+1] = OL[i+1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(0);
  OL[NumOps-2+1].set(0);
  NumOperands = NumOps-2;

  // If the PHI node is dead, because it has zero entries, nuke it now.
  if (NumOps == 2 && DeletePHIIfEmpty) {
    // If anyone is using this PHI, make them use a dummy value instead...
    replaceAllUsesWith(UndefValue::get(getType()));
    eraseFromParent();
  }
  return Removed;
}

/// resizeOperands - resize operands - This adjusts the length of the operands
/// list according to the following behavior:
///   1. If NumOps == 0, grow the operand list in response to a push_back style
///      of operation.  This grows the number of ops by 1.5 times.
///   2. If NumOps > NumOperands, reserve space for NumOps operands.
///   3. If NumOps == NumOperands, trim the reserved space.
///
void PHINode::resizeOperands(unsigned NumOps) {
  if (NumOps == 0) {
    NumOps = (getNumOperands())*3/2;
    if (NumOps < 4) NumOps = 4;      // 4 op PHI nodes are VERY common.
  } else if (NumOps*2 > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *NewOps = new Use[NumOps];
  Use *OldOps = OperandList;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      NewOps[i].init(OldOps[i], this);
      OldOps[i].set(0);
  }
  delete [] OldOps;
  OperandList = NewOps;
}

/// hasConstantValue - If the specified PHI node always merges together the same
/// value, return the value, otherwise return null.
///
Value *PHINode::hasConstantValue(bool AllowNonDominatingInstruction) const {
  // If the PHI node only has one incoming value, eliminate the PHI node...
  if (getNumIncomingValues() == 1)
    if (getIncomingValue(0) != this)   // not  X = phi X
      return getIncomingValue(0);
    else
      return UndefValue::get(getType());  // Self cycle is dead.
      
  // Otherwise if all of the incoming values are the same for the PHI, replace
  // the PHI node with the incoming value.
  //
  Value *InVal = 0;
  bool HasUndefInput = false;
  for (unsigned i = 0, e = getNumIncomingValues(); i != e; ++i)
    if (isa<UndefValue>(getIncomingValue(i)))
      HasUndefInput = true;
    else if (getIncomingValue(i) != this)  // Not the PHI node itself...
      if (InVal && getIncomingValue(i) != InVal)
        return 0;  // Not the same, bail out.
      else
        InVal = getIncomingValue(i);
  
  // The only case that could cause InVal to be null is if we have a PHI node
  // that only has entries for itself.  In this case, there is no entry into the
  // loop, so kill the PHI.
  //
  if (InVal == 0) InVal = UndefValue::get(getType());
  
  // If we have a PHI node like phi(X, undef, X), where X is defined by some
  // instruction, we cannot always return X as the result of the PHI node.  Only
  // do this if X is not an instruction (thus it must dominate the PHI block),
  // or if the client is prepared to deal with this possibility.
  if (HasUndefInput && !AllowNonDominatingInstruction)
    if (Instruction *IV = dyn_cast<Instruction>(InVal))
      // If it's in the entry block, it dominates everything.
      if (IV->getParent() != &IV->getParent()->getParent()->front() ||
          isa<InvokeInst>(IV))
        return 0;   // Cannot guarantee that InVal dominates this PHINode.

  // All of the incoming values are the same, return the value now.
  return InVal;
}


//===----------------------------------------------------------------------===//
//                        CallInst Implementation
//===----------------------------------------------------------------------===//

CallInst::~CallInst() {
  delete [] OperandList;
}

void CallInst::init(Value *Func, const std::vector<Value*> &Params) {
  NumOperands = Params.size()+1;
  Use *OL = OperandList = new Use[Params.size()+1];
  OL[0].init(Func, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((Params.size() == FTy->getNumParams() ||
          (FTy->isVarArg() && Params.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature!");
  for (unsigned i = 0, e = Params.size(); i != e; ++i) {
    assert((i >= FTy->getNumParams() || 
            FTy->getParamType(i) == Params[i]->getType()) &&
           "Calling a function with a bad signature!");
    OL[i+1].init(Params[i], this);
  }
}

void CallInst::init(Value *Func, Value *Actual1, Value *Actual2) {
  NumOperands = 3;
  Use *OL = OperandList = new Use[3];
  OL[0].init(Func, this);
  OL[1].init(Actual1, this);
  OL[2].init(Actual2, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((FTy->getNumParams() == 2 ||
          (FTy->isVarArg() && FTy->getNumParams() < 2)) &&
         "Calling a function with bad signature");
  assert((0 >= FTy->getNumParams() || 
          FTy->getParamType(0) == Actual1->getType()) &&
         "Calling a function with a bad signature!");
  assert((1 >= FTy->getNumParams() || 
          FTy->getParamType(1) == Actual2->getType()) &&
         "Calling a function with a bad signature!");
}

void CallInst::init(Value *Func, Value *Actual) {
  NumOperands = 2;
  Use *OL = OperandList = new Use[2];
  OL[0].init(Func, this);
  OL[1].init(Actual, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((FTy->getNumParams() == 1 ||
          (FTy->isVarArg() && FTy->getNumParams() == 0)) &&
         "Calling a function with bad signature");
  assert((0 == FTy->getNumParams() || 
          FTy->getParamType(0) == Actual->getType()) &&
         "Calling a function with a bad signature!");
}

void CallInst::init(Value *Func) {
  NumOperands = 1;
  Use *OL = OperandList = new Use[1];
  OL[0].init(Func, this);

  const FunctionType *MTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert(MTy->getNumParams() == 0 && "Calling a function with bad signature");
}

CallInst::CallInst(Value *Func, const std::vector<Value*> &Params,
                   const std::string &Name, Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                 ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertBefore) {
  init(Func, Params);
}

CallInst::CallInst(Value *Func, const std::vector<Value*> &Params,
                   const std::string &Name, BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                 ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertAtEnd) {
  init(Func, Params);
}

CallInst::CallInst(Value *Func, Value *Actual1, Value *Actual2,
                   const std::string &Name, Instruction  *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertBefore) {
  init(Func, Actual1, Actual2);
}

CallInst::CallInst(Value *Func, Value *Actual1, Value *Actual2,
                   const std::string &Name, BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertAtEnd) {
  init(Func, Actual1, Actual2);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   Instruction  *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertBefore) {
  init(Func, Actual);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertAtEnd) {
  init(Func, Actual);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertBefore) {
  init(Func);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, Name, InsertAtEnd) {
  init(Func);
}

CallInst::CallInst(const CallInst &CI)
  : Instruction(CI.getType(), Instruction::Call, new Use[CI.getNumOperands()],
                CI.getNumOperands()) {
  SubclassData = CI.SubclassData;
  Use *OL = OperandList;
  Use *InOL = CI.OperandList;
  for (unsigned i = 0, e = CI.getNumOperands(); i != e; ++i)
    OL[i].init(InOL[i], this);
}


//===----------------------------------------------------------------------===//
//                        InvokeInst Implementation
//===----------------------------------------------------------------------===//

InvokeInst::~InvokeInst() {
  delete [] OperandList;
}

void InvokeInst::init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
                      const std::vector<Value*> &Params) {
  NumOperands = 3+Params.size();
  Use *OL = OperandList = new Use[3+Params.size()];
  OL[0].init(Fn, this);
  OL[1].init(IfNormal, this);
  OL[2].init(IfException, this);
  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType());

  assert((Params.size() == FTy->getNumParams()) ||
         (FTy->isVarArg() && Params.size() > FTy->getNumParams()) &&
         "Calling a function with bad signature");

  for (unsigned i = 0, e = Params.size(); i != e; i++) {
    assert((i >= FTy->getNumParams() || 
            FTy->getParamType(i) == Params[i]->getType()) &&
           "Invoking a function with a bad signature!");
    
    OL[i+3].init(Params[i], this);
  }
}

InvokeInst::InvokeInst(Value *Fn, BasicBlock *IfNormal,
                       BasicBlock *IfException,
                       const std::vector<Value*> &Params,
                       const std::string &Name, Instruction *InsertBefore)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Fn->getType())
                                    ->getElementType())->getReturnType(),
                   Instruction::Invoke, 0, 0, Name, InsertBefore) {
  init(Fn, IfNormal, IfException, Params);
}

InvokeInst::InvokeInst(Value *Fn, BasicBlock *IfNormal,
                       BasicBlock *IfException,
                       const std::vector<Value*> &Params,
                       const std::string &Name, BasicBlock *InsertAtEnd)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Fn->getType())
                                    ->getElementType())->getReturnType(),
                   Instruction::Invoke, 0, 0, Name, InsertAtEnd) {
  init(Fn, IfNormal, IfException, Params);
}

InvokeInst::InvokeInst(const InvokeInst &II)
  : TerminatorInst(II.getType(), Instruction::Invoke,
                   new Use[II.getNumOperands()], II.getNumOperands()) {
  SubclassData = II.SubclassData;
  Use *OL = OperandList, *InOL = II.OperandList;
  for (unsigned i = 0, e = II.getNumOperands(); i != e; ++i)
    OL[i].init(InOL[i], this);
}

BasicBlock *InvokeInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned InvokeInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void InvokeInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  return setSuccessor(idx, B);
}


//===----------------------------------------------------------------------===//
//                        ReturnInst Implementation
//===----------------------------------------------------------------------===//

void ReturnInst::init(Value *retVal) {
  if (retVal && retVal->getType() != Type::VoidTy) {
    assert(!isa<BasicBlock>(retVal) &&
           "Cannot return basic block.  Probably using the incorrect ctor");
    NumOperands = 1;
    RetVal.init(retVal, this);
  }
}

unsigned ReturnInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

// Out-of-line ReturnInst method, put here so the C++ compiler can choose to
// emit the vtable for the class in this translation unit.
void ReturnInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "ReturnInst has no successors!");
}

BasicBlock *ReturnInst::getSuccessorV(unsigned idx) const {
  assert(0 && "ReturnInst has no successors!");
  abort();
  return 0;
}


//===----------------------------------------------------------------------===//
//                        UnwindInst Implementation
//===----------------------------------------------------------------------===//

unsigned UnwindInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

void UnwindInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnwindInst has no successors!");
}

BasicBlock *UnwindInst::getSuccessorV(unsigned idx) const {
  assert(0 && "UnwindInst has no successors!");
  abort();
  return 0;
}

//===----------------------------------------------------------------------===//
//                      UnreachableInst Implementation
//===----------------------------------------------------------------------===//

unsigned UnreachableInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

void UnreachableInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnwindInst has no successors!");
}

BasicBlock *UnreachableInst::getSuccessorV(unsigned idx) const {
  assert(0 && "UnwindInst has no successors!");
  abort();
  return 0;
}

//===----------------------------------------------------------------------===//
//                        BranchInst Implementation
//===----------------------------------------------------------------------===//

void BranchInst::AssertOK() {
  if (isConditional())
    assert(getCondition()->getType() == Type::BoolTy &&
           "May only branch on boolean predicates!");
}

BranchInst::BranchInst(const BranchInst &BI) :
  TerminatorInst(Instruction::Br, Ops, BI.getNumOperands()) {
  OperandList[0].init(BI.getOperand(0), this);
  if (BI.getNumOperands() != 1) {
    assert(BI.getNumOperands() == 3 && "BR can have 1 or 3 operands!");
    OperandList[1].init(BI.getOperand(1), this);
    OperandList[2].init(BI.getOperand(2), this);
  }
}

BasicBlock *BranchInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned BranchInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void BranchInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  setSuccessor(idx, B);
}


//===----------------------------------------------------------------------===//
//                        AllocationInst Implementation
//===----------------------------------------------------------------------===//

static Value *getAISize(Value *Amt) {
  if (!Amt)
    Amt = ConstantUInt::get(Type::UIntTy, 1);
  else {
    assert(!isa<BasicBlock>(Amt) &&
           "Passed basic block into allocation size parameter!  Ue other ctor");
    assert(Amt->getType() == Type::UIntTy &&
           "Malloc/Allocation array size != UIntTy!");
  }
  return Amt;
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy,
                               unsigned Align, const std::string &Name,
                               Instruction *InsertBefore)
  : UnaryInstruction(PointerType::get(Ty), iTy, getAISize(ArraySize),
                     Name, InsertBefore), Alignment(Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  assert(Ty != Type::VoidTy && "Cannot allocate void!");
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy,
                               unsigned Align, const std::string &Name,
                               BasicBlock *InsertAtEnd)
  : UnaryInstruction(PointerType::get(Ty), iTy, getAISize(ArraySize),
                     Name, InsertAtEnd), Alignment(Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  assert(Ty != Type::VoidTy && "Cannot allocate void!");
}

// Out of line virtual method, so the vtable, etc has a home.
AllocationInst::~AllocationInst() {
}

bool AllocationInst::isArrayAllocation() const {
  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(getOperand(0)))
    return CUI->getValue() != 1;
  return true;
}

const Type *AllocationInst::getAllocatedType() const {
  return getType()->getElementType();
}

AllocaInst::AllocaInst(const AllocaInst &AI)
  : AllocationInst(AI.getType()->getElementType(), (Value*)AI.getOperand(0),
                   Instruction::Alloca, AI.getAlignment()) {
}

MallocInst::MallocInst(const MallocInst &MI)
  : AllocationInst(MI.getType()->getElementType(), (Value*)MI.getOperand(0),
                   Instruction::Malloc, MI.getAlignment()) {
}

//===----------------------------------------------------------------------===//
//                             FreeInst Implementation
//===----------------------------------------------------------------------===//

void FreeInst::AssertOK() {
  assert(isa<PointerType>(getOperand(0)->getType()) &&
         "Can not free something of nonpointer type!");
}

FreeInst::FreeInst(Value *Ptr, Instruction *InsertBefore)
  : UnaryInstruction(Type::VoidTy, Free, Ptr, "", InsertBefore) {
  AssertOK();
}

FreeInst::FreeInst(Value *Ptr, BasicBlock *InsertAtEnd)
  : UnaryInstruction(Type::VoidTy, Free, Ptr, "", InsertAtEnd) {
  AssertOK();
}


//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

void LoadInst::AssertOK() {
  assert(isa<PointerType>(getOperand(0)->getType()) &&
         "Ptr must have pointer type.");
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, Name, InsertBef) {
  setVolatile(false);
  AssertOK();
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, Name, InsertAE) {
  setVolatile(false);
  AssertOK();
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
                   Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, Name, InsertBef) {
  setVolatile(isVolatile);
  AssertOK();
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
                   BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, Name, InsertAE) {
  setVolatile(isVolatile);
  AssertOK();
}


//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

void StoreInst::AssertOK() {
  assert(isa<PointerType>(getOperand(1)->getType()) &&
         "Ptr must have pointer type!");
  assert(getOperand(0)->getType() ==
                 cast<PointerType>(getOperand(1)->getType())->getElementType()
         && "Ptr must be a pointer to Val type!");
}


StoreInst::StoreInst(Value *val, Value *addr, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, Ops, 2, "", InsertBefore) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(false);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, Ops, 2, "", InsertAtEnd) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(false);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, Ops, 2, "", InsertBefore) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(isVolatile);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, Ops, 2, "", InsertAtEnd) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(isVolatile);
  AssertOK();
}

//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

// checkType - Simple wrapper function to give a better assertion failure
// message on bad indexes for a gep instruction.
//
static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Invalid GetElementPtrInst indices for type!");
  return Ty;
}

void GetElementPtrInst::init(Value *Ptr, const std::vector<Value*> &Idx) {
  NumOperands = 1+Idx.size();
  Use *OL = OperandList = new Use[NumOperands];
  OL[0].init(Ptr, this);

  for (unsigned i = 0, e = Idx.size(); i != e; ++i)
    OL[i+1].init(Idx[i], this);
}

void GetElementPtrInst::init(Value *Ptr, Value *Idx0, Value *Idx1) {
  NumOperands = 3;
  Use *OL = OperandList = new Use[3];
  OL[0].init(Ptr, this);
  OL[1].init(Idx0, this);
  OL[2].init(Idx1, this);
}

void GetElementPtrInst::init(Value *Ptr, Value *Idx) {
  NumOperands = 2;
  Use *OL = OperandList = new Use[2];
  OL[0].init(Ptr, this);
  OL[1].init(Idx, this);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
                                     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx, true))),
                GetElementPtr, 0, 0, Name, InBe) {
  init(Ptr, Idx);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
                                     const std::string &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx, true))),
                GetElementPtr, 0, 0, Name, IAE) {
  init(Ptr, Idx);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx,
                                     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),Idx))),
                GetElementPtr, 0, 0, Name, InBe) {
  init(Ptr, Idx);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx,
                                     const std::string &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),Idx))),
                GetElementPtr, 0, 0, Name, IAE) {
  init(Ptr, Idx);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx0, Value *Idx1,
                                     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx0, Idx1, true))),
                GetElementPtr, 0, 0, Name, InBe) {
  init(Ptr, Idx0, Idx1);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx0, Value *Idx1,
                                     const std::string &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx0, Idx1, true))),
                GetElementPtr, 0, 0, Name, IAE) {
  init(Ptr, Idx0, Idx1);
}

GetElementPtrInst::~GetElementPtrInst() {
  delete[] OperandList;
}

// getIndexedType - Returns the type of the element that would be loaded with
// a load instruction with the specified parameters.
//
// A null type is returned if the indices are invalid for the specified
// pointer type.
//
const Type* GetElementPtrInst::getIndexedType(const Type *Ptr,
                                              const std::vector<Value*> &Idx,
                                              bool AllowCompositeLeaf) {
  if (!isa<PointerType>(Ptr)) return 0;   // Type isn't a pointer type!

  // Handle the special case of the empty set index set...
  if (Idx.empty())
    if (AllowCompositeLeaf ||
        cast<PointerType>(Ptr)->getElementType()->isFirstClassType())
      return cast<PointerType>(Ptr)->getElementType();
    else
      return 0;

  unsigned CurIdx = 0;
  while (const CompositeType *CT = dyn_cast<CompositeType>(Ptr)) {
    if (Idx.size() == CurIdx) {
      if (AllowCompositeLeaf || CT->isFirstClassType()) return Ptr;
      return 0;   // Can't load a whole structure or array!?!?
    }

    Value *Index = Idx[CurIdx++];
    if (isa<PointerType>(CT) && CurIdx != 1)
      return 0;  // Can only index into pointer types at the first index!
    if (!CT->indexValid(Index)) return 0;
    Ptr = CT->getTypeAtIndex(Index);

    // If the new type forwards to another type, then it is in the middle
    // of being refined to another type (and hence, may have dropped all
    // references to what it was using before).  So, use the new forwarded
    // type.
    if (const Type * Ty = Ptr->getForwardedType()) {
      Ptr = Ty;
    }
  }
  return CurIdx == Idx.size() ? Ptr : 0;
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr,
                                              Value *Idx0, Value *Idx1,
                                              bool AllowCompositeLeaf) {
  const PointerType *PTy = dyn_cast<PointerType>(Ptr);
  if (!PTy) return 0;   // Type isn't a pointer type!

  // Check the pointer index.
  if (!PTy->indexValid(Idx0)) return 0;

  const CompositeType *CT = dyn_cast<CompositeType>(PTy->getElementType());
  if (!CT || !CT->indexValid(Idx1)) return 0;

  const Type *ElTy = CT->getTypeAtIndex(Idx1);
  if (AllowCompositeLeaf || ElTy->isFirstClassType())
    return ElTy;
  return 0;
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr, Value *Idx) {
  const PointerType *PTy = dyn_cast<PointerType>(Ptr);
  if (!PTy) return 0;   // Type isn't a pointer type!

  // Check the pointer index.
  if (!PTy->indexValid(Idx)) return 0;

  return PTy->getElementType();
}

//===----------------------------------------------------------------------===//
//                           ExtractElementInst Implementation
//===----------------------------------------------------------------------===//

ExtractElementInst::ExtractElementInst(Value *Val, Value *Index,
                                       const std::string &Name,
                                       Instruction *InsertBef)
  : Instruction(cast<PackedType>(Val->getType())->getElementType(),
                ExtractElement, Ops, 2, Name, InsertBef) {
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");
  Ops[0].init(Val, this);
  Ops[1].init(Index, this);
}

ExtractElementInst::ExtractElementInst(Value *Val, Value *Index,
                                       const std::string &Name,
                                       BasicBlock *InsertAE)
  : Instruction(cast<PackedType>(Val->getType())->getElementType(),
                ExtractElement, Ops, 2, Name, InsertAE) {
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");

  Ops[0].init(Val, this);
  Ops[1].init(Index, this);
}

bool ExtractElementInst::isValidOperands(const Value *Val, const Value *Index) {
  if (!isa<PackedType>(Val->getType()) || Index->getType() != Type::UIntTy)
    return false;
  return true;
}


//===----------------------------------------------------------------------===//
//                           InsertElementInst Implementation
//===----------------------------------------------------------------------===//

InsertElementInst::InsertElementInst(const InsertElementInst &IE)
    : Instruction(IE.getType(), InsertElement, Ops, 3) {
  Ops[0].init(IE.Ops[0], this);
  Ops[1].init(IE.Ops[1], this);
  Ops[2].init(IE.Ops[2], this);
}
InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, Value *Index,
                                     const std::string &Name,
                                     Instruction *InsertBef)
  : Instruction(Vec->getType(), InsertElement, Ops, 3, Name, InsertBef) {
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");
  Ops[0].init(Vec, this);
  Ops[1].init(Elt, this);
  Ops[2].init(Index, this);
}

InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, Value *Index,
                                     const std::string &Name,
                                     BasicBlock *InsertAE)
  : Instruction(Vec->getType(), InsertElement, Ops, 3, Name, InsertAE) {
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");

  Ops[0].init(Vec, this);
  Ops[1].init(Elt, this);
  Ops[2].init(Index, this);
}

bool InsertElementInst::isValidOperands(const Value *Vec, const Value *Elt, 
                                        const Value *Index) {
  if (!isa<PackedType>(Vec->getType()))
    return false;   // First operand of insertelement must be packed type.
  
  if (Elt->getType() != cast<PackedType>(Vec->getType())->getElementType())
    return false;// Second operand of insertelement must be packed element type.
    
  if (Index->getType() != Type::UIntTy)
    return false;  // Third operand of insertelement must be uint.
  return true;
}


//===----------------------------------------------------------------------===//
//                      ShuffleVectorInst Implementation
//===----------------------------------------------------------------------===//

ShuffleVectorInst::ShuffleVectorInst(const ShuffleVectorInst &SV) 
    : Instruction(SV.getType(), ShuffleVector, Ops, 3) {
  Ops[0].init(SV.Ops[0], this);
  Ops[1].init(SV.Ops[1], this);
  Ops[2].init(SV.Ops[2], this);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const std::string &Name,
                                     Instruction *InsertBefore)
  : Instruction(V1->getType(), ShuffleVector, Ops, 3, Name, InsertBefore) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");
  Ops[0].init(V1, this);
  Ops[1].init(V2, this);
  Ops[2].init(Mask, this);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const std::string &Name, 
                                     BasicBlock *InsertAtEnd)
  : Instruction(V1->getType(), ShuffleVector, Ops, 3, Name, InsertAtEnd) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");

  Ops[0].init(V1, this);
  Ops[1].init(V2, this);
  Ops[2].init(Mask, this);
}

bool ShuffleVectorInst::isValidOperands(const Value *V1, const Value *V2, 
                                        const Value *Mask) {
  if (!isa<PackedType>(V1->getType())) return false;
  if (V1->getType() != V2->getType()) return false;
  if (!isa<PackedType>(Mask->getType()) ||
         cast<PackedType>(Mask->getType())->getElementType() != Type::UIntTy ||
         cast<PackedType>(Mask->getType())->getNumElements() !=
         cast<PackedType>(V1->getType())->getNumElements())
    return false;
  return true;
}


//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

void BinaryOperator::init(BinaryOps iType)
{
  Value *LHS = getOperand(0), *RHS = getOperand(1);
  assert(LHS->getType() == RHS->getType() &&
         "Binary operator operand types must match!");
#ifndef NDEBUG
  switch (iType) {
  case Add: case Sub:
  case Mul: case Div:
  case Rem:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || getType()->isFloatingPoint() ||
            isa<PackedType>(getType())) &&
          "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case And: case Or:
  case Xor:
    assert(getType() == LHS->getType() &&
           "Logical operation should return same type as operands!");
    assert((getType()->isIntegral() ||
            (isa<PackedType>(getType()) && 
             cast<PackedType>(getType())->getElementType()->isIntegral())) &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    assert(getType() == Type::BoolTy && "Setcc must return bool!");
  default:
    break;
  }
#endif
}

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
                                       const std::string &Name,
                                       Instruction *InsertBefore) {
  assert(S1->getType() == S2->getType() &&
         "Cannot create binary operator with two operands of differing type!");
  switch (Op) {
  // Binary comparison operators...
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    return new SetCondInst(Op, S1, S2, Name, InsertBefore);

  default:
    return new BinaryOperator(Op, S1, S2, S1->getType(), Name, InsertBefore);
  }
}

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
                                       const std::string &Name,
                                       BasicBlock *InsertAtEnd) {
  BinaryOperator *Res = create(Op, S1, S2, Name);
  InsertAtEnd->getInstList().push_back(Res);
  return Res;
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  if (!Op->getType()->isFloatingPoint())
    return new BinaryOperator(Instruction::Sub,
                              Constant::getNullValue(Op->getType()), Op,
                              Op->getType(), Name, InsertBefore);
  else
    return new BinaryOperator(Instruction::Sub,
                              ConstantFP::get(Op->getType(), -0.0), Op,
                              Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          BasicBlock *InsertAtEnd) {
  if (!Op->getType()->isFloatingPoint())
    return new BinaryOperator(Instruction::Sub,
                              Constant::getNullValue(Op->getType()), Op,
                              Op->getType(), Name, InsertAtEnd);
  else
    return new BinaryOperator(Instruction::Sub,
                              ConstantFP::get(Op->getType(), -0.0), Op,
                              Op->getType(), Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  Constant *C;
  if (const PackedType *PTy = dyn_cast<PackedType>(Op->getType())) {
    C = ConstantIntegral::getAllOnesValue(PTy->getElementType());
    C = ConstantPacked::get(std::vector<Constant*>(PTy->getNumElements(), C));
  } else {
    C = ConstantIntegral::getAllOnesValue(Op->getType());
  }
  
  return new BinaryOperator(Instruction::Xor, Op, C,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          BasicBlock *InsertAtEnd) {
  Constant *AllOnes;
  if (const PackedType *PTy = dyn_cast<PackedType>(Op->getType())) {
    // Create a vector of all ones values.
    Constant *Elt = ConstantIntegral::getAllOnesValue(PTy->getElementType());
    AllOnes = 
      ConstantPacked::get(std::vector<Constant*>(PTy->getNumElements(), Elt));
  } else {
    AllOnes = ConstantIntegral::getAllOnesValue(Op->getType());
  }
  
  return new BinaryOperator(Instruction::Xor, Op, AllOnes,
                            Op->getType(), Name, InsertAtEnd);
}


// isConstantAllOnes - Helper function for several functions below
static inline bool isConstantAllOnes(const Value *V) {
  return isa<ConstantIntegral>(V) &&cast<ConstantIntegral>(V)->isAllOnesValue();
}

bool BinaryOperator::isNeg(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    if (Bop->getOpcode() == Instruction::Sub)
      if (!V->getType()->isFloatingPoint())
        return Bop->getOperand(0) == Constant::getNullValue(Bop->getType());
      else
        return Bop->getOperand(0) == ConstantFP::get(Bop->getType(), -0.0);
  return false;
}

bool BinaryOperator::isNot(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    return (Bop->getOpcode() == Instruction::Xor &&
            (isConstantAllOnes(Bop->getOperand(1)) ||
             isConstantAllOnes(Bop->getOperand(0))));
  return false;
}

Value *BinaryOperator::getNegArgument(Value *BinOp) {
  assert(isNeg(BinOp) && "getNegArgument from non-'neg' instruction!");
  return cast<BinaryOperator>(BinOp)->getOperand(1);
}

const Value *BinaryOperator::getNegArgument(const Value *BinOp) {
  return getNegArgument(const_cast<Value*>(BinOp));
}

Value *BinaryOperator::getNotArgument(Value *BinOp) {
  assert(isNot(BinOp) && "getNotArgument on non-'not' instruction!");
  BinaryOperator *BO = cast<BinaryOperator>(BinOp);
  Value *Op0 = BO->getOperand(0);
  Value *Op1 = BO->getOperand(1);
  if (isConstantAllOnes(Op0)) return Op1;

  assert(isConstantAllOnes(Op1));
  return Op0;
}

const Value *BinaryOperator::getNotArgument(const Value *BinOp) {
  return getNotArgument(const_cast<Value*>(BinOp));
}


// swapOperands - Exchange the two operands to this instruction.  This
// instruction is safe to use on any binary instruction and does not
// modify the semantics of the instruction.  If the instruction is
// order dependent (SetLT f.e.) the opcode is changed.
//
bool BinaryOperator::swapOperands() {
  if (isCommutative())
    ;  // If the instruction is commutative, it is safe to swap the operands
  else if (SetCondInst *SCI = dyn_cast<SetCondInst>(this))
    /// FIXME: SetCC instructions shouldn't all have different opcodes.
    setOpcode(SCI->getSwappedCondition());
  else
    return true;   // Can't commute operands

  std::swap(Ops[0], Ops[1]);
  return false;
}


//===----------------------------------------------------------------------===//
//                             SetCondInst Class
//===----------------------------------------------------------------------===//

SetCondInst::SetCondInst(BinaryOps Opcode, Value *S1, Value *S2,
                         const std::string &Name, Instruction *InsertBefore)
  : BinaryOperator(Opcode, S1, S2, Type::BoolTy, Name, InsertBefore) {

  // Make sure it's a valid type... getInverseCondition will assert out if not.
  assert(getInverseCondition(Opcode));
}

SetCondInst::SetCondInst(BinaryOps Opcode, Value *S1, Value *S2,
                         const std::string &Name, BasicBlock *InsertAtEnd)
  : BinaryOperator(Opcode, S1, S2, Type::BoolTy, Name, InsertAtEnd) {

  // Make sure it's a valid type... getInverseCondition will assert out if not.
  assert(getInverseCondition(Opcode));
}

// getInverseCondition - Return the inverse of the current condition opcode.
// For example seteq -> setne, setgt -> setle, setlt -> setge, etc...
//
Instruction::BinaryOps SetCondInst::getInverseCondition(BinaryOps Opcode) {
  switch (Opcode) {
  default:
    assert(0 && "Unknown setcc opcode!");
  case SetEQ: return SetNE;
  case SetNE: return SetEQ;
  case SetGT: return SetLE;
  case SetLT: return SetGE;
  case SetGE: return SetLT;
  case SetLE: return SetGT;
  }
}

// getSwappedCondition - Return the condition opcode that would be the result
// of exchanging the two operands of the setcc instruction without changing
// the result produced.  Thus, seteq->seteq, setle->setge, setlt->setgt, etc.
//
Instruction::BinaryOps SetCondInst::getSwappedCondition(BinaryOps Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case SetEQ: case SetNE: return Opcode;
  case SetGT: return SetLT;
  case SetLT: return SetGT;
  case SetGE: return SetLE;
  case SetLE: return SetGE;
  }
}

//===----------------------------------------------------------------------===//
//                        SwitchInst Implementation
//===----------------------------------------------------------------------===//

void SwitchInst::init(Value *Value, BasicBlock *Default, unsigned NumCases) {
  assert(Value && Default);
  ReservedSpace = 2+NumCases*2;
  NumOperands = 2;
  OperandList = new Use[ReservedSpace];

  OperandList[0].init(Value, this);
  OperandList[1].init(Default, this);
}

SwitchInst::SwitchInst(const SwitchInst &SI)
  : TerminatorInst(Instruction::Switch, new Use[SI.getNumOperands()],
                   SI.getNumOperands()) {
  Use *OL = OperandList, *InOL = SI.OperandList;
  for (unsigned i = 0, E = SI.getNumOperands(); i != E; i+=2) {
    OL[i].init(InOL[i], this);
    OL[i+1].init(InOL[i+1], this);
  }
}

SwitchInst::~SwitchInst() {
  delete [] OperandList;
}


/// addCase - Add an entry to the switch instruction...
///
void SwitchInst::addCase(ConstantInt *OnVal, BasicBlock *Dest) {
  unsigned OpNo = NumOperands;
  if (OpNo+2 > ReservedSpace)
    resizeOperands(0);  // Get more space!
  // Initialize some new operands.
  assert(OpNo+1 < ReservedSpace && "Growing didn't work!");
  NumOperands = OpNo+2;
  OperandList[OpNo].init(OnVal, this);
  OperandList[OpNo+1].init(Dest, this);
}

/// removeCase - This method removes the specified successor from the switch
/// instruction.  Note that this cannot be used to remove the default
/// destination (successor #0).
///
void SwitchInst::removeCase(unsigned idx) {
  assert(idx != 0 && "Cannot remove the default case!");
  assert(idx*2 < getNumOperands() && "Successor index out of range!!!");

  unsigned NumOps = getNumOperands();
  Use *OL = OperandList;

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // client might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  for (unsigned i = (idx+1)*2; i != NumOps; i += 2) {
    OL[i-2] = OL[i];
    OL[i-2+1] = OL[i+1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(0);
  OL[NumOps-2+1].set(0);
  NumOperands = NumOps-2;
}

/// resizeOperands - resize operands - This adjusts the length of the operands
/// list according to the following behavior:
///   1. If NumOps == 0, grow the operand list in response to a push_back style
///      of operation.  This grows the number of ops by 1.5 times.
///   2. If NumOps > NumOperands, reserve space for NumOps operands.
///   3. If NumOps == NumOperands, trim the reserved space.
///
void SwitchInst::resizeOperands(unsigned NumOps) {
  if (NumOps == 0) {
    NumOps = getNumOperands()/2*6;
  } else if (NumOps*2 > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *NewOps = new Use[NumOps];
  Use *OldOps = OperandList;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      NewOps[i].init(OldOps[i], this);
      OldOps[i].set(0);
  }
  delete [] OldOps;
  OperandList = NewOps;
}


BasicBlock *SwitchInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned SwitchInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void SwitchInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  setSuccessor(idx, B);
}


// Define these methods here so vtables don't get emitted into every translation
// unit that uses these classes.

GetElementPtrInst *GetElementPtrInst::clone() const {
  return new GetElementPtrInst(*this);
}

BinaryOperator *BinaryOperator::clone() const {
  return create(getOpcode(), Ops[0], Ops[1]);
}

MallocInst *MallocInst::clone() const { return new MallocInst(*this); }
AllocaInst *AllocaInst::clone() const { return new AllocaInst(*this); }
FreeInst   *FreeInst::clone()   const { return new FreeInst(getOperand(0)); }
LoadInst   *LoadInst::clone()   const { return new LoadInst(*this); }
StoreInst  *StoreInst::clone()  const { return new StoreInst(*this); }
CastInst   *CastInst::clone()   const { return new CastInst(*this); }
CallInst   *CallInst::clone()   const { return new CallInst(*this); }
ShiftInst  *ShiftInst::clone()  const { return new ShiftInst(*this); }
SelectInst *SelectInst::clone() const { return new SelectInst(*this); }
VAArgInst  *VAArgInst::clone()  const { return new VAArgInst(*this); }
ExtractElementInst *ExtractElementInst::clone() const {
  return new ExtractElementInst(*this);
}
InsertElementInst *InsertElementInst::clone() const {
  return new InsertElementInst(*this);
}
ShuffleVectorInst *ShuffleVectorInst::clone() const {
  return new ShuffleVectorInst(*this);
}
PHINode    *PHINode::clone()    const { return new PHINode(*this); }
ReturnInst *ReturnInst::clone() const { return new ReturnInst(*this); }
BranchInst *BranchInst::clone() const { return new BranchInst(*this); }
SwitchInst *SwitchInst::clone() const { return new SwitchInst(*this); }
InvokeInst *InvokeInst::clone() const { return new InvokeInst(*this); }
UnwindInst *UnwindInst::clone() const { return new UnwindInst(); }
UnreachableInst *UnreachableInst::clone() const { return new UnreachableInst();}
