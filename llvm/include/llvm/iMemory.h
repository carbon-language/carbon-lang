//===-- llvm/iMemory.h - Memory Operator node definitions --------*- C++ -*--=//
//
// This file contains the declarations of all of the memory related operators.
// This includes: malloc, free, alloca, load, store, getfield, putfield
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IMEMORY_H
#define LLVM_IMEMORY_H

#include "llvm/Instruction.h"
class PointerType;

//===----------------------------------------------------------------------===//
//                             AllocationInst Class
//===----------------------------------------------------------------------===//
//
// AllocationInst - This class is the common base class of MallocInst and
// AllocaInst.
//
class AllocationInst : public Instruction {
protected:
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
		 const std::string &Name = "");
public:

  // isArrayAllocation - Return true if there is an allocation size parameter
  // to the allocation instruction that is not 1.
  //
  bool isArrayAllocation() const;

  // getArraySize - Get the number of element allocated, for a simple allocation
  // of a single element, this will return a constant 1 value.
  //
  inline const Value *getArraySize() const { return Operands[0]; }
  inline Value *getArraySize() { return Operands[0]; }

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (const PointerType*)Instruction::getType(); 
  }

  // getAllocatedType - Return the type that is being allocated by the
  // instruction.
  //
  const Type *getAllocatedType() const;

  virtual Instruction *clone() const = 0;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AllocationInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Alloca ||
           I->getOpcode() == Instruction::Malloc;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                MallocInst Class
//===----------------------------------------------------------------------===//

struct MallocInst : public AllocationInst {
  MallocInst(const Type *Ty, Value *ArraySize = 0, const std::string &Name = "")
    : AllocationInst(Ty, ArraySize, Malloc, Name) {}

  virtual Instruction *clone() const { 
    return new MallocInst((Type*)getType(), (Value*)Operands[0].get());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MallocInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Malloc);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                AllocaInst Class
//===----------------------------------------------------------------------===//

struct AllocaInst : public AllocationInst {
  AllocaInst(const Type *Ty, Value *ArraySize = 0, const std::string &Name = "")
    : AllocationInst(Ty, ArraySize, Alloca, Name) {}

  virtual Instruction *clone() const { 
    return new AllocaInst((Type*)getType(), (Value*)Operands[0].get());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AllocaInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Alloca);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                 FreeInst Class
//===----------------------------------------------------------------------===//

struct FreeInst : public Instruction {
  FreeInst(Value *Ptr);

  virtual Instruction *clone() const { return new FreeInst(Operands[0]); }

  virtual bool hasSideEffects() const { return true; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FreeInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Free);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                              MemAccessInst Class
//===----------------------------------------------------------------------===//
//
// MemAccessInst - Common base class of LoadInst, StoreInst, and
// GetElementPtrInst...
//
class MemAccessInst : public Instruction {
protected:
  inline MemAccessInst(const Type *Ty, unsigned Opcode,
		       const std::string &Nam = "")
    : Instruction(Ty, Opcode, Nam) {}
public:
  // getIndexedType - Returns the type of the element that would be loaded with
  // a load instruction with the specified parameters.
  //
  // A null type is returned if the indices are invalid for the specified 
  // pointer type.
  //
  static const Type *getIndexedType(const Type *Ptr, 
				    const std::vector<Value*> &Indices,
				    bool AllowStructLeaf = false);

  inline op_iterator       idx_begin()       {
    return op_begin()+getFirstIndexOperandNumber();
  }
  inline const_op_iterator idx_begin() const {
    return op_begin()+getFirstIndexOperandNumber();
  }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }


  std::vector<Value*> copyIndices() const {
    return std::vector<Value*>(idx_begin(), idx_end());
  }

  Value *getPointerOperand() {
    return getOperand(getFirstIndexOperandNumber()-1);
  }
  const Value *getPointerOperand() const {
    return getOperand(getFirstIndexOperandNumber()-1);
  }
  
  virtual unsigned getFirstIndexOperandNumber() const = 0;

  inline unsigned getNumIndices() const {  // Note: always non-negative
    return (getNumOperands() - getFirstIndexOperandNumber());
  }
  
  inline bool hasIndices() const {
    return getNumIndices() > 0;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MemAccessInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Load || I->getOpcode() == Store ||
           I->getOpcode() == GetElementPtr;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                LoadInst Class
//===----------------------------------------------------------------------===//

class LoadInst : public MemAccessInst {
  LoadInst(const LoadInst &LI) : MemAccessInst(LI.getType(), Load) {
    Operands.reserve(LI.Operands.size());
    for (unsigned i = 0, E = LI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(LI.Operands[i], this));
  }
public:
  LoadInst(Value *Ptr, const std::vector<Value*> &Ix, const std::string & = "");
  LoadInst(Value *Ptr, const std::string &Name = "");

  virtual Instruction *clone() const { return new LoadInst(*this); }

  virtual unsigned getFirstIndexOperandNumber() const { return 1; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Load);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                StoreInst Class
//===----------------------------------------------------------------------===//

class StoreInst : public MemAccessInst {
  StoreInst(const StoreInst &SI) : MemAccessInst(SI.getType(), Store) {
    Operands.reserve(SI.Operands.size());
    for (unsigned i = 0, E = SI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(SI.Operands[i], this));
  }
public:
  StoreInst(Value *Val, Value *Ptr, const std::vector<Value*> &Idx);
  StoreInst(Value *Val, Value *Ptr);
  virtual Instruction *clone() const { return new StoreInst(*this); }

  virtual bool hasSideEffects() const { return true; }
  virtual unsigned getFirstIndexOperandNumber() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StoreInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Store);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                             GetElementPtrInst Class
//===----------------------------------------------------------------------===//

class GetElementPtrInst : public MemAccessInst {
  GetElementPtrInst(const GetElementPtrInst &EPI)
    : MemAccessInst((Type*)EPI.getType(), GetElementPtr) {
    Operands.reserve(EPI.Operands.size());
    for (unsigned i = 0, E = EPI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(EPI.Operands[i], this));
  }
public:
  GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
		    const std::string &Name = "");
  virtual Instruction *clone() const { return new GetElementPtrInst(*this); }
  virtual unsigned getFirstIndexOperandNumber() const { return 1; }
  
  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (PointerType*)Instruction::getType();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GetElementPtrInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::GetElementPtr);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

#endif // LLVM_IMEMORY_H
