//===-- llvm/iMemory.h - Memory Operator node definitions --------*- C++ -*--=//
//
// This file contains the declarations of all of the memory related operators.
// This includes: malloc, free, alloca, load, store, getfield, putfield
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IMEMORY_H
#define LLVM_IMEMORY_H

#include "llvm/Instruction.h"
#include "llvm/DerivedTypes.h"

//===----------------------------------------------------------------------===//
//                             AllocationInst Class
//===----------------------------------------------------------------------===//
//
// AllocationInst - This class is the common base class of MallocInst and
// AllocaInst.
//
class AllocationInst : public Instruction {
public:
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
		 const std::string &Name = "")
    : Instruction(Ty, iTy, Name) {
    assert(Ty->isPointerType() && "Can't allocate a non pointer type!");

    if (ArraySize) {
      assert(ArraySize->getType() == Type::UIntTy &&
             "Malloc/Allocation array size != UIntTy!");

      Operands.reserve(1);
      Operands.push_back(Use(ArraySize, this));
    }
  }

  // isArrayAllocation - Return true if there is an allocation size parameter
  // to the allocation instruction that is not 1.
  //
  bool isArrayAllocation() const;

  inline const Value *getArraySize() const {
    assert(isArrayAllocation()); return Operands[0];
  }
  inline Value *getArraySize() {
    assert(isArrayAllocation()); return Operands[0];
  }

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (const PointerType*)Instruction::getType(); 
  }

  // getAllocatedType - Return the type that is being allocated by the
  // instruction.
  inline const Type *getAllocatedType() const {
    return getType()->getElementType();
  }

  virtual Instruction *clone() const = 0;
};


//===----------------------------------------------------------------------===//
//                                MallocInst Class
//===----------------------------------------------------------------------===//

class MallocInst : public AllocationInst {
public:
  MallocInst(const Type *Ty, Value *ArraySize = 0, const std::string &Name = "")
    : AllocationInst(Ty, ArraySize, Malloc, Name) {}

  virtual Instruction *clone() const { 
    return new MallocInst(getType(), 
			  Operands.size() ? (Value*)Operands[0].get() : 0);
  }

  virtual const char *getOpcodeName() const { return "malloc"; }

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

class AllocaInst : public AllocationInst {
public:
  AllocaInst(const Type *Ty, Value *ArraySize = 0, const std::string &Name = "")
    : AllocationInst(Ty, ArraySize, Alloca, Name) {}

  virtual Instruction *clone() const { 
    return new AllocaInst(getType(),
			  Operands.size() ? (Value*)Operands[0].get() : 0);
  }

  virtual const char *getOpcodeName() const { return "alloca"; }

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

class FreeInst : public Instruction {
public:
  FreeInst(Value *Ptr) : Instruction(Type::VoidTy, Free, "") {
    assert(Ptr->getType()->isPointerType() && "Can't free nonpointer!");
    Operands.reserve(1);
    Operands.push_back(Use(Ptr, this));
  }

  virtual Instruction *clone() const { return new FreeInst(Operands[0]); }

  virtual const char *getOpcodeName() const { return "free"; }

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

  inline bool hasIndices() const {
    return getNumOperands() > getFirstIndexOperandNumber();
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
  virtual const char *getOpcodeName() const { return "load"; }  

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
  StoreInst(Value *Val, Value *Ptr, const std::vector<Value*> &Idx,
	    const std::string &Name = "");
  StoreInst(Value *Val, Value *Ptr, const std::string &Name = "");
  virtual Instruction *clone() const { return new StoreInst(*this); }

  virtual const char *getOpcodeName() const { return "store"; }  
  
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
    : MemAccessInst(EPI.getType(), GetElementPtr) {
    Operands.reserve(EPI.Operands.size());
    for (unsigned i = 0, E = EPI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(EPI.Operands[i], this));
  }
public:
  GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
		    const std::string &Name = "");
  virtual Instruction *clone() const { return new GetElementPtrInst(*this); }
  virtual const char *getOpcodeName() const { return "getelementptr"; }  
  virtual unsigned getFirstIndexOperandNumber() const { return 1; }
  
  inline bool isArraySelector() const { return !isStructSelector(); }
  bool isStructSelector() const;

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return cast<const PointerType>(Instruction::getType());
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
