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
		 const string &Name = "")
    : Instruction(Ty, iTy, Name) {
    assert(Ty->isPointerType() && "Can't allocate a non pointer type!");

    if (ArraySize) {
      // Make sure they didn't try to specify a size for !(unsized array) type
      assert(getType()->getValueType()->isArrayType() && 
             cast<ArrayType>(getType()->getValueType())->isUnsized() && 
           "Trying to allocate something other than unsized array, with size!");

      Operands.reserve(1);
      Operands.push_back(Use(ArraySize, this));
    } else {
      // Make sure that the pointer is not to an unsized array!
      assert(!getType()->getValueType()->isArrayType() ||
	     cast<const ArrayType>(getType()->getValueType())->isSized() && 
	     "Trying to allocate unsized array without size!");
    }
  }

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (const PointerType*)Instruction::getType(); 
  }

  virtual Instruction *clone() const = 0;
};


//===----------------------------------------------------------------------===//
//                                MallocInst Class
//===----------------------------------------------------------------------===//

class MallocInst : public AllocationInst {
public:
  MallocInst(const Type *Ty, Value *ArraySize = 0, const string &Name = "") 
    : AllocationInst(Ty, ArraySize, Malloc, Name) {}

  virtual Instruction *clone() const { 
    return new MallocInst(getType(), Operands.size() ? Operands[1] : 0);
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
  AllocaInst(const Type *Ty, Value *ArraySize = 0, const string &Name = "") 
    : AllocationInst(Ty, ArraySize, Alloca, Name) {}

  virtual Instruction *clone() const { 
    return new AllocaInst(getType(), Operands.size() ? Operands[1] : 0);
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
  FreeInst(Value *Ptr, const string &Name = "") 
    : Instruction(Type::VoidTy, Free, Name) {
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
		       const vector<ConstPoolVal*> &Idx,
		       const string &Nam = "")
    : Instruction(Ty, Opcode, Nam),
      indexVec(Idx)
  {}
  
protected:
  vector<ConstPoolVal*> indexVec;
  
public:
  // getIndexedType - Returns the type of the element that would be loaded with
  // a load instruction with the specified parameters.
  //
  // A null type is returned if the indices are invalid for the specified 
  // pointer type.
  //
  static const Type *getIndexedType(const Type *Ptr, 
				    const vector<ConstPoolVal*> &Indices,
				    bool AllowStructLeaf = false);
  
  const vector<ConstPoolVal*>& getIndexVec() const { return indexVec; }
  
  virtual Value* getPtrOperand() = 0;
  
  virtual int	getFirstOffsetIdx() const = 0;
};


//===----------------------------------------------------------------------===//
//                                LoadInst Class
//===----------------------------------------------------------------------===//

class LoadInst : public MemAccessInst {
  LoadInst(const LoadInst &LI) : MemAccessInst(LI.getType(), Load, LI.getIndexVec()) {
    Operands.reserve(LI.Operands.size());
    for (unsigned i = 0, E = LI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(LI.Operands[i], this));
  }
public:
  LoadInst(Value *Ptr, const vector<ConstPoolVal*> &Idx,
	   const string &Name = "");
  virtual Instruction*	clone() const { return new LoadInst(*this); }
  virtual const char*	getOpcodeName() const { return "load"; }  
  virtual Value*	getPtrOperand() { return this->getOperand(0); }
  virtual int getFirstOffsetIdx() const { return (this->getNumOperands() > 1)? 1 : -1;}

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
  StoreInst(const StoreInst &SI) : MemAccessInst(SI.getType(), Store, SI.getIndexVec()) {
    Operands.reserve(SI.Operands.size());
    for (unsigned i = 0, E = SI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(SI.Operands[i], this));
  }
public:
  StoreInst(Value *Val, Value *Ptr, const vector<ConstPoolVal*> &Idx,
	    const string &Name = "");
  virtual Instruction *clone() const { return new StoreInst(*this); }
  virtual const char *getOpcodeName() const { return "store"; }  
  
  virtual bool hasSideEffects() const { return true; }
  virtual Value*	getPtrOperand()	{ return this->getOperand(1); }
  virtual int getFirstOffsetIdx() const { return (this->getNumOperands() > 2)? 2 : -1;}

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
    : MemAccessInst(EPI.getType(), GetElementPtr, EPI.getIndexVec()) {
    Operands.reserve(EPI.Operands.size());
    for (unsigned i = 0, E = EPI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(EPI.Operands[i], this));
  }
public:
  GetElementPtrInst(Value *Ptr, const vector<ConstPoolVal*> &Idx,
		    const string &Name = "");
  virtual Instruction *clone() const { return new GetElementPtrInst(*this); }
  virtual const char *getOpcodeName() const { return "getelementptr"; }  
  virtual Value*	getPtrOperand()	{ return this->getOperand(0); }
  virtual int getFirstOffsetIdx() const { return (this->getNumOperands() > 1)? 1 : -1;}
  
  inline bool isArraySelector() const { return !isStructSelector(); }
  bool isStructSelector() const;


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
