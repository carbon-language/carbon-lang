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
#include "llvm/ConstPoolVals.h"

class ConstPoolType;

class AllocationInst : public Instruction {
protected:
  UseTy<ConstPoolType> TyVal;
  Use ArraySize;
public:
  AllocationInst(ConstPoolType *tyVal, Value *arrSize, unsigned iTy, 
		 const string &Name = "") 
    : Instruction(tyVal->getValue(), iTy, Name),
      TyVal(tyVal, this), ArraySize(arrSize, this) {

    // Make sure they didn't try to specify a size for an invalid type...
    assert(arrSize == 0 || 
	   (getType()->getValueType()->isArrayType() && 
	    ((const ArrayType*)getType()->getValueType())->isUnsized()) && 
	   "Trying to allocate something other than unsized array, with size!");

    // Make sure that if a size is specified, that it is a uint!
    assert(arrSize == 0 || arrSize->getType() == Type::UIntTy &&
	   "Malloc SIZE is not a 'uint'!");
  }
  inline ~AllocationInst() {}

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (const PointerType*)Instruction::getType(); 
  }

  virtual Instruction *clone() const = 0;

  inline virtual void dropAllReferences() { TyVal = 0; ArraySize = 0; }
  virtual bool setOperand(unsigned i, Value *Val) { 
    if (i == 0) {
      assert(!Val || Val->getValueType() == Value::ConstantVal);
      TyVal = (ConstPoolType*)Val;
      return true;
    } else if (i == 1) {
      // Make sure they didn't try to specify a size for an invalid type...
      assert(Val == 0 || 
	     (getType()->getValueType()->isArrayType() && 
	      ((const ArrayType*)getType()->getValueType())->isUnsized()) && 
           "Trying to allocate something other than unsized array, with size!");
      
      // Make sure that if a size is specified, that it is a uint!
      assert(Val == 0 || Val->getType() == Type::UIntTy &&
	     "Malloc SIZE is not a 'uint'!");
      
      ArraySize = Val;
      return true;
    }
    return false; 
  }

  virtual unsigned getNumOperands() const { return 2; }

  virtual const Value *getOperand(unsigned i) const { 
    return i == 0 ? TyVal : (i == 1 ? ArraySize : 0); 
  }
};

class MallocInst : public AllocationInst {
public:
  MallocInst(ConstPoolType *tyVal, Value *ArraySize = 0, 
	     const string &Name = "") 
    : AllocationInst(tyVal, ArraySize, Instruction::Malloc, Name) {}
  inline ~MallocInst() {}

  virtual Instruction *clone() const { 
    return new MallocInst(TyVal, ArraySize);
  }

  virtual string getOpcode() const { return "malloc"; }
};

class AllocaInst : public AllocationInst {
public:
  AllocaInst(ConstPoolType *tyVal, Value *ArraySize = 0, 
	     const string &Name = "") 
    : AllocationInst(tyVal, ArraySize, Instruction::Alloca, Name) {}
  inline ~AllocaInst() {}

  virtual Instruction *clone() const { 
    return new AllocaInst(TyVal, ArraySize);
  }

  virtual string getOpcode() const { return "alloca"; }
};



class FreeInst : public Instruction {
protected:
  Use Pointer;
public:
  FreeInst(Value *Ptr, const string &Name = "") 
    : Instruction(Type::VoidTy, Instruction::Free, Name),
      Pointer(Ptr, this) {

    assert(Ptr->getType()->isPointerType() && "Can't free nonpointer!");
  }
  inline ~FreeInst() {}

  virtual Instruction *clone() const { return new FreeInst(Pointer); }

  inline virtual void dropAllReferences() { Pointer = 0;  }

  virtual bool setOperand(unsigned i, Value *Val) { 
    if (i == 0) {
      assert(!Val || Val->getType()->isPointerType() &&
	     "Can't free nonpointer!");
      Pointer = Val;
      return true;
    }
    return false; 
  }

  virtual unsigned getNumOperands() const { return 1; }
  virtual const Value *getOperand(unsigned i) const { 
    return i == 0 ? Pointer : 0;
  }

  virtual string getOpcode() const { return "free"; }
};

#endif // LLVM_IMEMORY_H
