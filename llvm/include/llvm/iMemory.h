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

class AllocationInst : public Instruction {
public:
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
		 const string &Name = "")
    : Instruction(Ty, iTy, Name) {
    assert(Ty->isPointerType() && "Can't allocate a non pointer type!");

    if (ArraySize) {
      // Make sure they didn't try to specify a size for !(unsized array) type...
      assert((getType()->getValueType()->isArrayType() && 
	      ((const ArrayType*)getType()->getValueType())->isUnsized()) && 
            "Trying to allocate something other than unsized array, with size!");

      Operands.reserve(1);
      Operands.push_back(Use(ArraySize, this));
    }
  }

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (const PointerType*)Instruction::getType(); 
  }

  virtual Instruction *clone() const = 0;
};

class MallocInst : public AllocationInst {
public:
  MallocInst(const Type *Ty, Value *ArraySize = 0, const string &Name = "") 
    : AllocationInst(Ty, ArraySize, Instruction::Malloc, Name) {}

  virtual Instruction *clone() const { 
    return new MallocInst(getType(), Operands.size() ? Operands[1] : 0);
  }

  virtual string getOpcode() const { return "malloc"; }
};

class AllocaInst : public AllocationInst {
public:
  AllocaInst(const Type *Ty, Value *ArraySize = 0, const string &Name = "") 
    : AllocationInst(Ty, ArraySize, Instruction::Alloca, Name) {}

  virtual Instruction *clone() const { 
    return new AllocaInst(getType(), Operands.size() ? Operands[1] : 0);
  }

  virtual string getOpcode() const { return "alloca"; }
};



class FreeInst : public Instruction {
public:
  FreeInst(Value *Ptr, const string &Name = "") 
    : Instruction(Type::VoidTy, Instruction::Free, Name) {
    assert(Ptr->getType()->isPointerType() && "Can't free nonpointer!");
    Operands.reserve(1);
    Operands.push_back(Use(Ptr, this));
  }
  inline ~FreeInst() {}

  virtual Instruction *clone() const { return new FreeInst(Operands[0]); }

  virtual string getOpcode() const { return "free"; }
};

#endif // LLVM_IMEMORY_H
