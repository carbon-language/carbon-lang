//===-- llvm/iMemory.h - Memory Operator node definitions -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of all of the memory related operators.
// This includes: malloc, free, alloca, load, store, and getelementptr
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IMEMORY_H
#define LLVM_IMEMORY_H

#include "llvm/Instruction.h"

namespace llvm {

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
		 const std::string &Name = "", Instruction *InsertBefore = 0);
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

class MallocInst : public AllocationInst {
  MallocInst(const MallocInst &MI);
public:
  MallocInst(const Type *Ty, Value *ArraySize = 0, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Malloc, Name, InsertBefore) {}

  virtual Instruction *clone() const { 
    return new MallocInst(*this);
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

class AllocaInst : public AllocationInst {
  AllocaInst(const AllocaInst &);
public:
  AllocaInst(const Type *Ty, Value *ArraySize = 0, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Alloca, Name, InsertBefore) {}

  virtual Instruction *clone() const { 
    return new AllocaInst(*this);
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
  FreeInst(Value *Ptr, Instruction *InsertBefore = 0);

  virtual Instruction *clone() const { return new FreeInst(Operands[0]); }

  virtual bool mayWriteToMemory() const { return true; }

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
//                                LoadInst Class
//===----------------------------------------------------------------------===//

class LoadInst : public Instruction {
  LoadInst(const LoadInst &LI) : Instruction(LI.getType(), Load) {
    Volatile = LI.isVolatile();
    Operands.reserve(1);
    Operands.push_back(Use(LI.Operands[0], this));
  }
  bool Volatile;   // True if this is a volatile load
public:
  LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const std::string &Name = "", bool isVolatile = false,
           Instruction *InsertBefore = 0);

  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  bool isVolatile() const { return Volatile; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) { Volatile = V; }

  virtual Instruction *clone() const { return new LoadInst(*this); }

  virtual bool mayWriteToMemory() const { return isVolatile(); }

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Load;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                StoreInst Class
//===----------------------------------------------------------------------===//

class StoreInst : public Instruction {
  StoreInst(const StoreInst &SI) : Instruction(SI.getType(), Store) {
    Volatile = SI.isVolatile();
    Operands.reserve(2);
    Operands.push_back(Use(SI.Operands[0], this));
    Operands.push_back(Use(SI.Operands[1], this));
  }
  bool Volatile;   // True if this is a volatile store
public:
  StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile = false,
            Instruction *InsertBefore = 0);


  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  bool isVolatile() const { return Volatile; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) { Volatile = V; }

  virtual Instruction *clone() const { return new StoreInst(*this); }

  virtual bool mayWriteToMemory() const { return true; }

  Value *getPointerOperand() { return getOperand(1); }
  const Value *getPointerOperand() const { return getOperand(1); }
  static unsigned getPointerOperandIndex() { return 1U; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StoreInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Store;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                             GetElementPtrInst Class
//===----------------------------------------------------------------------===//

class GetElementPtrInst : public Instruction {
  GetElementPtrInst(const GetElementPtrInst &EPI)
    : Instruction((Type*)EPI.getType(), GetElementPtr) {
    Operands.reserve(EPI.Operands.size());
    for (unsigned i = 0, E = EPI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(EPI.Operands[i], this));
  }
public:
  GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
		    const std::string &Name = "", Instruction *InsertBefore =0);
  virtual Instruction *clone() const { return new GetElementPtrInst(*this); }
  
  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return (PointerType*)Instruction::getType();
  }

  /// getIndexedType - Returns the type of the element that would be loaded with
  /// a load instruction with the specified parameters.
  ///
  /// A null type is returned if the indices are invalid for the specified 
  /// pointer type.
  ///
  static const Type *getIndexedType(const Type *Ptr, 
				    const std::vector<Value*> &Indices,
				    bool AllowStructLeaf = false);
  
  inline op_iterator       idx_begin()       {
    return op_begin()+1;
  }
  inline const_op_iterator idx_begin() const {
    return op_begin()+1;
  }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }

  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  inline unsigned getNumIndices() const {  // Note: always non-negative
    return getNumOperands() - 1;
  }
  
  inline bool hasIndices() const {
    return getNumOperands() > 1;
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

} // End llvm namespace

#endif // LLVM_IMEMORY_H
