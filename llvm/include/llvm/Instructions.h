//===-- llvm/Instructions.h - Instruction subclass definitions --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file exposes the class definitions of all of the subclasses of the
// Instruction class.  This is meant to be an easy way to get access to all
// instruction subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTIONS_H
#define LLVM_INSTRUCTIONS_H

#include "llvm/Instruction.h"
#include "llvm/InstrTypes.h"

namespace llvm {

class BasicBlock;
class PointerType;

//===----------------------------------------------------------------------===//
//                             AllocationInst Class
//===----------------------------------------------------------------------===//

/// AllocationInst - This class is the common base class of MallocInst and
/// AllocaInst.
///
class AllocationInst : public Instruction {
protected:
  void init(const Type *Ty, Value *ArraySize, unsigned iTy);
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
		 const std::string &Name = "", Instruction *InsertBefore = 0);
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
		 const std::string &Name, BasicBlock *InsertAtEnd);

public:

  /// isArrayAllocation - Return true if there is an allocation size parameter
  /// to the allocation instruction that is not 1.
  ///
  bool isArrayAllocation() const;

  /// getArraySize - Get the number of element allocated, for a simple
  /// allocation of a single element, this will return a constant 1 value.
  ///
  inline const Value *getArraySize() const { return Operands[0]; }
  inline Value *getArraySize() { return Operands[0]; }

  /// getType - Overload to return most specific pointer type
  ///
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Instruction::getType()); 
  }

  /// getAllocatedType - Return the type that is being allocated by the
  /// instruction.
  ///
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

/// MallocInst - an instruction to allocated memory on the heap
///
class MallocInst : public AllocationInst {
  MallocInst(const MallocInst &MI);
public:
  explicit MallocInst(const Type *Ty, Value *ArraySize = 0,
                      const std::string &Name = "",
                      Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Malloc, Name, InsertBefore) {}
  MallocInst(const Type *Ty, Value *ArraySize, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, ArraySize, Malloc, Name, InsertAtEnd) {}

  virtual MallocInst *clone() const;

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

/// AllocaInst - an instruction to allocate memory on the stack
///
class AllocaInst : public AllocationInst {
  AllocaInst(const AllocaInst &);
public:
  explicit AllocaInst(const Type *Ty, Value *ArraySize = 0,
                      const std::string &Name = "",
                      Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Alloca, Name, InsertBefore) {}
  AllocaInst(const Type *Ty, Value *ArraySize, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, ArraySize, Alloca, Name, InsertAtEnd) {}

  virtual AllocaInst *clone() const;

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

/// FreeInst - an instruction to deallocate memory
///
class FreeInst : public Instruction {
  void init(Value *Ptr);

public:
  explicit FreeInst(Value *Ptr, Instruction *InsertBefore = 0);
  FreeInst(Value *Ptr, BasicBlock *InsertAfter);

  virtual FreeInst *clone() const;

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

/// LoadInst - an instruction for reading from memory 
///
class LoadInst : public Instruction {
  LoadInst(const LoadInst &LI) : Instruction(LI.getType(), Load) {
    Volatile = LI.isVolatile();
    init(LI.Operands[0]);
  }
  bool Volatile;   // True if this is a volatile load
  void init(Value *Ptr);
public:
  LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const std::string &Name, BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const std::string &Name = "", bool isVolatile = false,
           Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
           BasicBlock *InsertAtEnd);

  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  ///
  bool isVolatile() const { return Volatile; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) { Volatile = V; }

  virtual LoadInst *clone() const;

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

/// StoreInst - an instruction for storing to memory 
///
class StoreInst : public Instruction {
  StoreInst(const StoreInst &SI) : Instruction(SI.getType(), Store) {
    Volatile = SI.isVolatile();
    init(SI.Operands[0], SI.Operands[1]);
  }
  bool Volatile;   // True if this is a volatile store
  void init(Value *Val, Value *Ptr);
public:
  StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore);
  StoreInst(Value *Val, Value *Ptr, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile = false,
            Instruction *InsertBefore = 0);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile, BasicBlock *InsertAtEnd);


  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  ///
  bool isVolatile() const { return Volatile; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) { Volatile = V; }

  virtual StoreInst *clone() const;

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

/// GetElementPtrInst - an instruction for type-safe pointer arithmetic to
/// access elements of arrays and structs
///
class GetElementPtrInst : public Instruction {
  GetElementPtrInst(const GetElementPtrInst &EPI)
    : Instruction((static_cast<const Instruction*>(&EPI)->getType()),
                  GetElementPtr) {
    Operands.reserve(EPI.Operands.size());
    for (unsigned i = 0, E = (unsigned)EPI.Operands.size(); i != E; ++i)
      Operands.push_back(Use(EPI.Operands[i], this));
  }
  void init(Value *Ptr, const std::vector<Value*> &Idx);
  void init(Value *Ptr, Value *Idx0, Value *Idx1);
public:
  /// Constructors - Create a getelementptr instruction with a base pointer an
  /// list of indices.  The first ctor can optionally insert before an existing
  /// instruction, the second appends the new instruction to the specified
  /// BasicBlock.
  GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
		    const std::string &Name = "", Instruction *InsertBefore =0);
  GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
		    const std::string &Name, BasicBlock *InsertAtEnd);

  /// Constructors - These two constructors are convenience methods because two
  /// index getelementptr instructions are so common.
  GetElementPtrInst(Value *Ptr, Value *Idx0, Value *Idx1,
		    const std::string &Name = "", Instruction *InsertBefore =0);
  GetElementPtrInst(Value *Ptr, Value *Idx0, Value *Idx1,
		    const std::string &Name, BasicBlock *InsertAtEnd);

  virtual GetElementPtrInst *clone() const;
  
  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Instruction::getType());
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
  static const Type *getIndexedType(const Type *Ptr, Value *Idx0, Value *Idx1,
				    bool AllowStructLeaf = false);
  
  inline op_iterator       idx_begin()       { return op_begin()+1; }
  inline const_op_iterator idx_begin() const { return op_begin()+1; }
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

//===----------------------------------------------------------------------===//
//                            SetCondInst Class
//===----------------------------------------------------------------------===//

/// SetCondInst class - Represent a setCC operator, where CC is eq, ne, lt, gt,
/// le, or ge.
///
class SetCondInst : public BinaryOperator {
  BinaryOps OpType;
public:
  SetCondInst(BinaryOps Opcode, Value *LHS, Value *RHS,
	      const std::string &Name = "", Instruction *InsertBefore = 0);
  SetCondInst(BinaryOps Opcode, Value *LHS, Value *RHS,
	      const std::string &Name, BasicBlock *InsertAtEnd);

  /// getInverseCondition - Return the inverse of the current condition opcode.
  /// For example seteq -> setne, setgt -> setle, setlt -> setge, etc...
  ///
  BinaryOps getInverseCondition() const {
    return getInverseCondition(getOpcode());
  }

  /// getInverseCondition - Static version that you can use without an
  /// instruction available.
  ///
  static BinaryOps getInverseCondition(BinaryOps Opcode);

  /// getSwappedCondition - Return the condition opcode that would be the result
  /// of exchanging the two operands of the setcc instruction without changing
  /// the result produced.  Thus, seteq->seteq, setle->setge, setlt->setgt, etc.
  ///
  BinaryOps getSwappedCondition() const {
    return getSwappedCondition(getOpcode());
  }

  /// getSwappedCondition - Static version that you can use without an
  /// instruction available.
  ///
  static BinaryOps getSwappedCondition(BinaryOps Opcode);


  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SetCondInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == SetEQ || I->getOpcode() == SetNE ||
           I->getOpcode() == SetLE || I->getOpcode() == SetGE ||
           I->getOpcode() == SetLT || I->getOpcode() == SetGT;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 CastInst Class
//===----------------------------------------------------------------------===//

/// CastInst - This class represents a cast from Operand[0] to the type of
/// the instruction (i->getType()).
///
class CastInst : public Instruction {
  CastInst(const CastInst &CI) : Instruction(CI.getType(), Cast) {
    Operands.reserve(1);
    Operands.push_back(Use(CI.Operands[0], this));
  }
  void init(Value *S) {
    Operands.reserve(1);
    Operands.push_back(Use(S, this));
  }
public:
  CastInst(Value *S, const Type *Ty, const std::string &Name = "",
           Instruction *InsertBefore = 0)
    : Instruction(Ty, Cast, Name, InsertBefore) {
    init(S);
  }
  CastInst(Value *S, const Type *Ty, const std::string &Name,
           BasicBlock *InsertAtEnd)
    : Instruction(Ty, Cast, Name, InsertAtEnd) {
    init(S);
  }

  virtual CastInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CastInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Cast;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                 CallInst Class
//===----------------------------------------------------------------------===//

/// CallInst - This class represents a function call, abstracting a target
/// machine's calling convention.
///
class CallInst : public Instruction {
  CallInst(const CallInst &CI);
  void init(Value *Func, const std::vector<Value*> &Params);
  void init(Value *Func, Value *Actual1, Value *Actual2);
  void init(Value *Func, Value *Actual);
  void init(Value *Func);

public:
  CallInst(Value *F, const std::vector<Value*> &Par,
           const std::string &Name = "", Instruction *InsertBefore = 0);
  CallInst(Value *F, const std::vector<Value*> &Par,
           const std::string &Name, BasicBlock *InsertAtEnd);

  // Alternate CallInst ctors w/ two actuals, w/ one actual and no
  // actuals, respectively.
  CallInst(Value *F, Value *Actual1, Value *Actual2,
           const std::string& Name = "", Instruction *InsertBefore = 0);
  CallInst(Value *F, Value *Actual1, Value *Actual2,
           const std::string& Name, BasicBlock *InsertAtEnd);
  CallInst(Value *F, Value *Actual, const std::string& Name = "",
           Instruction *InsertBefore = 0);
  CallInst(Value *F, Value *Actual, const std::string& Name,
           BasicBlock *InsertAtEnd);
  explicit CallInst(Value *F, const std::string &Name = "", 
                    Instruction *InsertBefore = 0);
  explicit CallInst(Value *F, const std::string &Name, 
                    BasicBlock *InsertAtEnd);

  virtual CallInst *clone() const;
  bool mayWriteToMemory() const { return true; }

  // FIXME: These methods should be inline once we eliminate
  // ConstantPointerRefs!
  const Function *getCalledFunction() const;
  Function *getCalledFunction();

  // getCalledValue - Get a pointer to a method that is invoked by this inst.
  inline const Value *getCalledValue() const { return Operands[0]; }
  inline       Value *getCalledValue()       { return Operands[0]; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Call; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                 ShiftInst Class
//===----------------------------------------------------------------------===//

/// ShiftInst - This class represents left and right shift instructions.
///
class ShiftInst : public Instruction {
  ShiftInst(const ShiftInst &SI) : Instruction(SI.getType(), SI.getOpcode()) {
    Operands.reserve(2);
    Operands.push_back(Use(SI.Operands[0], this));
    Operands.push_back(Use(SI.Operands[1], this));
  }
  void init(OtherOps Opcode, Value *S, Value *SA) {
    assert((Opcode == Shl || Opcode == Shr) && "ShiftInst Opcode invalid!");
    Operands.reserve(2);
    Operands.push_back(Use(S, this));
    Operands.push_back(Use(SA, this));
  }

public:
  ShiftInst(OtherOps Opcode, Value *S, Value *SA, const std::string &Name = "",
            Instruction *InsertBefore = 0)
    : Instruction(S->getType(), Opcode, Name, InsertBefore) {
    init(Opcode, S, SA);
  }
  ShiftInst(OtherOps Opcode, Value *S, Value *SA, const std::string &Name,
            BasicBlock *InsertAtEnd)
    : Instruction(S->getType(), Opcode, Name, InsertAtEnd) {
    init(Opcode, S, SA);
  }

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  virtual ShiftInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ShiftInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Shr) | 
           (I->getOpcode() == Instruction::Shl);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               SelectInst Class
//===----------------------------------------------------------------------===//

/// SelectInst - This class represents the LLVM 'select' instruction.
///
class SelectInst : public Instruction {
  SelectInst(const SelectInst &SI) : Instruction(SI.getType(), SI.getOpcode()) {
    Operands.reserve(3);
    Operands.push_back(Use(SI.Operands[0], this));
    Operands.push_back(Use(SI.Operands[1], this));
    Operands.push_back(Use(SI.Operands[2], this));
  }
  void init(Value *C, Value *S1, Value *S2) {
    Operands.reserve(3);
    Operands.push_back(Use(C, this));
    Operands.push_back(Use(S1, this));
    Operands.push_back(Use(S2, this));
  }

public:
  SelectInst(Value *C, Value *S1, Value *S2, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(S1->getType(), Instruction::Select, Name, InsertBefore) {
    init(C, S1, S2);
  }
  SelectInst(Value *C, Value *S1, Value *S2, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : Instruction(S1->getType(), Instruction::Select, Name, InsertAtEnd) {
    init(C, S1, S2);
  }

  Value *getCondition() const { return Operands[0]; }
  Value *getTrueValue() const { return Operands[1]; }
  Value *getFalseValue() const { return Operands[2]; }

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  virtual SelectInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SelectInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Select;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                VANextInst Class
//===----------------------------------------------------------------------===//

/// VANextInst - This class represents the va_next llvm instruction, which
/// advances a vararg list passed an argument of the specified type, returning
/// the resultant list.
///
class VANextInst : public Instruction {
  PATypeHolder ArgTy;
  void init(Value *List) {
    Operands.reserve(1);
    Operands.push_back(Use(List, this));
  }
  VANextInst(const VANextInst &VAN)
    : Instruction(VAN.getType(), VANext), ArgTy(VAN.getArgType()) {
    init(VAN.Operands[0]);
  }

public:
  VANextInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(List->getType(), VANext, Name, InsertBefore), ArgTy(Ty) {
    init(List);
  }
  VANextInst(Value *List, const Type *Ty, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : Instruction(List->getType(), VANext, Name, InsertAtEnd), ArgTy(Ty) {
    init(List);
  }

  const Type *getArgType() const { return ArgTy; }

  virtual VANextInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VANextInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VANext;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                VAArgInst Class
//===----------------------------------------------------------------------===//

/// VAArgInst - This class represents the va_arg llvm instruction, which returns
/// an argument of the specified type given a va_list.
///
class VAArgInst : public Instruction {
  void init(Value* List) {
    Operands.reserve(1);
    Operands.push_back(Use(List, this));
  }
  VAArgInst(const VAArgInst &VAA)
    : Instruction(VAA.getType(), VAArg) {
    init(VAA.Operands[0]);
  }
public:
  VAArgInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(Ty, VAArg, Name, InsertBefore) {
    init(List);
  }
  VAArgInst(Value *List, const Type *Ty, const std::string &Name,
            BasicBlock *InsertAtEnd)
    : Instruction(Ty, VAArg, Name, InsertAtEnd) {
    init(List);
  }

  virtual VAArgInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VAArgInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VAArg;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

// PHINode - The PHINode class is used to represent the magical mystical PHI
// node, that can not exist in nature, but can be synthesized in a computer
// scientist's overactive imagination.
//
class PHINode : public Instruction {
  PHINode(const PHINode &PN);
public:
  PHINode(const Type *Ty, const std::string &Name = "",
          Instruction *InsertBefore = 0)
    : Instruction(Ty, Instruction::PHI, Name, InsertBefore) {
  }

  PHINode(const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd)
    : Instruction(Ty, Instruction::PHI, Name, InsertAtEnd) {
  }

  virtual PHINode *clone() const;

  /// getNumIncomingValues - Return the number of incoming edges
  ///
  unsigned getNumIncomingValues() const { return (unsigned)Operands.size()/2; }

  /// getIncomingValue - Return incoming value #x
  ///
  Value *getIncomingValue(unsigned i) const {
    assert(i*2 < Operands.size() && "Invalid value number!");
    return Operands[i*2];
  }
  void setIncomingValue(unsigned i, Value *V) {
    assert(i*2 < Operands.size() && "Invalid value number!");
    Operands[i*2] = V;
  }
  inline unsigned getOperandNumForIncomingValue(unsigned i) {
    return i*2;
  }

  /// getIncomingBlock - Return incoming basic block #x
  ///
  BasicBlock *getIncomingBlock(unsigned i) const { 
    assert(i*2+1 < Operands.size() && "Invalid value number!");
    return reinterpret_cast<BasicBlock*>(Operands[i*2+1].get());
  }
  void setIncomingBlock(unsigned i, BasicBlock *BB) {
    assert(i*2+1 < Operands.size() && "Invalid value number!");
    Operands[i*2+1] = reinterpret_cast<Value*>(BB);
  }
  unsigned getOperandNumForIncomingBlock(unsigned i) {
    return i*2+1;
  }

  /// addIncoming - Add an incoming value to the end of the PHI list
  ///
  void addIncoming(Value *V, BasicBlock *BB) {
    assert(getType() == V->getType() &&
           "All operands to PHI node must be the same type as the PHI node!");
    Operands.push_back(Use(V, this));
    Operands.push_back(Use(reinterpret_cast<Value*>(BB), this));
  }
  
  /// removeIncomingValue - Remove an incoming value.  This is useful if a
  /// predecessor basic block is deleted.  The value removed is returned.
  ///
  /// If the last incoming value for a PHI node is removed (and DeletePHIIfEmpty
  /// is true), the PHI node is destroyed and any uses of it are replaced with
  /// dummy values.  The only time there should be zero incoming values to a PHI
  /// node is when the block is dead, so this strategy is sound.
  ///
  Value *removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty = true);

  Value *removeIncomingValue(const BasicBlock *BB, bool DeletePHIIfEmpty =true){
    int Idx = getBasicBlockIndex(BB);
    assert(Idx >= 0 && "Invalid basic block argument to remove!");
    return removeIncomingValue(Idx, DeletePHIIfEmpty);
  }

  /// getBasicBlockIndex - Return the first index of the specified basic 
  /// block in the value list for this PHI.  Returns -1 if no instance.
  ///
  int getBasicBlockIndex(const BasicBlock *BB) const {
    for (unsigned i = 0; i < Operands.size()/2; ++i) 
      if (getIncomingBlock(i) == BB) return i;
    return -1;
  }

  Value *getIncomingValueForBlock(const BasicBlock *BB) const {
    return getIncomingValue(getBasicBlockIndex(BB));
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PHINode *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::PHI; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               ReturnInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// ReturnInst - Return a value (possibly void), from a function.  Execution
/// does not continue in this function any longer.
///
class ReturnInst : public TerminatorInst {
  ReturnInst(const ReturnInst &RI) : TerminatorInst(Instruction::Ret) {
    if (RI.Operands.size()) {
      assert(RI.Operands.size() == 1 && "Return insn can only have 1 operand!");
      Operands.reserve(1);
      Operands.push_back(Use(RI.Operands[0], this));
    }
  }

  void init(Value *RetVal);

public:
  // ReturnInst constructors:
  // ReturnInst()                  - 'ret void' instruction
  // ReturnInst(    null)          - 'ret void' instruction
  // ReturnInst(Value* X)          - 'ret X'    instruction
  // ReturnInst(    null, Inst *)  - 'ret void' instruction, insert before I
  // ReturnInst(Value* X, Inst *I) - 'ret X'    instruction, insert before I
  // ReturnInst(    null, BB *B)   - 'ret void' instruction, insert @ end of BB
  // ReturnInst(Value* X, BB *B)   - 'ret X'    instruction, insert @ end of BB
  //
  // NOTE: If the Value* passed is of type void then the constructor behaves as
  // if it was passed NULL.
  ReturnInst(Value *RetVal = 0, Instruction *InsertBefore = 0)
    : TerminatorInst(Instruction::Ret, InsertBefore) {
    init(RetVal);
  }
  ReturnInst(Value *RetVal, BasicBlock *InsertAtEnd)
    : TerminatorInst(Instruction::Ret, InsertAtEnd) {
    init(RetVal);
  }
  ReturnInst(BasicBlock *InsertAtEnd)
    : TerminatorInst(Instruction::Ret, InsertAtEnd) {
  }

  virtual ReturnInst *clone() const;

  inline const Value *getReturnValue() const {
    return Operands.size() ? Operands[0].get() : 0; 
  }
  inline       Value *getReturnValue()       {
    return Operands.size() ? Operands[0].get() : 0;
  }

  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    assert(0 && "ReturnInst has no successors!");
    abort();
    return 0;
  }
  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc);
  virtual unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ReturnInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Ret);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               BranchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// BranchInst - Conditional or Unconditional Branch instruction.
///
class BranchInst : public TerminatorInst {
  BranchInst(const BranchInst &BI);
  void init(BasicBlock *IfTrue);
  void init(BasicBlock *True, BasicBlock *False, Value *Cond);
public:
  // BranchInst constructors (where {B, T, F} are blocks, and C is a condition):
  // BranchInst(BB *B)                           - 'br B'
  // BranchInst(BB* T, BB *F, Value *C)          - 'br C, T, F'
  // BranchInst(BB* B, Inst *I)                  - 'br B'        insert before I
  // BranchInst(BB* T, BB *F, Value *C, Inst *I) - 'br C, T, F', insert before I
  // BranchInst(BB* B, BB *I)                    - 'br B'        insert at end
  // BranchInst(BB* T, BB *F, Value *C, BB *I)   - 'br C, T, F', insert at end
  BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore = 0)
    : TerminatorInst(Instruction::Br, InsertBefore) {
    init(IfTrue);
  }
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             Instruction *InsertBefore = 0)
    : TerminatorInst(Instruction::Br, InsertBefore) {
    init(IfTrue, IfFalse, Cond);
  }

  BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd)
    : TerminatorInst(Instruction::Br, InsertAtEnd) {
    init(IfTrue);
  }

  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             BasicBlock *InsertAtEnd)
    : TerminatorInst(Instruction::Br, InsertAtEnd) {
    init(IfTrue, IfFalse, Cond);
  }

  virtual BranchInst *clone() const;

  inline bool isUnconditional() const { return Operands.size() == 1; }
  inline bool isConditional()   const { return Operands.size() == 3; }

  inline Value *getCondition() const {
    assert(isConditional() && "Cannot get condition of an uncond branch!");
    return Operands[2].get();
  }

  void setCondition(Value *V) {
    assert(isConditional() && "Cannot set condition of unconditional branch!");
    setOperand(2, V);
  }

  // setUnconditionalDest - Change the current branch to an unconditional branch
  // targeting the specified block.
  //
  void setUnconditionalDest(BasicBlock *Dest) {
    if (isConditional()) Operands.erase(Operands.begin()+1, Operands.end());
    Operands[0] = reinterpret_cast<Value*>(Dest);
  }

  virtual const BasicBlock *getSuccessor(unsigned i) const {
    assert(i < getNumSuccessors() && "Successor # out of range for Branch!");
    return (i == 0) ? cast<BasicBlock>(Operands[0].get()) : 
                      cast<BasicBlock>(Operands[1].get());
  }
  inline BasicBlock *getSuccessor(unsigned idx) {
    const BranchInst *BI = this;
    return const_cast<BasicBlock*>(BI->getSuccessor(idx));
  }

  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for Branch!");
    Operands[idx] = reinterpret_cast<Value*>(NewSucc);
  }

  virtual unsigned getNumSuccessors() const { return 1+isConditional(); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BranchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Br);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               SwitchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// SwitchInst - Multiway switch
///
class SwitchInst : public TerminatorInst {
  // Operand[0]    = Value to switch on
  // Operand[1]    = Default basic block destination
  // Operand[2n  ] = Value to match
  // Operand[2n+1] = BasicBlock to go to on match
  SwitchInst(const SwitchInst &RI);
  void init(Value *Value, BasicBlock *Default);

public:
  SwitchInst(Value *Value, BasicBlock *Default, Instruction *InsertBefore = 0) 
    : TerminatorInst(Instruction::Switch, InsertBefore) {
    init(Value, Default);
  }
  SwitchInst(Value *Value, BasicBlock *Default, BasicBlock  *InsertAtEnd) 
    : TerminatorInst(Instruction::Switch, InsertAtEnd) {
    init(Value, Default);
  }

  virtual SwitchInst *clone() const;

  // Accessor Methods for Switch stmt
  //
  inline const Value *getCondition() const { return Operands[0]; }
  inline       Value *getCondition()       { return Operands[0]; }
  inline const BasicBlock *getDefaultDest() const {
    return cast<BasicBlock>(Operands[1].get());
  }
  inline       BasicBlock *getDefaultDest()       {
    return cast<BasicBlock>(Operands[1].get());
  }

  /// getNumCases - return the number of 'cases' in this switch instruction.
  /// Note that case #0 is always the default case.
  unsigned getNumCases() const {
    return (unsigned)Operands.size()/2;
  }

  /// getCaseValue - Return the specified case value.  Note that case #0, the
  /// default destination, does not have a case value.
  Constant *getCaseValue(unsigned i) {
    assert(i && i < getNumCases() && "Illegal case value to get!");
    return getSuccessorValue(i);
  }

  /// getCaseValue - Return the specified case value.  Note that case #0, the
  /// default destination, does not have a case value.
  const Constant *getCaseValue(unsigned i) const {
    assert(i && i < getNumCases() && "Illegal case value to get!");
    return getSuccessorValue(i);
  }

  /// findCaseValue - Search all of the case values for the specified constant.
  /// If it is explicitly handled, return the case number of it, otherwise
  /// return 0 to indicate that it is handled by the default handler.
  unsigned findCaseValue(const Constant *C) const {
    for (unsigned i = 1, e = getNumCases(); i != e; ++i)
      if (getCaseValue(i) == C)
        return i;
    return 0;
  }

  /// addCase - Add an entry to the switch instruction...
  ///
  void addCase(Constant *OnVal, BasicBlock *Dest);

  /// removeCase - This method removes the specified successor from the switch
  /// instruction.  Note that this cannot be used to remove the default
  /// destination (successor #0).
  ///
  void removeCase(unsigned idx);

  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(Operands[idx*2+1].get());
  }
  inline BasicBlock *getSuccessor(unsigned idx) {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(Operands[idx*2+1].get());
  }

  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for switch!");
    Operands[idx*2+1] = reinterpret_cast<Value*>(NewSucc);
  }

  // getSuccessorValue - Return the value associated with the specified
  // successor.
  inline const Constant *getSuccessorValue(unsigned idx) const {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return cast<Constant>(Operands[idx*2].get());
  }
  inline Constant *getSuccessorValue(unsigned idx) {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return cast<Constant>(Operands[idx*2].get());
  }
  virtual unsigned getNumSuccessors() const { return (unsigned)Operands.size()/2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SwitchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Switch);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               InvokeInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// InvokeInst - Invoke instruction
///
class InvokeInst : public TerminatorInst {
  InvokeInst(const InvokeInst &BI);
  void init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
            const std::vector<Value*> &Params);
public:
  InvokeInst(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
	     const std::vector<Value*> &Params, const std::string &Name = "",
             Instruction *InsertBefore = 0);
  InvokeInst(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
	     const std::vector<Value*> &Params, const std::string &Name,
             BasicBlock *InsertAtEnd);

  virtual InvokeInst *clone() const;

  bool mayWriteToMemory() const { return true; }

  /// getCalledFunction - Return the function called, or null if this is an
  /// indirect function invocation... 
  ///
  /// FIXME: These should be inlined once we get rid of ConstantPointerRefs!
  ///
  const Function *getCalledFunction() const;
  Function *getCalledFunction();

  // getCalledValue - Get a pointer to a function that is invoked by this inst.
  inline const Value *getCalledValue() const { return Operands[0]; }
  inline       Value *getCalledValue()       { return Operands[0]; }

  // get*Dest - Return the destination basic blocks...
  inline const BasicBlock *getNormalDest() const {
    return cast<BasicBlock>(Operands[1].get());
  }
  inline       BasicBlock *getNormalDest() {
    return cast<BasicBlock>(Operands[1].get());
  }
  inline const BasicBlock *getUnwindDest() const {
    return cast<BasicBlock>(Operands[2].get());
  }
  inline       BasicBlock *getUnwindDest() {
    return cast<BasicBlock>(Operands[2].get());
  }

  inline void setNormalDest(BasicBlock *B){
    Operands[1] = reinterpret_cast<Value*>(B);
  }

  inline void setUnwindDest(BasicBlock *B){
    Operands[2] = reinterpret_cast<Value*>(B);
  }

  virtual const BasicBlock *getSuccessor(unsigned i) const {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getUnwindDest();
  }
  inline BasicBlock *getSuccessor(unsigned i) {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getUnwindDest();
  }

  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < 2 && "Successor # out of range for invoke!");
    Operands[idx+1] = reinterpret_cast<Value*>(NewSucc);
  }

  virtual unsigned getNumSuccessors() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InvokeInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Invoke);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                              UnwindInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// UnwindInst - Immediately exit the current function, unwinding the stack
/// until an invoke instruction is found.
///
class UnwindInst : public TerminatorInst {
public:
  UnwindInst(Instruction *InsertBefore = 0)
    : TerminatorInst(Instruction::Unwind, InsertBefore) {
  }
  UnwindInst(BasicBlock *InsertAtEnd)
    : TerminatorInst(Instruction::Unwind, InsertAtEnd) {
  }

  virtual UnwindInst *clone() const;

  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    assert(0 && "UnwindInst has no successors!");
    abort();
    return 0;
  }
  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc);
  virtual unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnwindInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unwind;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                           UnreachableInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// UnreachableInst - This function has undefined behavior.  In particular, the
/// presence of this instruction indicates some higher level knowledge that the
/// end of the block cannot be reached.
///
class UnreachableInst : public TerminatorInst {
public:
  UnreachableInst(Instruction *InsertBefore = 0)
    : TerminatorInst(Instruction::Unreachable, InsertBefore) {
  }
  UnreachableInst(BasicBlock *InsertAtEnd)
    : TerminatorInst(Instruction::Unreachable, InsertAtEnd) {
  }

  virtual UnreachableInst *clone() const;

  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    assert(0 && "UnreachableInst has no successors!");
    abort();
    return 0;
  }
  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc);
  virtual unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnreachableInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unreachable;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
