//===-- llvm/Instruction.h - Instruction class definition --------*- C++ -*--=//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the VM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTION_H
#define LLVM_INSTRUCTION_H

#include "llvm/User.h"

class Type;
class BasicBlock;
class Method;

class Instruction : public User {
  BasicBlock *Parent;
  unsigned iType;      // InstructionType

  friend class ValueHolder<Instruction,BasicBlock>;
  inline void setParent(BasicBlock *P) { Parent = P; }

public:
  Instruction(const Type *Ty, unsigned iType, const string &Name = "");
  virtual ~Instruction();  // Virtual dtor == good.

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name);

  // clone() - Create a copy of 'this' instruction that is identical in all ways
  // except the following:
  //   * The instruction has no parent
  //   * The instruction has no name
  //
  virtual Instruction *clone() const = 0;

  // Accessor methods...
  //
  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }
  bool hasSideEffects() const { return false; }  // Memory & Call insts = true

  // ---------------------------------------------------------------------------
  // Implement the User interface 
  // if i > the number of operands, then getOperand() returns 0, and setOperand
  // returns false.  setOperand() may also return false if the operand is of
  // the wrong type.
  //
  inline Value *getOperand(unsigned i) {
    return (Value*)((const Instruction *)this)->getOperand(i);
  }
  virtual const Value *getOperand(unsigned i) const = 0;
  virtual bool setOperand(unsigned i, Value *Val) = 0;
  virtual unsigned getNumOperands() const = 0;

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  template <class _Inst, class _Val> class OperandIterator;
  typedef OperandIterator<Instruction *, Value *> op_iterator;
  typedef OperandIterator<const Instruction *, const Value *> op_const_iterator;

  inline op_iterator       op_begin()      ;
  inline op_const_iterator op_begin() const;
  inline op_iterator       op_end()        ;
  inline op_const_iterator op_end()   const;


  // ---------------------------------------------------------------------------
  // Subclass classification... getInstType() returns a member of 
  // one of the enums that is coming soon (down below)...
  //
  virtual string getOpcode() const = 0;

  unsigned getInstType() const { return iType; }
  inline bool isTerminator() const {   // Instance of TerminatorInst?
    return iType >= FirstTermOp && iType < NumTermOps; 
  }
  inline bool isDefinition() const { return !isTerminator(); }
  inline bool isUnaryOp() const {
    return iType >= FirstUnaryOp && iType < NumUnaryOps;
  }
  inline bool isBinaryOp() const {
    return iType >= FirstBinaryOp && iType < NumBinaryOps;
  }

  // isPHINode() - This is used frequently enough to allow it to exist
  inline bool isPHINode() const { return iType == PHINode; }


  //----------------------------------------------------------------------
  // Exported enumerations...
  //
  enum TermOps {       // These terminate basic blocks
    FirstTermOp = 1,
    Ret = 1, Br, Switch, 
    NumTermOps         // Must remain at end of enum
  };

  enum UnaryOps {
    FirstUnaryOp = NumTermOps,
    Neg          = NumTermOps, Not, 
    
    // Type conversions...
    ToBoolTy  , 
    ToUByteTy , ToSByteTy,  ToUShortTy, ToShortTy,
    ToUInt    , ToInt,      ToULongTy , ToLongTy,

    ToFloatTy , ToDoubleTy, ToArrayTy , ToPointerTy,

    NumUnaryOps        // Must remain at end of enum
  };

  enum BinaryOps {
    // Standard binary operators...
    FirstBinaryOp = NumUnaryOps,
    Add = NumUnaryOps, Sub, Mul, Div, Rem,

    // Logical operators...
    And, Or, Xor,

    // Binary comparison operators...
    SetEQ, SetNE, SetLE, SetGE, SetLT, SetGT,

    NumBinaryOps
  };

  enum MemoryOps {
    FirstMemoryOp = NumBinaryOps,
    Malloc = NumBinaryOps, Free,     // Heap management instructions
    Alloca,                          // Stack management instruction

    Load, Store,                     // Memory manipulation instructions.

    GetField, PutField,              // Structure manipulation instructions

    NumMemoryOps
  };

  enum OtherOps {
    FirstOtherOp = NumMemoryOps,
    PHINode      = NumMemoryOps,     // PHI node instruction
    Call,                            // Call a function

    Shl, Shr,                        // Shift operations...

    NumOps,                          // Must be the last 'op' defined.
    UserOp1, UserOp2                 // May be used internally to a pass...
  };

public:
  template <class _Inst, class _Val>         // Operand Iterator Implementation
  class OperandIterator {
    const _Inst Inst;
    unsigned idx;
  public:
    typedef OperandIterator<_Inst, _Val> _Self;
    typedef bidirectional_iterator_tag iterator_category;
    typedef _Val pointer;
    
    inline OperandIterator(_Inst T) : Inst(T), idx(0) {}    // begin iterator
    inline OperandIterator(_Inst T, bool) 
      : Inst(T), idx(Inst->getNumOperands()) {}             // end iterator
    
    inline bool operator==(const _Self& x) const { return idx == x.idx; }
    inline bool operator!=(const _Self& x) const { return !operator==(x); }

    inline pointer operator*() const { return Inst->getOperand(idx); }
    inline pointer *operator->() const { return &(operator*()); }
    
    inline _Self& operator++() { ++idx; return *this; } // Preincrement
    inline _Self operator++(int) { // Postincrement
      _Self tmp = *this; ++*this; return tmp; 
    }

    inline _Self& operator--() { --idx; return *this; }  // Predecrement
    inline _Self operator--(int) { // Postdecrement
      _Self tmp = *this; --*this; return tmp;
    }
  };

};

inline Instruction::op_iterator       Instruction::op_begin()       {
  return op_iterator(this);
}
inline Instruction::op_const_iterator Instruction::op_begin() const {
  return op_const_iterator(this);
}
inline Instruction::op_iterator       Instruction::op_end()         {
  return op_iterator(this,true);
}
inline Instruction::op_const_iterator Instruction::op_end()   const {
  return op_const_iterator(this,true);
}


#endif
