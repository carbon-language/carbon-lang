//===-- llvm/iOther.h - "Other" instruction node definitions ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for instructions that fall into the 
// grandiose 'other' catagory...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IOTHER_H
#define LLVM_IOTHER_H

#include "llvm/InstrTypes.h"

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
public:
  CastInst(Value *S, const Type *Ty, const std::string &Name = "",
           Instruction *InsertBefore = 0)
    : Instruction(Ty, Cast, Name, InsertBefore) {
    Operands.reserve(1);
    Operands.push_back(Use(S, this));
  }

  virtual Instruction *clone() const { return new CastInst(*this); }

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

class CallInst : public Instruction {
  CallInst(const CallInst &CI);
public:
  CallInst(Value *F, const std::vector<Value*> &Par,
           const std::string &Name = "", Instruction *InsertBefore = 0);

  // Alternate CallInst ctors w/ no actuals & one actual, respectively.
  CallInst(Value *F, const std::string &Name = "",
           Instruction  *InsertBefore = 0);
  CallInst(Value *F, Value *Actual, const std::string& Name = "",
           Instruction* InsertBefore = 0);

  virtual Instruction *clone() const { return new CallInst(*this); }
  bool mayWriteToMemory() const { return true; }

  const Function *getCalledFunction() const {
    return dyn_cast<Function>(Operands[0].get());
  }
  Function *getCalledFunction() {
    return dyn_cast<Function>(Operands[0].get());
  }

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

// ShiftInst - This class represents left and right shift instructions.
//
class ShiftInst : public Instruction {
  ShiftInst(const ShiftInst &SI) : Instruction(SI.getType(), SI.getOpcode()) {
    Operands.reserve(2);
    Operands.push_back(Use(SI.Operands[0], this));
    Operands.push_back(Use(SI.Operands[1], this));
  }
public:
  ShiftInst(OtherOps Opcode, Value *S, Value *SA, const std::string &Name = "",
            Instruction *InsertBefore = 0)
    : Instruction(S->getType(), Opcode, Name, InsertBefore) {
    assert((Opcode == Shl || Opcode == Shr) && "ShiftInst Opcode invalid!");
    Operands.reserve(2);
    Operands.push_back(Use(S, this));
    Operands.push_back(Use(SA, this));
  }

  OtherOps getOpcode() const { return (OtherOps)Instruction::getOpcode(); }

  virtual Instruction *clone() const { return new ShiftInst(*this); }

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
//                                VANextInst Class
//===----------------------------------------------------------------------===//

/// VANextInst - This class represents the va_next llvm instruction, which
/// advances a vararg list passed an argument of the specified type, returning
/// the resultant list.
///
class VANextInst : public Instruction {
  PATypeHolder ArgTy;
  VANextInst(const VANextInst &VAN)
    : Instruction(VAN.getType(), VANext), ArgTy(VAN.getArgType()) {
    Operands.reserve(1);
    Operands.push_back(Use(VAN.Operands[0], this));
  }
public:
  VANextInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(List->getType(), VANext, Name, InsertBefore), ArgTy(Ty) {
    Operands.reserve(1);
    Operands.push_back(Use(List, this));
  }

  const Type *getArgType() const { return ArgTy; }

  virtual Instruction *clone() const { return new VANextInst(*this); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VANextInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VANext;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

/// VAArgInst - This class represents the va_arg llvm instruction, which returns
/// an argument of the specified type given a va_list.
///
class VAArgInst : public Instruction {
  VAArgInst(const VAArgInst &VAA)
    : Instruction(VAA.getType(), VAArg) {
    Operands.reserve(1);
    Operands.push_back(Use(VAA.Operands[0], this));
  }
public:
  VAArgInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(Ty, VAArg, Name, InsertBefore) {
    Operands.reserve(1);
    Operands.push_back(Use(List, this));
  }

  virtual Instruction *clone() const { return new VAArgInst(*this); }

  bool mayWriteToMemory() const { return true; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VAArgInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VAArg;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

#endif
