//===-- llvm/iPHINode.h - PHI instruction definition -------------*- C++ -*--=//
//
// This file defines the PHINode class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IPHINODE_H
#define LLVM_IPHINODE_H

#include "llvm/Instruction.h"
class BasicBlock;

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
  PHINode(const Type *Ty, const std::string &Name = "");

  virtual Instruction *clone() const { return new PHINode(*this); }

  // getNumIncomingValues - Return the number of incoming edges the PHI node has
  inline unsigned getNumIncomingValues() const { return Operands.size()/2; }

  // getIncomingValue - Return incoming value #x
  inline const Value *getIncomingValue(unsigned i) const {
    return Operands[i*2];
  }
  inline Value *getIncomingValue(unsigned i) {
    return Operands[i*2];
  }
  inline void setIncomingValue(unsigned i, Value *V) {
    Operands[i*2] = V;
  }

  // getIncomingBlock - Return incoming basic block #x
  inline const BasicBlock *getIncomingBlock(unsigned i) const { 
    return (const BasicBlock*)Operands[i*2+1].get();
  }
  inline BasicBlock *getIncomingBlock(unsigned i) { 
    return (BasicBlock*)Operands[i*2+1].get();
  }
  inline void setIncomingBlock(unsigned i, BasicBlock *BB) {
    Operands[i*2+1] = (Value*)BB;
  }

  // addIncoming - Add an incoming value to the end of the PHI list
  void addIncoming(Value *D, BasicBlock *BB);

  // removeIncomingValue - Remove an incoming value.  This is useful if a
  // predecessor basic block is deleted.  The value removed is returned.
  Value *removeIncomingValue(const BasicBlock *BB);

  // getBasicBlockIndex - Return the first index of the specified basic 
  // block in the value list for this PHI.  Returns -1 if no instance.
  //
  int getBasicBlockIndex(const BasicBlock *BB) const {
    for (unsigned i = 0; i < Operands.size()/2; ++i) 
      if (getIncomingBlock(i) == BB) return i;
    return -1;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PHINode *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::PHINode; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

#endif
