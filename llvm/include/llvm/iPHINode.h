//===-- llvm/iPHINode.h - PHI instruction definition ------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
  PHINode(const Type *Ty, const std::string &Name = "",
          Instruction *InsertBefore = 0)
    : Instruction(Ty, Instruction::PHI, Name, InsertBefore) {
  }

  virtual Instruction *clone() const { return new PHINode(*this); }

  /// getNumIncomingValues - Return the number of incoming edges the PHI node
  /// has
  unsigned getNumIncomingValues() const { return Operands.size()/2; }

  /// getIncomingValue - Return incoming value #x
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
  BasicBlock *getIncomingBlock(unsigned i) const { 
    assert(i*2+1 < Operands.size() && "Invalid value number!");
    return (BasicBlock*)Operands[i*2+1].get();
  }
  void setIncomingBlock(unsigned i, BasicBlock *BB) {
    assert(i*2+1 < Operands.size() && "Invalid value number!");
    Operands[i*2+1] = (Value*)BB;
  }
  unsigned getOperandNumForIncomingBlock(unsigned i) {
    return i*2+1;
  }

  /// addIncoming - Add an incoming value to the end of the PHI list
  void addIncoming(Value *D, BasicBlock *BB) {
    assert(getType() == D->getType() &&
           "All operands to PHI node must be the same type as the PHI node!");
    Operands.push_back(Use(D, this));
    Operands.push_back(Use((Value*)BB, this));
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

#endif
