//===-- BBLiveVar.h - Live Variable Analysis for a BasicBlock ----*- C++ -*--=//
//
// This is a wrapper class for BasicBlock which is used by live var analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LIVE_VAR_BB_H
#define LIVE_VAR_BB_H

#include "llvm/Analysis/LiveVar/ValueSet.h"
#include <map>
class Method;
class BasicBlock;
class Value;

class BBLiveVar {
  const BasicBlock *BB;         // pointer to BasicBlock
  unsigned POID;                // Post-Order ID

  ValueSet DefSet;            // Def set for LV analysis
  ValueSet InSet, OutSet;     // In & Out for LV analysis
  bool InSetChanged, OutSetChanged;   // set if the InSet/OutSet is modified

                                // map that contains phi args->BB they came
                                // set by calcDefUseSets & used by setPropagate
  std::map<const Value *, const BasicBlock *> PhiArgMap;  

  // method to propogate an InSet to OutSet of a predecessor
  bool setPropagate(ValueSet *OutSetOfPred, 
                    const ValueSet *InSetOfThisBB,
                    const BasicBlock *PredBB);

  // To add an operand which is a def
  void  addDef(const Value *Op); 

  // To add an operand which is a use
  void  addUse(const Value *Op);

 public:
  BBLiveVar(const BasicBlock *BB, unsigned POID);

  inline bool isInSetChanged() const  { return InSetChanged; }    
  inline bool isOutSetChanged() const { return OutSetChanged; }

  inline unsigned getPOId() const { return POID; }

  void calcDefUseSets();         // calculates the Def & Use sets for this BB
  bool applyTransferFunc();      // calcultes the In in terms of Out 

  // calculates Out set using In sets of the predecessors
  bool applyFlowFunc(std::map<const BasicBlock *, BBLiveVar *> &LVMap);    

  inline const ValueSet *getOutSet() const { return &OutSet; }
  inline const ValueSet  *getInSet() const { return &InSet; }

  void printAllSets() const;      // for printing Def/In/Out sets
  void printInOutSets() const;    // for printing In/Out sets
};

#endif

