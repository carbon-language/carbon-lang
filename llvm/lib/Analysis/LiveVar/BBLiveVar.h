/* Title:   BBLiveVar.h                  -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: This is a wrapper class for BasicBlock which is used by live 
            variable anaysis.
*/

#ifndef LIVE_VAR_BB_H
#define LIVE_VAR_BB_H

#include "LiveVarSet.h"
#include "LiveVarMap.h"

#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"
#include "llvm/iOther.h"


class BBLiveVar 
{
  const BasicBlock* BaseBB;     // pointer to BasicBlock
  unsigned int POId;            // Post-Order ID

  LiveVarSet DefSet;            // Def set for LV analysis
  LiveVarSet InSet, OutSet;     // In & Out for LV analysis
  bool InSetChanged, OutSetChanged;   // set if the InSet/OutSet is modified

                                // map that contains phi args->BB they came
                                // set by calcDefUseSets & used by setPropagate
  std::hash_map<const Value *, const BasicBlock *> PhiArgMap;  

  // method to propogate an InSet to OutSet of a predecessor
  bool setPropagate( LiveVarSet *const OutSetOfPred, 
		     const LiveVarSet *const InSetOfThisBB,
		     const BasicBlock *const PredBB);

  // To add an operand which is a def
  void  addDef(const Value *Op); 

  // To add an operand which is a use
  void  addUse(const Value *Op);

 public:

  BBLiveVar( const BasicBlock* baseBB, unsigned int POId);

  inline bool isInSetChanged() const { return InSetChanged; }    
  inline bool isOutSetChanged() const { return OutSetChanged; }

  inline unsigned int getPOId() const { return POId; }

  void calcDefUseSets() ;         // calculates the Def & Use sets for this BB
  bool  applyTransferFunc();      // calcultes the In in terms of Out 

  // calculates Out set using In sets of the predecessors
  bool applyFlowFunc(BBToBBLiveVarMapType LVMap);    

  inline const LiveVarSet* getOutSet()  const { return &OutSet; }
  inline const LiveVarSet*  getInSet() const { return &InSet; }

  void printAllSets() const;      // for printing Def/In/Out sets
  void printInOutSets() const;    // for printing In/Out sets

  ~BBLiveVar() { }                // nothing to do since only composite objects



};







#endif

