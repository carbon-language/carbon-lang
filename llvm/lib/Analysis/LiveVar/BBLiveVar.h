//===-- BBLiveVar.h - Live Variable Analysis for a BasicBlock ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a BasicBlock annotation class that is used by live var analysis to
// hold data flow information for a basic block.
//
//===----------------------------------------------------------------------===//

#ifndef LIVE_VAR_BB_H
#define LIVE_VAR_BB_H

#include "llvm/CodeGen/ValueSet.h"
#include "Support/hash_map"
class BasicBlock;
class Value;
class MachineBasicBlock;

enum LiveVarDebugLevel_t {
  LV_DEBUG_None,
  LV_DEBUG_Normal,
  LV_DEBUG_Instr,
  LV_DEBUG_Verbose
};

extern LiveVarDebugLevel_t DEBUG_LV;

class BBLiveVar {
  const BasicBlock &BB;         // pointer to BasicBlock
  MachineBasicBlock &MBB;       // Pointer to MachineBasicBlock
  unsigned POID;                // Post-Order ID

  ValueSet DefSet;           // Def set (with no preceding uses) for LV analysis
  ValueSet InSet, OutSet;       // In & Out for LV analysis
  bool InSetChanged, OutSetChanged;   // set if the InSet/OutSet is modified

                                // map that contains PredBB -> Phi arguments
                                // coming in on that edge.  such uses have to be
                                // treated differently from ordinary uses.
  hash_map<const BasicBlock *, ValueSet> PredToEdgeInSetMap;
  
  // method to propagate an InSet to OutSet of a predecessor
  bool setPropagate(ValueSet *OutSetOfPred, 
                    const ValueSet *InSetOfThisBB,
                    const BasicBlock *PredBB);

  // To add an operand which is a def
  void addDef(const Value *Op); 

  // To add an operand which is a use
  void addUse(const Value *Op);

  void calcDefUseSets();         // calculates the Def & Use sets for this BB
public:

  BBLiveVar(const BasicBlock &BB, MachineBasicBlock &MBB, unsigned POID);

  inline bool isInSetChanged() const  { return InSetChanged; }    
  inline bool isOutSetChanged() const { return OutSetChanged; }

  MachineBasicBlock &getMachineBasicBlock() const { return MBB; }

  inline unsigned getPOId() const { return POID; }

  bool applyTransferFunc();      // calcultes the In in terms of Out 

  // calculates Out set using In sets of the predecessors
  bool applyFlowFunc(hash_map<const BasicBlock*, BBLiveVar*> &BBLiveVarInfo);

  inline const ValueSet &getOutSet() const { return OutSet; }
  inline       ValueSet &getOutSet()       { return OutSet; }

  inline const ValueSet  &getInSet() const { return InSet; }
  inline       ValueSet  &getInSet()       { return InSet; }

  void printAllSets() const;      // for printing Def/In/Out sets
  void printInOutSets() const;    // for printing In/Out sets
};

#endif
