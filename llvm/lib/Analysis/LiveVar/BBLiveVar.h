//===-- BBLiveVar.h - Live Variable Analysis for a BasicBlock ----*- C++ -*--=//
//
// This is a BasicBlock annotation class that is used by live var analysis to
// hold data flow information for a basic block.
//
//===----------------------------------------------------------------------===//

#ifndef LIVE_VAR_BB_H
#define LIVE_VAR_BB_H

#include "llvm/CodeGen/ValueSet.h"
#include "Support/Annotation.h"
#include <map>
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

class BBLiveVar : public Annotation {
  const BasicBlock &BB;         // pointer to BasicBlock
  MachineBasicBlock &MBB;       // Pointer to MachineBasicBlock
  unsigned POID;                // Post-Order ID

  ValueSet DefSet;           // Def set (with no preceding uses) for LV analysis
  ValueSet InSet, OutSet;       // In & Out for LV analysis
  bool InSetChanged, OutSetChanged;   // set if the InSet/OutSet is modified

                                // map that contains PredBB -> Phi arguments
                                // coming in on that edge.  such uses have to be
                                // treated differently from ordinary uses.
  std::map<const BasicBlock *, ValueSet> PredToEdgeInSetMap;
  
  // method to propagate an InSet to OutSet of a predecessor
  bool setPropagate(ValueSet *OutSetOfPred, 
                    const ValueSet *InSetOfThisBB,
                    const BasicBlock *PredBB);

  // To add an operand which is a def
  void addDef(const Value *Op); 

  // To add an operand which is a use
  void addUse(const Value *Op);

  void calcDefUseSets();         // calculates the Def & Use sets for this BB

  BBLiveVar(const BasicBlock &BB, MachineBasicBlock &MBB, unsigned POID);
  ~BBLiveVar() {}                // make dtor private
 public:
  static BBLiveVar *CreateOnBB(const BasicBlock &BB, MachineBasicBlock &MBB,
                               unsigned POID);
  static BBLiveVar *GetFromBB(const BasicBlock &BB);
  static void RemoveFromBB(const BasicBlock &BB);

  inline bool isInSetChanged() const  { return InSetChanged; }    
  inline bool isOutSetChanged() const { return OutSetChanged; }

  MachineBasicBlock &getMachineBasicBlock() const { return MBB; }

  inline unsigned getPOId() const { return POID; }

  bool applyTransferFunc();      // calcultes the In in terms of Out 

  // calculates Out set using In sets of the predecessors
  bool applyFlowFunc();

  inline const ValueSet &getOutSet() const { return OutSet; }
  inline const ValueSet  &getInSet() const { return InSet; }

  void printAllSets() const;      // for printing Def/In/Out sets
  void printInOutSets() const;    // for printing In/Out sets
};

#endif
