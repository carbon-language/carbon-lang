//===-- CodeGen/FunctionLiveVarInfo.h - LiveVar Analysis --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is the interface for live variable info of a function that is required 
// by any other part of the compiler
//
// After the analysis, getInSetOfBB or getOutSetofBB can be called to get 
// live var info of a BB.
//
// The live var set before an instruction can be obtained in 2 ways:
//
// 1. Use the method getLiveVarSetAfterInst(Instruction *) to get the LV Info 
//    just after an instruction. (also exists getLiveVarSetBeforeInst(..))
//
//    This function caluclates the LV info for a BB only once and caches that 
//    info. If the cache does not contain the LV info of the instruction, it 
//    calculates the LV info for the whole BB and caches them.
//
//    Getting liveVar info this way uses more memory since, LV info should be 
//    cached. However, if you need LV info of nearly all the instructions of a
//    BB, this is the best and simplest interfrace.
//
// 2. Use the OutSet and applyTranferFuncForInst(const Instruction *const Inst) 
//    declared in LiveVarSet and  traverse the instructions of a basic block in 
//    reverse (using const_reverse_iterator in the BB class). 
//
//===----------------------------------------------------------------------===//

#ifndef FUNCTION_LIVE_VAR_INFO_H
#define FUNCTION_LIVE_VAR_INFO_H

#include "Support/hash_map"
#include "llvm/Pass.h"
#include "llvm/CodeGen/ValueSet.h"

class BBLiveVar;
class MachineInstr;

class FunctionLiveVarInfo : public FunctionPass {
  // Machine Instr to LiveVarSet Map for providing LVset BEFORE each inst
  // These sets are owned by this map and will be freed in releaseMemory().
  hash_map<const MachineInstr *, ValueSet *> MInst2LVSetBI; 

  // Machine Instr to LiveVarSet Map for providing LVset AFTER each inst.
  // These sets are just pointers to sets in MInst2LVSetBI or BBLiveVar.
  hash_map<const MachineInstr *, ValueSet *> MInst2LVSetAI; 

  // Stored Function that the data is computed with respect to
  const Function *M;

  // --------- private methods -----------------------------------------

  // constructs BBLiveVars and init Def and In sets
  void constructBBs(const Function *F);
    
  // do one backward pass over the CFG
  bool doSingleBackwardPass(const Function *F, unsigned int iter); 

  // calculates live var sets for instructions in a BB
  void calcLiveVarSetsForBB(const BasicBlock *BB);
  
public:
  // --------- Implement the FunctionPass interface ----------------------

  // runOnFunction - Perform analysis, update internal data structures.
  virtual bool runOnFunction(Function &F);

  // releaseMemory - After LiveVariable analysis has been used, forget!
  virtual void releaseMemory();

  // getAnalysisUsage - Provide self!
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  // --------- Functions to access analysis results -------------------

  // get OutSet of a BB
  const ValueSet &getOutSetOfBB(const BasicBlock *BB) const;
        ValueSet &getOutSetOfBB(const BasicBlock *BB)      ;

  // get InSet of a BB
  const ValueSet &getInSetOfBB(const BasicBlock *BB) const;
        ValueSet &getInSetOfBB(const BasicBlock *BB)      ;

  // gets the Live var set BEFORE an instruction.
  // if BB is specified and the live var set has not yet been computed,
  // it will be computed on demand.
  const ValueSet &getLiveVarSetBeforeMInst(const MachineInstr *MI,
                                           const BasicBlock *BB = 0);

  // gets the Live var set AFTER an instruction
  // if BB is specified and the live var set has not yet been computed,
  // it will be computed on demand.
  const ValueSet &getLiveVarSetAfterMInst(const MachineInstr *MI,
                                          const BasicBlock *BB = 0);
};

#endif
