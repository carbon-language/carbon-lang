/* Title:   MethodLiveVarInfo.h             -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: 

   This is the interface for live variable info of a method that is required 
   by any other part of the compiler

   It must be called like:

       MethodLiveVarInfo MLVI( Mehtod *);  // initializes data structures
       MLVI.analyze();                     // do the actural live variable anal

 After the analysis, getInSetOfBB or getOutSetofBB can be called to get 
 live var info of a BB.

 The live var set before an instruction can be obtained in 2 ways:

 1. Use the method getLiveVarSetAfterInst(Instruction *) to get the LV Info 
    just after an instruction. (also exists getLiveVarSetBeforeInst(..))

    This function caluclates the LV info for a BB only once and caches that 
    info. If the cache does not contain the LV info of the instruction, it 
    calculates the LV info for the whole BB and caches them.

    Getting liveVar info this way uses more memory since, LV info should be 
    cached. However, if you need LV info of nearly all the instructions of a
    BB, this is the best and simplest interfrace.


 2. Use the OutSet and applyTranferFuncForInst(const Instruction *const Inst) 
    declared in LiveVarSet and  traverse the instructions of a basic block in 
    reverse (using const_reverse_iterator in the BB class). 

    This is the most memory efficient method if you need LV info for 
    only several instructions in a BasicBlock. An example is given below:


    LiveVarSet LVSet;  // this will be the set used to traverse through each BB

    // Initialize LVSet so that it is the same as OutSet of the BB
    LVSet.setUnion( LVI->getOutSetOfBB( *BBI ) );  
 
    BasicBlock::InstListType::const_reverse_iterator 
      InstIterator = InstListInBB.rbegin(); // get the rev iter for inst in BB

      // iterate over all the instructions in BB in reverse
    for( ; InstIterator != InstListInBB.rend(); InstIterator++) {  

      //...... all  code here which uses LVSet ........

      LVSet.applyTranferFuncForInst(*InstIterator);

      // Now LVSet contains live vars ABOVE the current instrution
    }

    See buildInterferenceGraph() for the above example.


DOCUMENTATION:
-------------

See README.    

*/


#ifndef METH_LIVE_VAR_INFO_H
#define METH_LIVE_VAR_INFO_H

// set DEBUG_LV for printing out debug messages
// if DEBUG_LV is 1 normal output messages
// if DEBUG_LV is 2 extensive debug info for each instr

static const int DEBUG_LV = 0;

#include "LiveVarSet.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/Method.h"

#include "LiveVarMap.h"
#include "BBLiveVar.h"


class MethodLiveVarInfo
{
 private:

  // Live var anal is done on this method - set by constructor
  const Method *const Meth;   

  // A map betwn the BasicBlock and BBLiveVar
  BBToBBLiveVarMapType  BB2BBLVMap;  

  // Machine Instr to LiveVarSet Map for providing LVset BEFORE each inst
  MInstToLiveVarSetMapType MInst2LVSetBI; 

  // Machine Instr to LiveVarSet Map for providing LVset AFTER each inst
  MInstToLiveVarSetMapType MInst2LVSetAI; 

  // True if the analyze() method has been called. This is checked when
  // getInSet/OutSet is called to prevent calling those methods before analyze
  bool HasAnalyzed;


  // --------- private methods -----------------------------------------

  // constructs BBLiveVars and init Def and In sets
  void constructBBs();      
    
  // do one backward pass over the CFG
  bool  doSingleBackwardPass(); 

  // calculates live var sets for instructions in a BB
  void calcLiveVarSetsForBB(const BasicBlock *const BB);
  

 public:
  MethodLiveVarInfo(const Method *const Meth);    // constructor 

  ~MethodLiveVarInfo();                           // destructor

  // performs a liver var analysis of a single method
  void analyze();            

  // gets OutSet of a BB
  inline const LiveVarSet *getOutSetOfBB( const BasicBlock *const BB) const { 
    assert( HasAnalyzed && "call analyze() before calling this" );
    return  ( (* (BB2BBLVMap.find(BB)) ).second ) ->getOutSet();
  }

  // gets InSet of a BB
  inline const LiveVarSet *getInSetOfBB( const BasicBlock *const BB)  const { 
    assert( HasAnalyzed && "call analyze() before calling this" );
    return (   (* (BB2BBLVMap.find(BB)) ).second  )->getInSet();
  }

  // gets the Live var set BEFORE an instruction
  const LiveVarSet * getLiveVarSetBeforeMInst(const MachineInstr *const Inst,
					      const BasicBlock *const CurBB);

  // gets the Live var set AFTER an instruction
  const LiveVarSet * getLiveVarSetAfterMInst(const MachineInstr *const MInst,
					     const BasicBlock *const CurBB);

};





#endif




