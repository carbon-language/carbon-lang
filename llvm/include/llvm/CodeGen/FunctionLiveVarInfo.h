/* Title:   MethodLiveVarInfo.h
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: 

   This is the interface for live variable info of a method that is required by 
   any other part of the compiler

   It should be called like:

       MethodLiveVarInfo MLVI( Mehtod *);  // initializes data structures
       MLVI.analyze();                    // do the actural live variable anal

 After the analysis, getInSetOfBB or getOutSetofBB can be called to get 
 live var info of a BB

 The live var set before an instruction can be constructed in several ways:

 1. Use the OutSet and applyTranferFuncForInst(const Instruction *const Inst) 
    declared in LiveVarSet and  traverse the instructions of a basic block in 
    reverse (using const_reverse_iterator in the BB class). 

    This is the most efficient method if you need LV info for several (all) 
    instructions in a BasicBlock. An example is given below:


    LiveVarSet LVSet;  // this will be the set used to traverse through each BB

                   // Initialize LVSet so that it is the same as OutSet of the BB
    LVSet.setUnion( LVI->getOutSetOfBB( *BBI ) );  
 
    BasicBlock::InstListType::const_reverse_iterator 
      InstIterator = InstListInBB.rbegin();  // get the reverse it for inst in BB

                            // iterate over all the instructions in BB in reverse
    for( ; InstIterator != InstListInBB.rend(); InstIterator++) {  

      //...... all  code here which uses LVSet ........

      LVSet.applyTranferFuncForInst(*InstIterator);

      // Now LVSet contains live vars ABOVE the current instrution
    }

    See buildInterferenceGraph() for the above example.


 2. Use the function getLiveVarSetBeforeInst(Instruction *) to get the LV Info 
    just before an instruction.

    This function caluclates the LV info for a BB only once and caches that 
    info. If the cache does not contain the LV info of the instruction, it 
    calculates the LV info for the whole BB and caches them.

    Getting liveVar info this way uses more memory since, LV info should be 
    cached.


 **BUGS: Cannot be called on a method prototype because the BB front() 
   iterator causes a seg fault in CFG.h (Chris will fix this)
   So, currently, DO NOT call this for method prototypes. 

*/


#ifndef METH_LIVE_VAR_INFO_H
#define METH_LIVE_VAR_INFO_H

        // for printing out debug messages
#define DEBUG_LV (1)

#include "LiveVarSet.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/Method.h"
#include "llvm/CFG.h"

#include "LiveVarMap.h"
#include "BBLiveVar.h"


class MethodLiveVarInfo
{
 private:
  const Method *Meth;   // Live var anal is done on this method 
                        // set by constructor

  BBToBBLiveVarMapType  BB2BBLVMap;  // A map betwn the BasicBlock and BBLiveVar

  InstToLiveVarSetMapType Inst2LVSetMap; // Instruction to LiveVarSet Map 
                                         //- for providing LV info for each inst

  void constructBBs();          // constructs BBLiveVars and init Def and In sets
  bool  doSingleBackwardPass(); // do one backward pass over the CFG

  

 public:
  MethodLiveVarInfo(Method *const Meth);    // constructor 

  ~MethodLiveVarInfo();                     // destructor

  void analyze();             // performs a liver var analysis of a single method

                                                           // gets OutSet of a BB
  inline const LiveVarSet *getOutSetOfBB( const BasicBlock *const BB)  const {   
    return (   (* (BB2BBLVMap.find(BB)) ).second  )->getOutSet();
  }

                                                            // gets InSet of a BB
  inline const LiveVarSet *getInSetOfBB( const BasicBlock *const BB)  const { 
    return (   (* (BB2BBLVMap.find(BB)) ).second  )->getInSet();
  }
                                   // gets the Live var set before an instruction
  const LiveVarSet * 
    MethodLiveVarInfo::getLiveVarSetBeforeInst(const Instruction *const Inst);

 
};





#endif




