/* Title:   LiveRangeInfo.h
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: 

   This file constructs and keeps the LiveRang map which contains all the live
   ranges used in a method.

   Assumptions: 

   All variables (llvm Values) are defined before they are used. However, a 
   constant may not be defined in the machine instruction stream if it can be
   used as an immediate value within a machine instruction. However, register
   allocation does not have to worry about immediate constants since they
   do not require registers.

   Since an llvm Value has a list of uses associated, it is sufficient to
   record only the defs in a Live Range.

*/


#ifndef LIVE_RANGE_INFO_H
#define LIVE_RANGE_INFO_H


#include "llvm/Type.h"
#include "llvm/Method.h"
#include "llvm/CodeGen/MachineInstr.h"

#include "llvm/Analysis/LiveVar/LiveVarSet.h"

#include "llvm/CodeGen/IGNode.h"
#include "llvm/CodeGen/LiveRange.h"
#include "llvm/CodeGen/RegClass.h"

/*
 #ifndef size_type 
 #define size_type (unsigned int)
 #endif
*/


typedef hash_map <const Value *,  LiveRange *, hashFuncValue> LiveRangeMapType;


class LiveRangeInfo 
{

private:

  const Method *const Meth;         // Method for which live range info is held

  LiveRangeMapType  LiveRangeMap;   // A map from Value * to LiveRange * 
                                    // created by constructLiveRanges

  void unionAndUpdateLRs(LiveRange *L1, LiveRange *L2);

  void addInterference(const Instruction *const Inst, 
		       const LiveVarSet *const LVSet);
  

public:
  
  LiveRangeInfo(const Method *const M);

  void constructLiveRanges();

  inline const LiveRangeMapType *const getLiveRangeMap() const 
    { return &LiveRangeMap; }

  inline LiveRange *getLiveRangeForValue( const Value *const Val) 
    { return LiveRangeMap[ Val ]; }




  void coalesceLRs();  

  void printLiveRanges();

};




#endif 
