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
typedef vector <const Instruction *> CallRetInstrListType;

class LiveRangeInfo 
{

private:

  const Method *const Meth;         // Method for which live range info is held

  LiveRangeMapType  LiveRangeMap;   // A map from Value * to LiveRange * 
                                    // created by constructLiveRanges

  const TargetMachine& TM;          // target machine description
  vector<RegClass *> & RegClassList;// a vector containing register classess
  const MachineRegInfo& MRI;        // machine reg info

  CallRetInstrListType  CallRetInstrList;  // a list of all call/ret instrs

  void unionAndUpdateLRs(LiveRange *L1, LiveRange *L2);

  void addInterference(const Instruction *const Inst, 
		       const LiveVarSet *const LVSet);
  

public:
  
  LiveRangeInfo(const Method *const M, 
		const TargetMachine& tm,
		vector<RegClass *> & RCList);

  void constructLiveRanges();

  inline void addLRToMap(const Value *Val, LiveRange *LR) {
    assert( Val && LR && "Val/LR is NULL!\n");
    assert( (! LiveRangeMap[ Val ]) && "LR already set in map");
    LiveRangeMap[ Val ] = LR;
  }
  

  inline const LiveRangeMapType *const getLiveRangeMap() const 
    { return &LiveRangeMap; }

  inline LiveRange *getLiveRangeForValue( const Value *const Val) 
    { return LiveRangeMap[ Val ]; }

  inline  CallRetInstrListType &getCallRetInstrList() {
    return CallRetInstrList;
  }



  void coalesceLRs();  

  void printLiveRanges();

};




#endif 
