/* Title:   LiveRangeInfo.h
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: 

   This file contains the class LiveRangeInfo which constructs and keeps 
   the LiveRangMap which contains all the live ranges used in a method.

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


typedef hash_map <const Value *,  LiveRange *, hashFuncValue> LiveRangeMapType;
typedef vector <const MachineInstr *> CallRetInstrListType;



//----------------------------------------------------------------------------
// Class LiveRangeInfo
//
// Constructs and keeps the LiveRangMap which contains all the live 
// ranges used in a method. Also contain methods to coalesce live ranges.
//----------------------------------------------------------------------------

class LiveRangeInfo 
{

private:

  const Method *const Meth;         // Method for which live range info is held

  LiveRangeMapType  LiveRangeMap;   // A map from Value * to LiveRange * to 
                                    // record all live ranges in a method
                                    // created by constructLiveRanges
  
  const TargetMachine& TM;          // target machine description

  vector<RegClass *> & RegClassList;// a vector containing register classess

  const MachineRegInfo& MRI;        // machine reg info

  CallRetInstrListType  CallRetInstrList;  // a list of all call/ret instrs


  //------------ Private methods (see LiveRangeInfo.cpp for description)-------

  void unionAndUpdateLRs(LiveRange *L1, LiveRange *L2);

  void addInterference(const Instruction *const Inst, 
		       const LiveVarSet *const LVSet);
  
  void suggestRegs4CallRets();

  const Method* getMethod() { return Meth; }


public:
  
  LiveRangeInfo(const Method *const M, 
		const TargetMachine& tm,
		vector<RegClass *> & RCList);


  // Destructor to destroy all LiveRanges in the LiveRange Map
  ~LiveRangeInfo();

  // Main entry point for live range construction
  //
  void constructLiveRanges();

  // This method is used to add a live range created elsewhere (e.g.,
  // in machine specific code) to the common live range map
  //
  inline void addLRToMap(const Value *Val, LiveRange *LR) {
    assert( Val && LR && "Val/LR is NULL!\n");
    assert( (! LiveRangeMap[ Val ]) && "LR already set in map");
    LiveRangeMap[ Val ] = LR;
  }
  
  // return the common live range map for this method
  //
  inline const LiveRangeMapType *const getLiveRangeMap() const 
    { return &LiveRangeMap; }

  // Method sed to get the corresponding live range of a Value
  //
  inline LiveRange *getLiveRangeForValue( const Value *const Val) 
    { return LiveRangeMap[ Val ]; }

  // Method used to get the Call and Return instruction list
  //
  inline  CallRetInstrListType &getCallRetInstrList() {
    return CallRetInstrList;
  }

  // Method for coalescing live ranges. Called only after interference info
  // is calculated.
  //
  void coalesceLRs();  

  // debugging method to print the live ranges
  //
  void printLiveRanges();

};




#endif 
