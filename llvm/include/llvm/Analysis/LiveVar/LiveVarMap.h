/* Title:   LiveVarMap.h
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: This file contains the class for a map between the BasicBlock class
            and the BBLiveVar class, which is a wrapper class of BasicBlock
	    used for the live variable analysis. The reverse mapping can
	    be found in the BBLiveVar class (It has a pointer to the 
	    corresponding BasicBlock)
*/


#ifndef LIVE_VAR_MAP_H
#define LIVE_VAR_MAP_H

#include <ext/hash_map>

class BasicBlock;
class BBLiveVar;


struct hashFuncMInst {  // sturcture containing the hash function for MInst
  inline size_t operator () (const MachineInstr *val) const { 
    return (size_t) val;  
  }
};


struct hashFuncBB {          // sturcture containing the hash function for BB
  inline size_t operator () (const BasicBlock *val) const { 
    return (size_t) val; 
  }
};




typedef std::hash_map<const BasicBlock *,  
                      BBLiveVar *, hashFuncBB > BBToBBLiveVarMapType;

typedef std::hash_map<const MachineInstr *,  const LiveVarSet *, 
                      hashFuncMInst> MInstToLiveVarSetMapType;


#endif


