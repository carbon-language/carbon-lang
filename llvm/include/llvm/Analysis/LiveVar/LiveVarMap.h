/* Title:   ValueSet.h
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

#include <hash_map>

class BasicBlock;
class BBLiveVar;


struct hashFuncInst {        // sturcture containing the hash function for Inst
  inline size_t operator () (const Instruction *val) const { 
    return (size_t) val;  
  }
};


struct hashFuncBB {          // sturcture containing the hash function for BB
  inline size_t operator () (const BasicBlock *val) const { 
    return (size_t) val; 
  }
};




typedef hash_map<const BasicBlock *,  
		 BBLiveVar *, hashFuncBB > BBToBBLiveVarMapType;

typedef hash_map<const Instruction *,  const LiveVarSet *, 
		 hashFuncInst> InstToLiveVarSetMapType;


#endif


