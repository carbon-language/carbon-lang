/* Title:   LiveVarMap.h   -*- C++ -*-
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

#include "Support/HashExtras.h"

class MachineInstr;
class BasicBlock;
class BBLiveVar;
class LiveVarSet;

typedef std::hash_map<const BasicBlock *, BBLiveVar *> BBToBBLiveVarMapType;
typedef std::hash_map<const MachineInstr *, const LiveVarSet *> MInstToLiveVarSetMapType;

#endif
