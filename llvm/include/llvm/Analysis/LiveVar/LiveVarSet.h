/* Title:   ValueSet.h
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: Contains the class definition of LiveVarSet which is used for
            live variable analysis.
*/

#ifndef LIVE_VAR_SET_H
#define LIVE_VAR_SET_H

#include "ValueSet.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"

class LiveVarSet : public ValueSet
{

 public:
  void applyTranferFuncForInst(const Instruction *const Inst); 
  
};


#endif


