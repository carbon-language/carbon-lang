/* Title:   LiveVarSet.h
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

  // This function applies a machine instr to a live var set (accepts OutSet)
  // and makes necessary changes to it (produces InSet).

  void applyTranferFuncForMInst(const MachineInstr *const MInst);

};


#endif


