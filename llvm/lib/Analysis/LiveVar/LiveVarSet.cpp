#include "llvm/Analysis/LiveVar/LiveVarSet.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Type.h"

// This function applies a machine instr to a live var set (accepts OutSet) and
// makes necessary changes to it (produces InSet). Note that two for loops are
// used to first kill all defs and then to add all uses. This is because there
// can be instructions like Val = Val + 1 since we allow multipe defs to a 
// machine instruction operand.


void LiveVarSet::applyTranferFuncForMInst(const MachineInstr *MInst) {
  for (MachineInstr::val_const_op_iterator OpI(MInst); !OpI.done(); ++OpI) {
    if (OpI.isDef())      // kill only if this operand is a def
         remove(*OpI);        // this definition kills any uses
  }

  // do for implicit operands as well
  for ( unsigned i=0; i < MInst->getNumImplicitRefs(); ++i) {
    if (MInst->implicitRefIsDefined(i))
      remove(MInst->getImplicitRef(i));
  }


  for (MachineInstr::val_const_op_iterator OpI(MInst); !OpI.done(); ++OpI) {
    if ((*OpI)->getType()->isLabelType()) continue; // don't process labels
    
    if (!OpI.isDef())      // add only if this operand is a use
       add(*OpI);            // An operand is a use - so add to use set
  }

  // do for implicit operands as well
  for (unsigned i=0; i < MInst->getNumImplicitRefs(); ++i) {
    if (!MInst->implicitRefIsDefined(i))
      add(MInst->getImplicitRef(i));
  }
}
