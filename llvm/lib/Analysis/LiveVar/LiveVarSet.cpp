#include "llvm/Analysis/LiveVar/LiveVarSet.h"
#include "llvm/CodeGen/MachineInstr.h"


// This function applies a machine instr to a live var set (accepts OutSet) and
// makes necessary changes to it (produces InSet). Note that two for loops are
// used to first kill all defs and then to add all uses. This is because there
// can be instructions like Val = Val + 1 since we allow multipe defs to a 
// machine instruction operand.


void LiveVarSet::applyTranferFuncForMInst(const MachineInstr *const MInst)
{

  for( MachineInstr::val_op_const_iterator OpI(MInst); !OpI.done() ; OpI++) {

    if( OpI.isDef() ) {     // kill only if this operand is a def
         remove(*OpI);        // this definition kills any uses
    }

  }

  for( MachineInstr::val_op_const_iterator OpI(MInst); !OpI.done() ; OpI++) {

    if ( ((*OpI)->getType())->isLabelType()) continue; // don't process labels

    if( ! OpI.isDef() ) {     // add only if this operand is a use
       add( *OpI );            // An operand is a use - so add to use set
    }
  }
}

  





#if 0
void LiveVarSet::applyTranferFuncForInst(const Instruction *const Inst) 
{

  if( Inst->isDefinition() ) {  // add to Defs iff this instr is a definition
       remove(Inst);            // this definition kills any uses
  }
  Instruction::op_const_iterator OpI = Inst->op_begin();  // get operand iterat

  for( ; OpI != Inst->op_end() ; OpI++) {              // iterate over operands
    if ( ((*OpI)->getType())->isLabelType()) continue; // don't process labels 
    add( *OpI );                     // An operand is a use - so add to use set
  }

}
#endif
