#include "llvm/Analysis/LiveVar/LiveVarSet.h"


// This function applies an instruction to a live var set (accepts OutSet) and
//  makes necessary changes to it (produces InSet)

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
