//===-- BBLiveVar.cpp - Live Variable Analysis for a BasicBlock -----------===//
//
// This is a wrapper class for BasicBlock which is used by live var analysis.
//
//===----------------------------------------------------------------------===//

#include "BBLiveVar.h"
#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/BasicBlock.h"

/// BROKEN: Should not include sparc stuff directly into here
#include "../../Target/Sparc/SparcInternals.h"  //  Only for PHI defn

using std::cerr;

BBLiveVar::BBLiveVar(const BasicBlock *bb, unsigned id)
  : BB(bb), POID(id) {
  InSetChanged = OutSetChanged = false;
}

//-----------------------------------------------------------------------------
// calculates def and use sets for each BB
// There are two passes over operands of a machine instruction. This is
// because, we can have instructions like V = V + 1, since we no longer
// assume single definition.
//-----------------------------------------------------------------------------

void BBLiveVar::calcDefUseSets() {
  // get the iterator for machine instructions
  const MachineCodeForBasicBlock &MIVec = BB->getMachineInstrVec();

  // iterate over all the machine instructions in BB
  for (MachineCodeForBasicBlock::const_reverse_iterator MII = MIVec.rbegin(),
         MIE = MIVec.rend(); MII != MIE; ++MII) {
    const MachineInstr *MI = *MII;
    
    if (DEBUG_LV > 1) {                            // debug msg
      cerr << " *Iterating over machine instr ";
      MI->dump();
      cerr << "\n";
    }

    // iterate over  MI operands to find defs
    for (MachineInstr::val_const_op_iterator OpI(MI); !OpI.done(); ++OpI)
      if (OpI.isDef())      // add to Defs only if this operand is a def
	addDef(*OpI);

    // do for implicit operands as well
    for (unsigned i = 0; i < MI->getNumImplicitRefs(); ++i)
      if (MI->implicitRefIsDefined(i))
	addDef(MI->getImplicitRef(i));
    
    bool IsPhi = MI->getOpCode() == PHI;
 
    // iterate over MI operands to find uses
    for (MachineInstr::val_const_op_iterator OpI(MI); !OpI.done(); ++OpI) {
      const Value *Op = *OpI;

      if (Op->getType()->isLabelType())    
	continue;             // don't process labels

      if (!OpI.isDef()) {   // add to Defs only if this operand is a use
	addUse(Op);

	if (IsPhi) {         // for a phi node
	  // put args into the PhiArgMap (Val -> BB)
          const Value *ArgVal = Op;
	  const Value *BBVal = *++OpI; // increment to point to BB of value
	  
	  PhiArgMap[ArgVal] = cast<BasicBlock>(BBVal); 
	  
	  if (DEBUG_LV > 1)
	    cerr << "   - phi operand " << RAV(ArgVal) << " came from BB "
                 << RAV(PhiArgMap[ArgVal]) << "\n";
	} // if( IsPhi )
      } // if a use
    } // for all operands

    // do for implicit operands as well
    for (unsigned i = 0; i < MI->getNumImplicitRefs(); ++i) {
      assert(!IsPhi && "Phi cannot have implicit opeands");
      const Value *Op = MI->getImplicitRef(i);

      if (Op->getType()->isLabelType())             // don't process labels
	continue;

      if (!MI->implicitRefIsDefined(i))
	addUse(Op);
    }
  } // for all machine instructions
} 


	
//-----------------------------------------------------------------------------
// To add an operand which is a def
//-----------------------------------------------------------------------------
void  BBLiveVar::addDef(const Value *Op) {
  DefSet.insert(Op);     // operand is a def - so add to def set
  InSet.erase(Op);       // this definition kills any uses
  InSetChanged = true; 

  if (DEBUG_LV > 1) cerr << "  +Def: " << RAV(Op) << "\n";
}


//-----------------------------------------------------------------------------
// To add an operand which is a use
//-----------------------------------------------------------------------------
void  BBLiveVar::addUse(const Value *Op) {
  InSet.insert(Op);   // An operand is a use - so add to use set
  OutSet.erase(Op);   // remove if there is a def below this use
  InSetChanged = true; 

  if (DEBUG_LV > 1) cerr << "   Use: " << RAV(Op) << "\n";
}


//-----------------------------------------------------------------------------
// Applies the transfer function to a basic block to produce the InSet using
// the outset. 
//-----------------------------------------------------------------------------

bool BBLiveVar::applyTransferFunc() {
  // IMPORTANT: caller should check whether the OutSet changed 
  //           (else no point in calling)

  ValueSet OutMinusDef = set_difference(OutSet, DefSet);
  InSetChanged = set_union(InSet, OutMinusDef);
 
  OutSetChanged = false;      // no change to OutSet since transf func applied
  return InSetChanged;
}


//-----------------------------------------------------------------------------
// calculates Out set using In sets of the predecessors
//-----------------------------------------------------------------------------

bool BBLiveVar::setPropagate(ValueSet *OutSet, const ValueSet *InSet, 
                             const BasicBlock *PredBB) {
  bool Changed = false;

  // for all all elements in InSet
  for (ValueSet::const_iterator InIt = InSet->begin(), InE = InSet->end();
       InIt != InE; ++InIt) {  
    const BasicBlock *PredBBOfPhiArg = PhiArgMap[*InIt];

    // if this var is not a phi arg OR 
    // it's a phi arg and the var went down from this BB
    if (!PredBBOfPhiArg || PredBBOfPhiArg == PredBB)
      if (OutSet->insert(*InIt).second)
        Changed = true;
  }

  return Changed;
} 


//-----------------------------------------------------------------------------
// propogates in set to OutSets of PREDECESSORs
//-----------------------------------------------------------------------------

bool BBLiveVar::applyFlowFunc(std::map<const BasicBlock *, BBLiveVar *> &LVMap){
  // IMPORTANT: caller should check whether inset changed 
  //            (else no point in calling)

  // If this BB changed any OutSets of preds whose POID is lower, than we need
  // another iteration...
  //
  bool needAnotherIt = false;  

  for (BasicBlock::pred_const_iterator PI = BB->pred_begin(),
         PE = BB->pred_begin(); PI != PE ; ++PI) {
    BBLiveVar *PredLVBB = LVMap[*PI];

    // do set union
    if (setPropagate(&PredLVBB->OutSet, &InSet, *PI)) {  
      PredLVBB->OutSetChanged = true;

      // if the predec POID is lower than mine
      if (PredLVBB->getPOId() <= POID)
	needAnotherIt = true;   
    }
  }  // for

  return needAnotherIt;
}



// ----------------- Methods For Debugging (Printing) -----------------

void BBLiveVar::printAllSets() const {
  cerr << "  Defs: "; printSet(DefSet);  cerr << "\n";
  cerr << "  In: ";  printSet(InSet);  cerr << "\n";
  cerr << "  Out: "; printSet(OutSet);  cerr << "\n";
}

void BBLiveVar::printInOutSets() const {
  cerr << "  In: ";   printSet(InSet);  cerr << "\n";
  cerr << "  Out: ";  printSet(OutSet);  cerr << "\n";
}




