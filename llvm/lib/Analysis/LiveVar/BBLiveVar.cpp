//===-- BBLiveVar.cpp - Live Variable Analysis for a BasicBlock -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a wrapper class for BasicBlock which is used by live var analysis.
//
//===----------------------------------------------------------------------===//

#include "BBLiveVar.h"
#include "llvm/CodeGen/FunctionLiveVarInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/CFG.h"
#include "Support/SetOperations.h"

/// BROKEN: Should not include sparc stuff directly into here
#include "../../Target/Sparc/SparcInternals.h"  //  Only for PHI defn

using std::cerr;

static AnnotationID AID(AnnotationManager::getID("Analysis::BBLiveVar"));

BBLiveVar *BBLiveVar::CreateOnBB(const BasicBlock &BB, MachineBasicBlock &MBB,
                                 unsigned POID) {
  BBLiveVar *Result = new BBLiveVar(BB, MBB, POID);
  BB.addAnnotation(Result);
  return Result;
}

BBLiveVar *BBLiveVar::GetFromBB(const BasicBlock &BB) {
  return (BBLiveVar*)BB.getAnnotation(AID);
}

void BBLiveVar::RemoveFromBB(const BasicBlock &BB) {
  bool Deleted = BB.deleteAnnotation(AID);
  assert(Deleted && "BBLiveVar annotation did not exist!");
}


BBLiveVar::BBLiveVar(const BasicBlock &bb, MachineBasicBlock &mbb, unsigned id)
  : Annotation(AID), BB(bb), MBB(mbb), POID(id) {
  InSetChanged = OutSetChanged = false;

  calcDefUseSets();
}

//-----------------------------------------------------------------------------
// calculates def and use sets for each BB
// There are two passes over operands of a machine instruction. This is
// because, we can have instructions like V = V + 1, since we no longer
// assume single definition.
//-----------------------------------------------------------------------------

void BBLiveVar::calcDefUseSets() {
  // iterate over all the machine instructions in BB
  for (MachineBasicBlock::const_reverse_iterator MII = MBB.rbegin(),
         MIE = MBB.rend(); MII != MIE; ++MII) {
    const MachineInstr *MI = *MII;
    
    if (DEBUG_LV >= LV_DEBUG_Verbose) {
      cerr << " *Iterating over machine instr ";
      MI->dump();
      cerr << "\n";
    }

    // iterate over  MI operands to find defs
    for (MachineInstr::const_val_op_iterator OpI = MI->begin(), OpE = MI->end();
         OpI != OpE; ++OpI)
      if (OpI.isDefOnly() || OpI.isDefAndUse()) // add to Defs if this operand is a def
	addDef(*OpI);

    // do for implicit operands as well
    for (unsigned i = 0; i < MI->getNumImplicitRefs(); ++i)
      if (MI->getImplicitOp(i).opIsDefOnly() || MI->getImplicitOp(i).opIsDefAndUse())
	addDef(MI->getImplicitRef(i));
    
    // iterate over MI operands to find uses
    for (MachineInstr::const_val_op_iterator OpI = MI->begin(), OpE = MI->end();
         OpI != OpE; ++OpI) {
      const Value *Op = *OpI;

      if (isa<BasicBlock>(Op))
	continue;             // don't process labels

      if (OpI.isUseOnly() || OpI.isDefAndUse()) {
                                // add to Uses only if this operand is a use
        //
        // *** WARNING: The following code for handling dummy PHI machine
        //     instructions is untested.  The previous code was broken and I
        //     fixed it, but it turned out to be unused as long as Phi
        //     elimination is performed during instruction selection.
        // 
        // Put Phi operands in UseSet for the incoming edge, not node.
        // They must not "hide" later defs, and must be handled specially
        // during set propagation over the CFG.
	if (MI->getOpCode() == V9::PHI) {         // for a phi node
          const Value *ArgVal = Op;
	  const BasicBlock *PredBB = cast<BasicBlock>(*++OpI); // next ptr is BB
	  
	  PredToEdgeInSetMap[PredBB].insert(ArgVal); 
	  
	  if (DEBUG_LV >= LV_DEBUG_Verbose)
	    cerr << "   - phi operand " << RAV(ArgVal) << " came from BB "
                 << RAV(PredBB) << "\n";
	} // if( IsPhi )
        else {
          // It is not a Phi use: add to regular use set and remove later defs.
          addUse(Op);
        }
      } // if a use
    } // for all operands

    // do for implicit operands as well
    for (unsigned i = 0; i < MI->getNumImplicitRefs(); ++i) {
      assert(MI->getOpCode() != V9::PHI && "Phi cannot have implicit operands");
      const Value *Op = MI->getImplicitRef(i);

      if (Op->getType() == Type::LabelTy)             // don't process labels
	continue;

      if (MI->getImplicitOp(i).opIsUse() || MI->getImplicitOp(i).opIsDefAndUse())
	addUse(Op);
    }
  } // for all machine instructions
} 


	
//-----------------------------------------------------------------------------
// To add an operand which is a def
//-----------------------------------------------------------------------------
void BBLiveVar::addDef(const Value *Op) {
  DefSet.insert(Op);     // operand is a def - so add to def set
  InSet.erase(Op);       // this definition kills any later uses
  InSetChanged = true; 

  if (DEBUG_LV >= LV_DEBUG_Verbose) cerr << "  +Def: " << RAV(Op) << "\n";
}


//-----------------------------------------------------------------------------
// To add an operand which is a use
//-----------------------------------------------------------------------------
void  BBLiveVar::addUse(const Value *Op) {
  InSet.insert(Op);   // An operand is a use - so add to use set
  DefSet.erase(Op);   // remove if there is a def below this use
  InSetChanged = true; 

  if (DEBUG_LV >= LV_DEBUG_Verbose) cerr << "   Use: " << RAV(Op) << "\n";
}


//-----------------------------------------------------------------------------
// Applies the transfer function to a basic block to produce the InSet using
// the OutSet. 
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
// calculates Out set using In sets of the successors
//-----------------------------------------------------------------------------

bool BBLiveVar::setPropagate(ValueSet *OutSet, const ValueSet *InSet, 
                             const BasicBlock *PredBB) {
  bool Changed = false;
  
  // merge all members of InSet into OutSet of the predecessor
  for (ValueSet::const_iterator InIt = InSet->begin(), InE = InSet->end();
       InIt != InE; ++InIt)
    if ((OutSet->insert(*InIt)).second)
      Changed = true;
  
  // 
  //**** WARNING: The following code for handling dummy PHI machine
  //     instructions is untested.  See explanation above.
  // 
  // then merge all members of the EdgeInSet for the predecessor into the OutSet
  const ValueSet& EdgeInSet = PredToEdgeInSetMap[PredBB];
  for (ValueSet::const_iterator InIt = EdgeInSet.begin(), InE = EdgeInSet.end();
       InIt != InE; ++InIt)
    if ((OutSet->insert(*InIt)).second)
      Changed = true;
  // 
  //****
  
  return Changed;
} 


//-----------------------------------------------------------------------------
// propagates in set to OutSets of PREDECESSORs
//-----------------------------------------------------------------------------

bool BBLiveVar::applyFlowFunc() {
  // IMPORTANT: caller should check whether inset changed 
  //            (else no point in calling)
  
  // If this BB changed any OutSets of preds whose POID is lower, than we need
  // another iteration...
  //
  bool needAnotherIt = false;  

  for (pred_const_iterator PI = pred_begin(&BB), PE = pred_end(&BB);
       PI != PE ; ++PI) {
    BBLiveVar *PredLVBB = BBLiveVar::GetFromBB(**PI);

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




