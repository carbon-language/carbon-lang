#include "llvm/Analysis/LiveVar/BBLiveVar.h"
#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

/// BROKEN: Should not include sparc stuff directly into here
#include "../../Target/Sparc/SparcInternals.h"  //  Only for PHI defn

using std::cerr;
using std::endl;
using std::pair;

//-----------------------------------------------------------------------------
// Constructor
//-----------------------------------------------------------------------------
BBLiveVar::BBLiveVar( const BasicBlock *const  baseBB, unsigned int RdfoId) 
                      : BaseBB(baseBB), DefSet(),  InSet(), 
			OutSet(), PhiArgMap() {  
    BaseBB = baseBB;   
    InSetChanged = OutSetChanged = false;
    POId = RdfoId;
}


//-----------------------------------------------------------------------------
// caluculates def and use sets for each BB
// There are two passes over operands of a machine instruction. This is
// because, we can have instructions like V = V + 1, since we no longer
// assume single definition.
//-----------------------------------------------------------------------------

void BBLiveVar::calcDefUseSets()  
{
  // get the iterator for machine instructions
  const MachineCodeForBasicBlock& MIVec = BaseBB->getMachineInstrVec();
  MachineCodeForBasicBlock::const_reverse_iterator 
    MInstIterator = MIVec.rbegin();

  // iterate over all the machine instructions in BB
  for( ; MInstIterator != MIVec.rend(); ++MInstIterator) {  

    const MachineInstr * MInst  = *MInstIterator;  // MInst is the machine inst
    assert(MInst);
    
    if( DEBUG_LV > 1) {                            // debug msg
      cerr << " *Iterating over machine instr ";
      MInst->dump();
      cerr << "\n";
    }

    // iterate over  MI operands to find defs
    for( MachineInstr::val_const_op_iterator OpI(MInst); !OpI.done() ; ++OpI) {

      if( OpI.isDef() )      // add to Defs only if this operand is a def
	addDef( *OpI );
    }

    // do for implicit operands as well
    for( unsigned i=0; i < MInst->getNumImplicitRefs(); ++i) {
      if(  MInst->implicitRefIsDefined(i) )
	addDef( MInst->getImplicitRef(i) );
     }

    
    bool IsPhi = ( MInst->getOpCode() == PHI );

 
    // iterate over  MI operands to find uses
    for (MachineInstr::val_const_op_iterator OpI(MInst); !OpI.done() ; ++OpI) {
      const Value *Op = *OpI;

      if ( ((Op)->getType())->isLabelType() )    
	continue;             // don't process labels

      if(! OpI.isDef() ) {   // add to Defs only if this operand is a use
	addUse( Op );

	if( IsPhi ) {         // for a phi node
	  // put args into the PhiArgMap (Val -> BB)
	
	  const Value * ArgVal = Op;
	  ++OpI;              // increment to point to BB of value
	  const Value * BBVal = *OpI; 
	  
	  
	  assert( (BBVal)->getValueType() == Value::BasicBlockVal );
	  
	  PhiArgMap[ ArgVal ] = (const BasicBlock *) (BBVal); 
	  assert( PhiArgMap[ ArgVal ] );
	  
	  if( DEBUG_LV > 1) {   // debug msg of level 2
	    cerr << "   - phi operand "; 
	    printValue( ArgVal ); 
	    cerr << " came from BB "; 
	    printValue( PhiArgMap[ ArgVal ]); 
	    cerr << "\n";
	  }

	} // if( IsPhi )

      } // if a use

    } // for all operands

    // do for implicit operands as well
    for( unsigned i=0; i < MInst->getNumImplicitRefs(); ++i) {

      assert( !IsPhi && "Phi cannot have implicit opeands");
      const Value *Op =  MInst->getImplicitRef(i);

      if ( ((Op)->getType())->isLabelType() )    
	continue;             // don't process labels
      if(  ! MInst->implicitRefIsDefined(i) )
	addUse( Op );
     }

  } // for all machine instructions
} 


	
//-----------------------------------------------------------------------------
// To add an operand which is a def
//-----------------------------------------------------------------------------
void  BBLiveVar::addDef(const Value *Op) 
{
  DefSet.add( Op );     // operand is a def - so add to def set
  InSet.remove( Op);    // this definition kills any uses
  InSetChanged = true; 

  if( DEBUG_LV > 1) {   
    cerr << "  +Def: "; printValue( Op ); cerr << "\n";
  }
}


//-----------------------------------------------------------------------------
// To add an operand which is a use
//-----------------------------------------------------------------------------
void  BBLiveVar::addUse(const Value *Op) 
{
  InSet.add( Op );      // An operand is a use - so add to use set
  OutSet.remove( Op );  // remove if there is a def below this use
  InSetChanged = true; 

  if( DEBUG_LV > 1) {   // debug msg of level 2
    cerr << "   Use: "; printValue( Op ); cerr << endl;
  }

}


//-----------------------------------------------------------------------------
// Applies the transfer function to a basic block to produce the InSet using
// the outset. 
//-----------------------------------------------------------------------------

bool BBLiveVar::applyTransferFunc() // calculates the InSet in terms of OutSet 
{

  // IMPORTANT: caller should check whether the OutSet changed 
  //           (else no point in calling)

  LiveVarSet OutMinusDef;     // set to hold (Out[B] - Def[B])
  OutMinusDef.setDifference( &OutSet, &DefSet);
  InSetChanged = InSet.setUnion( &OutMinusDef );
 
  OutSetChanged = false;      // no change to OutSet since transf func applied

  return InSetChanged;
}


//-----------------------------------------------------------------------------
// calculates Out set using In sets of the predecessors
//-----------------------------------------------------------------------------
bool BBLiveVar::setPropagate( LiveVarSet *const OutSet, 
			      const LiveVarSet *const InSet, 
			      const BasicBlock *const PredBB) {

  LiveVarSet::const_iterator InIt;
  pair<LiveVarSet::iterator, bool> result;
  bool changed = false;
  const BasicBlock *PredBBOfPhiArg;

  // for all all elements in InSet
  for( InIt = InSet->begin() ; InIt != InSet->end(); InIt++) {  
    PredBBOfPhiArg =  PhiArgMap[ *InIt ];

    // if this var is not a phi arg OR 
    // it's a phi arg and the var went down from this BB
    if( !PredBBOfPhiArg || PredBBOfPhiArg == PredBB) {  
      result = OutSet->insert( *InIt );               // insert to this set
      if( result.second == true) changed = true;
    }
  }

  return changed;
} 


//-----------------------------------------------------------------------------
// propogates in set to OutSets of PREDECESSORs
//-----------------------------------------------------------------------------
bool BBLiveVar::applyFlowFunc(BBToBBLiveVarMapType LVMap) 
{

  // IMPORTANT: caller should check whether inset changed 
  //            (else no point in calling)

  bool needAnotherIt= false;  // did this BB change any OutSets of pred.s 
                              // whose POId is lower


  BasicBlock::pred_const_iterator PredBBI = BaseBB->pred_begin();

  for( ; PredBBI != BaseBB->pred_end() ; PredBBI++) {
    assert( *PredBBI );       // assert that the predecessor is valid
    BBLiveVar  *PredLVBB = LVMap[*PredBBI];

                              // do set union
    if(  setPropagate( &(PredLVBB->OutSet), &InSet, *PredBBI ) == true) {  
      PredLVBB->OutSetChanged = true;

      // if the predec POId is lower than mine
      if( PredLVBB->getPOId() <= POId) 
	needAnotherIt = true;   
    }

  }  // for

  return needAnotherIt;

}



/* ----------------- Methods For Debugging (Printing) ----------------- */

void BBLiveVar::printAllSets() const
{
  cerr << "  Defs: ";   DefSet.printSet();  cerr << endl;
  cerr << "  In: ";   InSet.printSet();  cerr << endl;
  cerr << "  Out: ";   OutSet.printSet();  cerr << endl;
}

void BBLiveVar::printInOutSets() const
{
  cerr << "  In: ";   InSet.printSet();  cerr << endl;
  cerr << "  Out: ";   OutSet.printSet();  cerr << endl;
}




