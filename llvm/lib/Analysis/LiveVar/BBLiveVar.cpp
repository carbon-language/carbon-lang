#include "llvm/Analysis/LiveVar/BBLiveVar.h"


/********************* Implementation **************************************/

BBLiveVar::BBLiveVar( const BasicBlock* baseBB, unsigned int RdfoId) 
                      : DefSet(),  InSet(), OutSet(), PhiArgMap() {  
    BaseBB = baseBB;   
    InSetChanged = OutSetChanged = false;
    POId = RdfoId;
}



void BBLiveVar::calcDefUseSets()  // caluculates def and use sets for each BB
{
                                                // instructions in basic block 
  const BasicBlock::InstListType&  InstListInBB = BaseBB->getInstList();   

  BasicBlock::InstListType::const_reverse_iterator 
    InstIterator = InstListInBB.rbegin();  // get the iterator for instructions

                                     // iterate over all the instructions in BB
  for( ; InstIterator != InstListInBB.rend(); InstIterator++) {  

    const Instruction * Inst  = *InstIterator;     // Inst is the current instr
    assert(Inst);

    if( Inst->isDefinition() ) {  // add to Defs only if this instr is a def
  
      DefSet.add( Inst );   // nstruction is a def - so add to def set
      InSet.remove( Inst);  // this definition kills any uses
      InSetChanged = true; 
      //cout << " adding inst to def "; printValue( Inst ); cout << endl;
    }

    Instruction::op_const_iterator 
      OpI = Inst->op_begin();                // get iterator for operands

    bool IsPhi=( Inst->getOpcode() == Instruction::PHINode );  // Is this a phi

    for(int OpNum=0 ; OpI != Inst->op_end() ; OpI++) { // iterate over operands

      if ( ((*OpI)->getType())->isLabelType() ) 
	continue;                                      // don't process labels 

      InSet.add( *OpI );      // An operand is a use - so add to use set
      OutSet.remove( *OpI );  // remove if there is a definition below this use

      if( IsPhi ) {           // for a phi node
                              // put args into the PhiArgMap
	PhiArgMap[ *OpI ] = ((PHINode *) Inst )->getIncomingBlock( OpNum++ ); 
	assert( PhiArgMap[ *OpI ] );
      	//cout << " Phi operand "; printValue( *OpI ); 
	//cout  << " came from BB "; printValue(PhiArgMap[*OpI]); cout<<endl;
      }

      InSetChanged = true; 
      //cout << " adding operand to use "; printValue( *OpI ); cout << endl;
    }
    
  }
} 
	

 

bool BBLiveVar::applyTransferFunc() // calculates the InSet in terms of OutSet 
{

  // IMPORTANT: caller should check whether the OutSet changed 
  //           (else no point in calling)

  LiveVarSet OutMinusDef;                      // set to hold (Out[B] - Def[B])
  OutMinusDef.setDifference( &OutSet, &DefSet);
  InSetChanged = InSet.setUnion( &OutMinusDef );
 
  OutSetChanged = false;    // no change to OutSet since transfer func applied

  return InSetChanged;
}



                        // calculates Out set using In sets of the predecessors
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

                        // if this var is not a phi arg or it came from this BB
    if( !PredBBOfPhiArg || PredBBOfPhiArg == PredBB) {  
      result = OutSet->insert( *InIt );               // insert to this set
      if( result.second == true) changed = true;
    }
  }

  return changed;
} 



                                // propogates in set to OutSets of PREDECESSORs
bool BBLiveVar::applyFlowFunc(BBToBBLiveVarMapType LVMap) 
{

  // IMPORTANT: caller should check whether inset changed 
  //            (else no point in calling)

  bool needAnotherIt= false;  // did this BB change any OutSets of pred.s 
                              // whose POId is lower


  cfg::pred_const_iterator PredBBI = cfg::pred_begin(BaseBB);

  for( ; PredBBI != cfg::pred_end(BaseBB) ; PredBBI++) {
    assert( *PredBBI );                 // assert that the predecessor is valid
    BBLiveVar  *PredLVBB = LVMap[*PredBBI];

                                                               // do set union
    if(  setPropagate( &(PredLVBB->OutSet), &InSet, *PredBBI ) == true) {  
      PredLVBB->OutSetChanged = true;

      if( PredLVBB->getPOId() <= POId) // if the predec POId is lower than mine
	needAnotherIt = true;   
    }
  } // for

  return needAnotherIt;

}





/* ----------------- Methods For Debugging (Printing) ----------------- */

void BBLiveVar::printAllSets() const
{
  cout << "Defs: ";   DefSet.printSet();  cout << endl;
  cout << "In: ";   InSet.printSet();  cout << endl;
  cout << "Out: ";   OutSet.printSet();  cout << endl;
}

void BBLiveVar::printInOutSets() const
{
  cout << "In: ";   InSet.printSet();  cout << endl;
  cout << "Out: ";   OutSet.printSet();  cout << endl;
}
