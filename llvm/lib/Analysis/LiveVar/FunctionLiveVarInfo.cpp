/* Title:   ValueSet.h
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: 

   This is the interface for live variable info of a method that is required by 
   any other part of the compiler.

*/


#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"




/************************** Constructor/Destructor ***************************/


MethodLiveVarInfo::MethodLiveVarInfo(Method *const MethPtr) :  BB2BBLVMap()  
{
  Meth = MethPtr;  // init BB2BBLVMap and records Method for future use
}



MethodLiveVarInfo:: ~MethodLiveVarInfo()
{
  BBToBBLiveVarMapType::iterator HMI = BB2BBLVMap.begin();   // hash map iterator

  for( ; HMI != BB2BBLVMap.end() ; HMI ++ ) {  
    if( (*HMI).first )                    // delete all LiveVarSets in BB2BBLVMap
      delete (*HMI).second;
   }
}


// -------------------------- support functions -------------------------------



                                // constructs BBLiveVars and init Def and In sets
void MethodLiveVarInfo::constructBBs()   
{
  unsigned int POId = 0;   // Reverse Depth-first Order ID

  cfg::po_const_iterator BBI = cfg::po_begin(Meth);

  for(  ; BBI != cfg::po_end(Meth) ; ++BBI, ++POId) 
  { 

    if(DEBUG_LV) cout << "-- For BB " << (*BBI)->getName() << ":" << endl ;

    const BasicBlock *BB = *BBI;      // get the current BB 
    BBLiveVar * LVBB = new BBLiveVar( BB, POId );  // create a new BBLiveVar
    
    BB2BBLVMap[ BB ] = LVBB;  // insert the pair to Map
    
    LVBB->calcDefUseSets();  // calculates the def and in set

    if(DEBUG_LV) LVBB->printAllSets();
    //cout << "InSetChanged: " << LVBB->isInSetChanged() << endl; 
  }

 
}

                                             // do one backward pass over the CFG
bool MethodLiveVarInfo::doSingleBackwardPass()  
{
  bool ResultFlow, NeedAnotherIteration = false;

  if(DEBUG_LV) cout << endl <<  "------- After Backward Pass --------" << endl;

  cfg::po_const_iterator BBI = cfg::po_begin(Meth);

  for( ; BBI != cfg::po_end(Meth) ; ++BBI) 
  { 

    BBLiveVar* LVBB = BB2BBLVMap[*BBI];
    assert( LVBB );

    if(DEBUG_LV) cout << "-- For BB " << (*BBI)->getName() << ":"  << endl;
    // cout << " (POId=" << LVBB->getPOId() << ")" << endl ;

    ResultFlow = false;

    if( LVBB->isOutSetChanged() ) 
      LVBB->applyTransferFunc();   // apply the Transfer Func to calc the InSet
    if( LVBB->isInSetChanged() )  
      ResultFlow = LVBB->applyFlowFunc( BB2BBLVMap ); // to calc Outsets of preds

    if(DEBUG_LV) LVBB->printInOutSets();
    //cout << "InChanged = " << LVBB->isInSetChanged() 
    //cout << "   UpdatedBBwithLowerPOId = " << ResultFlow  << endl;

    if( ResultFlow ) NeedAnotherIteration = true;

  }

  return NeedAnotherIteration; // true if we need to reiterate over the CFG
}





void MethodLiveVarInfo::analyze()          // performs live var anal for a method
{
  //cout << "In analyze . . ." << cout;

  constructBBs();          // create and initialize all the BBLiveVars of the CFG

  bool NeedAnotherIteration = false;
  do {
    NeedAnotherIteration = doSingleBackwardPass( );   // do one  pass over  CFG
  } while (NeedAnotherIteration );      // repeat until we need more iterations
}




/* This function will give the LiveVar info for any instruction in a method. It 
   should be called after a call to analyze().

   This function calucluates live var info for all the instructions in a BB, 
   when LVInfo for one inst is requested. Hence, this function is useful when 
   live var info is required for many (or all) instructions in a basic block
   Also, the arguments to this method does not require specific iterators
*/


const LiveVarSet * 
MethodLiveVarInfo::getLiveVarSetBeforeInst(const Instruction *const Inst) 
{
                                   // get the BB corresponding to the instruction
  const BasicBlock *const CurBB = Inst->getParent();  

  const LiveVarSet *LVSet = Inst2LVSetMap[Inst];

  if( LVSet  ) return LVSet;                     // if found, just return the set

  const BasicBlock::InstListType&  InstListInBB = CurBB->getInstList();         
  BasicBlock::InstListType::const_reverse_iterator 
    InstItEnd= InstListInBB.rend() - 1;    // InstItEnd is set to the first instr

                                                  // LVSet of first instr = InSet
  Inst2LVSetMap[*InstItEnd] = getInSetOfBB( CurBB );  

                  // if the first instruction is requested, just return the InSet
  if( Inst == *InstItEnd) return  Inst2LVSetMap[Inst];      

                 // else calculate for all other instruction in the BB

  BasicBlock::InstListType::const_reverse_iterator 
    InstIt= InstListInBB.rbegin();  // get the iterator for instructions in BB

  LiveVarSet *CurSet = new LiveVarSet();
  CurSet->setUnion( getOutSetOfBB( CurBB ));   // LVSet now contains the OutSet

    // calculate LVSet for all instructions in the basic block (except the first)
  for( ; InstIt != InstItEnd ;  InstIt++) {    

    CurSet->applyTranferFuncForInst( *InstIt );  // apply the transfer Func
    LiveVarSet *NewSet = new LiveVarSet();       // create a new set and
    NewSet->setUnion( CurSet );                  // copy the set after T/F to it 
    Inst2LVSetMap[*InstIt] = NewSet;             // record that in the map
  }

  return Inst2LVSetMap[Inst];
}



/*
NOTES: delete all the LVBBs allocated by adding a destructor to the BB2BBLVMap???
            use the dfo_iterator in the doSingleBackwardPass  
*/











