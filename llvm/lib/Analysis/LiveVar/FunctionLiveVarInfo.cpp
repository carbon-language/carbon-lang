/* Title:   MethodLiveVarInfo.cpp
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: 

   This is the interface for live variable info of a method that is required 
   by any other part of the compiler.

*/


#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/PostOrderIterator.h"


/************************** Constructor/Destructor ***************************/


MethodLiveVarInfo::MethodLiveVarInfo(const Method *const M) : Meth(M),  
							      BB2BBLVMap()
{ 
  assert(! M->isExternal() );           // cannot be a prototype decleration
  HasAnalyzed = false;                  // still we haven't called analyze()
}



MethodLiveVarInfo:: ~MethodLiveVarInfo()
{
  // hash map iterator
  BBToBBLiveVarMapType::iterator HMI = BB2BBLVMap.begin(); 

  for( ; HMI != BB2BBLVMap.end() ; HMI ++ ) {  
    if( (*HMI).first )                  // delete all LiveVarSets in BB2BBLVMap
      delete (*HMI).second;
   }
}


// -------------------------- support functions -------------------------------



// constructs BBLiveVars and init Def and In sets
void MethodLiveVarInfo::constructBBs()   
{
  unsigned int POId = 0;                // Reverse Depth-first Order ID

  po_iterator<const Method*> BBI = po_begin(Meth);

  for(  ; BBI != po_end(Meth) ; ++BBI, ++POId) 
  { 

    const BasicBlock *BB = *BBI;        // get the current BB 

    if(DEBUG_LV) { cout << " For BB "; printValue(BB); cout << ":" << endl; }

                                        // create a new BBLiveVar
    BBLiveVar * LVBB = new BBLiveVar( BB, POId );  
    
    BB2BBLVMap[ BB ] = LVBB;            // insert the pair to Map
    
    LVBB->calcDefUseSets();             // calculates the def and in set

    if(DEBUG_LV) 
      LVBB->printAllSets();
  }
}



// do one backward pass over the CFG
bool MethodLiveVarInfo::doSingleBackwardPass()  
{
  bool ResultFlow, NeedAnotherIteration = false;

  if(DEBUG_LV) 
    cout << endl <<  " After Backward Pass ..." << endl;

  po_iterator<const Method*> BBI = po_begin(Meth);

  for( ; BBI != po_end(Meth) ; ++BBI) 
  { 

    BBLiveVar* LVBB = BB2BBLVMap[*BBI];
    assert( LVBB );

    if(DEBUG_LV) cout << " For BB " << (*BBI)->getName() << ":"  << endl;
    // cout << " (POId=" << LVBB->getPOId() << ")" << endl ;

    ResultFlow = false;

    if( LVBB->isOutSetChanged() ) 
      LVBB->applyTransferFunc();        // apply the Tran Func to calc InSet

    if( LVBB->isInSetChanged() )        // to calc Outsets of preds
      ResultFlow = LVBB->applyFlowFunc(BB2BBLVMap); 

    if(DEBUG_LV) LVBB->printInOutSets();


    if( ResultFlow ) NeedAnotherIteration = true;

  }

  // true if we need to reiterate over the CFG
  return NeedAnotherIteration;         
}





// performs live var anal for a method
void MethodLiveVarInfo::analyze()        
{
  // Don't analyze the same method twice!
  // Later, we need to add change notification here.
  if (HasAnalyzed)
    return;
  
  if( DEBUG_LV) cout << "Analysing live variables ..." << endl;

  // create and initialize all the BBLiveVars of the CFG
  constructBBs();        

  bool NeedAnotherIteration = false;
  do {                                // do one  pass over  CFG
    NeedAnotherIteration = doSingleBackwardPass( );   
  } while (NeedAnotherIteration );    // repeat until we need more iterations

  
  HasAnalyzed  = true;                // finished analysing

  if( DEBUG_LV) cout << "Live Variable Analysis complete!" << endl;
}




/* Thsese functions will give the LiveVar info for any machine instruction in
   a method. It should be called after a call to analyze().

   Thsese functions calucluates live var info for all the machine instrs in a 
   BB when LVInfo for one inst is requested. Hence, this function is useful 
   when live var info is required for many (or all) instructions in a basic 
   block. Also, the arguments to this method does not require specific 
   iterators.
*/


const LiveVarSet * 
MethodLiveVarInfo::getLiveVarSetBeforeMInst(const MachineInstr *const MInst,
					    const BasicBlock *const CurBB) 
{
  const LiveVarSet *LVSet = MInst2LVSetBI[MInst];

  if( LVSet  ) return LVSet;              // if found, just return the set
  else { 
    calcLiveVarSetsForBB( CurBB );        // else, calc for all instrs in BB
    assert( MInst2LVSetBI[ MInst ] );
    return  MInst2LVSetBI[ MInst ];
  }
}


const LiveVarSet * 
MethodLiveVarInfo::getLiveVarSetAfterMInst(const MachineInstr *const MInst,
					    const BasicBlock *const CurBB) 
{
  const LiveVarSet *LVSet = MInst2LVSetAI[MInst];

  if( LVSet  ) return LVSet;              // if found, just return the set
  else { 
    calcLiveVarSetsForBB( CurBB );        // else, calc for all instrs in BB
    assert( MInst2LVSetAI[ MInst ] );
    return  MInst2LVSetAI[ MInst ];
  }
}


void MethodLiveVarInfo::calcLiveVarSetsForBB(const BasicBlock *const BB)
{
  const MachineCodeForBasicBlock& MIVec = BB->getMachineInstrVec();
  MachineCodeForBasicBlock::const_reverse_iterator 
    MInstIterator = MIVec.rbegin();

  LiveVarSet *CurSet = new LiveVarSet();
  const LiveVarSet *SetAI = getOutSetOfBB(BB); // init SetAI with OutSet
  CurSet->setUnion(SetAI);                     // CurSet now contains OutSet

  // iterate over all the machine instructions in BB
  for( ; MInstIterator != MIVec.rend(); MInstIterator++) {  

    // MInst is cur machine inst
    const MachineInstr * MInst  = *MInstIterator;  

    MInst2LVSetAI[MInst] = SetAI;              // record in After Inst map
    
    CurSet->applyTranferFuncForMInst( MInst ); // apply the transfer Func
    LiveVarSet *NewSet = new LiveVarSet();     // create a new set and
    NewSet->setUnion( CurSet );                // copy the set after T/F to it
 
    MInst2LVSetBI[MInst] = NewSet;             // record in Before Inst map

    // SetAI will be used in the next iteration
    SetAI = NewSet;                 
  }
  
}

















