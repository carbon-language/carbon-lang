//===-- FunctionLiveVarInfo.cpp - Live Variable Analysis for a Function ---===//
//
// This is the interface to function level live variable information that is
// provided by live variable analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LiveVar/FunctionLiveVarInfo.h"
#include "BBLiveVar.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/CFG.h"
#include "Support/PostOrderIterator.h"
#include "Support/SetOperations.h"
#include "Support/CommandLine.h"
#include <iostream>

AnalysisID FunctionLiveVarInfo::ID(AnalysisID::create<FunctionLiveVarInfo>());

LiveVarDebugLevel_t DEBUG_LV;

static cl::Enum<LiveVarDebugLevel_t> DEBUG_LV_opt(DEBUG_LV, "dlivevar", cl::Hidden,
  "enable live-variable debugging information",
  clEnumValN(LV_DEBUG_None   , "n", "disable debug output"),
  clEnumValN(LV_DEBUG_Normal , "y", "enable debug output"),
  clEnumValN(LV_DEBUG_Instr,   "i", "print live-var sets before/after every machine instrn"),
  clEnumValN(LV_DEBUG_Verbose, "v", "print def, use sets for every instrn also"), 0);



//-----------------------------------------------------------------------------
// Accessor Functions
//-----------------------------------------------------------------------------

// gets OutSet of a BB
const ValueSet &FunctionLiveVarInfo::getOutSetOfBB(const BasicBlock *BB) const {
  return BBLiveVar::GetFromBB(*BB)->getOutSet();
}

// gets InSet of a BB
const ValueSet &FunctionLiveVarInfo::getInSetOfBB(const BasicBlock *BB) const {
  return BBLiveVar::GetFromBB(*BB)->getInSet();
}


//-----------------------------------------------------------------------------
// Performs live var analysis for a function
//-----------------------------------------------------------------------------

bool FunctionLiveVarInfo::runOnFunction(Function &F) {
  M = &F;
  if (DEBUG_LV) std::cerr << "Analysing live variables ...\n";

  // create and initialize all the BBLiveVars of the CFG
  constructBBs(M);

  unsigned int iter=0;
  while (doSingleBackwardPass(M, iter++))
    ; // Iterate until we are done.
  
  if (DEBUG_LV) std::cerr << "Live Variable Analysis complete!\n";
  return false;
}


//-----------------------------------------------------------------------------
// constructs BBLiveVars and init Def and In sets
//-----------------------------------------------------------------------------

void FunctionLiveVarInfo::constructBBs(const Function *M) {
  unsigned int POId = 0;                // Reverse Depth-first Order ID
  
  for(po_iterator<const Function*> BBI = po_begin(M), BBE = po_end(M);
      BBI != BBE; ++BBI, ++POId) { 
    const BasicBlock &BB = **BBI;        // get the current BB 

    if (DEBUG_LV) std::cerr << " For BB " << RAV(BB) << ":\n";

    // create a new BBLiveVar
    BBLiveVar *LVBB = BBLiveVar::CreateOnBB(BB, POId);  
    
    if (DEBUG_LV)
      LVBB->printAllSets();
  }

  // Since the PO iterator does not discover unreachable blocks,
  // go over the random iterator and init those blocks as well.
  // However, LV info is not correct for those blocks (they are not
  // analyzed)
  //
  for (Function::const_iterator BBRI = M->begin(), BBRE = M->end();
       BBRI != BBRE; ++BBRI, ++POId)
    if (!BBLiveVar::GetFromBB(*BBRI))                 // Not yet processed?
      BBLiveVar::CreateOnBB(*BBRI, POId);
}


//-----------------------------------------------------------------------------
// do one backward pass over the CFG (for iterative analysis)
//-----------------------------------------------------------------------------

bool FunctionLiveVarInfo::doSingleBackwardPass(const Function *M,
                                               unsigned iter) {
  if (DEBUG_LV) std::cerr << "\n After Backward Pass " << iter << "...\n";

  bool NeedAnotherIteration = false;
  for (po_iterator<const Function*> BBI = po_begin(M), BBE = po_end(M);
       BBI != BBE; ++BBI) {
    BBLiveVar *LVBB = BBLiveVar::GetFromBB(**BBI);
    assert(LVBB && "BasicBlock information not set for block!");

    if (DEBUG_LV) std::cerr << " For BB " << (*BBI)->getName() << ":\n";

    // InSets are initialized to "GenSet". Recompute only if OutSet changed.
    if(LVBB->isOutSetChanged()) 
      LVBB->applyTransferFunc();        // apply the Tran Func to calc InSet
    
    // OutSets are initialized to EMPTY.  Recompute on first iter or if InSet
    // changed.
    if (iter == 0 || LVBB->isInSetChanged())        // to calc Outsets of preds
      NeedAnotherIteration |= LVBB->applyFlowFunc();
    
    if (DEBUG_LV) LVBB->printInOutSets();
  }

  // true if we need to reiterate over the CFG
  return NeedAnotherIteration;         
}


void FunctionLiveVarInfo::releaseMemory() {
  // First remove all BBLiveVar annotations created in constructBBs().
  if (M)
    for (Function::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
      BBLiveVar::RemoveFromBB(*I);
  M = 0;

  // Then delete all objects of type ValueSet created in calcLiveVarSetsForBB
  // and entered into  MInst2LVSetBI and  MInst2LVSetAI (these are caches
  // to return ValueSet's before/after a machine instruction quickly). It
  // is sufficient to free up all ValueSet using only one cache since 
  // both caches refer to the same sets
  //
  for (std::map<const MachineInstr*, const ValueSet*>::iterator
         MI = MInst2LVSetBI.begin(),
         ME = MInst2LVSetBI.end(); MI != ME; ++MI)
    delete MI->second;           // delete all ValueSets in  MInst2LVSetBI

  MInst2LVSetBI.clear();
  MInst2LVSetAI.clear();
}




//-----------------------------------------------------------------------------
// Following functions will give the LiveVar info for any machine instr in
// a function. It should be called after a call to analyze().
//
// Thsese functions calucluates live var info for all the machine instrs in a 
// BB when LVInfo for one inst is requested. Hence, this function is useful 
// when live var info is required for many (or all) instructions in a basic 
// block. Also, the arguments to this function does not require specific 
// iterators.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Gives live variable information before a machine instruction
//-----------------------------------------------------------------------------

const ValueSet &
FunctionLiveVarInfo::getLiveVarSetBeforeMInst(const MachineInstr *MInst,
                                              const BasicBlock *BB) {
  if (const ValueSet *LVSet = MInst2LVSetBI[MInst]) {
    return *LVSet;                      // if found, just return the set
  } else { 
    calcLiveVarSetsForBB(BB);          // else, calc for all instrs in BB
    return *MInst2LVSetBI[MInst];
  }
}


//-----------------------------------------------------------------------------
// Gives live variable information after a machine instruction
//-----------------------------------------------------------------------------
const ValueSet & 
FunctionLiveVarInfo::getLiveVarSetAfterMInst(const MachineInstr *MI,
                                             const BasicBlock *BB) {

  if (const ValueSet *LVSet = MInst2LVSetAI[MI]) {
    return *LVSet;                      // if found, just return the set
  } else { 
    calcLiveVarSetsForBB(BB);           // else, calc for all instrs in BB
    return *MInst2LVSetAI[MI];
  }
}

// This function applies a machine instr to a live var set (accepts OutSet) and
// makes necessary changes to it (produces InSet). Note that two for loops are
// used to first kill all defs and then to add all uses. This is because there
// can be instructions like Val = Val + 1 since we allow multipe defs to a 
// machine instruction operand.
//
static void applyTranferFuncForMInst(ValueSet &LVS, const MachineInstr *MInst) {
  for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
         OpE = MInst->end(); OpI != OpE; ++OpI) {
    if (OpI.isDef())           // kill only if this operand is a def
      LVS.erase(*OpI);         // this definition kills any uses
  }

  // do for implicit operands as well
  for (unsigned i=0; i < MInst->getNumImplicitRefs(); ++i) {
    if (MInst->implicitRefIsDefined(i))
      LVS.erase(MInst->getImplicitRef(i));
  }

  for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
         OpE = MInst->end(); OpI != OpE; ++OpI) {
    if (!isa<BasicBlock>(*OpI))      // don't process labels
      if (!OpI.isDef())              // add only if this operand is a use
        LVS.insert(*OpI);            // An operand is a use - so add to use set
  }

  // do for implicit operands as well
  for (unsigned i = 0, e = MInst->getNumImplicitRefs(); i != e; ++i)
    if (!MInst->implicitRefIsDefined(i))
      LVS.insert(MInst->getImplicitRef(i));
}

//-----------------------------------------------------------------------------
// This method calculates the live variable information for all the 
// instructions in a basic block and enter the newly constructed live
// variable sets into a the caches (MInst2LVSetAI, MInst2LVSetBI)
//-----------------------------------------------------------------------------

void FunctionLiveVarInfo::calcLiveVarSetsForBB(const BasicBlock *BB) {
  const MachineCodeForBasicBlock &MIVec = BB->getMachineInstrVec();

  if (DEBUG_LV >= LV_DEBUG_Instr)
    std::cerr << "\n======For BB " << BB->getName()
              << ": Live var sets for instructions======\n";
  
  ValueSet CurSet;
  const ValueSet *SetAI = &getOutSetOfBB(BB);  // init SetAI with OutSet
  set_union(CurSet, *SetAI);                   // CurSet now contains OutSet

  // iterate over all the machine instructions in BB
  for (MachineCodeForBasicBlock::const_reverse_iterator MII = MIVec.rbegin(),
         MIE = MIVec.rend(); MII != MIE; ++MII) {  
    // MI is cur machine inst
    const MachineInstr *MI = *MII;  

    MInst2LVSetAI[MI] = SetAI;                 // record in After Inst map

    applyTranferFuncForMInst(CurSet, MI);      // apply the transfer Func
    ValueSet *NewSet = new ValueSet();         // create a new set and
    set_union(*NewSet, CurSet);                // copy the set after T/F to it
 
    MInst2LVSetBI[MI] = NewSet;                // record in Before Inst map

    if (DEBUG_LV >= LV_DEBUG_Instr) {
      std::cerr << "\nLive var sets before/after instruction " << *MI;
      std::cerr << "  Before: ";   printSet(*NewSet);  std::cerr << "\n";
      std::cerr << "  After : ";   printSet(*SetAI);   std::cerr << "\n";
    }

    // SetAI will be used in the next iteration
    SetAI = NewSet;                 
  }
}
