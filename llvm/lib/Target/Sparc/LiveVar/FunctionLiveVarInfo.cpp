//===-- FunctionLiveVarInfo.cpp - Live Variable Analysis for a Function ---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is the interface to function level live variable information that is
// provided by live variable analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/FunctionLiveVarInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CFG.h"
#include "Support/PostOrderIterator.h"
#include "Support/SetOperations.h"
#include "Support/CommandLine.h"
#include "BBLiveVar.h"

namespace llvm {

static RegisterAnalysis<FunctionLiveVarInfo>
X("livevar", "Live Variable Analysis");

LiveVarDebugLevel_t DEBUG_LV;

static cl::opt<LiveVarDebugLevel_t, true>
DEBUG_LV_opt("dlivevar", cl::Hidden, cl::location(DEBUG_LV),
             cl::desc("enable live-variable debugging information"),
             cl::values(
clEnumValN(LV_DEBUG_None   , "n", "disable debug output"),
clEnumValN(LV_DEBUG_Normal , "y", "enable debug output"),
clEnumValN(LV_DEBUG_Instr,   "i", "print live-var sets before/after "
           "every machine instrn"),
clEnumValN(LV_DEBUG_Verbose, "v", "print def, use sets for every instrn also"),
                        0));



//-----------------------------------------------------------------------------
// Accessor Functions
//-----------------------------------------------------------------------------

// gets OutSet of a BB
const ValueSet &FunctionLiveVarInfo::getOutSetOfBB(const BasicBlock *BB) const {
  return BBLiveVarInfo.find(BB)->second->getOutSet();
}
      ValueSet &FunctionLiveVarInfo::getOutSetOfBB(const BasicBlock *BB)       {
  return BBLiveVarInfo[BB]->getOutSet();
}

// gets InSet of a BB
const ValueSet &FunctionLiveVarInfo::getInSetOfBB(const BasicBlock *BB) const {
  return BBLiveVarInfo.find(BB)->second->getInSet();
}
ValueSet &FunctionLiveVarInfo::getInSetOfBB(const BasicBlock *BB) {
  return BBLiveVarInfo[BB]->getInSet();
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

void FunctionLiveVarInfo::constructBBs(const Function *F) {
  unsigned POId = 0;                // Reverse Depth-first Order ID
  std::map<const BasicBlock*, unsigned> PONumbering;

  for (po_iterator<const Function*> BBI = po_begin(M), BBE = po_end(M);
      BBI != BBE; ++BBI)
    PONumbering[*BBI] = POId++;

  MachineFunction &MF = MachineFunction::get(F);
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    const BasicBlock &BB = *I->getBasicBlock();        // get the current BB 
    if (DEBUG_LV) std::cerr << " For BB " << RAV(BB) << ":\n";

    BBLiveVar *LVBB;
    std::map<const BasicBlock*, unsigned>::iterator POI = PONumbering.find(&BB);
    if (POI != PONumbering.end()) {
      // create a new BBLiveVar
      LVBB = new BBLiveVar(BB, *I, POId);
    } else {
      // The PO iterator does not discover unreachable blocks, but the random
      // iterator later may access these blocks.  We must make sure to
      // initialize unreachable blocks as well.  However, LV info is not correct
      // for those blocks (they are not analyzed)
      //
      LVBB = new BBLiveVar(BB, *I, ++POId);
    }
    BBLiveVarInfo[&BB] = LVBB;
    
    if (DEBUG_LV)
      LVBB->printAllSets();
  }
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
    BBLiveVar *LVBB = BBLiveVarInfo[*BBI];
    assert(LVBB && "BasicBlock information not set for block!");

    if (DEBUG_LV) std::cerr << " For BB " << (*BBI)->getName() << ":\n";

    // InSets are initialized to "GenSet". Recompute only if OutSet changed.
    if(LVBB->isOutSetChanged()) 
      LVBB->applyTransferFunc();        // apply the Tran Func to calc InSet
    
    // OutSets are initialized to EMPTY.  Recompute on first iter or if InSet
    // changed.
    if (iter == 0 || LVBB->isInSetChanged())        // to calc Outsets of preds
      NeedAnotherIteration |= LVBB->applyFlowFunc(BBLiveVarInfo);
    
    if (DEBUG_LV) LVBB->printInOutSets();
  }

  // true if we need to reiterate over the CFG
  return NeedAnotherIteration;         
}


void FunctionLiveVarInfo::releaseMemory() {
  // First remove all BBLiveVars created in constructBBs().
  if (M) {
    for (Function::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
      delete BBLiveVarInfo[I];
    BBLiveVarInfo.clear();
  }
  M = 0;

  // Then delete all objects of type ValueSet created in calcLiveVarSetsForBB
  // and entered into MInst2LVSetBI and MInst2LVSetAI (these are caches
  // to return ValueSet's before/after a machine instruction quickly).
  // We do not need to free up ValueSets in MInst2LVSetAI because it holds
  // pointers to the same sets as in MInst2LVSetBI (for all instructions
  // except the last one in a BB) or in BBLiveVar (for the last instruction).
  //
  for (hash_map<const MachineInstr*, ValueSet*>::iterator
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
// These functions calculate live var info for all the machine instrs in a 
// BB when LVInfo for one inst is requested. Hence, this function is useful 
// when live var info is required for many (or all) instructions in a basic 
// block. Also, the arguments to this function does not require specific 
// iterators.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Gives live variable information before a machine instruction
//-----------------------------------------------------------------------------

const ValueSet &
FunctionLiveVarInfo::getLiveVarSetBeforeMInst(const MachineInstr *MI,
                                              const BasicBlock *BB) {
  ValueSet* &LVSet = MInst2LVSetBI[MI]; // ref. to map entry
  if (LVSet == NULL && BB != NULL) {    // if not found and BB provided
    calcLiveVarSetsForBB(BB);           // calc LVSet for all instrs in BB
    assert(LVSet != NULL);
  }
  return *LVSet;
}


//-----------------------------------------------------------------------------
// Gives live variable information after a machine instruction
//-----------------------------------------------------------------------------

const ValueSet & 
FunctionLiveVarInfo::getLiveVarSetAfterMInst(const MachineInstr *MI,
                                             const BasicBlock *BB) {

  ValueSet* &LVSet = MInst2LVSetAI[MI]; // ref. to map entry
  if (LVSet == NULL && BB != NULL) {    // if not found and BB provided 
    calcLiveVarSetsForBB(BB);           // calc LVSet for all instrs in BB
    assert(LVSet != NULL);
  }
  return *LVSet;
}

// This function applies a machine instr to a live var set (accepts OutSet) and
// makes necessary changes to it (produces InSet). Note that two for loops are
// used to first kill all defs and then to add all uses. This is because there
// can be instructions like Val = Val + 1 since we allow multiple defs to a 
// machine instruction operand.
//
static void applyTranferFuncForMInst(ValueSet &LVS, const MachineInstr *MInst) {
  for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
         OpE = MInst->end(); OpI != OpE; ++OpI) {
    if (OpI.isDef())                          // kill if this operand is a def
      LVS.erase(*OpI);                        // this definition kills any uses
  }

  // do for implicit operands as well
  for (unsigned i=0; i < MInst->getNumImplicitRefs(); ++i) {
    if (MInst->getImplicitOp(i).isDef())
      LVS.erase(MInst->getImplicitRef(i));
  }

  for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
         OpE = MInst->end(); OpI != OpE; ++OpI) {
    if (!isa<BasicBlock>(*OpI))      // don't process labels
      // add only if this operand is a use
      if (OpI.isUse())
        LVS.insert(*OpI);            // An operand is a use - so add to use set
  }

  // do for implicit operands as well
  for (unsigned i = 0, e = MInst->getNumImplicitRefs(); i != e; ++i)
    if (MInst->getImplicitOp(i).isUse())
      LVS.insert(MInst->getImplicitRef(i));
}

//-----------------------------------------------------------------------------
// This method calculates the live variable information for all the 
// instructions in a basic block and enter the newly constructed live
// variable sets into a the caches (MInst2LVSetAI, MInst2LVSetBI)
//-----------------------------------------------------------------------------

void FunctionLiveVarInfo::calcLiveVarSetsForBB(const BasicBlock *BB) {
  BBLiveVar *BBLV = BBLiveVarInfo[BB];
  assert(BBLV && "BBLiveVar annotation doesn't exist?");
  const MachineBasicBlock &MIVec = BBLV->getMachineBasicBlock();
  const MachineFunction &MF = MachineFunction::get(M);
  const TargetMachine &TM = MF.getTarget();

  if (DEBUG_LV >= LV_DEBUG_Instr)
    std::cerr << "\n======For BB " << BB->getName()
              << ": Live var sets for instructions======\n";
  
  ValueSet *SetAI = &getOutSetOfBB(BB);         // init SetAI with OutSet
  ValueSet CurSet(*SetAI);                      // CurSet now contains OutSet

  // iterate over all the machine instructions in BB
  for (MachineBasicBlock::const_reverse_iterator MII = MIVec.rbegin(),
         MIE = MIVec.rend(); MII != MIE; ++MII) {  
    // MI is cur machine inst
    const MachineInstr *MI = *MII;  

    MInst2LVSetAI[MI] = SetAI;                 // record in After Inst map

    applyTranferFuncForMInst(CurSet, MI);      // apply the transfer Func
    ValueSet *NewSet = new ValueSet(CurSet);   // create a new set with a copy
                                               // of the set after T/F
    MInst2LVSetBI[MI] = NewSet;                // record in Before Inst map

    // If the current machine instruction has delay slots, mark values
    // used by this instruction as live before and after each delay slot
    // instruction (After(MI) is the same as Before(MI+1) except for last MI).
    if (unsigned DS = TM.getInstrInfo().getNumDelaySlots(MI->getOpCode())) {
      MachineBasicBlock::const_iterator fwdMII = MII.base(); // ptr to *next* MI
      for (unsigned i = 0; i < DS; ++i, ++fwdMII) {
        assert(fwdMII != MIVec.end() && "Missing instruction in delay slot?");
        MachineInstr* DelaySlotMI = *fwdMII;
        if (! TM.getInstrInfo().isNop(DelaySlotMI->getOpCode())) {
          set_union(*MInst2LVSetBI[DelaySlotMI], *NewSet);
          if (i+1 == DS)
            set_union(*MInst2LVSetAI[DelaySlotMI], *NewSet);
        }
      }
    }

    if (DEBUG_LV >= LV_DEBUG_Instr) {
      std::cerr << "\nLive var sets before/after instruction " << *MI;
      std::cerr << "  Before: ";   printSet(*NewSet);  std::cerr << "\n";
      std::cerr << "  After : ";   printSet(*SetAI);   std::cerr << "\n";
    }

    // SetAI will be used in the next iteration
    SetAI = NewSet;                 
  }
}

} // End llvm namespace
