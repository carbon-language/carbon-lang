//===-- PhyRegAlloc.cpp ---------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Traditional graph-coloring global register allocator currently used
// by the SPARC back-end.
//
// NOTE: This register allocator has some special support
// for the Reoptimizer, such as not saving some registers on calls to
// the first-level instrumentation function.
//
// NOTE 2: This register allocator can save its state in a global
// variable in the module it's working on. This feature is not
// thread-safe; if you have doubts, leave it turned off.
// 
//===----------------------------------------------------------------------===//

#include "AllocInfo.h"
#include "IGNode.h"
#include "PhyRegAlloc.h"
#include "RegAllocCommon.h"
#include "RegClass.h"
#include "../LiveVar/FunctionLiveVarInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "../MachineInstrAnnot.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "Support/CommandLine.h"
#include "Support/SetOperations.h"
#include "Support/STLExtras.h"
#include <cmath>

namespace llvm {

RegAllocDebugLevel_t DEBUG_RA;

static cl::opt<RegAllocDebugLevel_t, true>
DRA_opt("dregalloc", cl::Hidden, cl::location(DEBUG_RA),
        cl::desc("enable register allocation debugging information"),
        cl::values(
  clEnumValN(RA_DEBUG_None   ,     "n", "disable debug output"),
  clEnumValN(RA_DEBUG_Results,     "y", "debug output for allocation results"),
  clEnumValN(RA_DEBUG_Coloring,    "c", "debug output for graph coloring step"),
  clEnumValN(RA_DEBUG_Interference,"ig","debug output for interference graphs"),
  clEnumValN(RA_DEBUG_LiveRanges , "lr","debug output for live ranges"),
  clEnumValN(RA_DEBUG_Verbose,     "v", "extra debug output"),
                   0));

/// The reoptimizer wants to be able to grovel through the register
/// allocator's state after it has done its job. This is a hack.
///
PhyRegAlloc::SavedStateMapTy ExportedFnAllocState;
bool SaveRegAllocState = false;
bool SaveStateToModule = true;
static cl::opt<bool, true>
SaveRegAllocStateOpt("save-ra-state", cl::Hidden,
                  cl::location (SaveRegAllocState),
                  cl::init(false),
                  cl::desc("write reg. allocator state into module"));

FunctionPass *getRegisterAllocator(TargetMachine &T) {
  return new PhyRegAlloc (T);
}

void PhyRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo> ();
  AU.addRequired<FunctionLiveVarInfo> ();
}


/// Initialize interference graphs (one in each reg class) and IGNodeLists
/// (one in each IG). The actual nodes will be pushed later.
///
void PhyRegAlloc::createIGNodeListsAndIGs() {
  if (DEBUG_RA >= RA_DEBUG_LiveRanges) std::cerr << "Creating LR lists ...\n";

  LiveRangeMapType::const_iterator HMI = LRI->getLiveRangeMap()->begin();   
  LiveRangeMapType::const_iterator HMIEnd = LRI->getLiveRangeMap()->end();   

  for (; HMI != HMIEnd ; ++HMI ) {
    if (HMI->first) { 
      LiveRange *L = HMI->second;   // get the LiveRange
      if (!L) { 
        if (DEBUG_RA && !isa<ConstantIntegral> (HMI->first))
          std::cerr << "\n**** ?!?WARNING: NULL LIVE RANGE FOUND FOR: "
               << RAV(HMI->first) << "****\n";
        continue;
      }

      // if the Value * is not null, and LR is not yet written to the IGNodeList
      if (!(L->getUserIGNode())  ) {  
        RegClass *const RC =           // RegClass of first value in the LR
          RegClassList[ L->getRegClassID() ];
        RC->addLRToIG(L);              // add this LR to an IG
      }
    }
  }
    
  // init RegClassList
  for ( unsigned rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[rc]->createInterferenceGraph();

  if (DEBUG_RA >= RA_DEBUG_LiveRanges) std::cerr << "LRLists Created!\n";
}


/// Add all interferences for a given instruction.  Interference occurs only
/// if the LR of Def (Inst or Arg) is of the same reg class as that of live
/// var. The live var passed to this function is the LVset AFTER the
/// instruction.
///
void PhyRegAlloc::addInterference(const Value *Def, const ValueSet *LVSet,
				  bool isCallInst) {
  ValueSet::const_iterator LIt = LVSet->begin();

  // get the live range of instruction
  const LiveRange *const LROfDef = LRI->getLiveRangeForValue( Def );   

  IGNode *const IGNodeOfDef = LROfDef->getUserIGNode();
  assert( IGNodeOfDef );

  RegClass *const RCOfDef = LROfDef->getRegClass(); 

  // for each live var in live variable set
  for ( ; LIt != LVSet->end(); ++LIt) {

    if (DEBUG_RA >= RA_DEBUG_Verbose)
      std::cerr << "< Def=" << RAV(Def) << ", Lvar=" << RAV(*LIt) << "> ";

    //  get the live range corresponding to live var
    LiveRange *LROfVar = LRI->getLiveRangeForValue(*LIt);

    // LROfVar can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if (LROfVar)
      if (LROfDef != LROfVar)                  // do not set interf for same LR
        if (RCOfDef == LROfVar->getRegClass()) // 2 reg classes are the same
          RCOfDef->setInterference( LROfDef, LROfVar);  
  }
}


/// For a call instruction, this method sets the CallInterference flag in 
/// the LR of each variable live in the Live Variable Set live after the
/// call instruction (except the return value of the call instruction - since
/// the return value does not interfere with that call itself).
///
void PhyRegAlloc::setCallInterferences(const MachineInstr *MInst, 
				       const ValueSet *LVSetAft) {
  if (DEBUG_RA >= RA_DEBUG_Interference)
    std::cerr << "\n For call inst: " << *MInst;

  // for each live var in live variable set after machine inst
  for (ValueSet::const_iterator LIt = LVSetAft->begin(), LEnd = LVSetAft->end();
       LIt != LEnd; ++LIt) {

    //  get the live range corresponding to live var
    LiveRange *const LR = LRI->getLiveRangeForValue(*LIt ); 

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if (LR ) {  
      if (DEBUG_RA >= RA_DEBUG_Interference) {
        std::cerr << "\n\tLR after Call: ";
        printSet(*LR);
      }
      LR->setCallInterference();
      if (DEBUG_RA >= RA_DEBUG_Interference) {
	std::cerr << "\n  ++After adding call interference for LR: " ;
	printSet(*LR);
      }
    }

  }

  // Now find the LR of the return value of the call
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but it does not interfere with call
  // (i.e., we can allocate a volatile register to the return value)
  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(MInst);
  
  if (const Value *RetVal = argDesc->getReturnValue()) {
    LiveRange *RetValLR = LRI->getLiveRangeForValue( RetVal );
    assert( RetValLR && "No LR for RetValue of call");
    RetValLR->clearCallInterference();
  }

  // If the CALL is an indirect call, find the LR of the function pointer.
  // That has a call interference because it conflicts with outgoing args.
  if (const Value *AddrVal = argDesc->getIndirectFuncPtr()) {
    LiveRange *AddrValLR = LRI->getLiveRangeForValue( AddrVal );
    assert( AddrValLR && "No LR for indirect addr val of call");
    AddrValLR->setCallInterference();
  }
}


/// Create interferences in the IG of each RegClass, and calculate the spill
/// cost of each Live Range (it is done in this method to save another pass
/// over the code).
///
void PhyRegAlloc::buildInterferenceGraphs() {
  if (DEBUG_RA >= RA_DEBUG_Interference)
    std::cerr << "Creating interference graphs ...\n";

  unsigned BBLoopDepthCost;
  for (MachineFunction::iterator BBI = MF->begin(), BBE = MF->end();
       BBI != BBE; ++BBI) {
    const MachineBasicBlock &MBB = *BBI;
    const BasicBlock *BB = MBB.getBasicBlock();

    // find the 10^(loop_depth) of this BB 
    BBLoopDepthCost = (unsigned)pow(10.0, LoopDepthCalc->getLoopDepth(BB));

    // get the iterator for machine instructions
    MachineBasicBlock::const_iterator MII = MBB.begin();

    // iterate over all the machine instructions in BB
    for ( ; MII != MBB.end(); ++MII) {
      const MachineInstr *MInst = MII;

      // get the LV set after the instruction
      const ValueSet &LVSetAI = LVI->getLiveVarSetAfterMInst(MInst, BB);
      bool isCallInst = TM.getInstrInfo().isCall(MInst->getOpcode());

      if (isCallInst) {
	// set the isCallInterference flag of each live range which extends
	// across this call instruction. This information is used by graph
	// coloring algorithm to avoid allocating volatile colors to live ranges
	// that span across calls (since they have to be saved/restored)
	setCallInterferences(MInst, &LVSetAI);
      }

      // iterate over all MI operands to find defs
      for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
             OpE = MInst->end(); OpI != OpE; ++OpI) {
       	if (OpI.isDef()) // create a new LR since def
	  addInterference(*OpI, &LVSetAI, isCallInst);

	// Calculate the spill cost of each live range
	LiveRange *LR = LRI->getLiveRangeForValue(*OpI);
	if (LR) LR->addSpillCost(BBLoopDepthCost);
      } 

      // Mark all operands of pseudo-instructions as interfering with one
      // another.  This must be done because pseudo-instructions may be
      // expanded to multiple instructions by the assembler, so all the
      // operands must get distinct registers.
      if (TM.getInstrInfo().isPseudoInstr(MInst->getOpcode()))
      	addInterf4PseudoInstr(MInst);

      // Also add interference for any implicit definitions in a machine
      // instr (currently, only calls have this).
      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      for (unsigned z=0; z < NumOfImpRefs; z++) 
        if (MInst->getImplicitOp(z).isDef())
	  addInterference( MInst->getImplicitRef(z), &LVSetAI, isCallInst );

    } // for all machine instructions in BB
  } // for all BBs in function

  // add interferences for function arguments. Since there are no explicit 
  // defs in the function for args, we have to add them manually
  addInterferencesForArgs();          

  if (DEBUG_RA >= RA_DEBUG_Interference)
    std::cerr << "Interference graphs calculated!\n";
}


/// Mark all operands of the given MachineInstr as interfering with one
/// another.
///
void PhyRegAlloc::addInterf4PseudoInstr(const MachineInstr *MInst) {
  bool setInterf = false;

  // iterate over MI operands to find defs
  for (MachineInstr::const_val_op_iterator It1 = MInst->begin(),
         ItE = MInst->end(); It1 != ItE; ++It1) {
    const LiveRange *LROfOp1 = LRI->getLiveRangeForValue(*It1); 
    assert((LROfOp1 || It1.isDef()) && "No LR for Def in PSEUDO insruction");

    MachineInstr::const_val_op_iterator It2 = It1;
    for (++It2; It2 != ItE; ++It2) {
      const LiveRange *LROfOp2 = LRI->getLiveRangeForValue(*It2); 

      if (LROfOp2) {
	RegClass *RCOfOp1 = LROfOp1->getRegClass(); 
	RegClass *RCOfOp2 = LROfOp2->getRegClass(); 
 
	if (RCOfOp1 == RCOfOp2 ){ 
	  RCOfOp1->setInterference( LROfOp1, LROfOp2 );  
	  setInterf = true;
	}
      } // if Op2 has a LR
    } // for all other defs in machine instr
  } // for all operands in an instruction

  if (!setInterf && MInst->getNumOperands() > 2) {
    std::cerr << "\nInterf not set for any operand in pseudo instr:\n";
    std::cerr << *MInst;
    assert(0 && "Interf not set for pseudo instr with > 2 operands" );
  }
} 


/// Add interferences for incoming arguments to a function.
///
void PhyRegAlloc::addInterferencesForArgs() {
  // get the InSet of root BB
  const ValueSet &InSet = LVI->getInSetOfBB(&Fn->front());  

  for (Function::const_aiterator AI = Fn->abegin(); AI != Fn->aend(); ++AI) {
    // add interferences between args and LVars at start 
    addInterference(AI, &InSet, false);
    
    if (DEBUG_RA >= RA_DEBUG_Interference)
      std::cerr << " - %% adding interference for argument " << RAV(AI) << "\n";
  }
}


/// The following are utility functions used solely by updateMachineCode and
/// the functions that it calls. They should probably be folded back into
/// updateMachineCode at some point.
///

// used by: updateMachineCode (1 time), PrependInstructions (1 time)
inline void InsertBefore(MachineInstr* newMI, MachineBasicBlock& MBB,
                         MachineBasicBlock::iterator& MII) {
  MII = MBB.insert(MII, newMI);
  ++MII;
}

// used by: AppendInstructions (1 time)
inline void InsertAfter(MachineInstr* newMI, MachineBasicBlock& MBB,
                        MachineBasicBlock::iterator& MII) {
  ++MII;    // insert before the next instruction
  MII = MBB.insert(MII, newMI);
}

// used by: updateMachineCode (2 times)
inline void PrependInstructions(std::vector<MachineInstr *> &IBef,
                                MachineBasicBlock& MBB,
                                MachineBasicBlock::iterator& MII,
                                const std::string& msg) {
  if (!IBef.empty()) {
      MachineInstr* OrigMI = MII;
      std::vector<MachineInstr *>::iterator AdIt; 
      for (AdIt = IBef.begin(); AdIt != IBef.end() ; ++AdIt) {
          if (DEBUG_RA) {
            if (OrigMI) std::cerr << "For MInst:\n  " << *OrigMI;
            std::cerr << msg << "PREPENDed instr:\n  " << **AdIt << "\n";
          }
          InsertBefore(*AdIt, MBB, MII);
        }
    }
}

// used by: updateMachineCode (1 time)
inline void AppendInstructions(std::vector<MachineInstr *> &IAft,
                               MachineBasicBlock& MBB,
                               MachineBasicBlock::iterator& MII,
                               const std::string& msg) {
  if (!IAft.empty()) {
      MachineInstr* OrigMI = MII;
      std::vector<MachineInstr *>::iterator AdIt; 
      for ( AdIt = IAft.begin(); AdIt != IAft.end() ; ++AdIt ) {
          if (DEBUG_RA) {
            if (OrigMI) std::cerr << "For MInst:\n  " << *OrigMI;
            std::cerr << msg << "APPENDed instr:\n  "  << **AdIt << "\n";
          }
          InsertAfter(*AdIt, MBB, MII);
        }
    }
}

/// Set the registers for operands in the given MachineInstr, if a register was
/// successfully allocated.  Return true if any of its operands has been marked
/// for spill.
///
bool PhyRegAlloc::markAllocatedRegs(MachineInstr* MInst)
{
  bool instrNeedsSpills = false;

  // First, set the registers for operands in the machine instruction
  // if a register was successfully allocated.  Do this first because we
  // will need to know which registers are already used by this instr'n.
  for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {
      MachineOperand& Op = MInst->getOperand(OpNum);
      if (Op.getType() ==  MachineOperand::MO_VirtualRegister || 
          Op.getType() ==  MachineOperand::MO_CCRegister) {
          const Value *const Val =  Op.getVRegValue();
          if (const LiveRange* LR = LRI->getLiveRangeForValue(Val)) {
            // Remember if any operand needs spilling
            instrNeedsSpills |= LR->isMarkedForSpill();

            // An operand may have a color whether or not it needs spilling
            if (LR->hasColor())
              MInst->SetRegForOperand(OpNum,
                          MRI.getUnifiedRegNum(LR->getRegClassID(),
                                               LR->getColor()));
          }
        }
    } // for each operand

  return instrNeedsSpills;
}

/// Mark allocated registers (using markAllocatedRegs()) on the instruction
/// that MII points to. Then, if it's a call instruction, insert caller-saving
/// code before and after it. Finally, insert spill code before and after it,
/// using insertCode4SpilledLR().
///
void PhyRegAlloc::updateInstruction(MachineBasicBlock::iterator& MII,
                                    MachineBasicBlock &MBB) {
  MachineInstr* MInst = MII;
  unsigned Opcode = MInst->getOpcode();

  // Reset tmp stack positions so they can be reused for each machine instr.
  MF->getInfo()->popAllTempValues();  

  // Mark the operands for which regs have been allocated.
  bool instrNeedsSpills = markAllocatedRegs(MII);

#ifndef NDEBUG
  // Mark that the operands have been updated.  Later,
  // setRelRegsUsedByThisInst() is called to find registers used by each
  // MachineInst, and it should not be used for an instruction until
  // this is done.  This flag just serves as a sanity check.
  OperandsColoredMap[MInst] = true;
#endif

  // Now insert caller-saving code before/after the call.
  // Do this before inserting spill code since some registers must be
  // used by save/restore and spill code should not use those registers.
  if (TM.getInstrInfo().isCall(Opcode)) {
    AddedInstrns &AI = AddedInstrMap[MInst];
    insertCallerSavingCode(AI.InstrnsBefore, AI.InstrnsAfter, MInst,
                           MBB.getBasicBlock());
  }

  // Now insert spill code for remaining operands not allocated to
  // registers.  This must be done even for call return instructions
  // since those are not handled by the special code above.
  if (instrNeedsSpills)
    for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {
        MachineOperand& Op = MInst->getOperand(OpNum);
        if (Op.getType() ==  MachineOperand::MO_VirtualRegister || 
            Op.getType() ==  MachineOperand::MO_CCRegister) {
            const Value* Val = Op.getVRegValue();
            if (const LiveRange *LR = LRI->getLiveRangeForValue(Val))
              if (LR->isMarkedForSpill())
                insertCode4SpilledLR(LR, MII, MBB, OpNum);
          }
      } // for each operand
}

/// Iterate over all the MachineBasicBlocks in the current function and set
/// the allocated registers for each instruction (using updateInstruction()),
/// after register allocation is complete. Then move code out of delay slots.
///
void PhyRegAlloc::updateMachineCode()
{
  // Insert any instructions needed at method entry
  MachineBasicBlock::iterator MII = MF->front().begin();
  PrependInstructions(AddedInstrAtEntry.InstrnsBefore, MF->front(), MII,
                      "At function entry: \n");
  assert(AddedInstrAtEntry.InstrnsAfter.empty() &&
         "InstrsAfter should be unnecessary since we are just inserting at "
         "the function entry point here.");
  
  for (MachineFunction::iterator BBI = MF->begin(), BBE = MF->end();
       BBI != BBE; ++BBI) {
    MachineBasicBlock &MBB = *BBI;

    // Iterate over all machine instructions in BB and mark operands with
    // their assigned registers or insert spill code, as appropriate. 
    // Also, fix operands of call/return instructions.
    for (MachineBasicBlock::iterator MII = MBB.begin(); MII != MBB.end(); ++MII)
      if (! TM.getInstrInfo().isDummyPhiInstr(MII->getOpcode()))
        updateInstruction(MII, MBB);

    // Now, move code out of delay slots of branches and returns if needed.
    // (Also, move "after" code from calls to the last delay slot instruction.)
    // Moving code out of delay slots is needed in 2 situations:
    // (1) If this is a branch and it needs instructions inserted after it,
    //     move any existing instructions out of the delay slot so that the
    //     instructions can go into the delay slot.  This only supports the
    //     case that #instrsAfter <= #delay slots.
    // 
    // (2) If any instruction in the delay slot needs
    //     instructions inserted, move it out of the delay slot and before the
    //     branch because putting code before or after it would be VERY BAD!
    // 
    // If the annul bit of the branch is set, neither of these is legal!
    // If so, we need to handle spill differently but annulling is not yet used.
    for (MachineBasicBlock::iterator MII = MBB.begin(); MII != MBB.end(); ++MII)
      if (unsigned delaySlots =
          TM.getInstrInfo().getNumDelaySlots(MII->getOpcode())) { 
          MachineBasicBlock::iterator DelaySlotMI = next(MII);
          assert(DelaySlotMI != MBB.end() && "no instruction for delay slot");
          
          // Check the 2 conditions above:
          // (1) Does a branch need instructions added after it?
          // (2) O/w does delay slot instr. need instrns before or after?
          bool isBranch = (TM.getInstrInfo().isBranch(MII->getOpcode()) ||
                           TM.getInstrInfo().isReturn(MII->getOpcode()));
          bool cond1 = (isBranch &&
                        AddedInstrMap.count(MII) &&
                        AddedInstrMap[MII].InstrnsAfter.size() > 0);
          bool cond2 = (AddedInstrMap.count(DelaySlotMI) &&
                        (AddedInstrMap[DelaySlotMI].InstrnsBefore.size() > 0 ||
                         AddedInstrMap[DelaySlotMI].InstrnsAfter.size()  > 0));

          if (cond1 || cond2) {
              assert(delaySlots==1 &&
                     "InsertBefore does not yet handle >1 delay slots!");

              if (DEBUG_RA) {
                std::cerr << "\nRegAlloc: Moved instr. with added code: "
                     << *DelaySlotMI
                     << "           out of delay slots of instr: " << *MII;
              }

              // move instruction before branch
              MBB.insert(MII, MBB.remove(DelaySlotMI));

              // On cond1 we are done (we already moved the
              // instruction out of the delay slot). On cond2 we need
              // to insert a nop in place of the moved instruction
              if (cond2) {
                MBB.insert(MII, BuildMI(TM.getInstrInfo().getNOPOpCode(),1));
              }
            }
          else {
            // For non-branch instr with delay slots (probably a call), move
            // InstrAfter to the instr. in the last delay slot.
            MachineBasicBlock::iterator tmp = next(MII, delaySlots);
            move2DelayedInstr(MII, tmp);
          }
      }

    // Finally iterate over all instructions in BB and insert before/after
    for (MachineBasicBlock::iterator MII=MBB.begin(); MII != MBB.end(); ++MII) {
      MachineInstr *MInst = MII; 

      // do not process Phis
      if (TM.getInstrInfo().isDummyPhiInstr(MInst->getOpcode()))
	continue;

      // if there are any added instructions...
      if (AddedInstrMap.count(MInst)) {
        AddedInstrns &CallAI = AddedInstrMap[MInst];

#ifndef NDEBUG
        bool isBranch = (TM.getInstrInfo().isBranch(MInst->getOpcode()) ||
                         TM.getInstrInfo().isReturn(MInst->getOpcode()));
        assert((!isBranch ||
                AddedInstrMap[MInst].InstrnsAfter.size() <=
                TM.getInstrInfo().getNumDelaySlots(MInst->getOpcode())) &&
               "Cannot put more than #delaySlots instrns after "
               "branch or return! Need to handle temps differently.");
#endif

#ifndef NDEBUG
        // Temporary sanity checking code to detect whether the same machine
        // instruction is ever inserted twice before/after a call.
        // I suspect this is happening but am not sure. --Vikram, 7/1/03.
        std::set<const MachineInstr*> instrsSeen;
        for (int i = 0, N = CallAI.InstrnsBefore.size(); i < N; ++i) {
          assert(instrsSeen.count(CallAI.InstrnsBefore[i]) == 0 &&
                 "Duplicate machine instruction in InstrnsBefore!");
          instrsSeen.insert(CallAI.InstrnsBefore[i]);
        } 
        for (int i = 0, N = CallAI.InstrnsAfter.size(); i < N; ++i) {
          assert(instrsSeen.count(CallAI.InstrnsAfter[i]) == 0 &&
                 "Duplicate machine instruction in InstrnsBefore/After!");
          instrsSeen.insert(CallAI.InstrnsAfter[i]);
        } 
#endif

        // Now add the instructions before/after this MI.
        // We do this here to ensure that spill for an instruction is inserted
        // as close as possible to an instruction (see above insertCode4Spill)
        if (! CallAI.InstrnsBefore.empty())
          PrependInstructions(CallAI.InstrnsBefore, MBB, MII,"");
        
        if (! CallAI.InstrnsAfter.empty())
          AppendInstructions(CallAI.InstrnsAfter, MBB, MII,"");

      } // if there are any added instructions
    } // for each machine instruction
  }
}


/// Insert spill code for AN operand whose LR was spilled.  May be called
/// repeatedly for a single MachineInstr if it has many spilled operands. On
/// each call, it finds a register which is not live at that instruction and
/// also which is not used by other spilled operands of the same
/// instruction. Then it uses this register temporarily to accommodate the
/// spilled value.
///
void PhyRegAlloc::insertCode4SpilledLR(const LiveRange *LR, 
                                       MachineBasicBlock::iterator& MII,
                                       MachineBasicBlock &MBB,
				       const unsigned OpNum) {
  MachineInstr *MInst = MII;
  const BasicBlock *BB = MBB.getBasicBlock();

  assert((! TM.getInstrInfo().isCall(MInst->getOpcode()) || OpNum == 0) &&
         "Outgoing arg of a call must be handled elsewhere (func arg ok)");
  assert(! TM.getInstrInfo().isReturn(MInst->getOpcode()) &&
	 "Return value of a ret must be handled elsewhere");

  MachineOperand& Op = MInst->getOperand(OpNum);
  bool isDef =  Op.isDef();
  bool isUse = Op.isUse();
  unsigned RegType = MRI.getRegTypeForLR(LR);
  int SpillOff = LR->getSpillOffFromFP();
  RegClass *RC = LR->getRegClass();

  // Get the live-variable set to find registers free before this instr.
  const ValueSet &LVSetBef = LVI->getLiveVarSetBeforeMInst(MInst, BB);

#ifndef NDEBUG
  // If this instr. is in the delay slot of a branch or return, we need to
  // include all live variables before that branch or return -- we don't want to
  // trample those!  Verify that the set is included in the LV set before MInst.
  if (MII != MBB.begin()) {
    MachineBasicBlock::iterator PredMI = prior(MII);
    if (unsigned DS = TM.getInstrInfo().getNumDelaySlots(PredMI->getOpcode()))
      assert(set_difference(LVI->getLiveVarSetBeforeMInst(PredMI), LVSetBef)
             .empty() && "Live-var set before branch should be included in "
             "live-var set of each delay slot instruction!");
  }
#endif

  MF->getInfo()->pushTempValue(MRI.getSpilledRegSize(RegType));
  
  std::vector<MachineInstr*> MIBef, MIAft;
  std::vector<MachineInstr*> AdIMid;
  
  // Choose a register to hold the spilled value, if one was not preallocated.
  // This may insert code before and after MInst to free up the value.  If so,
  // this code should be first/last in the spill sequence before/after MInst.
  int TmpRegU=(LR->hasColor()
               ? MRI.getUnifiedRegNum(LR->getRegClassID(),LR->getColor())
               : getUsableUniRegAtMI(RegType, &LVSetBef, MInst, MIBef,MIAft));
  
  // Set the operand first so that it this register does not get used
  // as a scratch register for later calls to getUsableUniRegAtMI below
  MInst->SetRegForOperand(OpNum, TmpRegU);
  
  // get the added instructions for this instruction
  AddedInstrns &AI = AddedInstrMap[MInst];

  // We may need a scratch register to copy the spilled value to/from memory.
  // This may itself have to insert code to free up a scratch register.  
  // Any such code should go before (after) the spill code for a load (store).
  // The scratch reg is not marked as used because it is only used
  // for the copy and not used across MInst.
  int scratchRegType = -1;
  int scratchReg = -1;
  if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType)) {
      scratchReg = getUsableUniRegAtMI(scratchRegType, &LVSetBef,
                                       MInst, MIBef, MIAft);
      assert(scratchReg != MRI.getInvalidRegNum());
    }
  
  if (isUse) {
    // for a USE, we have to load the value of LR from stack to a TmpReg
    // and use the TmpReg as one operand of instruction
    
    // actual loading instruction(s)
    MRI.cpMem2RegMI(AdIMid, MRI.getFramePointer(), SpillOff, TmpRegU,
                    RegType, scratchReg);
    
    // the actual load should be after the instructions to free up TmpRegU
    MIBef.insert(MIBef.end(), AdIMid.begin(), AdIMid.end());
    AdIMid.clear();
  }
  
  if (isDef) {   // if this is a Def
    // for a DEF, we have to store the value produced by this instruction
    // on the stack position allocated for this LR
    
    // actual storing instruction(s)
    MRI.cpReg2MemMI(AdIMid, TmpRegU, MRI.getFramePointer(), SpillOff,
                    RegType, scratchReg);
    
    MIAft.insert(MIAft.begin(), AdIMid.begin(), AdIMid.end());
  }  // if !DEF
  
  // Finally, insert the entire spill code sequences before/after MInst
  AI.InstrnsBefore.insert(AI.InstrnsBefore.end(), MIBef.begin(), MIBef.end());
  AI.InstrnsAfter.insert(AI.InstrnsAfter.begin(), MIAft.begin(), MIAft.end());
  
  if (DEBUG_RA) {
    std::cerr << "\nFor Inst:\n  " << *MInst;
    std::cerr << "SPILLED LR# " << LR->getUserIGNode()->getIndex();
    std::cerr << "; added Instructions:";
    for_each(MIBef.begin(), MIBef.end(), std::mem_fun(&MachineInstr::dump));
    for_each(MIAft.begin(), MIAft.end(), std::mem_fun(&MachineInstr::dump));
  }
}


/// Insert caller saving/restoring instructions before/after a call machine
/// instruction (before or after any other instructions that were inserted for
/// the call).
///
void
PhyRegAlloc::insertCallerSavingCode(std::vector<MachineInstr*> &instrnsBefore,
                                    std::vector<MachineInstr*> &instrnsAfter,
                                    MachineInstr *CallMI, 
                                    const BasicBlock *BB) {
  assert(TM.getInstrInfo().isCall(CallMI->getOpcode()));
  
  // hash set to record which registers were saved/restored
  hash_set<unsigned> PushedRegSet;

  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(CallMI);
  
  // if the call is to a instrumentation function, do not insert save and
  // restore instructions the instrumentation function takes care of save
  // restore for volatile regs.
  //
  // FIXME: this should be made general, not specific to the reoptimizer!
  const Function *Callee = argDesc->getCallInst()->getCalledFunction();
  bool isLLVMFirstTrigger = Callee && Callee->getName() == "llvm_first_trigger";

  // Now check if the call has a return value (using argDesc) and if so,
  // find the LR of the TmpInstruction representing the return value register.
  // (using the last or second-last *implicit operand* of the call MI).
  // Insert it to to the PushedRegSet since we must not save that register
  // and restore it after the call.
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but we must not save/restore it.
  if (const Value *origRetVal = argDesc->getReturnValue()) {
    unsigned retValRefNum = (CallMI->getNumImplicitRefs() -
                             (argDesc->getIndirectFuncPtr()? 1 : 2));
    const TmpInstruction* tmpRetVal =
      cast<TmpInstruction>(CallMI->getImplicitRef(retValRefNum));
    assert(tmpRetVal->getOperand(0) == origRetVal &&
           tmpRetVal->getType() == origRetVal->getType() &&
           "Wrong implicit ref?");
    LiveRange *RetValLR = LRI->getLiveRangeForValue(tmpRetVal);
    assert(RetValLR && "No LR for RetValue of call");

    if (! RetValLR->isMarkedForSpill())
      PushedRegSet.insert(MRI.getUnifiedRegNum(RetValLR->getRegClassID(),
                                               RetValLR->getColor()));
  }

  const ValueSet &LVSetAft =  LVI->getLiveVarSetAfterMInst(CallMI, BB);
  ValueSet::const_iterator LIt = LVSetAft.begin();

  // for each live var in live variable set after machine inst
  for( ; LIt != LVSetAft.end(); ++LIt) {
    // get the live range corresponding to live var
    LiveRange *const LR = LRI->getLiveRangeForValue(*LIt);

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if (LR) {  
      if (! LR->isMarkedForSpill()) {
        assert(LR->hasColor() && "LR is neither spilled nor colored?");
	unsigned RCID = LR->getRegClassID();
	unsigned Color = LR->getColor();

	if (MRI.isRegVolatile(RCID, Color) ) {
	  // if this is a call to the first-level reoptimizer
	  // instrumentation entry point, and the register is not
	  // modified by call, don't save and restore it.
	  if (isLLVMFirstTrigger && !MRI.modifiedByCall(RCID, Color))
	    continue;

	  // if the value is in both LV sets (i.e., live before and after 
	  // the call machine instruction)
	  unsigned Reg = MRI.getUnifiedRegNum(RCID, Color);
	  
	  // if we haven't already pushed this register...
	  if( PushedRegSet.find(Reg) == PushedRegSet.end() ) {
	    unsigned RegType = MRI.getRegTypeForLR(LR);

	    // Now get two instructions - to push on stack and pop from stack
	    // and add them to InstrnsBefore and InstrnsAfter of the
	    // call instruction
	    int StackOff =
              MF->getInfo()->pushTempValue(MRI.getSpilledRegSize(RegType));
            
	    //---- Insert code for pushing the reg on stack ----------
            
	    std::vector<MachineInstr*> AdIBef, AdIAft;
            
            // We may need a scratch register to copy the saved value
            // to/from memory.  This may itself have to insert code to
            // free up a scratch register.  Any such code should go before
            // the save code.  The scratch register, if any, is by default
            // temporary and not "used" by the instruction unless the
            // copy code itself decides to keep the value in the scratch reg.
            int scratchRegType = -1;
            int scratchReg = -1;
            if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType))
              { // Find a register not live in the LVSet before CallMI
                const ValueSet &LVSetBef =
                  LVI->getLiveVarSetBeforeMInst(CallMI, BB);
                scratchReg = getUsableUniRegAtMI(scratchRegType, &LVSetBef,
                                                 CallMI, AdIBef, AdIAft);
                assert(scratchReg != MRI.getInvalidRegNum());
              }
            
            if (AdIBef.size() > 0)
              instrnsBefore.insert(instrnsBefore.end(),
                                   AdIBef.begin(), AdIBef.end());
            
            MRI.cpReg2MemMI(instrnsBefore, Reg, MRI.getFramePointer(),
                            StackOff, RegType, scratchReg);
            
            if (AdIAft.size() > 0)
              instrnsBefore.insert(instrnsBefore.end(),
                                   AdIAft.begin(), AdIAft.end());
            
	    //---- Insert code for popping the reg from the stack ----------
	    AdIBef.clear();
            AdIAft.clear();
            
            // We may need a scratch register to copy the saved value
            // from memory.  This may itself have to insert code to
            // free up a scratch register.  Any such code should go
            // after the save code.  As above, scratch is not marked "used".
            scratchRegType = -1;
            scratchReg = -1;
            if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType))
              { // Find a register not live in the LVSet after CallMI
                scratchReg = getUsableUniRegAtMI(scratchRegType, &LVSetAft,
                                                 CallMI, AdIBef, AdIAft);
                assert(scratchReg != MRI.getInvalidRegNum());
              }
            
            if (AdIBef.size() > 0)
              instrnsAfter.insert(instrnsAfter.end(),
                                  AdIBef.begin(), AdIBef.end());
            
	    MRI.cpMem2RegMI(instrnsAfter, MRI.getFramePointer(), StackOff,
                            Reg, RegType, scratchReg);
            
            if (AdIAft.size() > 0)
              instrnsAfter.insert(instrnsAfter.end(),
                                  AdIAft.begin(), AdIAft.end());
	    
	    PushedRegSet.insert(Reg);
            
	    if(DEBUG_RA) {
	      std::cerr << "\nFor call inst:" << *CallMI;
	      std::cerr << " -inserted caller saving instrs: Before:\n\t ";
              for_each(instrnsBefore.begin(), instrnsBefore.end(),
                       std::mem_fun(&MachineInstr::dump));
	      std::cerr << " -and After:\n\t ";
              for_each(instrnsAfter.begin(), instrnsAfter.end(),
                       std::mem_fun(&MachineInstr::dump));
	    }	    
	  } // if not already pushed
	} // if LR has a volatile color
      } // if LR has color
    } // if there is a LR for Var
  } // for each value in the LV set after instruction
}


/// Returns the unified register number of a temporary register to be used
/// BEFORE MInst. If no register is available, it will pick one and modify
/// MIBef and MIAft to contain instructions used to free up this returned
/// register.
///
int PhyRegAlloc::getUsableUniRegAtMI(const int RegType,
                                     const ValueSet *LVSetBef,
                                     MachineInstr *MInst, 
                                     std::vector<MachineInstr*>& MIBef,
                                     std::vector<MachineInstr*>& MIAft) {
  RegClass* RC = getRegClassByID(MRI.getRegClassIDOfRegType(RegType));
  
  int RegU = getUnusedUniRegAtMI(RC, RegType, MInst, LVSetBef);
  
  if (RegU == -1) {
    // we couldn't find an unused register. Generate code to free up a reg by
    // saving it on stack and restoring after the instruction
    
    int TmpOff = MF->getInfo()->pushTempValue(MRI.getSpilledRegSize(RegType));
    
    RegU = getUniRegNotUsedByThisInst(RC, RegType, MInst);
    
    // Check if we need a scratch register to copy this register to memory.
    int scratchRegType = -1;
    if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType)) {
        int scratchReg = getUsableUniRegAtMI(scratchRegType, LVSetBef,
                                             MInst, MIBef, MIAft);
        assert(scratchReg != MRI.getInvalidRegNum());
        
        // We may as well hold the value in the scratch register instead
        // of copying it to memory and back.  But we have to mark the
        // register as used by this instruction, so it does not get used
        // as a scratch reg. by another operand or anyone else.
        ScratchRegsUsed.insert(std::make_pair(MInst, scratchReg));
        MRI.cpReg2RegMI(MIBef, RegU, scratchReg, RegType);
        MRI.cpReg2RegMI(MIAft, scratchReg, RegU, RegType);
    } else { // the register can be copied directly to/from memory so do it.
        MRI.cpReg2MemMI(MIBef, RegU, MRI.getFramePointer(), TmpOff, RegType);
        MRI.cpMem2RegMI(MIAft, MRI.getFramePointer(), TmpOff, RegU, RegType);
    }
  }
  
  return RegU;
}


/// Returns the register-class register number of a new unused register that
/// can be used to accommodate a temporary value.  May be called repeatedly
/// for a single MachineInstr.  On each call, it finds a register which is not
/// live at that instruction and which is not used by any spilled operands of
/// that instruction.
///
int PhyRegAlloc::getUnusedUniRegAtMI(RegClass *RC, const int RegType,
                                     const MachineInstr *MInst,
                                     const ValueSet* LVSetBef) {
  RC->clearColorsUsed();     // Reset array

  if (LVSetBef == NULL) {
      LVSetBef = &LVI->getLiveVarSetBeforeMInst(MInst);
      assert(LVSetBef != NULL && "Unable to get live-var set before MInst?");
  }

  ValueSet::const_iterator LIt = LVSetBef->begin();

  // for each live var in live variable set after machine inst
  for ( ; LIt != LVSetBef->end(); ++LIt) {
    // Get the live range corresponding to live var, and its RegClass
    LiveRange *const LRofLV = LRI->getLiveRangeForValue(*LIt );    

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if (LRofLV && LRofLV->getRegClass() == RC && LRofLV->hasColor())
      RC->markColorsUsed(LRofLV->getColor(),
                         MRI.getRegTypeForLR(LRofLV), RegType);
  }

  // It is possible that one operand of this MInst was already spilled
  // and it received some register temporarily. If that's the case,
  // it is recorded in machine operand. We must skip such registers.
  setRelRegsUsedByThisInst(RC, RegType, MInst);

  int unusedReg = RC->getUnusedColor(RegType);   // find first unused color
  if (unusedReg >= 0)
    return MRI.getUnifiedRegNum(RC->getID(), unusedReg);

  return -1;
}


/// Return the unified register number of a register in class RC which is not
/// used by any operands of MInst.
///
int PhyRegAlloc::getUniRegNotUsedByThisInst(RegClass *RC, 
                                            const int RegType,
                                            const MachineInstr *MInst) {
  RC->clearColorsUsed();

  setRelRegsUsedByThisInst(RC, RegType, MInst);

  // find the first unused color
  int unusedReg = RC->getUnusedColor(RegType);
  assert(unusedReg >= 0 &&
         "FATAL: No free register could be found in reg class!!");

  return MRI.getUnifiedRegNum(RC->getID(), unusedReg);
}


/// Modify the IsColorUsedArr of register class RC, by setting the bits
/// corresponding to register RegNo. This is a helper method of
/// setRelRegsUsedByThisInst().
///
static void markRegisterUsed(int RegNo, RegClass *RC, int RegType,
                             const TargetRegInfo &TRI) {
  unsigned classId = 0;
  int classRegNum = TRI.getClassRegNum(RegNo, classId);
  if (RC->getID() == classId)
    RC->markColorsUsed(classRegNum, RegType, RegType);
}

void PhyRegAlloc::setRelRegsUsedByThisInst(RegClass *RC, int RegType,
                                           const MachineInstr *MI) {
  assert(OperandsColoredMap[MI] == true &&
         "Illegal to call setRelRegsUsedByThisInst() until colored operands "
         "are marked for an instruction.");

  // Add the registers already marked as used by the instruction. Both
  // explicit and implicit operands are set.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
    if (MI->getOperand(i).hasAllocatedReg())
      markRegisterUsed(MI->getOperand(i).getReg(), RC, RegType,MRI);

  for (unsigned i = 0, e = MI->getNumImplicitRefs(); i != e; ++i)
    if (MI->getImplicitOp(i).hasAllocatedReg())
      markRegisterUsed(MI->getImplicitOp(i).getReg(), RC, RegType,MRI);

  // Add all of the scratch registers that are used to save values across the
  // instruction (e.g., for saving state register values).
  std::pair<ScratchRegsUsedTy::iterator, ScratchRegsUsedTy::iterator>
    IR = ScratchRegsUsed.equal_range(MI);
  for (ScratchRegsUsedTy::iterator I = IR.first; I != IR.second; ++I)
    markRegisterUsed(I->second, RC, RegType, MRI);

  // If there are implicit references, mark their allocated regs as well
  for (unsigned z=0; z < MI->getNumImplicitRefs(); z++)
    if (const LiveRange*
        LRofImpRef = LRI->getLiveRangeForValue(MI->getImplicitRef(z)))    
      if (LRofImpRef->hasColor())
        // this implicit reference is in a LR that received a color
        RC->markColorsUsed(LRofImpRef->getColor(),
                           MRI.getRegTypeForLR(LRofImpRef), RegType);
}


/// If there are delay slots for an instruction, the instructions added after
/// it must really go after the delayed instruction(s).  So, we Move the
/// InstrAfter of that instruction to the corresponding delayed instruction
/// using the following method.
///
void PhyRegAlloc::move2DelayedInstr(const MachineInstr *OrigMI,
                                    const MachineInstr *DelayedMI)
{
  // "added after" instructions of the original instr
  std::vector<MachineInstr *> &OrigAft = AddedInstrMap[OrigMI].InstrnsAfter;

  if (DEBUG_RA && OrigAft.size() > 0) {
    std::cerr << "\nRegAlloc: Moved InstrnsAfter for: " << *OrigMI;
    std::cerr << "         to last delay slot instrn: " << *DelayedMI;
  }

  // "added after" instructions of the delayed instr
  std::vector<MachineInstr *> &DelayedAft=AddedInstrMap[DelayedMI].InstrnsAfter;

  // go thru all the "added after instructions" of the original instruction
  // and append them to the "added after instructions" of the delayed
  // instructions
  DelayedAft.insert(DelayedAft.end(), OrigAft.begin(), OrigAft.end());

  // empty the "added after instructions" of the original instruction
  OrigAft.clear();
}


void PhyRegAlloc::colorIncomingArgs()
{
  MRI.colorMethodArgs(Fn, *LRI, AddedInstrAtEntry.InstrnsBefore,
                      AddedInstrAtEntry.InstrnsAfter);
}


/// Determine whether the suggested color of each live range is really usable,
/// and then call its setSuggestedColorUsable() method to record the answer. A
/// suggested color is NOT usable when the suggested color is volatile AND
/// when there are call interferences.
///
void PhyRegAlloc::markUnusableSugColors()
{
  LiveRangeMapType::const_iterator HMI = (LRI->getLiveRangeMap())->begin();   
  LiveRangeMapType::const_iterator HMIEnd = (LRI->getLiveRangeMap())->end();   

  for (; HMI != HMIEnd ; ++HMI ) {
    if (HMI->first) { 
      LiveRange *L = HMI->second;      // get the LiveRange
      if (L && L->hasSuggestedColor ())
        L->setSuggestedColorUsable
          (!(MRI.isRegVolatile (L->getRegClassID (), L->getSuggestedColor ())
             && L->isCallInterference ()));
    }
  } // for all LR's in hash map
}


/// For each live range that is spilled, allocates a new spill position on the
/// stack, and set the stack offsets of the live range that will be spilled to
/// that position. This must be called just after coloring the LRs.
///
void PhyRegAlloc::allocateStackSpace4SpilledLRs() {
  if (DEBUG_RA) std::cerr << "\nSetting LR stack offsets for spills...\n";

  LiveRangeMapType::const_iterator HMI    = LRI->getLiveRangeMap()->begin();   
  LiveRangeMapType::const_iterator HMIEnd = LRI->getLiveRangeMap()->end();   

  for ( ; HMI != HMIEnd ; ++HMI) {
    if (HMI->first && HMI->second) {
      LiveRange *L = HMI->second;       // get the LiveRange
      if (L->isMarkedForSpill()) {      // NOTE: allocating size of long Type **
        int stackOffset = MF->getInfo()->allocateSpilledValue(Type::LongTy);
        L->setSpillOffFromFP(stackOffset);
        if (DEBUG_RA)
          std::cerr << "  LR# " << L->getUserIGNode()->getIndex()
               << ": stack-offset = " << stackOffset << "\n";
      }
    }
  } // for all LR's in hash map
}


void PhyRegAlloc::saveStateForValue (std::vector<AllocInfo> &state,
                                     const Value *V, int Insn, int Opnd) {
  LiveRangeMapType::const_iterator HMI = LRI->getLiveRangeMap ()->find (V); 
  LiveRangeMapType::const_iterator HMIEnd = LRI->getLiveRangeMap ()->end ();   
  AllocInfo::AllocStateTy AllocState = AllocInfo::NotAllocated; 
  int Placement = -1; 
  if ((HMI != HMIEnd) && HMI->second) { 
    LiveRange *L = HMI->second; 
    assert ((L->hasColor () || L->isMarkedForSpill ()) 
            && "Live range exists but not colored or spilled"); 
    if (L->hasColor ()) { 
      AllocState = AllocInfo::Allocated; 
      Placement = MRI.getUnifiedRegNum (L->getRegClassID (), 
                                        L->getColor ()); 
    } else if (L->isMarkedForSpill ()) { 
      AllocState = AllocInfo::Spilled; 
      assert (L->hasSpillOffset () 
              && "Live range marked for spill but has no spill offset"); 
      Placement = L->getSpillOffFromFP (); 
    } 
  } 
  state.push_back (AllocInfo (Insn, Opnd, AllocState, Placement)); 
}


/// Save the global register allocation decisions made by the register
/// allocator so that they can be accessed later (sort of like "poor man's
/// debug info").
///
void PhyRegAlloc::saveState () {
  std::vector<AllocInfo> &state = FnAllocState[Fn];
  unsigned ArgNum = 0;
  // Arguments encoded as instruction # -1
  for (Function::const_aiterator i=Fn->abegin (), e=Fn->aend (); i != e; ++i) {
    const Argument *Arg = &*i;
    saveStateForValue (state, Arg, -1, ArgNum);
    ++ArgNum;
  }
  unsigned Insn = 0;
  // Instructions themselves encoded as operand # -1
  for (const_inst_iterator II=inst_begin (Fn), IE=inst_end (Fn); II!=IE; ++II){
    saveStateForValue (state, (*II), Insn, -1);
    for (unsigned i = 0; i < (*II)->getNumOperands (); ++i) {
      const Value *V = (*II)->getOperand (i);
      // Don't worry about it unless it's something whose reg. we'll need. 
      if (!isa<Argument> (V) && !isa<Instruction> (V)) 
        continue; 
      saveStateForValue (state, V, Insn, i);
    }
    ++Insn;
  }
}


/// Check the saved state filled in by saveState(), and abort if it looks
/// wrong. Only used when debugging. FIXME: Currently it just prints out
/// the state, which isn't quite as useful.
///
void PhyRegAlloc::verifySavedState () {
  std::vector<AllocInfo> &state = FnAllocState[Fn];
  int ArgNum = 0;
  for (Function::const_aiterator i=Fn->abegin (), e=Fn->aend (); i != e; ++i) {
    const Argument *Arg = &*i;
    std::cerr << "Argument:  " << *Arg << "\n"
              << "FnAllocState:\n";
    for (unsigned i = 0; i < state.size (); ++i) {
      AllocInfo &S = state[i];
      if (S.Instruction == -1 && S.Operand == ArgNum)
        std::cerr << "  " << S << "\n";
    }
    std::cerr << "----------\n";
    ++ArgNum;
  }
  int Insn = 0;
  for (const_inst_iterator II=inst_begin (Fn), IE=inst_end (Fn); II!=IE; ++II) {
    const Instruction *I = *II;
    MachineCodeForInstruction &Instrs = MachineCodeForInstruction::get (I);
    std::cerr << "Instruction: " << *I
              << "MachineCodeForInstruction:\n";
    for (unsigned i = 0, n = Instrs.size (); i != n; ++i)
      std::cerr << "  " << *Instrs[i];
    std::cerr << "FnAllocState:\n";
    for (unsigned i = 0; i < state.size (); ++i) {
      AllocInfo &S = state[i];
      if (Insn == S.Instruction)
        std::cerr << "  " << S << "\n";
    }
    std::cerr << "----------\n";
    ++Insn;
  }
}


bool PhyRegAlloc::doFinalization (Module &M) { 
  if (SaveRegAllocState) finishSavingState (M);
  return false;
}


/// Finish the job of saveState(), by collapsing FnAllocState into an LLVM
/// Constant and stuffing it inside the Module.
///
/// FIXME: There should be other, better ways of storing the saved
/// state; this one is cumbersome and does not work well with the JIT.
///
void PhyRegAlloc::finishSavingState (Module &M) {
  if (DEBUG_RA)
    std::cerr << "---- Saving reg. alloc state; SaveStateToModule = "
              << SaveStateToModule << " ----\n";

  // If saving state into the module, just copy new elements to the
  // correct global.
  if (!SaveStateToModule) {
    ExportedFnAllocState = FnAllocState;
    // FIXME: should ONLY copy new elements in FnAllocState
    return;
  }

  // Convert FnAllocState to a single Constant array and add it
  // to the Module.
  ArrayType *AT = ArrayType::get (AllocInfo::getConstantType (), 0);
  std::vector<const Type *> TV;
  TV.push_back (Type::UIntTy);
  TV.push_back (AT);
  PointerType *PT = PointerType::get (StructType::get (TV));

  std::vector<Constant *> allstate;
  for (Module::iterator I = M.begin (), E = M.end (); I != E; ++I) {
    Function *F = I;
    if (F->isExternal ()) continue;
    if (FnAllocState.find (F) == FnAllocState.end ()) {
      allstate.push_back (ConstantPointerNull::get (PT));
    } else {
      std::vector<AllocInfo> &state = FnAllocState[F];

      // Convert state into an LLVM ConstantArray, and put it in a
      // ConstantStruct (named S) along with its size.
      std::vector<Constant *> stateConstants;
      for (unsigned i = 0, s = state.size (); i != s; ++i)
        stateConstants.push_back (state[i].toConstant ());
      unsigned Size = stateConstants.size ();
      ArrayType *AT = ArrayType::get (AllocInfo::getConstantType (), Size);
      std::vector<const Type *> TV;
      TV.push_back (Type::UIntTy);
      TV.push_back (AT);
      StructType *ST = StructType::get (TV);
      std::vector<Constant *> CV;
      CV.push_back (ConstantUInt::get (Type::UIntTy, Size));
      CV.push_back (ConstantArray::get (AT, stateConstants));
      Constant *S = ConstantStruct::get (ST, CV);

      GlobalVariable *GV =
        new GlobalVariable (ST, true,
                            GlobalValue::InternalLinkage, S,
                            F->getName () + ".regAllocState", &M);

      // Have: { uint, [Size x { uint, int, uint, int }] } *
      // Cast it to: { uint, [0 x { uint, int, uint, int }] } *
      Constant *CE = ConstantExpr::getCast (ConstantPointerRef::get (GV), PT);
      allstate.push_back (CE);
    }
  }

  unsigned Size = allstate.size ();
  // Final structure type is:
  // { uint, [Size x { uint, [0 x { uint, int, uint, int }] } *] }
  std::vector<const Type *> TV2;
  TV2.push_back (Type::UIntTy);
  ArrayType *AT2 = ArrayType::get (PT, Size);
  TV2.push_back (AT2);
  StructType *ST2 = StructType::get (TV2);
  std::vector<Constant *> CV2;
  CV2.push_back (ConstantUInt::get (Type::UIntTy, Size));
  CV2.push_back (ConstantArray::get (AT2, allstate));
  new GlobalVariable (ST2, true, GlobalValue::ExternalLinkage,
                      ConstantStruct::get (ST2, CV2), "_llvm_regAllocState",
                      &M);
}


/// Allocate registers for the machine code previously generated for F using
/// the graph-coloring algorithm.
///
bool PhyRegAlloc::runOnFunction (Function &F) { 
  if (DEBUG_RA) 
    std::cerr << "\n********* Function "<< F.getName () << " ***********\n"; 
 
  Fn = &F; 
  MF = &MachineFunction::get (Fn); 
  LVI = &getAnalysis<FunctionLiveVarInfo> (); 
  LRI = new LiveRangeInfo (Fn, TM, RegClassList); 
  LoopDepthCalc = &getAnalysis<LoopInfo> (); 
 
  // Create each RegClass for the target machine and add it to the 
  // RegClassList.  This must be done before calling constructLiveRanges().
  for (unsigned rc = 0; rc != NumOfRegClasses; ++rc)   
    RegClassList.push_back (new RegClass (Fn, &TM.getRegInfo (), 
					  MRI.getMachineRegClass (rc))); 
     
  LRI->constructLiveRanges();            // create LR info
  if (DEBUG_RA >= RA_DEBUG_LiveRanges)
    LRI->printLiveRanges();
  
  createIGNodeListsAndIGs();            // create IGNode list and IGs

  buildInterferenceGraphs();            // build IGs in all reg classes
  
  if (DEBUG_RA >= RA_DEBUG_LiveRanges) {
    // print all LRs in all reg classes
    for ( unsigned rc=0; rc < NumOfRegClasses  ; rc++)  
      RegClassList[rc]->printIGNodeList(); 
    
    // print IGs in all register classes
    for ( unsigned rc=0; rc < NumOfRegClasses ; rc++)  
      RegClassList[rc]->printIG();       
  }

  LRI->coalesceLRs();                    // coalesce all live ranges

  if (DEBUG_RA >= RA_DEBUG_LiveRanges) {
    // print all LRs in all reg classes
    for (unsigned rc=0; rc < NumOfRegClasses; rc++)
      RegClassList[rc]->printIGNodeList();
    
    // print IGs in all register classes
    for (unsigned rc=0; rc < NumOfRegClasses; rc++)
      RegClassList[rc]->printIG();
  }

  // mark un-usable suggested color before graph coloring algorithm.
  // When this is done, the graph coloring algo will not reserve
  // suggested color unnecessarily - they can be used by another LR
  markUnusableSugColors(); 

  // color all register classes using the graph coloring algo
  for (unsigned rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[rc]->colorAllRegs();    

  // After graph coloring, if some LRs did not receive a color (i.e, spilled)
  // a position for such spilled LRs
  allocateStackSpace4SpilledLRs();

  // Reset the temp. area on the stack before use by the first instruction.
  // This will also happen after updating each instruction.
  MF->getInfo()->popAllTempValues();

  // color incoming args - if the correct color was not received
  // insert code to copy to the correct register
  colorIncomingArgs();

  // Save register allocation state for this function in a Constant.
  if (SaveRegAllocState) {
    saveState();
    if (DEBUG_RA) // Check our work.
      verifySavedState ();
    if (!SaveStateToModule)
      finishSavingState (const_cast<Module&> (*Fn->getParent ()));
  }

  // Now update the machine code with register names and add any additional
  // code inserted by the register allocator to the instruction stream.
  updateMachineCode(); 

  if (DEBUG_RA) {
    std::cerr << "\n**** Machine Code After Register Allocation:\n\n";
    MF->dump();
  }
 
  // Tear down temporary data structures 
  for (unsigned rc = 0; rc < NumOfRegClasses; ++rc) 
    delete RegClassList[rc]; 
  RegClassList.clear (); 
  AddedInstrMap.clear (); 
  OperandsColoredMap.clear (); 
  ScratchRegsUsed.clear (); 
  AddedInstrAtEntry.clear (); 
  delete LRI;

  if (DEBUG_RA) std::cerr << "\nRegister allocation complete!\n"; 
  return false;     // Function was not modified
} 

} // End llvm namespace
