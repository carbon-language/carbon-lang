//===-- PhyRegAlloc.cpp ---------------------------------------------------===//
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

#include "PhyRegAlloc.h"
#include "RegAllocCommon.h"
#include "RegClass.h"
#include "IGNode.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/FunctionLiveVarInfo.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Module.h"
#include "Support/STLExtras.h"
#include "Support/SetOperations.h"
#include "Support/CommandLine.h"
#include <cmath>

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

static cl::opt<bool>
SaveRegAllocState("save-ra-state", cl::Hidden,
                  cl::desc("write reg. allocator state into module"));

FunctionPass *getRegisterAllocator(TargetMachine &T) {
  return new PhyRegAlloc (T);
}

void PhyRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo> ();
  AU.addRequired<FunctionLiveVarInfo> ();
}



//----------------------------------------------------------------------------
// This method initially creates interference graphs (one in each reg class)
// and IGNodeList (one in each IG). The actual nodes will be pushed later. 
//----------------------------------------------------------------------------
void PhyRegAlloc::createIGNodeListsAndIGs() {
  if (DEBUG_RA >= RA_DEBUG_LiveRanges) std::cerr << "Creating LR lists ...\n";

  // hash map iterator
  LiveRangeMapType::const_iterator HMI = LRI->getLiveRangeMap()->begin();   

  // hash map end
  LiveRangeMapType::const_iterator HMIEnd = LRI->getLiveRangeMap()->end();   

  for (; HMI != HMIEnd ; ++HMI ) {
    if (HMI->first) { 
      LiveRange *L = HMI->second;   // get the LiveRange
      if (!L) { 
        if (DEBUG_RA)
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


//----------------------------------------------------------------------------
// This method will add all interferences at for a given instruction.
// Interference occurs only if the LR of Def (Inst or Arg) is of the same reg 
// class as that of live var. The live var passed to this function is the 
// LVset AFTER the instruction
//----------------------------------------------------------------------------

void PhyRegAlloc::addInterference(const Value *Def, 
				  const ValueSet *LVSet,
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


//----------------------------------------------------------------------------
// For a call instruction, this method sets the CallInterference flag in 
// the LR of each variable live int the Live Variable Set live after the
// call instruction (except the return value of the call instruction - since
// the return value does not interfere with that call itself).
//----------------------------------------------------------------------------

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


//----------------------------------------------------------------------------
// This method will walk thru code and create interferences in the IG of
// each RegClass. Also, this method calculates the spill cost of each
// Live Range (it is done in this method to save another pass over the code).
//----------------------------------------------------------------------------

void PhyRegAlloc::buildInterferenceGraphs()
{
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
      const MachineInstr *MInst = *MII;

      // get the LV set after the instruction
      const ValueSet &LVSetAI = LVI->getLiveVarSetAfterMInst(MInst, BB);
      bool isCallInst = TM.getInstrInfo().isCall(MInst->getOpCode());

      if (isCallInst ) {
	// set the isCallInterference flag of each live range which extends
	// across this call instruction. This information is used by graph
	// coloring algorithm to avoid allocating volatile colors to live ranges
	// that span across calls (since they have to be saved/restored)
	setCallInterferences(MInst, &LVSetAI);
      }

      // iterate over all MI operands to find defs
      for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
             OpE = MInst->end(); OpI != OpE; ++OpI) {
       	if (OpI.isDefOnly() || OpI.isDefAndUse()) // create a new LR since def
	  addInterference(*OpI, &LVSetAI, isCallInst);

	// Calculate the spill cost of each live range
	LiveRange *LR = LRI->getLiveRangeForValue(*OpI);
	if (LR) LR->addSpillCost(BBLoopDepthCost);
      } 

      // if there are multiple defs in this instruction e.g. in SETX
      if (TM.getInstrInfo().isPseudoInstr(MInst->getOpCode()))
      	addInterf4PseudoInstr(MInst);

      // Also add interference for any implicit definitions in a machine
      // instr (currently, only calls have this).
      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      for (unsigned z=0; z < NumOfImpRefs; z++) 
        if (MInst->getImplicitOp(z).opIsDefOnly() ||
	    MInst->getImplicitOp(z).opIsDefAndUse())
	  addInterference( MInst->getImplicitRef(z), &LVSetAI, isCallInst );

    } // for all machine instructions in BB
  } // for all BBs in function

  // add interferences for function arguments. Since there are no explicit 
  // defs in the function for args, we have to add them manually
  addInterferencesForArgs();          

  if (DEBUG_RA >= RA_DEBUG_Interference)
    std::cerr << "Interference graphs calculated!\n";
}


//--------------------------------------------------------------------------
// Pseudo-instructions may be expanded to multiple instructions by the
// assembler. Consequently, all the operands must get distinct registers.
// Therefore, we mark all operands of a pseudo-instruction as interfering
// with one another.
//--------------------------------------------------------------------------

void PhyRegAlloc::addInterf4PseudoInstr(const MachineInstr *MInst) {
  bool setInterf = false;

  // iterate over MI operands to find defs
  for (MachineInstr::const_val_op_iterator It1 = MInst->begin(),
         ItE = MInst->end(); It1 != ItE; ++It1) {
    const LiveRange *LROfOp1 = LRI->getLiveRangeForValue(*It1); 
    assert((LROfOp1 || !It1.isUseOnly())&&"No LR for Def in PSEUDO insruction");

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


//----------------------------------------------------------------------------
// This method adds interferences for incoming arguments to a function.
//----------------------------------------------------------------------------

void PhyRegAlloc::addInterferencesForArgs() {
  // get the InSet of root BB
  const ValueSet &InSet = LVI->getInSetOfBB(&Fn->front());  

  for (Function::const_aiterator AI = Fn->abegin(); AI != Fn->aend(); ++AI) {
    // add interferences between args and LVars at start 
    addInterference(AI, &InSet, false);
    
    if (DEBUG_RA >= RA_DEBUG_Interference)
      std::cerr << " - %% adding interference for  argument " << RAV(AI) << "\n";
  }
}


//----------------------------------------------------------------------------
// This method is called after register allocation is complete to set the
// allocated registers in the machine code. This code will add register numbers
// to MachineOperands that contain a Value. Also it calls target specific
// methods to produce caller saving instructions. At the end, it adds all
// additional instructions produced by the register allocator to the 
// instruction stream. 
//----------------------------------------------------------------------------

//-----------------------------
// Utility functions used below
//-----------------------------
inline void
InsertBefore(MachineInstr* newMI,
             MachineBasicBlock& MBB,
             MachineBasicBlock::iterator& MII)
{
  MII = MBB.insert(MII, newMI);
  ++MII;
}

inline void
InsertAfter(MachineInstr* newMI,
            MachineBasicBlock& MBB,
            MachineBasicBlock::iterator& MII)
{
  ++MII;    // insert before the next instruction
  MII = MBB.insert(MII, newMI);
}

inline void
DeleteInstruction(MachineBasicBlock& MBB,
                  MachineBasicBlock::iterator& MII)
{
  MII = MBB.erase(MII);
}

inline void
SubstituteInPlace(MachineInstr* newMI,
                  MachineBasicBlock& MBB,
                  MachineBasicBlock::iterator MII)
{
  *MII = newMI;
}

inline void
PrependInstructions(std::vector<MachineInstr *> &IBef,
                    MachineBasicBlock& MBB,
                    MachineBasicBlock::iterator& MII,
                    const std::string& msg)
{
  if (!IBef.empty())
    {
      MachineInstr* OrigMI = *MII;
      std::vector<MachineInstr *>::iterator AdIt; 
      for (AdIt = IBef.begin(); AdIt != IBef.end() ; ++AdIt)
        {
          if (DEBUG_RA) {
            if (OrigMI) std::cerr << "For MInst:\n  " << *OrigMI;
            std::cerr << msg << "PREPENDed instr:\n  " << **AdIt << "\n";
          }
          InsertBefore(*AdIt, MBB, MII);
        }
    }
}

inline void
AppendInstructions(std::vector<MachineInstr *> &IAft,
                   MachineBasicBlock& MBB,
                   MachineBasicBlock::iterator& MII,
                   const std::string& msg)
{
  if (!IAft.empty())
    {
      MachineInstr* OrigMI = *MII;
      std::vector<MachineInstr *>::iterator AdIt; 
      for ( AdIt = IAft.begin(); AdIt != IAft.end() ; ++AdIt )
        {
          if (DEBUG_RA) {
            if (OrigMI) std::cerr << "For MInst:\n  " << *OrigMI;
            std::cerr << msg << "APPENDed instr:\n  "  << **AdIt << "\n";
          }
          InsertAfter(*AdIt, MBB, MII);
        }
    }
}

bool PhyRegAlloc::markAllocatedRegs(MachineInstr* MInst)
{
  bool instrNeedsSpills = false;

  // First, set the registers for operands in the machine instruction
  // if a register was successfully allocated.  Do this first because we
  // will need to know which registers are already used by this instr'n.
  for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum)
    {
      MachineOperand& Op = MInst->getOperand(OpNum);
      if (Op.getType() ==  MachineOperand::MO_VirtualRegister || 
          Op.getType() ==  MachineOperand::MO_CCRegister)
        {
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

void PhyRegAlloc::updateInstruction(MachineBasicBlock::iterator& MII,
                                    MachineBasicBlock &MBB)
{
  MachineInstr* MInst = *MII;
  unsigned Opcode = MInst->getOpCode();

  // Reset tmp stack positions so they can be reused for each machine instr.
  MF->getInfo()->popAllTempValues();  

  // Mark the operands for which regs have been allocated.
  bool instrNeedsSpills = markAllocatedRegs(*MII);

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
    for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum)
      {
        MachineOperand& Op = MInst->getOperand(OpNum);
        if (Op.getType() ==  MachineOperand::MO_VirtualRegister || 
            Op.getType() ==  MachineOperand::MO_CCRegister)
          {
            const Value* Val = Op.getVRegValue();
            if (const LiveRange *LR = LRI->getLiveRangeForValue(Val))
              if (LR->isMarkedForSpill())
                insertCode4SpilledLR(LR, MII, MBB, OpNum);
          }
      } // for each operand
}

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
      if (! TM.getInstrInfo().isDummyPhiInstr((*MII)->getOpCode()))
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
    for (MachineBasicBlock::iterator MII = MBB.begin();
         MII != MBB.end(); ++MII)
      if (unsigned delaySlots =
          TM.getInstrInfo().getNumDelaySlots((*MII)->getOpCode()))
        { 
          MachineInstr *MInst = *MII, *DelaySlotMI = *(MII+1);
          
          // Check the 2 conditions above:
          // (1) Does a branch need instructions added after it?
          // (2) O/w does delay slot instr. need instrns before or after?
          bool isBranch = (TM.getInstrInfo().isBranch(MInst->getOpCode()) ||
                           TM.getInstrInfo().isReturn(MInst->getOpCode()));
          bool cond1 = (isBranch &&
                        AddedInstrMap.count(MInst) &&
                        AddedInstrMap[MInst].InstrnsAfter.size() > 0);
          bool cond2 = (AddedInstrMap.count(DelaySlotMI) &&
                        (AddedInstrMap[DelaySlotMI].InstrnsBefore.size() > 0 ||
                         AddedInstrMap[DelaySlotMI].InstrnsAfter.size()  > 0));

          if (cond1 || cond2)
            {
              assert((MInst->getOpCodeFlags() & AnnulFlag) == 0 &&
                     "FIXME: Moving an annulled delay slot instruction!"); 
              assert(delaySlots==1 &&
                     "InsertBefore does not yet handle >1 delay slots!");
              InsertBefore(DelaySlotMI, MBB, MII); // MII pts back to branch

              // In case (1), delete it and don't replace with anything!
              // Otherwise (i.e., case (2) only) replace it with a NOP.
              if (cond1) {
                DeleteInstruction(MBB, ++MII); // MII now points to next inst.
                --MII;                         // reset MII for ++MII of loop
              }
              else
                SubstituteInPlace(BuildMI(TM.getInstrInfo().getNOPOpCode(),1),
                                  MBB, MII+1);        // replace with NOP

              if (DEBUG_RA) {
                std::cerr << "\nRegAlloc: Moved instr. with added code: "
                     << *DelaySlotMI
                     << "           out of delay slots of instr: " << *MInst;
              }
            }
          else
            // For non-branch instr with delay slots (probably a call), move
            // InstrAfter to the instr. in the last delay slot.
            move2DelayedInstr(*MII, *(MII+delaySlots));
        }

    // Finally iterate over all instructions in BB and insert before/after
    for (MachineBasicBlock::iterator MII=MBB.begin(); MII != MBB.end(); ++MII) {
      MachineInstr *MInst = *MII; 

      // do not process Phis
      if (TM.getInstrInfo().isDummyPhiInstr(MInst->getOpCode()))
	continue;

      // if there are any added instructions...
      if (AddedInstrMap.count(MInst)) {
        AddedInstrns &CallAI = AddedInstrMap[MInst];

#ifndef NDEBUG
        bool isBranch = (TM.getInstrInfo().isBranch(MInst->getOpCode()) ||
                         TM.getInstrInfo().isReturn(MInst->getOpCode()));
        assert((!isBranch ||
                AddedInstrMap[MInst].InstrnsAfter.size() <=
                TM.getInstrInfo().getNumDelaySlots(MInst->getOpCode())) &&
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


//----------------------------------------------------------------------------
// This method inserts spill code for AN operand whose LR was spilled.
// This method may be called several times for a single machine instruction
// if it contains many spilled operands. Each time it is called, it finds
// a register which is not live at that instruction and also which is not
// used by other spilled operands of the same instruction. Then it uses
// this register temporarily to accommodate the spilled value.
//----------------------------------------------------------------------------

void PhyRegAlloc::insertCode4SpilledLR(const LiveRange *LR, 
                                       MachineBasicBlock::iterator& MII,
                                       MachineBasicBlock &MBB,
				       const unsigned OpNum) {
  MachineInstr *MInst = *MII;
  const BasicBlock *BB = MBB.getBasicBlock();

  assert((! TM.getInstrInfo().isCall(MInst->getOpCode()) || OpNum == 0) &&
         "Outgoing arg of a call must be handled elsewhere (func arg ok)");
  assert(! TM.getInstrInfo().isReturn(MInst->getOpCode()) &&
	 "Return value of a ret must be handled elsewhere");

  MachineOperand& Op = MInst->getOperand(OpNum);
  bool isDef =  Op.opIsDefOnly();
  bool isDefAndUse = Op.opIsDefAndUse();
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
    MachineInstr *PredMI = *(MII-1);
    if (unsigned DS = TM.getInstrInfo().getNumDelaySlots(PredMI->getOpCode()))
      assert(set_difference(LVI->getLiveVarSetBeforeMInst(PredMI), LVSetBef)
             .empty() && "Live-var set before branch should be included in "
             "live-var set of each delay slot instruction!");
  }
#endif

  MF->getInfo()->pushTempValue(MRI.getSpilledRegSize(RegType) );
  
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
  if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType))
    {
      scratchReg = getUsableUniRegAtMI(scratchRegType, &LVSetBef,
                                       MInst, MIBef, MIAft);
      assert(scratchReg != MRI.getInvalidRegNum());
    }
  
  if (!isDef || isDefAndUse) {
    // for a USE, we have to load the value of LR from stack to a TmpReg
    // and use the TmpReg as one operand of instruction
    
    // actual loading instruction(s)
    MRI.cpMem2RegMI(AdIMid, MRI.getFramePointer(), SpillOff, TmpRegU,
                    RegType, scratchReg);
    
    // the actual load should be after the instructions to free up TmpRegU
    MIBef.insert(MIBef.end(), AdIMid.begin(), AdIMid.end());
    AdIMid.clear();
  }
  
  if (isDef || isDefAndUse) {   // if this is a Def
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


//----------------------------------------------------------------------------
// This method inserts caller saving/restoring instructions before/after
// a call machine instruction. The caller saving/restoring instructions are
// inserted like:
//    ** caller saving instructions
//    other instructions inserted for the call by ColorCallArg
//    CALL instruction
//    other instructions inserted for the call ColorCallArg
//    ** caller restoring instructions
//----------------------------------------------------------------------------

void
PhyRegAlloc::insertCallerSavingCode(std::vector<MachineInstr*> &instrnsBefore,
                                    std::vector<MachineInstr*> &instrnsAfter,
                                    MachineInstr *CallMI, 
                                    const BasicBlock *BB)
{
  assert(TM.getInstrInfo().isCall(CallMI->getOpCode()));
  
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
    if( LR )   {  
      if(! LR->isMarkedForSpill()) {
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


//----------------------------------------------------------------------------
// We can use the following method to get a temporary register to be used
// BEFORE any given machine instruction. If there is a register available,
// this method will simply return that register and set MIBef = MIAft = NULL.
// Otherwise, it will return a register and MIAft and MIBef will contain
// two instructions used to free up this returned register.
// Returned register number is the UNIFIED register number
//----------------------------------------------------------------------------

int PhyRegAlloc::getUsableUniRegAtMI(const int RegType,
                                     const ValueSet *LVSetBef,
                                     MachineInstr *MInst, 
                                     std::vector<MachineInstr*>& MIBef,
                                     std::vector<MachineInstr*>& MIAft) {
  RegClass* RC = getRegClassByID(MRI.getRegClassIDOfRegType(RegType));
  
  int RegU =  getUnusedUniRegAtMI(RC, RegType, MInst, LVSetBef);
  
  if (RegU == -1) {
    // we couldn't find an unused register. Generate code to free up a reg by
    // saving it on stack and restoring after the instruction
    
    int TmpOff = MF->getInfo()->pushTempValue(MRI.getSpilledRegSize(RegType));
    
    RegU = getUniRegNotUsedByThisInst(RC, RegType, MInst);
    
    // Check if we need a scratch register to copy this register to memory.
    int scratchRegType = -1;
    if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType))
      {
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
      }
    else
      { // the register can be copied directly to/from memory so do it.
        MRI.cpReg2MemMI(MIBef, RegU, MRI.getFramePointer(), TmpOff, RegType);
        MRI.cpMem2RegMI(MIAft, MRI.getFramePointer(), TmpOff, RegU, RegType);
      }
  }
  
  return RegU;
}


//----------------------------------------------------------------------------
// This method is called to get a new unused register that can be used
// to accommodate a temporary value.  This method may be called several times
// for a single machine instruction.  Each time it is called, it finds a
// register which is not live at that instruction and also which is not used
// by other spilled operands of the same instruction.  Return register number
// is relative to the register class, NOT the unified number.
//----------------------------------------------------------------------------

int PhyRegAlloc::getUnusedUniRegAtMI(RegClass *RC, 
                                     const int RegType,
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


//----------------------------------------------------------------------------
// Get any other register in a register class, other than what is used
// by operands of a machine instruction. Returns the unified reg number.
//----------------------------------------------------------------------------

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


//----------------------------------------------------------------------------
// This method modifies the IsColorUsedArr of the register class passed to it.
// It sets the bits corresponding to the registers used by this machine
// instructions. Both explicit and implicit operands are set.
//----------------------------------------------------------------------------

static void markRegisterUsed(int RegNo, RegClass *RC, int RegType,
                             const TargetRegInfo &TRI) {
  unsigned classId = 0;
  int classRegNum = TRI.getClassRegNum(RegNo, classId);
  if (RC->getID() == classId)
    RC->markColorsUsed(classRegNum, RegType, RegType);
}

void PhyRegAlloc::setRelRegsUsedByThisInst(RegClass *RC, int RegType,
                                           const MachineInstr *MI)
{
  assert(OperandsColoredMap[MI] == true &&
         "Illegal to call setRelRegsUsedByThisInst() until colored operands "
         "are marked for an instruction.");

  // Add the registers already marked as used by the instruction.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
    if (MI->getOperand(i).hasAllocatedReg())
      markRegisterUsed(MI->getOperand(i).getAllocatedRegNum(), RC, RegType,MRI);

  for (unsigned i = 0, e = MI->getNumImplicitRefs(); i != e; ++i)
    if (MI->getImplicitOp(i).hasAllocatedReg())
      markRegisterUsed(MI->getImplicitOp(i).getAllocatedRegNum(), RC,
                       RegType,MRI);

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


//----------------------------------------------------------------------------
// If there are delay slots for an instruction, the instructions
// added after it must really go after the delayed instruction(s).
// So, we move the InstrAfter of that instruction to the 
// corresponding delayed instruction using the following method.
//----------------------------------------------------------------------------

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


//----------------------------------------------------------------------------
// This method determines whether the suggested color of each live range
// is really usable, and then calls its setSuggestedColorUsable() method to
// record the answer. A suggested color is NOT usable when the suggested color
// is volatile AND when there are call interferences.
//----------------------------------------------------------------------------

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


//----------------------------------------------------------------------------
// The following method will set the stack offsets of the live ranges that
// are decided to be spilled. This must be called just after coloring the
// LRs using the graph coloring algo. For each live range that is spilled,
// this method allocate a new spill position on the stack.
//----------------------------------------------------------------------------

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


namespace {
  /// AllocInfo - Structure representing one instruction's
  /// operand's-worth of register allocation state. We create tables
  /// made out of these data structures to generate mapping information
  /// for this register allocator. (FIXME: This might move to a header
  /// file at some point.)
  ///
  struct AllocInfo {
    unsigned Instruction;
    unsigned Operand;
    unsigned AllocState;
    int Placement;
    AllocInfo (unsigned Instruction_, unsigned Operand_,
               unsigned AllocState_, int Placement_) :
      Instruction (Instruction_), Operand (Operand_),
      AllocState (AllocState_), Placement (Placement_) { }
    /// getConstantType - Return a StructType representing an AllocInfo
    /// object.
    ///
    static StructType *getConstantType () {
      std::vector<const Type *> TV;
      TV.push_back (Type::UIntTy);
      TV.push_back (Type::UIntTy);
      TV.push_back (Type::UIntTy);
      TV.push_back (Type::IntTy);
      return StructType::get (TV);
    }
    /// toConstant - Convert this AllocInfo into an LLVM Constant of type
    /// getConstantType(), and return the Constant.
    ///
    Constant *toConstant () const {
      StructType *ST = getConstantType ();
      std::vector<Constant *> CV;
      CV.push_back (ConstantUInt::get (Type::UIntTy, Instruction));
      CV.push_back (ConstantUInt::get (Type::UIntTy, Operand));
      CV.push_back (ConstantUInt::get (Type::UIntTy, AllocState));
      CV.push_back (ConstantSInt::get (Type::IntTy, Placement));
      return ConstantStruct::get (ST, CV);
    }
  };
}

void PhyRegAlloc::saveState ()
{
  std::vector<Constant *> state;
  unsigned Insn = 0;
  LiveRangeMapType::const_iterator HMIEnd = LRI->getLiveRangeMap ()->end ();   
  for (const_inst_iterator II=inst_begin (Fn), IE=inst_end (Fn); II != IE; ++II)
    for (unsigned i = 0; i < (*II)->getNumOperands (); ++i) {
      const Value *V = (*II)->getOperand (i);
      // Don't worry about it unless it's something whose reg. we'll need.
      if (!isa<Argument> (V) && !isa<Instruction> (V))
        continue;
      LiveRangeMapType::const_iterator HMI = LRI->getLiveRangeMap ()->find (V);
      static const unsigned NotAllocated = 0, Allocated = 1, Spilled = 2;
      unsigned AllocState = NotAllocated;
      int Placement = -1;
      if ((HMI != HMIEnd) && HMI->second) {
        LiveRange *L = HMI->second;
        assert ((L->hasColor () || L->isMarkedForSpill ())
                && "Live range exists but not colored or spilled");
        if (L->hasColor()) {
          AllocState = Allocated;
          Placement = MRI.getUnifiedRegNum (L->getRegClassID (),
                                            L->getColor ());
        } else if (L->isMarkedForSpill ()) {
          AllocState = Spilled;
          assert (L->hasSpillOffset ()
                  && "Live range marked for spill but has no spill offset");
          Placement = L->getSpillOffFromFP ();
        }
      }
      state.push_back (AllocInfo (Insn, i, AllocState,
                                  Placement).toConstant ());
    }
  // Convert state into an LLVM ConstantArray, and put it in a
  // ConstantStruct (named S) along with its size.
  unsigned Size = state.size ();
  ArrayType *AT = ArrayType::get (AllocInfo::getConstantType (), Size);
  std::vector<const Type *> TV;
  TV.push_back (Type::UIntTy);
  TV.push_back (AT);
  StructType *ST = StructType::get (TV);
  std::vector<Constant *> CV;
  CV.push_back (ConstantUInt::get (Type::UIntTy, Size));
  CV.push_back (ConstantArray::get (AT, state));
  Constant *S = ConstantStruct::get (ST, CV);
  // Save S in the map containing register allocator state for this module.
  FnAllocState[Fn] = S;
}


bool PhyRegAlloc::doFinalization (Module &M) { 
  if (!SaveRegAllocState)
    return false; // Nothing to do here, unless we're saving state.

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
    if (FnAllocState.find (F) == FnAllocState.end ()) {
      allstate.push_back (ConstantPointerNull::get (PT));
    } else {
      GlobalVariable *GV =
        new GlobalVariable (FnAllocState[F]->getType (), true,
                            GlobalValue::InternalLinkage, FnAllocState[F],
                            F->getName () + ".regAllocState", &M);
      // Have: { uint, [Size x { uint, uint, uint, int }] } *
      // Cast it to: { uint, [0 x { uint, uint, uint, int }] } *
      Constant *CE = ConstantExpr::getCast (ConstantPointerRef::get (GV), PT);
      allstate.push_back (CE);
    }
  }

  unsigned Size = allstate.size ();
  // Final structure type is:
  // { uint, [Size x { uint, [0 x { uint, uint, uint, int }] } *] }
  std::vector<const Type *> TV2;
  TV2.push_back (Type::UIntTy);
  ArrayType *AT2 = ArrayType::get (PT, Size);
  TV2.push_back (AT2);
  StructType *ST2 = StructType::get (TV2);
  std::vector<Constant *> CV2;
  CV2.push_back (ConstantUInt::get (Type::UIntTy, Size));
  CV2.push_back (ConstantArray::get (AT2, allstate));
  new GlobalVariable (ST2, true, GlobalValue::InternalLinkage,
                      ConstantStruct::get (ST2, CV2), "_llvm_regAllocState",
                      &M);
  return false; // No error.
}


//----------------------------------------------------------------------------
// The entry point to Register Allocation
//----------------------------------------------------------------------------

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
  if (SaveRegAllocState)
    saveState();

  // Now update the machine code with register names and add any 
  // additional code inserted by the register allocator to the instruction
  // stream
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
