//===-- PhyRegAlloc.cpp ---------------------------------------------------===//
// 
//  Register allocation for LLVM.
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterAllocation.h"
#include "llvm/CodeGen/PhyRegAlloc.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/Analysis/LiveVar/FunctionLiveVarInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/iOther.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "Support/CommandLine.h"
#include "Support/STLExtras.h"
#include <math.h>
using std::cerr;
using std::vector;

RegAllocDebugLevel_t DEBUG_RA;
static cl::opt<RegAllocDebugLevel_t, true>
DRA_opt("dregalloc", cl::Hidden, cl::location(DEBUG_RA),
        cl::desc("enable register allocation debugging information"),
        cl::values(
  clEnumValN(RA_DEBUG_None   , "n", "disable debug output"),
  clEnumValN(RA_DEBUG_Normal , "y", "enable debug output"),
  clEnumValN(RA_DEBUG_Verbose, "v", "enable extra debug output"),
                   0));


//----------------------------------------------------------------------------
// RegisterAllocation pass front end...
//----------------------------------------------------------------------------
namespace {
  class RegisterAllocator : public FunctionPass {
    TargetMachine &Target;
  public:
    inline RegisterAllocator(TargetMachine &T) : Target(T) {}

    const char *getPassName() const { return "Register Allocation"; }
    
    bool runOnFunction(Function &F) {
      if (DEBUG_RA)
        cerr << "\n********* Function "<< F.getName() << " ***********\n";
      
      PhyRegAlloc PRA(&F, Target, &getAnalysis<FunctionLiveVarInfo>(),
                      &getAnalysis<LoopInfo>());
      PRA.allocateRegisters();
      
      if (DEBUG_RA) cerr << "\nRegister allocation complete!\n";
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addRequired<FunctionLiveVarInfo>();
    }
  };
}

Pass *getRegisterAllocator(TargetMachine &T) {
  return new RegisterAllocator(T);
}

//----------------------------------------------------------------------------
// Constructor: Init local composite objects and create register classes.
//----------------------------------------------------------------------------
PhyRegAlloc::PhyRegAlloc(Function *F, const TargetMachine& tm, 
			 FunctionLiveVarInfo *Lvi, LoopInfo *LDC) 
                       :  TM(tm), Meth(F),
                          mcInfo(MachineCodeForMethod::get(F)),
                          LVI(Lvi), LRI(F, tm, RegClassList), 
			  MRI(tm.getRegInfo()),
                          NumOfRegClasses(MRI.getNumOfRegClasses()),
			  LoopDepthCalc(LDC) {

  // create each RegisterClass and put in RegClassList
  //
  for (unsigned rc=0; rc < NumOfRegClasses; rc++)  
    RegClassList.push_back(new RegClass(F, MRI.getMachineRegClass(rc),
                                        &ResColList));
}


//----------------------------------------------------------------------------
// Destructor: Deletes register classes
//----------------------------------------------------------------------------
PhyRegAlloc::~PhyRegAlloc() { 
  for ( unsigned rc=0; rc < NumOfRegClasses; rc++)
    delete RegClassList[rc];

  AddedInstrMap.clear();
} 

//----------------------------------------------------------------------------
// This method initally creates interference graphs (one in each reg class)
// and IGNodeList (one in each IG). The actual nodes will be pushed later. 
//----------------------------------------------------------------------------
void PhyRegAlloc::createIGNodeListsAndIGs() {
  if (DEBUG_RA) cerr << "Creating LR lists ...\n";

  // hash map iterator
  LiveRangeMapType::const_iterator HMI = LRI.getLiveRangeMap()->begin();   

  // hash map end
  LiveRangeMapType::const_iterator HMIEnd = LRI.getLiveRangeMap()->end();   

  for (; HMI != HMIEnd ; ++HMI ) {
    if (HMI->first) { 
      LiveRange *L = HMI->second;   // get the LiveRange
      if (!L) { 
        if (DEBUG_RA) {
          cerr << "\n*?!?Warning: Null liver range found for: "
               << RAV(HMI->first) << "\n";
        }
        continue;
      }
                                        // if the Value * is not null, and LR  
                                        // is not yet written to the IGNodeList
      if (!(L->getUserIGNode())  ) {  
        RegClass *const RC =           // RegClass of first value in the LR
          RegClassList[ L->getRegClass()->getID() ];
        
        RC->addLRToIG(L);              // add this LR to an IG
      }
    }
  }
    
  // init RegClassList
  for ( unsigned rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[rc]->createInterferenceGraph();

  if (DEBUG_RA)
    cerr << "LRLists Created!\n";
}




//----------------------------------------------------------------------------
// This method will add all interferences at for a given instruction.
// Interence occurs only if the LR of Def (Inst or Arg) is of the same reg 
// class as that of live var. The live var passed to this function is the 
// LVset AFTER the instruction
//----------------------------------------------------------------------------
void PhyRegAlloc::addInterference(const Value *Def, 
				  const ValueSet *LVSet,
				  bool isCallInst) {

  ValueSet::const_iterator LIt = LVSet->begin();

  // get the live range of instruction
  //
  const LiveRange *const LROfDef = LRI.getLiveRangeForValue( Def );   

  IGNode *const IGNodeOfDef = LROfDef->getUserIGNode();
  assert( IGNodeOfDef );

  RegClass *const RCOfDef = LROfDef->getRegClass(); 

  // for each live var in live variable set
  //
  for ( ; LIt != LVSet->end(); ++LIt) {

    if (DEBUG_RA >= RA_DEBUG_Verbose)
      cerr << "< Def=" << RAV(Def) << ", Lvar=" << RAV(*LIt) << "> ";

    //  get the live range corresponding to live var
    //
    LiveRange *LROfVar = LRI.getLiveRangeForValue(*LIt);

    // LROfVar can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    //
    if (LROfVar) {  
      if (LROfDef == LROfVar)            // do not set interf for same LR
	continue;

      // if 2 reg classes are the same set interference
      //
      if (RCOfDef == LROfVar->getRegClass()) {
	RCOfDef->setInterference( LROfDef, LROfVar);  
      } else if (DEBUG_RA >= RA_DEBUG_Verbose)  { 
        // we will not have LRs for values not explicitly allocated in the
        // instruction stream (e.g., constants)
        cerr << " warning: no live range for " << RAV(*LIt) << "\n";
      }
    }
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

  if (DEBUG_RA)
    cerr << "\n For call inst: " << *MInst;

  ValueSet::const_iterator LIt = LVSetAft->begin();

  // for each live var in live variable set after machine inst
  //
  for ( ; LIt != LVSetAft->end(); ++LIt) {

    //  get the live range corresponding to live var
    //
    LiveRange *const LR = LRI.getLiveRangeForValue(*LIt ); 

    if (LR && DEBUG_RA) {
      cerr << "\n\tLR Aft Call: ";
      printSet(*LR);
    }
   
    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    //
    if (LR )   {  
      LR->setCallInterference();
      if (DEBUG_RA) {
	cerr << "\n  ++Added call interf for LR: " ;
	printSet(*LR);
      }
    }

  }

  // Now find the LR of the return value of the call
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but it does not interfere with call
  // (i.e., we can allocate a volatile register to the return value)
  //
  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(MInst);
  
  if (const Value *RetVal = argDesc->getReturnValue()) {
    LiveRange *RetValLR = LRI.getLiveRangeForValue( RetVal );
    assert( RetValLR && "No LR for RetValue of call");
    RetValLR->clearCallInterference();
  }

  // If the CALL is an indirect call, find the LR of the function pointer.
  // That has a call interference because it conflicts with outgoing args.
  if (const Value *AddrVal = argDesc->getIndirectFuncPtr()) {
    LiveRange *AddrValLR = LRI.getLiveRangeForValue( AddrVal );
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

  if (DEBUG_RA) cerr << "Creating interference graphs ...\n";

  unsigned BBLoopDepthCost;
  for (Function::const_iterator BBI = Meth->begin(), BBE = Meth->end();
       BBI != BBE; ++BBI) {

    // find the 10^(loop_depth) of this BB 
    //
    BBLoopDepthCost = (unsigned)pow(10.0, LoopDepthCalc->getLoopDepth(BBI));

    // get the iterator for machine instructions
    //
    const MachineCodeForBasicBlock& MIVec = MachineCodeForBasicBlock::get(BBI);
    MachineCodeForBasicBlock::const_iterator MII = MIVec.begin();

    // iterate over all the machine instructions in BB
    //
    for ( ; MII != MIVec.end(); ++MII) {  

      const MachineInstr *MInst = *MII; 

      // get the LV set after the instruction
      //
      const ValueSet &LVSetAI = LVI->getLiveVarSetAfterMInst(MInst, BBI);
    
      const bool isCallInst = TM.getInstrInfo().isCall(MInst->getOpCode());

      if (isCallInst ) {
	// set the isCallInterference flag of each live range wich extends
	// accross this call instruction. This information is used by graph
	// coloring algo to avoid allocating volatile colors to live ranges
	// that span across calls (since they have to be saved/restored)
	//
	setCallInterferences(MInst, &LVSetAI);
      }


      // iterate over all MI operands to find defs
      //
      for (MachineInstr::const_val_op_iterator OpI = MInst->begin(),
             OpE = MInst->end(); OpI != OpE; ++OpI) {
       	if (OpI.isDef())    // create a new LR iff this operand is a def
	  addInterference(*OpI, &LVSetAI, isCallInst);

	// Calculate the spill cost of each live range
	//
	LiveRange *LR = LRI.getLiveRangeForValue(*OpI);
	if (LR) LR->addSpillCost(BBLoopDepthCost);
      } 


      // if there are multiple defs in this instruction e.g. in SETX
      //   
      if (TM.getInstrInfo().isPseudoInstr(MInst->getOpCode()))
      	addInterf4PseudoInstr(MInst);


      // Also add interference for any implicit definitions in a machine
      // instr (currently, only calls have this).
      //
      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      if ( NumOfImpRefs > 0 ) {
	for (unsigned z=0; z < NumOfImpRefs; z++) 
	  if (MInst->implicitRefIsDefined(z) )
	    addInterference( MInst->getImplicitRef(z), &LVSetAI, isCallInst );
      }


    } // for all machine instructions in BB
  } // for all BBs in function


  // add interferences for function arguments. Since there are no explict 
  // defs in the function for args, we have to add them manually
  //  
  addInterferencesForArgs();          

  if (DEBUG_RA)
    cerr << "Interference graphs calculted!\n";

}



//--------------------------------------------------------------------------
// Pseudo instructions will be exapnded to multiple instructions by the
// assembler. Consequently, all the opernds must get distinct registers.
// Therefore, we mark all operands of a pseudo instruction as they interfere
// with one another.
//--------------------------------------------------------------------------
void PhyRegAlloc::addInterf4PseudoInstr(const MachineInstr *MInst) {

  bool setInterf = false;

  // iterate over  MI operands to find defs
  //
  for (MachineInstr::const_val_op_iterator It1 = MInst->begin(),
         ItE = MInst->end(); It1 != ItE; ++It1) {
    const LiveRange *LROfOp1 = LRI.getLiveRangeForValue(*It1); 
    assert((LROfOp1 || !It1.isDef()) && "No LR for Def in PSEUDO insruction");

    MachineInstr::const_val_op_iterator It2 = It1;
    for (++It2; It2 != ItE; ++It2) {
      const LiveRange *LROfOp2 = LRI.getLiveRangeForValue(*It2); 

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
    cerr << "\nInterf not set for any operand in pseudo instr:\n";
    cerr << *MInst;
    assert(0 && "Interf not set for pseudo instr with > 2 operands" );
  }
} 



//----------------------------------------------------------------------------
// This method will add interferences for incoming arguments to a function.
//----------------------------------------------------------------------------
void PhyRegAlloc::addInterferencesForArgs() {
  // get the InSet of root BB
  const ValueSet &InSet = LVI->getInSetOfBB(&Meth->front());  

  for (Function::const_aiterator AI = Meth->abegin(); AI != Meth->aend(); ++AI) {
    // add interferences between args and LVars at start 
    addInterference(AI, &InSet, false);
    
    if (DEBUG_RA >= RA_DEBUG_Verbose)
      cerr << " - %% adding interference for  argument " << RAV(AI) << "\n";
  }
}


//----------------------------------------------------------------------------
// This method is called after register allocation is complete to set the
// allocated reisters in the machine code. This code will add register numbers
// to MachineOperands that contain a Value. Also it calls target specific
// methods to produce caller saving instructions. At the end, it adds all
// additional instructions produced by the register allocator to the 
// instruction stream. 
//----------------------------------------------------------------------------

//-----------------------------
// Utility functions used below
//-----------------------------
inline void
PrependInstructions(vector<MachineInstr *> &IBef,
                    MachineCodeForBasicBlock& MIVec,
                    MachineCodeForBasicBlock::iterator& MII,
                    const std::string& msg)
{
  if (!IBef.empty())
    {
      MachineInstr* OrigMI = *MII;
      std::vector<MachineInstr *>::iterator AdIt; 
      for (AdIt = IBef.begin(); AdIt != IBef.end() ; ++AdIt)
        {
          if (DEBUG_RA) {
            if (OrigMI) cerr << "For MInst: " << *OrigMI;
            cerr << msg << " PREPENDed instr: " << **AdIt << "\n";
          }
          MII = MIVec.insert(MII, *AdIt);
          ++MII;
        }
    }
}

inline void
AppendInstructions(std::vector<MachineInstr *> &IAft,
                   MachineCodeForBasicBlock& MIVec,
                   MachineCodeForBasicBlock::iterator& MII,
                   const std::string& msg)
{
  if (!IAft.empty())
    {
      MachineInstr* OrigMI = *MII;
      std::vector<MachineInstr *>::iterator AdIt; 
      for ( AdIt = IAft.begin(); AdIt != IAft.end() ; ++AdIt )
        {
          if (DEBUG_RA) {
            if (OrigMI) cerr << "For MInst: " << *OrigMI;
            cerr << msg << " APPENDed instr: "  << **AdIt << "\n";
          }
          ++MII;    // insert before the next instruction
          MII = MIVec.insert(MII, *AdIt);
        }
    }
}


void PhyRegAlloc::updateMachineCode()
{
  MachineCodeForBasicBlock& MIVec = MachineCodeForBasicBlock::get(&Meth->getEntryNode());
    
  // Insert any instructions needed at method entry
  MachineCodeForBasicBlock::iterator MII = MIVec.begin();
  PrependInstructions(AddedInstrAtEntry.InstrnsBefore, MIVec, MII,
                      "At function entry: \n");
  assert(AddedInstrAtEntry.InstrnsAfter.empty() &&
         "InstrsAfter should be unnecessary since we are just inserting at "
         "the function entry point here.");
  
  for (Function::const_iterator BBI = Meth->begin(), BBE = Meth->end();
       BBI != BBE; ++BBI) {
    
    // iterate over all the machine instructions in BB
    MachineCodeForBasicBlock &MIVec = MachineCodeForBasicBlock::get(BBI);
    for (MachineCodeForBasicBlock::iterator MII = MIVec.begin();
        MII != MIVec.end(); ++MII) {  
      
      MachineInstr *MInst = *MII; 
      
      unsigned Opcode =  MInst->getOpCode();
    
      // do not process Phis
      if (TM.getInstrInfo().isDummyPhiInstr(Opcode))
	continue;

      // Reset tmp stack positions so they can be reused for each machine instr.
      mcInfo.popAllTempValues(TM);  
	
      // Now insert speical instructions (if necessary) for call/return
      // instructions. 
      //
      if (TM.getInstrInfo().isCall(Opcode) ||
	  TM.getInstrInfo().isReturn(Opcode)) {

	AddedInstrns &AI = AddedInstrMap[MInst];
	
	if (TM.getInstrInfo().isCall(Opcode))
	  MRI.colorCallArgs(MInst, LRI, &AI, *this, BBI);
	else if (TM.getInstrInfo().isReturn(Opcode))
	  MRI.colorRetValue(MInst, LRI, &AI);
      }
      
      // Set the registers for operands in the machine instruction
      // if a register was successfully allocated.  If not, insert
      // code to spill the register value.
      // 
      for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum)
        {
          MachineOperand& Op = MInst->getOperand(OpNum);
          if (Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
              Op.getOperandType() ==  MachineOperand::MO_CCRegister)
            {
              const Value *const Val =  Op.getVRegValue();
          
              LiveRange *const LR = LRI.getLiveRangeForValue(Val);
              if (!LR)              // consts or labels will have no live range
                {
                  // if register is not allocated, mark register as invalid
                  if (Op.getAllocatedRegNum() == -1)
                    MInst->SetRegForOperand(OpNum, MRI.getInvalidRegNum()); 
                  continue;
                }
          
              if (LR->hasColor() )
                MInst->SetRegForOperand(OpNum,
                                MRI.getUnifiedRegNum(LR->getRegClass()->getID(),
                                                     LR->getColor()));
              else
                // LR did NOT receive a color (register). Insert spill code.
                insertCode4SpilledLR(LR, MInst, BBI, OpNum );
            }
        } // for each operand
      
      
      // Now add instructions that the register allocator inserts before/after 
      // this machine instructions (done only for calls/rets/incoming args)
      // We do this here, to ensure that spill for an instruction is inserted
      // closest as possible to an instruction (see above insertCode4Spill...)
      // 
      // If there are instructions to be added, *before* this machine
      // instruction, add them now.
      //      
      if (AddedInstrMap.count(MInst)) {
        PrependInstructions(AddedInstrMap[MInst].InstrnsBefore, MIVec, MII,"");
      }
      
      // If there are instructions to be added *after* this machine
      // instruction, add them now
      //
      if (!AddedInstrMap[MInst].InstrnsAfter.empty()) {

	// if there are delay slots for this instruction, the instructions
	// added after it must really go after the delayed instruction(s)
	// So, we move the InstrAfter of the current instruction to the 
	// corresponding delayed instruction
	
	unsigned delay;
	if ((delay=TM.getInstrInfo().getNumDelaySlots(MInst->getOpCode())) >0){ 
	  move2DelayedInstr(MInst,  *(MII+delay) );
	}
	else {
	  // Here we can add the "instructions after" to the current
	  // instruction since there are no delay slots for this instruction
	  AppendInstructions(AddedInstrMap[MInst].InstrnsAfter, MIVec, MII,"");
	}  // if not delay
      }
      
    } // for each machine instruction
  }
}



//----------------------------------------------------------------------------
// This method inserts spill code for AN operand whose LR was spilled.
// This method may be called several times for a single machine instruction
// if it contains many spilled operands. Each time it is called, it finds
// a register which is not live at that instruction and also which is not
// used by other spilled operands of the same instruction. Then it uses
// this register temporarily to accomodate the spilled value.
//----------------------------------------------------------------------------
void PhyRegAlloc::insertCode4SpilledLR(const LiveRange *LR, 
				       MachineInstr *MInst,
				       const BasicBlock *BB,
				       const unsigned OpNum) {

  assert(! TM.getInstrInfo().isCall(MInst->getOpCode()) &&
	 (! TM.getInstrInfo().isReturn(MInst->getOpCode())) &&
	 "Arg of a call/ret must be handled elsewhere");

  MachineOperand& Op = MInst->getOperand(OpNum);
  bool isDef =  MInst->operandIsDefined(OpNum);
  bool isDefAndUse =  MInst->operandIsDefinedAndUsed(OpNum);
  unsigned RegType = MRI.getRegType( LR );
  int SpillOff = LR->getSpillOffFromFP();
  RegClass *RC = LR->getRegClass();
  const ValueSet &LVSetBef = LVI->getLiveVarSetBeforeMInst(MInst, BB);

  mcInfo.pushTempValue(TM, MRI.getSpilledRegSize(RegType) );
  
  vector<MachineInstr*> MIBef, MIAft;
  vector<MachineInstr*> AdIMid;
  
  // Choose a register to hold the spilled value.  This may insert code
  // before and after MInst to free up the value.  If so, this code should
  // be first and last in the spill sequence before/after MInst.
  int TmpRegU = getUsableUniRegAtMI(RegType, &LVSetBef, MInst, MIBef, MIAft);
  
  // Set the operand first so that it this register does not get used
  // as a scratch register for later calls to getUsableUniRegAtMI below
  MInst->SetRegForOperand(OpNum, TmpRegU);
  
  // get the added instructions for this instruction
  AddedInstrns &AI = AddedInstrMap[MInst];

  // We may need a scratch register to copy the spilled value to/from memory.
  // This may itself have to insert code to free up a scratch register.  
  // Any such code should go before (after) the spill code for a load (store).
  int scratchRegType = -1;
  int scratchReg = -1;
  if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType))
    {
      scratchReg = this->getUsableUniRegAtMI(scratchRegType, &LVSetBef,
                                             MInst, MIBef, MIAft);
      assert(scratchReg != MRI.getInvalidRegNum());
      MInst->getRegsUsed().insert(scratchReg); 
    }
  
  if (!isDef || isDefAndUse) {
    // for a USE, we have to load the value of LR from stack to a TmpReg
    // and use the TmpReg as one operand of instruction
    
    // actual loading instruction(s)
    MRI.cpMem2RegMI(AdIMid, MRI.getFramePointer(), SpillOff, TmpRegU, RegType,
                    scratchReg);
    
    // the actual load should be after the instructions to free up TmpRegU
    MIBef.insert(MIBef.end(), AdIMid.begin(), AdIMid.end());
    AdIMid.clear();
  }
  
  if (isDef) {   // if this is a Def
    // for a DEF, we have to store the value produced by this instruction
    // on the stack position allocated for this LR
    
    // actual storing instruction(s)
    MRI.cpReg2MemMI(AdIMid, TmpRegU, MRI.getFramePointer(), SpillOff, RegType,
                    scratchReg);
    
    MIAft.insert(MIAft.begin(), AdIMid.begin(), AdIMid.end());
  }  // if !DEF
  
  // Finally, insert the entire spill code sequences before/after MInst
  AI.InstrnsBefore.insert(AI.InstrnsBefore.end(), MIBef.begin(), MIBef.end());
  AI.InstrnsAfter.insert(AI.InstrnsAfter.begin(), MIAft.begin(), MIAft.end());
  
  if (DEBUG_RA) {
    cerr << "\nFor Inst " << *MInst;
    cerr << " - SPILLED LR: "; printSet(*LR);
    cerr << "\n - Added Instructions:";
    for_each(MIBef.begin(), MIBef.end(), std::mem_fun(&MachineInstr::dump));
    for_each(MIAft.begin(), MIAft.end(), std::mem_fun(&MachineInstr::dump));
  }
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
  
  RegClass* RC = this->getRegClassByID(MRI.getRegClassIDOfRegType(RegType));
  
  int RegU =  getUnusedUniRegAtMI(RC, MInst, LVSetBef);
  
  if (RegU == -1) {
    // we couldn't find an unused register. Generate code to free up a reg by
    // saving it on stack and restoring after the instruction
    
    int TmpOff = mcInfo.pushTempValue(TM,  MRI.getSpilledRegSize(RegType) );
    
    RegU = getUniRegNotUsedByThisInst(RC, MInst);
    
    // Check if we need a scratch register to copy this register to memory.
    int scratchRegType = -1;
    if (MRI.regTypeNeedsScratchReg(RegType, scratchRegType))
      {
        int scratchReg = this->getUsableUniRegAtMI(scratchRegType, LVSetBef,
                                                   MInst, MIBef, MIAft);
        assert(scratchReg != MRI.getInvalidRegNum());
        
        // We may as well hold the value in the scratch register instead
        // of copying it to memory and back.  But we have to mark the
        // register as used by this instruction, so it does not get used
        // as a scratch reg. by another operand or anyone else.
        MInst->getRegsUsed().insert(scratchReg); 
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
// This method is called to get a new unused register that can be used to
// accomodate a spilled value. 
// This method may be called several times for a single machine instruction
// if it contains many spilled operands. Each time it is called, it finds
// a register which is not live at that instruction and also which is not
// used by other spilled operands of the same instruction.
// Return register number is relative to the register class. NOT
// unified number
//----------------------------------------------------------------------------
int PhyRegAlloc::getUnusedUniRegAtMI(RegClass *RC, 
				  const MachineInstr *MInst, 
				  const ValueSet *LVSetBef) {

  unsigned NumAvailRegs =  RC->getNumOfAvailRegs();
  
  std::vector<bool> &IsColorUsedArr = RC->getIsColorUsedArr();
  
  for (unsigned i=0; i <  NumAvailRegs; i++)     // Reset array
      IsColorUsedArr[i] = false;
      
  ValueSet::const_iterator LIt = LVSetBef->begin();

  // for each live var in live variable set after machine inst
  for ( ; LIt != LVSetBef->end(); ++LIt) {

   //  get the live range corresponding to live var
    LiveRange *const LRofLV = LRI.getLiveRangeForValue(*LIt );    

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if (LRofLV && LRofLV->getRegClass() == RC && LRofLV->hasColor() ) 
      IsColorUsedArr[ LRofLV->getColor() ] = true;
  }

  // It is possible that one operand of this MInst was already spilled
  // and it received some register temporarily. If that's the case,
  // it is recorded in machine operand. We must skip such registers.

  setRelRegsUsedByThisInst(RC, MInst);

  for (unsigned c=0; c < NumAvailRegs; c++)   // find first unused color
     if (!IsColorUsedArr[c])
       return MRI.getUnifiedRegNum(RC->getID(), c);
  
  return -1;
}


//----------------------------------------------------------------------------
// Get any other register in a register class, other than what is used
// by operands of a machine instruction. Returns the unified reg number.
//----------------------------------------------------------------------------
int PhyRegAlloc::getUniRegNotUsedByThisInst(RegClass *RC, 
                                            const MachineInstr *MInst) {

  vector<bool> &IsColorUsedArr = RC->getIsColorUsedArr();
  unsigned NumAvailRegs =  RC->getNumOfAvailRegs();

  for (unsigned i=0; i < NumAvailRegs ; i++)   // Reset array
    IsColorUsedArr[i] = false;

  setRelRegsUsedByThisInst(RC, MInst);

  for (unsigned c=0; c < RC->getNumOfAvailRegs(); c++)// find first unused color
    if (!IsColorUsedArr[c])
      return  MRI.getUnifiedRegNum(RC->getID(), c);

  assert(0 && "FATAL: No free register could be found in reg class!!");
  return 0;
}


//----------------------------------------------------------------------------
// This method modifies the IsColorUsedArr of the register class passed to it.
// It sets the bits corresponding to the registers used by this machine
// instructions. Both explicit and implicit operands are set.
//----------------------------------------------------------------------------
void PhyRegAlloc::setRelRegsUsedByThisInst(RegClass *RC, 
                                           const MachineInstr *MInst ) {

  vector<bool> &IsColorUsedArr = RC->getIsColorUsedArr();
  
  // Add the registers already marked as used by the instruction. 
  // This should include any scratch registers that are used to save
  // values across the instruction (e.g., for saving state register values).
  const hash_set<int>& regsUsed = MInst->getRegsUsed();
  for (hash_set<int>::const_iterator SI=regsUsed.begin(), SE=regsUsed.end();
       SI != SE; ++SI)
    {
      unsigned classId = 0;
      int classRegNum = MRI.getClassRegNum(*SI, classId);
      if (RC->getID() == classId)
        {
          assert(classRegNum < (int) IsColorUsedArr.size() &&
                 "Illegal register number for this reg class?");
          IsColorUsedArr[classRegNum] = true;
        }
    }
  
  // Now add registers allocated to the live ranges of values used in
  // the instruction.  These are not yet recorded in the instruction.
  for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum)
    {
      const MachineOperand& Op = MInst->getOperand(OpNum);
      
      if (Op.getOperandType() == MachineOperand::MO_VirtualRegister || 
          Op.getOperandType() == MachineOperand::MO_CCRegister)
        if (const Value* Val = Op.getVRegValue())
          if (MRI.getRegClassIDOfValue(Val) == RC->getID())
            if (Op.getAllocatedRegNum() == -1)
              if (LiveRange *LROfVal = LRI.getLiveRangeForValue(Val))
                if (LROfVal->hasColor() )
                  // this operand is in a LR that received a color
                  IsColorUsedArr[LROfVal->getColor()] = true;
    }
  
  // If there are implicit references, mark their allocated regs as well
  // 
  for (unsigned z=0; z < MInst->getNumImplicitRefs(); z++)
    if (const LiveRange*
        LRofImpRef = LRI.getLiveRangeForValue(MInst->getImplicitRef(z)))    
      if (LRofImpRef->hasColor())
        // this implicit reference is in a LR that received a color
        IsColorUsedArr[LRofImpRef->getColor()] = true;
}


//----------------------------------------------------------------------------
// If there are delay slots for an instruction, the instructions
// added after it must really go after the delayed instruction(s).
// So, we move the InstrAfter of that instruction to the 
// corresponding delayed instruction using the following method.

//----------------------------------------------------------------------------
void PhyRegAlloc::move2DelayedInstr(const MachineInstr *OrigMI,
                                    const MachineInstr *DelayedMI) {

  // "added after" instructions of the original instr
  std::vector<MachineInstr *> &OrigAft = AddedInstrMap[OrigMI].InstrnsAfter;

  // "added instructions" of the delayed instr
  AddedInstrns &DelayAdI = AddedInstrMap[DelayedMI];

  // "added after" instructions of the delayed instr
  std::vector<MachineInstr *> &DelayedAft = DelayAdI.InstrnsAfter;

  // go thru all the "added after instructions" of the original instruction
  // and append them to the "addded after instructions" of the delayed
  // instructions
  DelayedAft.insert(DelayedAft.end(), OrigAft.begin(), OrigAft.end());

  // empty the "added after instructions" of the original instruction
  OrigAft.clear();
}

//----------------------------------------------------------------------------
// This method prints the code with registers after register allocation is
// complete.
//----------------------------------------------------------------------------
void PhyRegAlloc::printMachineCode()
{

  cerr << "\n;************** Function " << Meth->getName()
       << " *****************\n";

  for (Function::const_iterator BBI = Meth->begin(), BBE = Meth->end();
       BBI != BBE; ++BBI) {
    cerr << "\n"; printLabel(BBI); cerr << ": ";

    // get the iterator for machine instructions
    MachineCodeForBasicBlock& MIVec = MachineCodeForBasicBlock::get(BBI);
    MachineCodeForBasicBlock::iterator MII = MIVec.begin();

    // iterate over all the machine instructions in BB
    for ( ; MII != MIVec.end(); ++MII) {  
      MachineInstr *const MInst = *MII; 

      cerr << "\n\t";
      cerr << TargetInstrDescriptors[MInst->getOpCode()].opCodeString;

      for (unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {
	MachineOperand& Op = MInst->getOperand(OpNum);

	if (Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister /*|| 
	    Op.getOperandType() ==  MachineOperand::MO_PCRelativeDisp*/ ) {

	  const Value *const Val = Op.getVRegValue () ;
	  // ****this code is temporary till NULL Values are fixed
	  if (! Val ) {
	    cerr << "\t<*NULL*>";
	    continue;
	  }

	  // if a label or a constant
	  if (isa<BasicBlock>(Val)) {
	    cerr << "\t"; printLabel(	Op.getVRegValue	() );
	  } else {
	    // else it must be a register value
	    const int RegNum = Op.getAllocatedRegNum();

	    cerr << "\t" << "%" << MRI.getUnifiedRegName( RegNum );
	    if (Val->hasName() )
	      cerr << "(" << Val->getName() << ")";
	    else 
	      cerr << "(" << Val << ")";

	    if (Op.opIsDef() )
	      cerr << "*";

	    const LiveRange *LROfVal = LRI.getLiveRangeForValue(Val);
	    if (LROfVal )
	      if (LROfVal->hasSpillOffset() )
		cerr << "$";
	  }

	} 
	else if (Op.getOperandType() ==  MachineOperand::MO_MachineRegister) {
	  cerr << "\t" << "%" << MRI.getUnifiedRegName(Op.getMachineRegNum());
	}

	else 
	  cerr << "\t" << Op;      // use dump field
      }

    

      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      if (NumOfImpRefs > 0) {
	cerr << "\tImplicit:";

	for (unsigned z=0; z < NumOfImpRefs; z++)
	  cerr << RAV(MInst->getImplicitRef(z)) << "\t";
      }

    } // for all machine instructions

    cerr << "\n";

  } // for all BBs

  cerr << "\n";
}


//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void PhyRegAlloc::colorIncomingArgs()
{
  const BasicBlock &FirstBB = Meth->front();
  const MachineInstr *FirstMI = MachineCodeForBasicBlock::get(&FirstBB).front();
  assert(FirstMI && "No machine instruction in entry BB");

  MRI.colorMethodArgs(Meth, LRI, &AddedInstrAtEntry);
}


//----------------------------------------------------------------------------
// Used to generate a label for a basic block
//----------------------------------------------------------------------------
void PhyRegAlloc::printLabel(const Value *const Val) {
  if (Val->hasName())
    cerr  << Val->getName();
  else
    cerr << "Label" <<  Val;
}


//----------------------------------------------------------------------------
// This method calls setSugColorUsable method of each live range. This
// will determine whether the suggested color of LR is  really usable.
// A suggested color is not usable when the suggested color is volatile
// AND when there are call interferences
//----------------------------------------------------------------------------

void PhyRegAlloc::markUnusableSugColors()
{
  if (DEBUG_RA ) cerr << "\nmarking unusable suggested colors ...\n";

  // hash map iterator
  LiveRangeMapType::const_iterator HMI = (LRI.getLiveRangeMap())->begin();   
  LiveRangeMapType::const_iterator HMIEnd = (LRI.getLiveRangeMap())->end();   

    for (; HMI != HMIEnd ; ++HMI ) {
      if (HMI->first) { 
	LiveRange *L = HMI->second;      // get the LiveRange
	if (L) { 
	  if (L->hasSuggestedColor()) {
	    int RCID = L->getRegClass()->getID();
	    if (MRI.isRegVolatile( RCID,  L->getSuggestedColor()) &&
		L->isCallInterference() )
	      L->setSuggestedColorUsable( false );
	    else
	      L->setSuggestedColorUsable( true );
	  }
	} // if L->hasSuggestedColor()
      }
    } // for all LR's in hash map
}



//----------------------------------------------------------------------------
// The following method will set the stack offsets of the live ranges that
// are decided to be spillled. This must be called just after coloring the
// LRs using the graph coloring algo. For each live range that is spilled,
// this method allocate a new spill position on the stack.
//----------------------------------------------------------------------------

void PhyRegAlloc::allocateStackSpace4SpilledLRs() {
  if (DEBUG_RA) cerr << "\nsetting LR stack offsets ...\n";

  LiveRangeMapType::const_iterator HMI    = LRI.getLiveRangeMap()->begin();   
  LiveRangeMapType::const_iterator HMIEnd = LRI.getLiveRangeMap()->end();   

  for ( ; HMI != HMIEnd ; ++HMI) {
    if (HMI->first && HMI->second) {
      LiveRange *L = HMI->second;      // get the LiveRange
      if (!L->hasColor())   //  NOTE: ** allocating the size of long Type **
        L->setSpillOffFromFP(mcInfo.allocateSpilledValue(TM, Type::LongTy));
    }
  } // for all LR's in hash map
}



//----------------------------------------------------------------------------
// The entry pont to Register Allocation
//----------------------------------------------------------------------------

void PhyRegAlloc::allocateRegisters()
{

  // make sure that we put all register classes into the RegClassList 
  // before we call constructLiveRanges (now done in the constructor of 
  // PhyRegAlloc class).
  //
  LRI.constructLiveRanges();            // create LR info

  if (DEBUG_RA)
    LRI.printLiveRanges();
  
  createIGNodeListsAndIGs();            // create IGNode list and IGs

  buildInterferenceGraphs();            // build IGs in all reg classes
  
  
  if (DEBUG_RA) {
    // print all LRs in all reg classes
    for ( unsigned rc=0; rc < NumOfRegClasses  ; rc++)  
      RegClassList[rc]->printIGNodeList(); 
    
    // print IGs in all register classes
    for ( unsigned rc=0; rc < NumOfRegClasses ; rc++)  
      RegClassList[rc]->printIG();       
  }
  

  LRI.coalesceLRs();                    // coalesce all live ranges
  

  if (DEBUG_RA) {
    // print all LRs in all reg classes
    for ( unsigned rc=0; rc < NumOfRegClasses  ; rc++)  
      RegClassList[ rc ]->printIGNodeList(); 
    
    // print IGs in all register classes
    for ( unsigned rc=0; rc < NumOfRegClasses ; rc++)  
      RegClassList[ rc ]->printIG();       
  }


  // mark un-usable suggested color before graph coloring algorithm.
  // When this is done, the graph coloring algo will not reserve
  // suggested color unnecessarily - they can be used by another LR
  //
  markUnusableSugColors(); 

  // color all register classes using the graph coloring algo
  for (unsigned rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[ rc ]->colorAllRegs();    

  // Atter grpah coloring, if some LRs did not receive a color (i.e, spilled)
  // a poistion for such spilled LRs
  //
  allocateStackSpace4SpilledLRs();

  mcInfo.popAllTempValues(TM);  // TODO **Check

  // color incoming args - if the correct color was not received
  // insert code to copy to the correct register
  //
  colorIncomingArgs();

  // Now update the machine code with register names and add any 
  // additional code inserted by the register allocator to the instruction
  // stream
  //
  updateMachineCode(); 

  if (DEBUG_RA) {
    MachineCodeForMethod::get(Meth).dump();
    printMachineCode();                   // only for DEBUGGING
  }
}



