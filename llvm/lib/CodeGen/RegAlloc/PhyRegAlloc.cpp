// $Id$
//***************************************************************************
// File:
//	PhyRegAlloc.cpp
// 
// Purpose:
//      Register allocation for LLVM.
//	
// History:
//	9/10/01	 -  Ruchira Sasanka - created.
//**************************************************************************/

#include "llvm/CodeGen/RegisterAllocation.h"
#include "llvm/CodeGen/PhyRegAlloc.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Type.h"
#include <iostream>
#include <math.h>
using std::cerr;


// ***TODO: There are several places we add instructions. Validate the order
//          of adding these instructions.

cl::Enum<RegAllocDebugLevel_t> DEBUG_RA("dregalloc", cl::NoFlags,
  "enable register allocation debugging information",
  clEnumValN(RA_DEBUG_None   , "n", "disable debug output"),
  clEnumValN(RA_DEBUG_Normal , "y", "enable debug output"),
  clEnumValN(RA_DEBUG_Verbose, "v", "enable extra debug output"), 0);


//----------------------------------------------------------------------------
// RegisterAllocation pass front end...
//----------------------------------------------------------------------------
namespace {
  class RegisterAllocator : public MethodPass {
    TargetMachine &Target;
  public:
    inline RegisterAllocator(TargetMachine &T) : Target(T) {}
    
    bool runOnMethod(Method *M) {
      if (DEBUG_RA)
        cerr << "\n******************** Method "<< M->getName()
             << " ********************\n";
      
      PhyRegAlloc PRA(M, Target, &getAnalysis<MethodLiveVarInfo>(),
                      &getAnalysis<cfg::LoopInfo>());
      PRA.allocateRegisters();
      
      if (DEBUG_RA) cerr << "\nRegister allocation complete!\n";
      return false;
    }

    virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
                                      Pass::AnalysisSet &Destroyed,
                                      Pass::AnalysisSet &Provided) {
      Requires.push_back(cfg::LoopInfo::ID);
      Requires.push_back(MethodLiveVarInfo::ID);
    }
  };
}

MethodPass *getRegisterAllocator(TargetMachine &T) {
  return new RegisterAllocator(T);
}

//----------------------------------------------------------------------------
// Constructor: Init local composite objects and create register classes.
//----------------------------------------------------------------------------
PhyRegAlloc::PhyRegAlloc(Method *M, 
			 const TargetMachine& tm, 
			 MethodLiveVarInfo *Lvi,
                         cfg::LoopInfo *LDC) 
                       :  TM(tm), Meth(M),
                          mcInfo(MachineCodeForMethod::get(M)),
                          LVI(Lvi), LRI(M, tm, RegClassList), 
			  MRI( tm.getRegInfo() ),
                          NumOfRegClasses(MRI.getNumOfRegClasses()),
			  LoopDepthCalc(LDC) {

  // create each RegisterClass and put in RegClassList
  //
  for(unsigned int rc=0; rc < NumOfRegClasses; rc++)  
    RegClassList.push_back( new RegClass(M, MRI.getMachineRegClass(rc), 
					 &ResColList) );
}


//----------------------------------------------------------------------------
// Destructor: Deletes register classes
//----------------------------------------------------------------------------
PhyRegAlloc::~PhyRegAlloc() { 
  for( unsigned int rc=0; rc < NumOfRegClasses; rc++)
    delete RegClassList[rc];
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
        if( DEBUG_RA) {
          cerr << "\n*?!?Warning: Null liver range found for: "
               << RAV(HMI->first) << "\n";
        }
        continue;
      }
                                        // if the Value * is not null, and LR  
                                        // is not yet written to the IGNodeList
      if( !(L->getUserIGNode())  ) {  
        RegClass *const RC =           // RegClass of first value in the LR
          RegClassList[ L->getRegClass()->getID() ];
        
        RC->addLRToIG(L);              // add this LR to an IG
      }
    }
  }
    
  // init RegClassList
  for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[rc]->createInterferenceGraph();

  if( DEBUG_RA)
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
  for( ; LIt != LVSet->end(); ++LIt) {

    if (DEBUG_RA > 1)
      cerr << "< Def=" << RAV(Def) << ", Lvar=" << RAV(*LIt) << "> ";

    //  get the live range corresponding to live var
    //
    LiveRange *LROfVar = LRI.getLiveRangeForValue(*LIt);

    // LROfVar can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    //
    if (LROfVar) {  
      if(LROfDef == LROfVar)            // do not set interf for same LR
	continue;

      // if 2 reg classes are the same set interference
      //
      if (RCOfDef == LROfVar->getRegClass()) {
	RCOfDef->setInterference( LROfDef, LROfVar);  
      } else if (DEBUG_RA > 1)  { 
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

  // Now find the LR of the return value of the call
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but it does not interfere with call
  // (i.e., we can allocate a volatile register to the return value)
  //
  LiveRange *RetValLR = NULL;
  const Value *RetVal = MRI.getCallInstRetVal( MInst );

  if( RetVal ) {
    RetValLR = LRI.getLiveRangeForValue( RetVal );
    assert( RetValLR && "No LR for RetValue of call");
  }

  if( DEBUG_RA)
    cerr << "\n For call inst: " << *MInst;

  ValueSet::const_iterator LIt = LVSetAft->begin();

  // for each live var in live variable set after machine inst
  //
  for( ; LIt != LVSetAft->end(); ++LIt) {

    //  get the live range corresponding to live var
    //
    LiveRange *const LR = LRI.getLiveRangeForValue(*LIt ); 

    if( LR && DEBUG_RA) {
      cerr << "\n\tLR Aft Call: ";
      printSet(*LR);
    }
   

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    //
    if( LR && (LR != RetValLR) )   {  
      LR->setCallInterference();
      if( DEBUG_RA) {
	cerr << "\n  ++Added call interf for LR: " ;
	printSet(*LR);
      }
    }

  }

}




//----------------------------------------------------------------------------
// This method will walk thru code and create interferences in the IG of
// each RegClass. Also, this method calculates the spill cost of each
// Live Range (it is done in this method to save another pass over the code).
//----------------------------------------------------------------------------
void PhyRegAlloc::buildInterferenceGraphs()
{

  if(DEBUG_RA) cerr << "Creating interference graphs ...\n";

  unsigned BBLoopDepthCost;
  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    // find the 10^(loop_depth) of this BB 
    //
    BBLoopDepthCost = (unsigned) pow( 10.0, LoopDepthCalc->getLoopDepth(*BBI));

    // get the iterator for machine instructions
    //
    const MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::const_iterator 
      MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    //
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  

      const MachineInstr *MInst = *MInstIterator; 

      // get the LV set after the instruction
      //
      const ValueSet &LVSetAI = LVI->getLiveVarSetAfterMInst(MInst, *BBI);
    
      const bool isCallInst = TM.getInstrInfo().isCall(MInst->getOpCode());

      if( isCallInst ) {
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
      if(  NumOfImpRefs > 0 ) {
	for(unsigned z=0; z < NumOfImpRefs; z++) 
	  if( MInst->implicitRefIsDefined(z) )
	    addInterference( MInst->getImplicitRef(z), &LVSetAI, isCallInst );
      }


    } // for all machine instructions in BB
    
  } // for all BBs in method


  // add interferences for method arguments. Since there are no explict 
  // defs in method for args, we have to add them manually
  //  
  addInterferencesForArgs();          

  if( DEBUG_RA)
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
    for(++It2; It2 != ItE; ++It2) {
      const LiveRange *LROfOp2 = LRI.getLiveRangeForValue(*It2); 

      if (LROfOp2) {
	RegClass *RCOfOp1 = LROfOp1->getRegClass(); 
	RegClass *RCOfOp2 = LROfOp2->getRegClass(); 
 
	if( RCOfOp1 == RCOfOp2 ){ 
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
// This method will add interferences for incoming arguments to a method.
//----------------------------------------------------------------------------
void PhyRegAlloc::addInterferencesForArgs() {
  // get the InSet of root BB
  const ValueSet &InSet = LVI->getInSetOfBB(Meth->front());  

  // get the argument list
  const Method::ArgumentListType& ArgList = Meth->getArgumentList();  

  // get an iterator to arg list
  Method::ArgumentListType::const_iterator ArgIt = ArgList.begin();          


  for( ; ArgIt != ArgList.end() ; ++ArgIt) {  // for each argument
    addInterference((Value*)*ArgIt, &InSet, false);// add interferences between 
                                              // args and LVars at start
    if( DEBUG_RA > 1)
      cerr << " - %% adding interference for  argument "
           << RAV((const Value *)*ArgIt) << "\n";
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
void PhyRegAlloc::updateMachineCode()
{

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    // get the iterator for machine instructions
    //
    MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::iterator MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    //
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      MachineInstr *MInst = *MInstIterator; 
      
      unsigned Opcode =  MInst->getOpCode();
    
      // do not process Phis
      if (TM.getInstrInfo().isPhi(Opcode))
	continue;

      // Now insert speical instructions (if necessary) for call/return
      // instructions. 
      //
      if (TM.getInstrInfo().isCall(Opcode) ||
	  TM.getInstrInfo().isReturn(Opcode)) {

	AddedInstrns *AI = AddedInstrMap[ MInst];
	if ( !AI ) { 
	  AI = new AddedInstrns();
	  AddedInstrMap[ MInst ] = AI;
	}
	
	// Tmp stack poistions are needed by some calls that have spilled args
	// So reset it before we call each such method
	//
	mcInfo.popAllTempValues(TM);  
	
	if (TM.getInstrInfo().isCall(Opcode))
	  MRI.colorCallArgs(MInst, LRI, AI, *this, *BBI);
	else if (TM.getInstrInfo().isReturn(Opcode))
	  MRI.colorRetValue(MInst, LRI, AI);
      }
      

      /* -- Using above code instead of this

      // if this machine instr is call, insert caller saving code

      if( (TM.getInstrInfo()).isCall( MInst->getOpCode()) )
	MRI.insertCallerSavingCode(MInst,  *BBI, *this );
	
      */

      
      // reset the stack offset for temporary variables since we may
      // need that to spill
      // mcInfo.popAllTempValues(TM);
      // TODO ** : do later
      
      //for(MachineInstr::val_const_op_iterator OpI(MInst);!OpI.done();++OpI) {


      // Now replace set the registers for operands in the machine instruction
      //
      for(unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {

	MachineOperand& Op = MInst->getOperand(OpNum);

	if( Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister) {

	  const Value *const Val =  Op.getVRegValue();

	  // delete this condition checking later (must assert if Val is null)
	  if( !Val) {
            if (DEBUG_RA)
              cerr << "Warning: NULL Value found for operand\n";
	    continue;
	  }
	  assert( Val && "Value is NULL");   

	  LiveRange *const LR = LRI.getLiveRangeForValue(Val);

	  if ( !LR ) {

	    // nothing to worry if it's a const or a label

            if (DEBUG_RA) {
              cerr << "*NO LR for operand : " << Op ;
	      cerr << " [reg:" <<  Op.getAllocatedRegNum() << "]";
	      cerr << " in inst:\t" << *MInst << "\n";
            }

	    // if register is not allocated, mark register as invalid
	    if( Op.getAllocatedRegNum() == -1)
	      Op.setRegForValue( MRI.getInvalidRegNum()); 
	    

	    continue;
	  }
	
	  unsigned RCID = (LR->getRegClass())->getID();

	  if( LR->hasColor() ) {
	    Op.setRegForValue( MRI.getUnifiedRegNum(RCID, LR->getColor()) );
	  }
	  else {

	    // LR did NOT receive a color (register). Now, insert spill code
	    // for spilled opeands in this machine instruction

	    //assert(0 && "LR must be spilled");
	    insertCode4SpilledLR(LR, MInst, *BBI, OpNum );

	  }
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
      if( AddedInstrMap[ MInst ] ) {
	std::deque<MachineInstr *> &IBef = AddedInstrMap[MInst]->InstrnsBefore;

	if( ! IBef.empty() ) {
	  std::deque<MachineInstr *>::iterator AdIt; 

	  for( AdIt = IBef.begin(); AdIt != IBef.end() ; ++AdIt ) {

	    if( DEBUG_RA) {
	      cerr << "For inst " << *MInst;
	      cerr << " PREPENDed instr: " << **AdIt << "\n";
	    }
	  	    
	    MInstIterator = MIVec.insert( MInstIterator, *AdIt );
	    ++MInstIterator;
	  }

	}

      }

      // If there are instructions to be added *after* this machine
      // instruction, add them now
      //
      if(AddedInstrMap[MInst] && 
         !AddedInstrMap[MInst]->InstrnsAfter.empty() ) {

	// if there are delay slots for this instruction, the instructions
	// added after it must really go after the delayed instruction(s)
	// So, we move the InstrAfter of the current instruction to the 
	// corresponding delayed instruction
	
	unsigned delay;
	if ((delay=TM.getInstrInfo().getNumDelaySlots(MInst->getOpCode())) >0){ 
	  move2DelayedInstr(MInst,  *(MInstIterator+delay) );

	  if(DEBUG_RA)  cerr<< "\nMoved an added instr after the delay slot";
	}
       
	else {
	

	  // Here we can add the "instructions after" to the current
	  // instruction since there are no delay slots for this instruction

	  std::deque<MachineInstr *> &IAft = AddedInstrMap[MInst]->InstrnsAfter;
	  
	  if( ! IAft.empty() ) {     
	    
	    std::deque<MachineInstr *>::iterator AdIt; 
	    
	    ++MInstIterator;   // advance to the next instruction
	    
	    for( AdIt = IAft.begin(); AdIt != IAft.end() ; ++AdIt ) {
	      
	      if(DEBUG_RA) {
		cerr << "For inst " << *MInst;
		cerr << " APPENDed instr: "  << **AdIt << "\n";
	      }	      

	      MInstIterator = MIVec.insert( MInstIterator, *AdIt );
	      ++MInstIterator;
	    }

	    // MInsterator already points to the next instr. Since the
	    // for loop also increments it, decrement it to point to the
	    // instruction added last
	    --MInstIterator;  
	    
	  }
	  
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
  unsigned RegType = MRI.getRegType( LR );
  int SpillOff = LR->getSpillOffFromFP();
  RegClass *RC = LR->getRegClass();
  const ValueSet &LVSetBef = LVI->getLiveVarSetBeforeMInst(MInst, BB);

  mcInfo.pushTempValue(TM, MRI.getSpilledRegSize(RegType) );
  
  MachineInstr *MIBef=NULL,  *AdIMid=NULL, *MIAft=NULL;
  
  int TmpRegU = getUsableUniRegAtMI(RC, RegType, MInst,&LVSetBef, MIBef, MIAft);
  
  // get the added instructions for this instruciton
  AddedInstrns *AI = AddedInstrMap[ MInst ];
  if ( !AI ) { 
    AI = new AddedInstrns();
    AddedInstrMap[ MInst ] = AI;
  }

    
  if( !isDef ) {

    // for a USE, we have to load the value of LR from stack to a TmpReg
    // and use the TmpReg as one operand of instruction

    // actual loading instruction
    AdIMid = MRI.cpMem2RegMI(MRI.getFramePointer(), SpillOff, TmpRegU,RegType);

    if(MIBef)
      AI->InstrnsBefore.push_back(MIBef);

    AI->InstrnsBefore.push_back(AdIMid);

    if(MIAft)
      AI->InstrnsAfter.push_front(MIAft);
    
  } else {   // if this is a Def
    // for a DEF, we have to store the value produced by this instruction
    // on the stack position allocated for this LR

    // actual storing instruction
    AdIMid = MRI.cpReg2MemMI(TmpRegU, MRI.getFramePointer(), SpillOff,RegType);

    if (MIBef)
      AI->InstrnsBefore.push_back(MIBef);

    AI->InstrnsAfter.push_front(AdIMid);

    if (MIAft)
      AI->InstrnsAfter.push_front(MIAft);

  }  // if !DEF

  cerr << "\nFor Inst " << *MInst;
  cerr << " - SPILLED LR: "; printSet(*LR);
  cerr << "\n - Added Instructions:";
  if (MIBef) cerr <<  *MIBef;
  cerr <<  *AdIMid;
  if (MIAft) cerr <<  *MIAft;

  Op.setRegForValue(TmpRegU);    // set the opearnd
}



//----------------------------------------------------------------------------
// We can use the following method to get a temporary register to be used
// BEFORE any given machine instruction. If there is a register available,
// this method will simply return that register and set MIBef = MIAft = NULL.
// Otherwise, it will return a register and MIAft and MIBef will contain
// two instructions used to free up this returned register.
// Returned register number is the UNIFIED register number
//----------------------------------------------------------------------------

int PhyRegAlloc::getUsableUniRegAtMI(RegClass *RC, 
				  const int RegType,
				  const MachineInstr *MInst, 
				  const ValueSet *LVSetBef,
				  MachineInstr *MIBef,
				  MachineInstr *MIAft) {

  int RegU =  getUnusedUniRegAtMI(RC, MInst, LVSetBef);


  if( RegU != -1) {
    // we found an unused register, so we can simply use it
    MIBef = MIAft = NULL;
  }
  else {
    // we couldn't find an unused register. Generate code to free up a reg by
    // saving it on stack and restoring after the instruction

    int TmpOff = mcInfo.pushTempValue(TM,  MRI.getSpilledRegSize(RegType) );
    
    RegU = getUniRegNotUsedByThisInst(RC, MInst);
    MIBef = MRI.cpReg2MemMI(RegU, MRI.getFramePointer(), TmpOff, RegType );
    MIAft = MRI.cpMem2RegMI(MRI.getFramePointer(), TmpOff, RegU, RegType );
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
  
  bool *IsColorUsedArr = RC->getIsColorUsedArr();
  
  for(unsigned i=0; i <  NumAvailRegs; i++)     // Reset array
      IsColorUsedArr[i] = false;
      
  ValueSet::const_iterator LIt = LVSetBef->begin();

  // for each live var in live variable set after machine inst
  for( ; LIt != LVSetBef->end(); ++LIt) {

   //  get the live range corresponding to live var
    LiveRange *const LRofLV = LRI.getLiveRangeForValue(*LIt );    

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if( LRofLV )     
      if( LRofLV->hasColor() ) 
	IsColorUsedArr[ LRofLV->getColor() ] = true;
  }

  // It is possible that one operand of this MInst was already spilled
  // and it received some register temporarily. If that's the case,
  // it is recorded in machine operand. We must skip such registers.

  setRelRegsUsedByThisInst(RC, MInst);

  unsigned c;                         // find first unused color
  for( c=0; c < NumAvailRegs; c++)  
     if( ! IsColorUsedArr[ c ] ) break;
   
  if(c < NumAvailRegs) 
    return  MRI.getUnifiedRegNum(RC->getID(), c);
  else 
    return -1;


}


//----------------------------------------------------------------------------
// Get any other register in a register class, other than what is used
// by operands of a machine instruction. Returns the unified reg number.
//----------------------------------------------------------------------------
int PhyRegAlloc::getUniRegNotUsedByThisInst(RegClass *RC, 
					 const MachineInstr *MInst) {

  bool *IsColorUsedArr = RC->getIsColorUsedArr();
  unsigned NumAvailRegs =  RC->getNumOfAvailRegs();


  for(unsigned i=0; i < NumAvailRegs ; i++)   // Reset array
    IsColorUsedArr[i] = false;

  setRelRegsUsedByThisInst(RC, MInst);

  unsigned c;                         // find first unused color
  for( c=0; c <  RC->getNumOfAvailRegs(); c++)  
     if( ! IsColorUsedArr[ c ] ) break;
   
  if(c < NumAvailRegs) 
    return  MRI.getUnifiedRegNum(RC->getID(), c);
  else 
    assert( 0 && "FATAL: No free register could be found in reg class!!");
  return 0;
}


//----------------------------------------------------------------------------
// This method modifies the IsColorUsedArr of the register class passed to it.
// It sets the bits corresponding to the registers used by this machine
// instructions. Both explicit and implicit operands are set.
//----------------------------------------------------------------------------
void PhyRegAlloc::setRelRegsUsedByThisInst(RegClass *RC, 
				       const MachineInstr *MInst ) {

 bool *IsColorUsedArr = RC->getIsColorUsedArr();
  
 for(unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {
    
   const MachineOperand& Op = MInst->getOperand(OpNum);

    if( Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	Op.getOperandType() ==  MachineOperand::MO_CCRegister ) {

      const Value *const Val =  Op.getVRegValue();

      if( Val ) 
	if( MRI.getRegClassIDOfValue(Val) == RC->getID() ) {   
	  int Reg;
	  if( (Reg=Op.getAllocatedRegNum()) != -1) {
	    IsColorUsedArr[ Reg ] = true;
	  }
	  else {
	    // it is possilbe that this operand still is not marked with
	    // a register but it has a LR and that received a color

	    LiveRange *LROfVal =  LRI.getLiveRangeForValue(Val);
	    if( LROfVal) 
	      if( LROfVal->hasColor() )
		IsColorUsedArr[ LROfVal->getColor() ] = true;
	  }
	
	} // if reg classes are the same
    }
    else if (Op.getOperandType() ==  MachineOperand::MO_MachineRegister) {
      IsColorUsedArr[ Op.getMachineRegNum() ] = true;
    }
 }
 
 // If there are implicit references, mark them as well

 for(unsigned z=0; z < MInst->getNumImplicitRefs(); z++) {

   LiveRange *const LRofImpRef = 
     LRI.getLiveRangeForValue( MInst->getImplicitRef(z)  );    
   
   if(LRofImpRef && LRofImpRef->hasColor())
     IsColorUsedArr[LRofImpRef->getColor()] = true;
 }
}








//----------------------------------------------------------------------------
// If there are delay slots for an instruction, the instructions
// added after it must really go after the delayed instruction(s).
// So, we move the InstrAfter of that instruction to the 
// corresponding delayed instruction using the following method.

//----------------------------------------------------------------------------
void PhyRegAlloc:: move2DelayedInstr(const MachineInstr *OrigMI,
				     const MachineInstr *DelayedMI) {

  // "added after" instructions of the original instr
  std::deque<MachineInstr *> &OrigAft = AddedInstrMap[OrigMI]->InstrnsAfter;

  // "added instructions" of the delayed instr
  AddedInstrns *DelayAdI = AddedInstrMap[DelayedMI];

  if(! DelayAdI )  {                // create a new "added after" if necessary
    DelayAdI = new AddedInstrns();
    AddedInstrMap[DelayedMI] =  DelayAdI;
  }

  // "added after" instructions of the delayed instr
  std::deque<MachineInstr *> &DelayedAft = DelayAdI->InstrnsAfter;

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

  cerr << "\n;************** Method " << Meth->getName()
       << " *****************\n";

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    cerr << "\n"; printLabel( *BBI); cerr << ": ";

    // get the iterator for machine instructions
    MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::iterator MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      MachineInstr *const MInst = *MInstIterator; 


      cerr << "\n\t";
      cerr << TargetInstrDescriptors[MInst->getOpCode()].opCodeString;
      

      //for(MachineInstr::val_const_op_iterator OpI(MInst);!OpI.done();++OpI) {

      for(unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {

	MachineOperand& Op = MInst->getOperand(OpNum);

	if( Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister /*|| 
	    Op.getOperandType() ==  MachineOperand::MO_PCRelativeDisp*/ ) {

	  const Value *const Val = Op.getVRegValue () ;
	  // ****this code is temporary till NULL Values are fixed
	  if( ! Val ) {
	    cerr << "\t<*NULL*>";
	    continue;
	  }

	  // if a label or a constant
	  if(isa<BasicBlock>(Val)) {
	    cerr << "\t"; printLabel(	Op.getVRegValue	() );
	  } else {
	    // else it must be a register value
	    const int RegNum = Op.getAllocatedRegNum();

	    cerr << "\t" << "%" << MRI.getUnifiedRegName( RegNum );
	    if (Val->hasName() )
	      cerr << "(" << Val->getName() << ")";
	    else 
	      cerr << "(" << Val << ")";

	    if( Op.opIsDef() )
	      cerr << "*";

	    const LiveRange *LROfVal = LRI.getLiveRangeForValue(Val);
	    if( LROfVal )
	      if( LROfVal->hasSpillOffset() )
		cerr << "$";
	  }

	} 
	else if(Op.getOperandType() ==  MachineOperand::MO_MachineRegister) {
	  cerr << "\t" << "%" << MRI.getUnifiedRegName(Op.getMachineRegNum());
	}

	else 
	  cerr << "\t" << Op;      // use dump field
      }

    

      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      if( NumOfImpRefs > 0) {
	cerr << "\tImplicit:";

	for(unsigned z=0; z < NumOfImpRefs; z++)
	  cerr << RAV(MInst->getImplicitRef(z)) << "\t";
      }

    } // for all machine instructions

    cerr << "\n";

  } // for all BBs

  cerr << "\n";
}


#if 0

//----------------------------------------------------------------------------
//
//----------------------------------------------------------------------------

void PhyRegAlloc::colorCallRetArgs()
{

  CallRetInstrListType &CallRetInstList = LRI.getCallRetInstrList();
  CallRetInstrListType::const_iterator It = CallRetInstList.begin();

  for( ; It != CallRetInstList.end(); ++It ) {

    const MachineInstr *const CRMI = *It;
    unsigned OpCode =  CRMI->getOpCode();
 
    // get the added instructions for this Call/Ret instruciton
    AddedInstrns *AI = AddedInstrMap[ CRMI ];
    if ( !AI ) { 
      AI = new AddedInstrns();
      AddedInstrMap[ CRMI ] = AI;
    }

    // Tmp stack poistions are needed by some calls that have spilled args
    // So reset it before we call each such method
    //mcInfo.popAllTempValues(TM);  


    
    if (TM.getInstrInfo().isCall(OpCode))
      MRI.colorCallArgs(CRMI, LRI, AI, *this);
    else if (TM.getInstrInfo().isReturn(OpCode)) 
      MRI.colorRetValue( CRMI, LRI, AI );
    else
      assert(0 && "Non Call/Ret instrn in CallRetInstrList\n");
  }
}

#endif 

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void PhyRegAlloc::colorIncomingArgs()
{
  const BasicBlock *const FirstBB = Meth->front();
  const MachineInstr *FirstMI = FirstBB->getMachineInstrVec().front();
  assert(FirstMI && "No machine instruction in entry BB");

  AddedInstrns *AI = AddedInstrMap[FirstMI];
  if (!AI)
    AddedInstrMap[FirstMI] = AI = new AddedInstrns();

  MRI.colorMethodArgs(Meth, LRI, AI);
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
  if(DEBUG_RA ) cerr << "\nmarking unusable suggested colors ...\n";

  // hash map iterator
  LiveRangeMapType::const_iterator HMI = (LRI.getLiveRangeMap())->begin();   
  LiveRangeMapType::const_iterator HMIEnd = (LRI.getLiveRangeMap())->end();   

    for(; HMI != HMIEnd ; ++HMI ) {
      if (HMI->first) { 
	LiveRange *L = HMI->second;      // get the LiveRange
	if (L) { 
	  if(L->hasSuggestedColor()) {
	    int RCID = L->getRegClass()->getID();
	    if( MRI.isRegVolatile( RCID,  L->getSuggestedColor()) &&
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

  for( ; HMI != HMIEnd ; ++HMI) {
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
    for( unsigned int rc=0; rc < NumOfRegClasses  ; rc++)  
      RegClassList[ rc ]->printIGNodeList(); 
    
    // print IGs in all register classes
    for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
      RegClassList[ rc ]->printIG();       
  }
  

  LRI.coalesceLRs();                    // coalesce all live ranges
  

  if( DEBUG_RA) {
    // print all LRs in all reg classes
    for( unsigned int rc=0; rc < NumOfRegClasses  ; rc++)  
      RegClassList[ rc ]->printIGNodeList(); 
    
    // print IGs in all register classes
    for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
      RegClassList[ rc ]->printIG();       
  }


  // mark un-usable suggested color before graph coloring algorithm.
  // When this is done, the graph coloring algo will not reserve
  // suggested color unnecessarily - they can be used by another LR
  //
  markUnusableSugColors(); 

  // color all register classes using the graph coloring algo
  for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
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



