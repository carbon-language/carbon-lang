//***************************************************************************
// File:
//	Sparc.cpp
// 
// Purpose:
//	
// History:
//	7/15/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#include "llvm/Target/Sparc.h"
#include "SparcInternals.h"
#include "llvm/Method.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/InstrSelection.h"

#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/PhyRegAlloc.h"


// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend. (the llvm/CodeGen/Sparc.h interface)
//
TargetMachine *allocateSparcTargetMachine() { return new UltraSparc(); }


//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class MachineInstrInfo. 
//---------------------------------------------------------------------------

/*ctor*/
UltraSparcInstrInfo::UltraSparcInstrInfo()
  : MachineInstrInfo(SparcMachineInstrDesc,
		     /*descSize = */ NUM_TOTAL_OPCODES,
		     /*numRealOpCodes = */ NUM_REAL_OPCODES)
{
}


//---------------------------------------------------------------------------
// class UltraSparcSchedInfo 
// 
// Purpose:
//   Scheduling information for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class MachineSchedInfo.
//---------------------------------------------------------------------------

/*ctor*/
UltraSparcSchedInfo::UltraSparcSchedInfo(const MachineInstrInfo* mii)
  : MachineSchedInfo((unsigned int) SPARC_NUM_SCHED_CLASSES,
		     mii,
		     SparcRUsageDesc,
		     SparcInstrUsageDeltas,
		     SparcInstrIssueDeltas,
		     sizeof(SparcInstrUsageDeltas)/sizeof(InstrRUsageDelta),
		     sizeof(SparcInstrIssueDeltas)/sizeof(InstrIssueDelta))
{
  maxNumIssueTotal = 4;
  longestIssueConflict = 0;		// computed from issuesGaps[]
  
  branchMispredictPenalty = 4;		// 4 for SPARC IIi
  branchTargetUnknownPenalty = 2;	// 2 for SPARC IIi
  l1DCacheMissPenalty = 8;		// 7 or 9 for SPARC IIi
  l1ICacheMissPenalty = 8;		// ? for SPARC IIi
  
  inOrderLoads = true;			// true for SPARC IIi
  inOrderIssue = true;			// true for SPARC IIi
  inOrderExec  = false;			// false for most architectures
  inOrderRetire= true;			// true for most architectures
  
  // must be called after above parameters are initialized.
  this->initializeResources();
}

void
UltraSparcSchedInfo::initializeResources()
{
  // Compute MachineSchedInfo::instrRUsages and MachineSchedInfo::issueGaps
  MachineSchedInfo::initializeResources();
  
  // Machine-dependent fixups go here.  None for now.
}



//---------------------------------------------------------------------------
// class UltraSparcRegInfo 
//
// Purpose:
//   This class provides info about sparc register classes.
//--------------------------------------------------------------------------

#if 0
UltraSparcRegInfo::UltraSparcRegInfo(const UltraSparc *const USI ) : 
                                                      UltraSparcInfo(USI), 
                                                      NumOfIntArgRegs(6), 
                                                      NumOfFloatArgRegs(6) 
  {    
    MachineRegClassArr.push_back( new SparcIntRegClass(IntRegClassID) );
    MachineRegClassArr.push_back( new SparcFloatRegClass(FloatRegClassID) );
    MachineRegClassArr.push_back( new SparcIntCCRegClass(IntCCRegClassID) );
    MachineRegClassArr.push_back( new SparcFloatCCRegClass(FloatCCRegClassID));

    assert( SparcFloatRegOrder::StartOfNonVolatileRegs == 6 && 
	    "6 Float regs are used for float arg passing");
  }

  // ***** TODO  insert deletes for reg classes 
UltraSparcRegInfo::~UltraSparcRegInfo(void) { }    // empty destructor 

#endif

//---------------------------------------------------------------------------
// UltraSparcRegInfo
// Purpose:
//   This method will color incoming args to a method. If there are more
//   args than that can fit in regs, code will be inserted to pop them from
//   stack
//---------------------------------------------------------------------------


void UltraSparcRegInfo::colorArgs(const Method *const Meth, 
				  LiveRangeInfo& LRI) const 
{

                                                 // get the argument list
  const Method::ArgumentListType& ArgList = Meth->getArgumentList();           
                                                 // get an iterator to arg list
  Method::ArgumentListType::const_iterator ArgIt = ArgList.begin(); 
  unsigned intArgNo=0;

  // to keep track of which float regs are allocated for argument passing
  bool FloatArgUsedArr[NumOfFloatArgRegs];

  // init float arg used array
  for(unsigned i=0; i < NumOfFloatArgRegs; ++i) 
    FloatArgUsedArr[i] = false;

  // for each argument
  for( ; ArgIt != ArgList.end() ; ++ArgIt) {    

    // get the LR of arg
    LiveRange *const LR = LRI.getLiveRangeForValue((const Value *) *ArgIt); 
    unsigned RegClassID = (LR->getRegClass())->getID();

    // if the arg is in int class - allocate a reg for an int arg
    if( RegClassID == IntRegClassID ) {

      if( intArgNo < NumOfIntArgRegs) {
	LR->setColor( SparcIntRegOrder::i0 + intArgNo );

	if( DEBUG_RA) printReg( LR );
      }
  
      else {
	// TODO: Insert push code here
	assert( 0 && "Insert push code here!");
      }
      ++intArgNo;
    }

    // if the arg is float/double 
    else if ( RegClassID == FloatRegClassID) {

      if( LR->getTypeID() == Type::DoubleTyID ) {

	// find the first reg # we can pass a double arg
	for(unsigned i=0; i < NumOfFloatArgRegs; i+= 2) {
	  if ( !FloatArgUsedArr[i] && !FloatArgUsedArr[i+1] ) {
	    LR->setColor( SparcFloatRegOrder::f0 + i );
	    FloatArgUsedArr[i] = true;
	    FloatArgUsedArr[i+1] = true;
	    if( DEBUG_RA) printReg( LR );
	    break;
	  }
	}
	if( ! LR->hasColor() ) { // if LR was not colored above

	  assert(0 && "insert push code here for a double");

	}

      }
      else if( LR->getTypeID() == Type::FloatTyID ) { 

	// find the first reg # we can pass a float arg
	for(unsigned i=0; i < NumOfFloatArgRegs; ++i) {
	  if ( !FloatArgUsedArr[i] ) {
	    LR->setColor( SparcFloatRegOrder::f0 + i );
	    FloatArgUsedArr[i] = true;
	    if( DEBUG_RA) printReg( LR );
	    break;
	  }
	}
	if( ! LR->hasColor() ) { // if LR was not colored above
	  assert(0 && "insert push code here for a float");
	}

      }
      else 
	assert(0 && "unknown float type in method arg");

    } // float register class

    else 
      assert(0 && "Unknown RegClassID");
  }
  
}






void UltraSparcRegInfo::printReg(const LiveRange *const LR) {

  unsigned RegClassID = (LR->getRegClass())->getID();

  cout << " *Node " << (LR->getUserIGNode())->getIndex();

  if( ! LR->hasColor() ) {
    cout << " - could not find a color" << endl;
    return;
  }
  
  // if a color is found

  cout << " colored with color "<< LR->getColor();

  if( RegClassID == IntRegClassID ) {

    cout<< " [" << SparcIntRegOrder::getRegName(LR->getColor()) ;
    cout << "]" << endl;
  }
  else if ( RegClassID == FloatRegClassID) {
    cout << "[" << SparcFloatRegOrder::getRegName(LR->getColor());
    if( LR->getTypeID() == Type::DoubleTyID )
      cout << "+" << SparcFloatRegOrder::getRegName(LR->getColor()+1);
    cout << "]" << endl;
  }


}


void UltraSparcRegInfo::colorCallArgs(vector<const Instruction *> & 
				      CallInstrList, LiveRangeInfo& LRI,
				      AddedInstrMapType &AddedInstrMap) const
{

  vector<const Instruction *>::const_iterator InstIt = CallInstrList.begin();

  for( ; InstIt != CallInstrList.end(); ++InstIt) {

    // Inst = LLVM call instruction
    const Instruction *const CallI = *InstIt;

    MachineCodeForVMInstr &  MInstVec = CallI->getMachineInstrVec();
    MachineCodeForVMInstr::const_iterator MIIt = MInstVec.begin();

    // find the CALL/JMMPL machine instruction
    for( ; MIIt != MInstVec.end() && 
	   ! getUltraSparcInfo().getInstrInfo().isCall((*MIIt)->getOpCode()); 
	 ++MIIt );

    assert( (MIIt != MInstVec.end())  && "CALL/JMPL not found");

    // CallMI = CALL/JMPL machine isntruction
    const MachineInstr *const CallMI = *MIIt;

    Instruction::op_const_iterator OpIt = CallI->op_begin();

    unsigned intArgNo=0;
    //unsigned NumOfCallInterfs = LR->getNumOfCallInterferences();

    // to keep track of which float regs are allocated for argument passing
    bool FloatArgUsedArr[NumOfFloatArgRegs];

    // init float arg used array
    for(unsigned i=0; i < NumOfFloatArgRegs; ++i) 
      FloatArgUsedArr[i] = false;

    // go thru all the operands of LLVM instruction
    for( ; OpIt != CallI->op_end(); ++OpIt ) {

      // get the LR of call operand (parameter)
      LiveRange *const LR = LRI.getLiveRangeForValue((const Value *) *OpIt); 

      if ( !LR ) {
	cout << " Warning: In call instr, no LR for arg: " ;
	printValue(*OpIt);
	cout << endl;
	continue;
      }

      unsigned RegClassID = (LR->getRegClass())->getID();
      
      // if the arg is in int class - allocate a reg for an int arg
      if( RegClassID == IntRegClassID ) {
	
	if( intArgNo < NumOfIntArgRegs) {
	  setCallArgColor( LR, SparcIntRegOrder::o0 + intArgNo );
	}
	
	else {
	  // TODO: Insert push code here
	  assert( 0 && "Insert push code here!");

	  AddedInstrns * AI = AddedInstrMap[ CallMI ];
	  if( ! AI ) AI = new AddedInstrns();

	  // AI->InstrnsBefore.push_back( getStackPushInstr(LR) );
	  AddedInstrMap[ CallMI ] = AI;
	  
	}
	++intArgNo;
      }
      
      // if the arg is float/double 
      else if ( RegClassID == FloatRegClassID) {
	
	if( LR->getTypeID() == Type::DoubleTyID ) {
	  
	  // find the first reg # we can pass a double arg
	  for(unsigned i=0; i < NumOfFloatArgRegs; i+= 2) {
	    if ( !FloatArgUsedArr[i] && !FloatArgUsedArr[i+1] ) {
	      setCallArgColor(LR, SparcFloatRegOrder::f0 + i );	    	    
	      FloatArgUsedArr[i] = true;
	      FloatArgUsedArr[i+1] = true;
	      //if( DEBUG_RA) printReg( LR );
	      break;
	    }
	  }
	  if( ! LR->hasColor() ) { // if LR was not colored above
	    
	    assert(0 && "insert push code here for a double");
	    
	  }
	  
	}
	else if( LR->getTypeID() == Type::FloatTyID ) { 
	  
	  // find the first reg # we can pass a float arg
	  for(unsigned i=0; i < NumOfFloatArgRegs; ++i) {
	    if ( !FloatArgUsedArr[i] ) {
	      setCallArgColor(LR, SparcFloatRegOrder::f0 + i );
	      FloatArgUsedArr[i] = true;
	      // LR->setColor( SparcFloatRegOrder::f0 + i );
	      // if( DEBUG_RA) printReg( LR );
	      break;
	    }
	  }
	  if( ! LR->hasColor() ) { // if LR was not colored above
	    assert(0 && "insert push code here for a float");
	  }
	  
	}
	else 
	  assert(0 && "unknown float type in method arg");
	
      } // float register class
      
      else 
	assert(0 && "Unknown RegClassID");


    } // for each operand in a call instruction

    


  } // for all call instrctions in CallInstrList

}


void UltraSparcRegInfo::setCallArgColor(LiveRange *const LR, 
					const unsigned RegNo) const {

  // if no call interference and LR is NOT previously colored (e.g., as an 
  // incoming arg)
  if( ! LR->getNumOfCallInterferences() && ! LR->hasColor() ) { 
    // we can directly allocate a %o register
    LR->setColor( RegNo);
    if( DEBUG_RA) printReg( LR );
  }
  else {                        // there are call interferences
    
    /* 
    // insert a copy machine instr to copy from LR to %o(reg)
    PreMInstrMap[ CallMI ] = 
    getNewCopyMInstr( LR->,  SparcIntRegOrder::o0 + intArgNo );
    */
    cout << " $$$ TODO: Insert a copy for call argument!: " << endl;

    // We don't color LR here. It's colored as any other normal LR
  }

}





//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as MachineInstrInfo. 
// 
//---------------------------------------------------------------------------

UltraSparc::UltraSparc() : TargetMachine("UltraSparc-Native"),
			   InstSchedulingInfo(&InstInfo),
			   RegInfo( this )  {
  optSizeForSubWordData = 4;
  minMemOpWordSize = 8; 
  maxAtomicMemOpWordSize = 8;
  zeroRegNum = 0;			// %g0 always gives 0 on Sparc
}



//----------------------------------------------------------------------------
// Entry point for register allocation for a module
//----------------------------------------------------------------------------

void AllocateRegisters(Method *M, TargetMachine &TM)
{
 
  if ( (M)->isExternal() )     // don't process prototypes
    return;
    
  if( DEBUG_RA ) {
    cout << endl << "******************** Method "<< (M)->getName();
    cout <<        " ********************" <<endl;
  }
    
  MethodLiveVarInfo LVI(M );   // Analyze live varaibles
  LVI.analyze();
  
    
  PhyRegAlloc PRA(M, TM , &LVI); // allocate registers
  PRA.allocateRegisters();
    

  if( DEBUG_RA )  cout << endl << "Register allocation complete!" << endl;

}





bool UltraSparc::compileMethod(Method *M) {
  if (SelectInstructionsForMethod(M, *this)) {
    cerr << "Instruction selection failed for method " << M->getName()
       << "\n\n";
    return true;
  }
  
  if (ScheduleInstructionsWithSSA(M, *this, InstSchedulingInfo)) {
    cerr << "Instruction scheduling before allocation failed for method "
       << M->getName() << "\n\n";
    return true;
  }

  AllocateRegisters(M, *this);    // allocate registers


  return false;
}

