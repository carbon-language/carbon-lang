#include "llvm/CodeGen/PhyRegAlloc.h"

cl::Enum<RegAllocDebugLevel_t> DEBUG_RA("dregalloc", cl::NoFlags,
  "enable register allocation debugging information",
  clEnumValN(RA_DEBUG_None   , "n", "disable debug output"),
  clEnumValN(RA_DEBUG_Normal , "y", "enable debug output"),
  clEnumValN(RA_DEBUG_Verbose, "v", "enable extra debug output"), 0);


//----------------------------------------------------------------------------
// Constructor: Init local composite objects and create register classes.
//----------------------------------------------------------------------------
PhyRegAlloc::PhyRegAlloc(const Method *const M, 
			 const TargetMachine& tm, 
			 MethodLiveVarInfo *const Lvi) 
                        : RegClassList(),
			  Meth(M), TM(tm), LVI(Lvi), LRI(M, tm, RegClassList), 
			  MRI( tm.getRegInfo() ),
                          NumOfRegClasses(MRI.getNumOfRegClasses()),
			  AddedInstrMap()

{
  // **TODO: use an actual reserved color list 
  ReservedColorListType *RCL = new ReservedColorListType();

  // create each RegisterClass and put in RegClassList
  for( unsigned int rc=0; rc < NumOfRegClasses; rc++)  
    RegClassList.push_back( new RegClass(M, MRI.getMachineRegClass(rc), RCL) );

}

//----------------------------------------------------------------------------
// This method initally creates interference graphs (one in each reg class)
// and IGNodeList (one in each IG). The actual nodes will be pushed later. 
//----------------------------------------------------------------------------

void PhyRegAlloc::createIGNodeListsAndIGs()
{
  if(DEBUG_RA ) cout << "Creating LR lists ..." << endl;

  // hash map iterator
  LiveRangeMapType::const_iterator HMI = (LRI.getLiveRangeMap())->begin();   

  // hash map end
  LiveRangeMapType::const_iterator HMIEnd = (LRI.getLiveRangeMap())->end();   

    for(  ; HMI != HMIEnd ; ++HMI ) {
      
      if( (*HMI).first ) { 

	LiveRange *L = (*HMI).second;      // get the LiveRange

	if( !L) { 
	  if( DEBUG_RA) {
	    cout << "\n*?!?Warning: Null liver range found for: ";
	    printValue( (*HMI).first) ; cout << endl;
	  }
	  continue;
	}
                                        // if the Value * is not null, and LR  
                                        // is not yet written to the IGNodeList
       if( !(L->getUserIGNode())  ) {  
	                           
	 RegClass *const RC =           // RegClass of first value in the LR
	   //RegClassList [MRI.getRegClassIDOfValue(*(L->begin()))];
	   RegClassList[ L->getRegClass()->getID() ];

	 RC-> addLRToIG( L );           // add this LR to an IG
       }
    }
  }

                                        // init RegClassList
  for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[ rc ]->createInterferenceGraph();

  if( DEBUG_RA)
    cout << "LRLists Created!" << endl;
}



//----------------------------------------------------------------------------
// This method will add all interferences at for a given instruction.
// Interence occurs only if the LR of Def (Inst or Arg) is of the same reg 
// class as that of live var. The live var passed to this function is the 
// LVset AFTER the instruction
//----------------------------------------------------------------------------

void PhyRegAlloc::addInterference(const Value *const Def, 
				  const LiveVarSet *const LVSet,
				  const bool isCallInst) {

  LiveVarSet::const_iterator LIt = LVSet->begin();

  // get the live range of instruction
  const LiveRange *const LROfDef = LRI.getLiveRangeForValue( Def );   

  IGNode *const IGNodeOfDef = LROfDef->getUserIGNode();
  assert( IGNodeOfDef );

  RegClass *const RCOfDef = LROfDef->getRegClass(); 

  // for each live var in live variable set
  for( ; LIt != LVSet->end(); ++LIt) {

    if( DEBUG_RA > 1) {
      cout << "< Def="; printValue(Def);     
      cout << ", Lvar=";  printValue( *LIt); cout  << "> ";
    }

    //  get the live range corresponding to live var
    LiveRange *const LROfVar = LRI.getLiveRangeForValue(*LIt );    

    // LROfVar can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if( LROfVar)   {  

      if(LROfDef == LROfVar)            // do not set interf for same LR
	continue;

      // if 2 reg classes are the same set interference
      if( RCOfDef == LROfVar->getRegClass() ){ 
	RCOfDef->setInterference( LROfDef, LROfVar);  

      }

      //the live range of this var interferes with this call
      if( isCallInst ) {
	LROfVar->addCallInterference( (const Instruction *const) Def );   
	// cout << "\n ++Added Call Interf to set:";
	//LROfVar->printSet();
      }
    }
    else if(DEBUG_RA > 1)  { 
      // we will not have LRs for values not explicitly allocated in the
      // instruction stream (e.g., constants)
      cout << " warning: no live range for " ; 
      printValue( *LIt); cout << endl; }
    
  }
 
}

//----------------------------------------------------------------------------
// This method will walk thru code and create interferences in the IG of
// each RegClass.
//----------------------------------------------------------------------------

void PhyRegAlloc::buildInterferenceGraphs()
{

  if(DEBUG_RA) cout << "Creating interference graphs ..." << endl;

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    // get the iterator for machine instructions
    const MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::const_iterator 
      MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  

      const MachineInstr *const MInst = *MInstIterator; 

      // get the LV set after the instruction
      const LiveVarSet *const LVSetAI = 
	LVI->getLiveVarSetAfterMInst(MInst, *BBI);
    
      const bool isCallInst = TM.getInstrInfo().isCall(MInst->getOpCode());

      // if( isCallInst) cout << "\n%%% Found call Inst:\n";

      // iterate over  MI operands to find defs
      for( MachineInstr::val_op_const_iterator OpI(MInst);!OpI.done(); ++OpI) {
	
	if( OpI.isDef() ) {     
	  // create a new LR iff this operand is a def
	  addInterference(*OpI, LVSetAI, isCallInst );

	} //if this is a def

      } // for all operands


      // Also add interference for any implicit definitions in a machine
      // instr (currently, only calls have this).

      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      if(  NumOfImpRefs > 0 ) {
	for(unsigned z=0; z < NumOfImpRefs; z++) 
	  if( MInst->implicitRefIsDefined(z) )
	    addInterference( MInst->getImplicitRef(z), LVSetAI, isCallInst );
      }

    } // for all machine instructions in BB


#if 0

    // go thru LLVM instructions in the basic block and record all CALL
    // instructions and Return instructions in the CallInstrList
    // This is done because since there are no reverse pointers in machine
    // instructions to find the llvm instruction, when we encounter a call
    // or a return whose args must be specailly colored (e.g., %o's for args)
    BasicBlock::const_iterator InstIt = (*BBI)->begin();

    for( ; InstIt != (*BBI)->end() ; ++ InstIt) {
      unsigned OpCode =  (*InstIt)->getOpcode();

      if( OpCode == Instruction::Call )
	CallInstrList.push_back( *InstIt );      

      else if( OpCode == Instruction::Ret )
	RetInstrList.push_back( *InstIt );
    }

#endif

    
  } // for all BBs in method


  // add interferences for method arguments. Since there are no explict 
  // defs in method for args, we have to add them manually
          
  addInterferencesForArgs();            // add interference for method args

  if( DEBUG_RA)
    cout << "Interference graphs calculted!" << endl;

}




//----------------------------------------------------------------------------
// This method will add interferences for incoming arguments to a method.
//----------------------------------------------------------------------------
void PhyRegAlloc::addInterferencesForArgs()
{
                                              // get the InSet of root BB
  const LiveVarSet *const InSet = LVI->getInSetOfBB( Meth->front() );  

                                              // get the argument list
  const Method::ArgumentListType& ArgList = Meth->getArgumentList();  

                                              // get an iterator to arg list
  Method::ArgumentListType::const_iterator ArgIt = ArgList.begin();          


  for( ; ArgIt != ArgList.end() ; ++ArgIt) {  // for each argument
    addInterference( *ArgIt, InSet, false );  // add interferences between 
                                              // args and LVars at start
    if( DEBUG_RA > 1) {
       cout << " - %% adding interference for  argument ";    
      printValue( (const Value *) *ArgIt); cout  << endl;
    }
  }
}



//----------------------------------------------------------------------------
// This method inserts caller saving/restoring instructons before/after
// a call machine instruction.
//----------------------------------------------------------------------------


void PhyRegAlloc::insertCallerSavingCode(const MachineInstr *MInst, 
					 const BasicBlock *BB  ) 
{
  assert( (TM.getInstrInfo()).isCall( MInst->getOpCode() ) );

  int StackOff = -8;  // ****TODO : Change
  hash_set<unsigned> PushedRegSet;

  // Now find the LR of the return value of the call
  // The last *implicit operand* is the return value of a call
  // Insert it to to he PushedRegSet since we must not save that register
  // and restore it after the call.
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but we must not save/restore it.

  unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
  if(  NumOfImpRefs > 0 ) {
    
    if(  MInst->implicitRefIsDefined(NumOfImpRefs-1) ) {

      const Value *RetVal = MInst->getImplicitRef(NumOfImpRefs-1); 
      LiveRange *RetValLR = LRI.getLiveRangeForValue( RetVal );
      assert( RetValLR && "No LR for RetValue of call");

      PushedRegSet.insert(  
	MRI.getUnifiedRegNum((RetValLR->getRegClass())->getID(), 
			     RetValLR->getColor() ) );
    }

  }


  const LiveVarSet *LVSetAft =  LVI->getLiveVarSetAfterMInst(MInst, BB);

  LiveVarSet::const_iterator LIt = LVSetAft->begin();

  // for each live var in live variable set after machine inst
  for( ; LIt != LVSetAft->end(); ++LIt) {

   //  get the live range corresponding to live var
    LiveRange *const LR = LRI.getLiveRangeForValue(*LIt );    

    // LROfVar can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if( LR )   {  
      
      if( LR->hasColor() ) {

	unsigned RCID = (LR->getRegClass())->getID();
	unsigned Color = LR->getColor();

	if ( MRI.isRegVolatile(RCID, Color) ) {

	  // if the value is in both LV sets (i.e., live before and after 
	  // the call machine instruction)

	  unsigned Reg =   MRI.getUnifiedRegNum(RCID, Color);
	  
	  if( PushedRegSet.find(Reg) == PushedRegSet.end() ) {
	    
	    // if we haven't already pushed that register

	    unsigned RegType = MRI.getRegType( LR );

	    // Now get two instructions - to push on stack and pop from stack
	    // and add them to InstrnsBefore and InstrnsAfter of the
	    // call instruction
	    
	    MachineInstr *AdIBef = 
	      MRI.cpReg2MemMI(Reg, MRI.getFramePointer(), StackOff, RegType ); 

	    MachineInstr *AdIAft = 
	      MRI.cpMem2RegMI(MRI.getFramePointer(), StackOff, Reg, RegType ); 

	    ((AddedInstrMap[MInst])->InstrnsBefore).push_front(AdIBef);
	    ((AddedInstrMap[MInst])->InstrnsAfter).push_back(AdIAft);
	    
	    PushedRegSet.insert( Reg );
	    StackOff -= 8; // ****TODO: Correct ??????

	    if(DEBUG_RA) {
	      cout << "For callee save call inst:" << *MInst  << endl;
	      cerr << "\n  -inserted caller saving instrs:\n\t ";
	      cerr << *AdIBef << "\n\t" << *AdIAft  ;
	    }	    
	  } // if not already pushed

	} // if LR has a volatile color
	
      } // if LR has color

    } // if there is a LR for Var
    
  } // for each value in the LV set after instruction
  
}


//----------------------------------------------------------------------------
// This method is called after register allocation is complete to set the
// allocated reisters in the machine code. This code will add register numbers
// to MachineOperands that contain a Value.
//----------------------------------------------------------------------------

void PhyRegAlloc::updateMachineCode()
{

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    // get the iterator for machine instructions
    MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::iterator MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      MachineInstr *MInst = *MInstIterator; 

      // if this machine instr is call, insert caller saving code

      if( (TM.getInstrInfo()).isCall( MInst->getOpCode()) )
	insertCallerSavingCode(MInst,  *BBI );

      // If there are instructions to be added, *before* this machine
      // instruction, add them now.
      
      if( AddedInstrMap[ MInst ] ) {

	deque<MachineInstr *> &IBef = (AddedInstrMap[MInst])->InstrnsBefore;

	if( ! IBef.empty() ) {

	  deque<MachineInstr *>::iterator AdIt; 

	  for( AdIt = IBef.begin(); AdIt != IBef.end() ; ++AdIt ) {

	    if( DEBUG_RA)
	      cerr << " *$* PREPENDed instr " << *AdIt << endl;
	  	    
	    MInstIterator = MIVec.insert( MInstIterator, *AdIt );
	    ++MInstIterator;
	  }

	}

      }



      //for(MachineInstr::val_op_const_iterator OpI(MInst);!OpI.done();++OpI) {

      for(unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {

	MachineOperand& Op = MInst->getOperand(OpNum);

	if( Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister) {

	  const Value *const Val =  Op.getVRegValue();

	  // delete this condition checking later (must assert if Val is null)
	  if( !Val) {
            if (DEBUG_RA)
              cout << "Warning: NULL Value found for operand" << endl;
	    continue;
	  }
	  assert( Val && "Value is NULL");   

	  const LiveRange *const LR = LRI.getLiveRangeForValue(Val);

	  if ( !LR ) {

	    // nothing to worry if it's a const or a label

            if (DEBUG_RA) {
              cout << "*NO LR for operand : " << Op ;
	      cout << " [reg:" <<  Op.getAllocatedRegNum() << "]";
	      cout << " in inst:\t" << *MInst << endl;
            }

	    // if register is not allocated, mark register as invalid
	    if( Op.getAllocatedRegNum() == -1)
	      Op.setRegForValue( MRI.getInvalidRegNum()); 
	    
#if 0
	    if(  ((Val->getType())->isLabelType()) || 
		 (Val->getValueType() == Value::ConstantVal)  )
	      ;                         // do nothing
	    
	    // The return address is not explicitly defined within a
	    // method. So, it is not colored by usual algorithm. In that case
	    // color it here.
	    
	    //else if (TM.getInstrInfo().isCall(MInst->getOpCode())) 
	    //Op.setRegForValue( MRI.getCallAddressReg() );

	    //TM.getInstrInfo().isReturn(MInst->getOpCode())
	    else if(TM.getInstrInfo().isReturn(MInst->getOpCode()) ) {
	      if (DEBUG_RA) cout << endl << "RETURN found" << endl;
 	      Op.setRegForValue( MRI.getReturnAddressReg() );

	    }

	    if (Val->getValueType() == Value::InstructionVal)
	    {
	      if( DEBUG_RA ) {
		cout << "!Warning: No LiveRange for: ";
		printValue( Val); cout << " Type: " << Val->getValueType();
		cout << " RegVal=" <<  Op.getAllocatedRegNum() << endl;
	      }
	    }

#endif

	    continue;
	  }
	
	  unsigned RCID = (LR->getRegClass())->getID();

	  Op.setRegForValue( MRI.getUnifiedRegNum(RCID, LR->getColor()) );

	  int RegNum = MRI.getUnifiedRegNum(RCID, LR->getColor());

	}

      } // for each operand


      // If there are instructions to be added *after* this machine
      // instruction, add them now
      
      if( AddedInstrMap[ MInst ] ) {

	deque<MachineInstr *> &IAft = (AddedInstrMap[MInst])->InstrnsAfter;

	if( ! IAft.empty() ) {

	  deque<MachineInstr *>::iterator AdIt; 

	  ++MInstIterator;   // advance to the next instruction

	  for( AdIt = IAft.begin(); AdIt != IAft.end() ; ++AdIt ) {

	    if(DEBUG_RA) 
	      cerr << " *#* APPENDed instr opcode: "  << *AdIt << endl;

	    MInstIterator = MIVec.insert( MInstIterator, *AdIt );
	    ++MInstIterator;
	  }

	  // MInsterator already points to the next instr. Since the
	  // for loop also increments it, decrement it to point to the
	  // instruction added last
	  --MInstIterator;  

	}

      }

    } // for each machine instruction
  }
}




//----------------------------------------------------------------------------
// This method prints the code with registers after register allocation is
// complete.
//----------------------------------------------------------------------------
void PhyRegAlloc::printMachineCode()
{

  cout << endl << ";************** Method ";
  cout << Meth->getName() << " *****************" << endl;

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    cout << endl ; printLabel( *BBI); cout << ": ";

    // get the iterator for machine instructions
    MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::iterator MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      MachineInstr *const MInst = *MInstIterator; 


      cout << endl << "\t";
      cout << TargetInstrDescriptors[MInst->getOpCode()].opCodeString;
      

      //for(MachineInstr::val_op_const_iterator OpI(MInst);!OpI.done();++OpI) {

      for(unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {

	MachineOperand& Op = MInst->getOperand(OpNum);

	if( Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister /*|| 
	    Op.getOperandType() ==  MachineOperand::MO_PCRelativeDisp*/ ) {

	  const Value *const Val = Op.getVRegValue () ;
	  // ****this code is temporary till NULL Values are fixed
	  if( ! Val ) {
	    cout << "\t<*NULL*>";
	    continue;
	  }

	  // if a label or a constant
	  if( (Val->getValueType() == Value::BasicBlockVal)  ) {

	    cout << "\t"; printLabel(	Op.getVRegValue	() );
	  }
	  else {
	    // else it must be a register value
	    const int RegNum = Op.getAllocatedRegNum();

	    //if( RegNum != 1000)

	      cout << "\t" << "%" << MRI.getUnifiedRegName( RegNum );
	    // else cout << "\t<*NoReg*>";

	  }

	} 
	else if(Op.getOperandType() ==  MachineOperand::MO_MachineRegister) {
	  cout << "\t" << "%" << MRI.getUnifiedRegName(Op.getMachineRegNum());
	}

	else 
	  cout << "\t" << Op;      // use dump field
      }

    

      unsigned NumOfImpRefs =  MInst->getNumImplicitRefs();
      if(  NumOfImpRefs > 0 ) {
	
	cout << "\tImplicit:";

	for(unsigned z=0; z < NumOfImpRefs; z++) {
	  printValue(  MInst->getImplicitRef(z) );
	  cout << "\t";
	}
	
      }

    } // for all machine instructions


    cout << endl;

  } // for all BBs

  cout << endl;
}


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

    if( (TM.getInstrInfo()).isCall( OpCode ) )
      MRI.colorCallArgs( CRMI, LRI, AI );
    
    else if (  (TM.getInstrInfo()).isReturn(OpCode) ) 
      MRI.colorRetValue( CRMI, LRI, AI );
    
    else assert( 0 && "Non Call/Ret instrn in CallRetInstrList\n" );

  }

}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void PhyRegAlloc::colorIncomingArgs()
{
  const BasicBlock *const FirstBB = Meth->front();
  const MachineInstr *FirstMI = *((FirstBB->getMachineInstrVec()).begin());
  assert( FirstMI && "No machine instruction in entry BB");

  AddedInstrns *AI = AddedInstrMap[ FirstMI ];
  if ( !AI ) { 
    AI = new AddedInstrns();
    AddedInstrMap[ FirstMI  ] = AI;
  }

  MRI.colorMethodArgs(Meth, LRI, AI );
}


//----------------------------------------------------------------------------
// Used to generate a label for a basic block
//----------------------------------------------------------------------------
void PhyRegAlloc::printLabel(const Value *const Val)
{
  if( Val->hasName() )
    cout  << Val->getName();
  else
    cout << "Label" <<  Val;
}


//----------------------------------------------------------------------------
// The entry pont to Register Allocation
//----------------------------------------------------------------------------

void PhyRegAlloc::allocateRegisters()
{

  // make sure that we put all register classes into the RegClassList 
  // before we call constructLiveRanges (now done in the constructor of 
  // PhyRegAlloc class).

  constructLiveRanges();                // create LR info

  if( DEBUG_RA )
    LRI.printLiveRanges();
  
  createIGNodeListsAndIGs();            // create IGNode list and IGs

  buildInterferenceGraphs();            // build IGs in all reg classes
  
  
  if( DEBUG_RA ) {
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

  // color all register classes
  for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[ rc ]->colorAllRegs();    


  // color incoming args and call args
  colorIncomingArgs();
  colorCallRetArgs();

 
  updateMachineCode(); 
  if (DEBUG_RA) {
    PrintMachineInstructions(Meth);
    printMachineCode();                   // only for DEBUGGING
  }
}




