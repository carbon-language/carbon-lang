#include "llvm/CodeGen/PhyRegAlloc.h"


PhyRegAlloc::PhyRegAlloc(const Method *const M, 
			 const TargetMachine& tm, 
			 MethodLiveVarInfo *const Lvi) 
                        : RegClassList(),
			  Meth(M), TM(tm), LVI(Lvi), LRI(M, tm, RegClassList), 
			  MRI( tm.getRegInfo() ),
                          NumOfRegClasses(MRI.getNumOfRegClasses()),
			  CallInstrList(),
			  AddedInstrMap()

{
  // **TODO: use an actual reserved color list 
  ReservedColorListType *RCL = new ReservedColorListType();

  // create each RegisterClass and put in RegClassList
  for( unsigned int rc=0; rc < NumOfRegClasses; rc++)  
    RegClassList.push_back( new RegClass(M, MRI.getMachineRegClass(rc), RCL) );

}






void PhyRegAlloc::createIGNodeListsAndIGs()
{
  if(DEBUG_RA ) cout << "Creating LR lists ..." << endl;

  // hash map iterator
  LiveRangeMapType::const_iterator HMI = (LRI.getLiveRangeMap())->begin();   

  // hash map end
  LiveRangeMapType::const_iterator HMIEnd = (LRI.getLiveRangeMap())->end();   

    for(  ; HMI != HMIEnd ; ++HMI ) {

     LiveRange *L = (*HMI).second;      // get the LiveRange

     if( (*HMI).first ) { 
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




// Interence occurs only if the LR of Def (Inst or Arg) is of the same reg 
// class as that of live var. The live var passed to this function is the 
// LVset AFTER the instruction


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
      if( isCallInst ) 
	LROfVar->addCallInterference( (const Instruction *const) Def );   
      
    }
    else if(DEBUG_RA > 1)  { 
      // we will not have LRs for values not explicitly allocated in the
      // instruction stream (e.g., constants)
      cout << " warning: no live range for " ; 
      printValue( *LIt); cout << endl; }
    
  }
 
}



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

      // iterate over  MI operands to find defs
      for( MachineInstr::val_op_const_iterator OpI(MInst);!OpI.done(); ++OpI) {
	
	if( OpI.isDef() ) {     
	  // create a new LR iff this operand is a def
	  addInterference(*OpI, LVSetAI, isCallInst );

	} //if this is a def

      } // for all operands

    } // for all machine instructions in BB


    // go thru LLVM instructions in the basic block and record all CALL
    // instructions in the CallInstrList
    BasicBlock::const_iterator InstIt = (*BBI)->begin();

    for( ; InstIt != (*BBI)->end() ; ++ InstIt) {

      if( (*InstIt)->getOpcode() == Instruction::Call )
	CallInstrList.push_back( *InstIt );
   }
    
  } // for all BBs in method


  // add interferences for method arguments. Since there are no explict 
  // defs in method for args, we have to add them manually
          
  addInterferencesForArgs();            // add interference for method args

  if( DEBUG_RA)
    cout << "Interference graphs calculted!" << endl;

}




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



void PhyRegAlloc::updateMachineCode()
{

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    // get the iterator for machine instructions
    MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::iterator MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      MachineInstr *const MInst = *MInstIterator; 

      //for(MachineInstr::val_op_const_iterator OpI(MInst);!OpI.done();++OpI) {

      for(unsigned OpNum=0; OpNum < MInst->getNumOperands(); ++OpNum) {

	MachineOperand& Op = MInst->getOperand(OpNum);

	if( Op.getOperandType() ==  MachineOperand::MO_VirtualRegister || 
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister) {

	  const Value *const Val =  Op.getVRegValue();

	  // delete this condition checking later (must assert if Val is null)
	  if( !Val ) { 
	    cout << "Error: NULL Value found in instr." << endl;
	    continue;
	  }
	  assert( Val && "Value is NULL");   

	  const LiveRange *const LR = LRI.getLiveRangeForValue(Val);

	  if ( !LR ) {
	    if( ! ( (Val->getType())->isLabelType() || 
		    (Val->getValueType() == Value::ConstantVal) ) ) {
	      cout << "Warning: No LiveRange for: ";
	      printValue( Val); cout << endl;
	    }

	    //assert( LR && "No LR found for Value");
	    continue;
	  }
	
	  unsigned RCID = (LR->getRegClass())->getID();

	  Op.setRegForValue( MRI.getUnifiedRegNum(RCID, LR->getColor()) );

	  int RegNum = MRI.getUnifiedRegNum(RCID, LR->getColor());

	} 
      }

    }
  }
}




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
	    Op.getOperandType() ==  MachineOperand::MO_CCRegister) {

	  const Value *const Val =  Op.getVRegValue();

	  if( !Val ) { 
	    cout << "\t<*NULL Value*>";
	    continue;
	  }
	  assert( Val && "Value is NULL");

	  const LiveRange *const LR = LRI.getLiveRangeForValue(Val);

	  if ( !LR ) {

	    
	    if( ! ( (Val->getType())->isLabelType() || 
		    (Val->getValueType() == Value::ConstantVal) ) ) {
	      cout <<  "\t" << "<*No LiveRange for: ";
	      printValue( Val); cout << "*>";
	    }


	    //assert( LR && "No LR found for Value");
	    continue;
	  }
	
	  unsigned RCID = (LR->getRegClass())->getID();

	  //cout << "Setting reg for value: "; printValue( Val );
	  //cout << endl;

	  Op.setRegForValue( MRI.getUnifiedRegNum(RCID, LR->getColor()) );

	  int RegNum = MRI.getUnifiedRegNum(RCID, LR->getColor());

	  cout << "\t" << "%" << MRI.getUnifiedRegName( RegNum );

	}
	else if( Op.getOperandType() ==  MachineOperand:: MO_MachineRegister) {
	  cout << "\t" << "%"<< MRI.getUnifiedRegName( Op.getMachineRegNum() );
	}
	else if( Op.getOperandType() ==  MachineOperand::MO_PCRelativeDisp ) {
	  const Value *const Val = Op.getVRegValue () ;
	  if( !Val ) {
	    cout << "\t<*NULL Value*>";
	    continue;
	  }
	  if( (Val->getValueType() == Value::BasicBlockVal))
	    { cout << "\t"; printLabel(	Op.getVRegValue	() ); }
	  else { cout << "\t"; printValue( Val ); }
	}

	else 
	  cout << "\t" << Op;      // use dump field

      }

    }

    cout << endl;

  }

  cout << endl;
}

void PhyRegAlloc::printLabel(const Value *const Val)
{
  if( Val->hasName() )
    cout  << Val->getName();
  else
    cout << "Label" <<  Val;
}








void PhyRegAlloc::allocateRegisters()
{
  constructLiveRanges();                // create LR info

  if( DEBUG_RA)
    LRI.printLiveRanges();

  createIGNodeListsAndIGs();            // create IGNode list and IGs

  buildInterferenceGraphs();            // build IGs in all reg classes

  
  if( DEBUG_RA) {
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

  MRI.colorArgs(Meth, LRI);             // color method args
                                        // color call args of call instrns
  MRI.colorCallArgs(CallInstrList, LRI, AddedInstrMap); 

                                        // color all register classes
  for( unsigned int rc=0; rc < NumOfRegClasses ; rc++)  
    RegClassList[ rc ]->colorAllRegs();    

  //updateMachineCode(); 
  PrintMachineInstructions(Meth);
  printMachineCode();                   // only for DEBUGGING
}




