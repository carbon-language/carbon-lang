#include "llvm/CodeGen/LiveRangeInfo.h"

LiveRangeInfo::LiveRangeInfo(const Method *const M, 
			     const TargetMachine& tm,
			     vector<RegClass *> &RCL) 
                             : Meth(M), LiveRangeMap(), 
			       TM(tm), RegClassList(RCL)
{ }


// union two live ranges into one. The 2nd LR is deleted. Used for coalescing.
// Note: the caller must make sure that L1 and L2 are distinct

void LiveRangeInfo::unionAndUpdateLRs(LiveRange *const L1, LiveRange *L2)
{
  assert( L1 != L2);
  L1->setUnion( L2 );             // add elements of L2 to L1
  ValueSet::iterator L2It;

  for( L2It = L2->begin() ; L2It != L2->end(); ++L2It) {

    //assert(( L1->getTypeID() == L2->getTypeID()) && "Merge:Different types");

    L1->add( *L2It );            // add the var in L2 to L1
    LiveRangeMap[ *L2It ] = L1;  // now the elements in L2 should map to L1    
  }
  delete ( L2 );                 // delete L2 as it is no longer needed
}



                                 
void LiveRangeInfo::constructLiveRanges()
{  

  if( DEBUG_RA) 
    cout << "Consturcting Live Ranges ..." << endl;

  // first find the live ranges for all incoming args of the method since
  // those LRs start from the start of the method
      
                                                 // get the argument list
  const Method::ArgumentListType& ArgList = Meth->getArgumentList();           
                                                 // get an iterator to arg list
  Method::ArgumentListType::const_iterator ArgIt = ArgList.begin(); 

             
  for( ; ArgIt != ArgList.end() ; ++ArgIt) {     // for each argument

    LiveRange * ArgRange = new LiveRange();      // creates a new LR and 
    const Value *const Val = (const Value *) *ArgIt;

    assert( Val);

    ArgRange->add( Val );     // add the arg (def) to it
    LiveRangeMap[ Val ] = ArgRange;

    // create a temp machine op to find the register class of value
    //const MachineOperand Op(MachineOperand::MO_VirtualRegister);

    unsigned rcid = (TM.getRegInfo()).getRegClassIDOfValue( Val );
    ArgRange->setRegClass(RegClassList[ rcid ] );

    			   
    if( DEBUG_RA > 1) {     
      cout << " adding LiveRange for argument ";    
      printValue( (const Value *) *ArgIt); cout  << endl;
    }
  }


  // Now find all LRs for machine the instructions. A new LR will be created 
  // only for defs in the machine instr since, we assume that all Values are
  // defined before they are used. However, there can be multiple defs for
  // the same Value in machine instructions.

  Method::const_iterator BBI = Meth->begin();    // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {            // go thru BBs in random order

    // get the iterator for machine instructions
    const MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::const_iterator 
      MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); MInstIterator++) {  
      
      const MachineInstr * MInst = *MInstIterator; 
      
      // iterate over  MI operands to find defs
      for( MachineInstr::val_op_const_iterator OpI(MInst);!OpI.done(); OpI++) {
	
	// create a new LR iff this operand is a def
	if( OpI.isDef() ) {     
	  
	  const Value *const Def = *OpI;
	  LiveRange *DefRange = LiveRangeMap[Def]; 

	  // see LR already there (because of multiple defs)
	  
	  if( !DefRange) {                  // if it is not in LiveRangeMap
	    
	    DefRange = new LiveRange();     // creates a new live range and 
	    DefRange->add( Def );           // add the instruction (def) to it
	    LiveRangeMap[ Def ] = DefRange; // update the map

	    if( DEBUG_RA > 1) { 	    
	      cout << "  creating a LR for def: ";    
	      printValue(Def); cout  << endl;
	    }

	    // set the register class of the new live range
	    //assert( RegClassList.size() );
	    MachineOperand::MachineOperandType OpTy = 
	      OpI.getMachineOperand().getOperandType();

	    bool isCC = ( OpTy == MachineOperand::MO_CCRegister);
	    unsigned rcid = (TM.getRegInfo()).getRegClassIDOfValue( 
			    OpI.getMachineOperand().getVRegValue(), isCC );


	    if(isCC )
	      cout << "\a" << "**created a LR for a CC reg**" << cout;

	    DefRange->setRegClass( RegClassList[ rcid ] );

	  }
	  else {
	    DefRange->add( Def );           // add the opearand to def range
                                            // update the map - Operand points 
	                                    // to the merged set
	    LiveRangeMap[ Def ] = DefRange; 

	    if( DEBUG_RA > 1) { 
	      cout << "   added to an existing LR for def: ";  
	      printValue( Def ); cout  << endl;
	    }
	  }




	} // if isDef()
	
      } // for all opereands in machine instructions

    } // for all machine instructions in the BB

  } // for all BBs in method

  if( DEBUG_RA) 
    cout << "Initial Live Ranges constructed!" << endl;

}



void LiveRangeInfo::coalesceLRs()  
{

/* Algorithm:
   for each BB in method
     for each machine instruction (inst)
       for each definition (def) in inst
         for each operand (op) of inst that is a use
           if the def and op are of the same type
	     if the def and op do not interfere //i.e., not simultaneously live
	       if (degree(LR of def) + degree(LR of op)) <= # avail regs
		 merge2IGNodes(def, op) // i.e., merge 2 LRs 

*/

  if( DEBUG_RA) 
    cout << endl << "Coalscing LRs ..." << endl;

  Method::const_iterator BBI = Meth->begin();  // random iterator for BBs   

  for( ; BBI != Meth->end(); ++BBI) {          // traverse BBs in random order

    // get the iterator for machine instructions
    const MachineCodeForBasicBlock& MIVec = (*BBI)->getMachineInstrVec();
    MachineCodeForBasicBlock::const_iterator 
      MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      const MachineInstr * MInst = *MInstIterator; 

      if( DEBUG_RA > 1) {
	cout << " *Iterating over machine instr ";
	MInst->dump();
	cout << endl;
      }


      // iterate over  MI operands to find defs
      for(MachineInstr::val_op_const_iterator DefI(MInst);!DefI.done();++DefI){
	
	if( DefI.isDef() ) {            // iff this operand is a def

	  LiveRange *const LROfDef = getLiveRangeForValue( *DefI );
	  assert( LROfDef );
	  RegClass *const RCOfDef = LROfDef->getRegClass();

	  MachineInstr::val_op_const_iterator UseI(MInst);
	  for( ; !UseI.done(); ++UseI){ // for all uses

 	    LiveRange *const LROfUse = getLiveRangeForValue( *UseI );

	    if( ! LROfUse ) {           // if LR of use is not found

	      //don't warn about labels
	      if (!((*UseI)->getType())->isLabelType() && DEBUG_RA) {
		cout<<" !! Warning: No LR for use "; printValue(*UseI);
		cout << endl;
	      }
	      continue;                 // ignore and continue
	    }

	    if( LROfUse == LROfDef)     // nothing to merge if they are same
	      continue;

	    // RegClass *const RCOfUse = LROfUse->getRegClass();

	    //if( RCOfDef == RCOfUse ) {  // if the reg classes are the same


	    if( LROfUse->getTypeID() == LROfDef->getTypeID() ) { 

	      if( ! RCOfDef->getInterference(LROfDef, LROfUse) ) {

		unsigned CombinedDegree =
		  LROfDef->getUserIGNode()->getNumOfNeighbors() + 
		  LROfUse->getUserIGNode()->getNumOfNeighbors();

		if( CombinedDegree <= RCOfDef->getNumOfAvailRegs() ) {

		  RCOfDef->mergeIGNodesOfLRs(LROfDef, LROfUse);
		  unionAndUpdateLRs(LROfDef, LROfUse);

		} // if combined degree is less than # of regs

	      } // if def and use do not interfere

	    } // if reg classes are the same

	  } // for all uses

	} // if def

      } // for all defs

    } // for all machine instructions

  } // for all BBs

  if( DEBUG_RA) 
    cout << endl << "Coalscing Done!" << endl;

}





/*--------------------------- Debug code for printing ---------------*/


void LiveRangeInfo::printLiveRanges()
{
  LiveRangeMapType::iterator HMI = LiveRangeMap.begin();   // hash map iterator
  cout << endl << "Printing Live Ranges from Hash Map:" << endl;
  for( ; HMI != LiveRangeMap.end() ; HMI ++ ) {
    if( (*HMI).first ) {
      cout <<" "; printValue((*HMI).first);  cout  << "\t: "; 
      ((*HMI).second)->printSet(); cout << endl;
    }
  }
}


