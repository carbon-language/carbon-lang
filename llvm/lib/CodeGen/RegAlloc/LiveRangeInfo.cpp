//===-- LiveRangeInfo.cpp -------------------------------------------------===//
// 
//  Live range construction for coloring-based register allocation for LLVM.
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveRangeInfo.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegClass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "Support/SetOperations.h"
using std::cerr;

LiveRangeInfo::LiveRangeInfo(const Function *F, const TargetMachine &tm,
			     std::vector<RegClass *> &RCL)
  : Meth(F), TM(tm), RegClassList(RCL), MRI(tm.getRegInfo()) { }


LiveRangeInfo::~LiveRangeInfo() {
  for (LiveRangeMapType::iterator MI = LiveRangeMap.begin(); 
       MI != LiveRangeMap.end(); ++MI) {  

    if (MI->first && MI->second) {
      LiveRange *LR = MI->second;

      // we need to be careful in deleting LiveRanges in LiveRangeMap
      // since two/more Values in the live range map can point to the same
      // live range. We have to make the other entries NULL when we delete
      // a live range.

      for(LiveRange::iterator LI = LR->begin(); LI != LR->end(); ++LI)
        LiveRangeMap[*LI] = 0;
      
      delete LR;
    }
  }
}


//---------------------------------------------------------------------------
// union two live ranges into one. The 2nd LR is deleted. Used for coalescing.
// Note: the caller must make sure that L1 and L2 are distinct and both
// LRs don't have suggested colors
//---------------------------------------------------------------------------

void LiveRangeInfo::unionAndUpdateLRs(LiveRange *L1, LiveRange *L2) {
  assert(L1 != L2 && (!L1->hasSuggestedColor() || !L2->hasSuggestedColor()));
  set_union(*L1, *L2);                   // add elements of L2 to L1

  for(ValueSet::iterator L2It = L2->begin(); L2It != L2->end(); ++L2It) {
    //assert(( L1->getTypeID() == L2->getTypeID()) && "Merge:Different types");

    L1->insert(*L2It);                  // add the var in L2 to L1
    LiveRangeMap[*L2It] = L1;           // now the elements in L2 should map 
                                        //to L1    
  }


  // Now if LROfDef(L1) has a suggested color, it will remain.
  // But, if LROfUse(L2) has a suggested color, the new range
  // must have the same color.

  if(L2->hasSuggestedColor())
    L1->setSuggestedColor(L2->getSuggestedColor());


  if (L2->isCallInterference())
    L1->setCallInterference();
  
  // add the spill costs
  L1->addSpillCost(L2->getSpillCost());
  
  delete L2;                        // delete L2 as it is no longer needed
}



//---------------------------------------------------------------------------
// Method for constructing all live ranges in a function. It creates live 
// ranges for all values defined in the instruction stream. Also, it
// creates live ranges for all incoming arguments of the function.
//---------------------------------------------------------------------------
void LiveRangeInfo::constructLiveRanges() {  

  if (DEBUG_RA >= RA_DEBUG_LiveRanges) 
    cerr << "Constructing Live Ranges ...\n";

  // first find the live ranges for all incoming args of the function since
  // those LRs start from the start of the function
  for (Function::const_aiterator AI = Meth->abegin(); AI != Meth->aend(); ++AI){
    LiveRange *ArgRange = new LiveRange();      // creates a new LR and 
    ArgRange->insert(AI);     // add the arg (def) to it
    LiveRangeMap[AI] = ArgRange;

    // create a temp machine op to find the register class of value
    //const MachineOperand Op(MachineOperand::MO_VirtualRegister);

    unsigned rcid = MRI.getRegClassIDOfValue(AI);
    ArgRange->setRegClass(RegClassList[rcid]);

    			   
    if( DEBUG_RA >= RA_DEBUG_LiveRanges)
      cerr << " Adding LiveRange for argument " << RAV(AI) << "\n";
  }

  // Now suggest hardware registers for these function args 
  MRI.suggestRegs4MethodArgs(Meth, *this);


  // Now find speical LLVM instructions (CALL, RET) and LRs in machine
  // instructions.
  //
  for (Function::const_iterator BBI=Meth->begin(); BBI != Meth->end(); ++BBI){
    // Now find all LRs for machine the instructions. A new LR will be created 
    // only for defs in the machine instr since, we assume that all Values are
    // defined before they are used. However, there can be multiple defs for
    // the same Value in machine instructions.

    // get the iterator for machine instructions
    MachineCodeForBasicBlock& MIVec = MachineCodeForBasicBlock::get(BBI);
    
    // iterate over all the machine instructions in BB
    for(MachineCodeForBasicBlock::iterator MInstIterator = MIVec.begin();
        MInstIterator != MIVec.end(); ++MInstIterator) {  
      MachineInstr *MInst = *MInstIterator; 

      // Now if the machine instruction is a  call/return instruction,
      // add it to CallRetInstrList for processing its implicit operands

      if(TM.getInstrInfo().isReturn(MInst->getOpCode()) ||
	 TM.getInstrInfo().isCall(MInst->getOpCode()))
	CallRetInstrList.push_back( MInst ); 
 
             
      // iterate over  MI operands to find defs
      for (MachineInstr::val_op_iterator OpI = MInst->begin(),
             OpE = MInst->end(); OpI != OpE; ++OpI) {
	if(DEBUG_RA >= RA_DEBUG_LiveRanges) {
	  MachineOperand::MachineOperandType OpTyp = 
	    OpI.getMachineOperand().getOperandType();

	  if (OpTyp == MachineOperand::MO_CCRegister)
	    cerr << "\n**CC reg found. Is Def=" << OpI.isDef() << " Val:"
                 << RAV(OpI.getMachineOperand().getVRegValue()) << "\n";
	}

	// create a new LR iff this operand is a def
	if (OpI.isDef()) {     
	  const Value *Def = *OpI;

	  // Only instruction values are accepted for live ranges here
	  if (Def->getValueType() != Value::InstructionVal ) {
	    cerr << "\n**%%Error: Def is not an instruction val. Def="
                 << RAV(Def) << "\n";
	    continue;
	  }

	  LiveRange *DefRange = LiveRangeMap[Def]; 

	  // see LR already there (because of multiple defs)
	  if( !DefRange) {                  // if it is not in LiveRangeMap
	    DefRange = new LiveRange();     // creates a new live range and 
	    DefRange->insert(Def);          // add the instruction (def) to it
	    LiveRangeMap[ Def ] = DefRange; // update the map

	    if (DEBUG_RA >= RA_DEBUG_LiveRanges)
	      cerr << "  creating a LR for def: " << RAV(Def) << "\n";

	    // set the register class of the new live range
	    //assert( RegClassList.size() );
	    MachineOperand::MachineOperandType OpTy = 
	      OpI.getMachineOperand().getOperandType();

	    bool isCC = ( OpTy == MachineOperand::MO_CCRegister);
	    unsigned rcid = MRI.getRegClassIDOfValue( 
			    OpI.getMachineOperand().getVRegValue(), isCC );


	    if (isCC && DEBUG_RA >= RA_DEBUG_LiveRanges)
	      cerr  << "\a**created a LR for a CC reg:"
                    << RAV(OpI.getMachineOperand().getVRegValue());

	    DefRange->setRegClass(RegClassList[rcid]);
	  } else {
	    DefRange->insert(Def);          // add the opearand to def range
                                            // update the map - Operand points 
	                                    // to the merged set
	    LiveRangeMap[Def] = DefRange; 

	    if (DEBUG_RA >= RA_DEBUG_LiveRanges)
	      cerr << "   Added to existing LR for def: " << RAV(Def) << "\n";
	  }

	} // if isDef()
	
      } // for all opereands in machine instructions

    } // for all machine instructions in the BB

  } // for all BBs in function
  

  // Now we have to suggest clors for call and return arg live ranges.
  // Also, if there are implicit defs (e.g., retun value of a call inst)
  // they must be added to the live range list

  suggestRegs4CallRets();

  if( DEBUG_RA >= RA_DEBUG_LiveRanges) 
    cerr << "Initial Live Ranges constructed!\n";

}


//---------------------------------------------------------------------------
// If some live ranges must be colored with specific hardware registers
// (e.g., for outgoing call args), suggesting of colors for such live
// ranges is done using target specific function. Those functions are called
// from this function. The target specific methods must:
//    1) suggest colors for call and return args. 
//    2) create new LRs for implicit defs in machine instructions
//---------------------------------------------------------------------------
void LiveRangeInfo::suggestRegs4CallRets()
{
  CallRetInstrListType::iterator It =  CallRetInstrList.begin();
  for( ; It !=  CallRetInstrList.end(); ++It ) {

    MachineInstr *MInst = *It;
    MachineOpCode OpCode =  MInst->getOpCode();

    if( (TM.getInstrInfo()).isReturn(OpCode)  )
      MRI.suggestReg4RetValue( MInst, *this);

    else if( (TM.getInstrInfo()).isCall( OpCode ) )
      MRI.suggestRegs4CallArgs( MInst, *this, RegClassList );
    
    else 
      assert( 0 && "Non call/ret instr in  CallRetInstrList" );
  }

}


//--------------------------------------------------------------------------
// The following method coalesces live ranges when possible. This method
// must be called after the interference graph has been constructed.


/* Algorithm:
   for each BB in function
     for each machine instruction (inst)
       for each definition (def) in inst
         for each operand (op) of inst that is a use
           if the def and op are of the same register type
	     if the def and op do not interfere //i.e., not simultaneously live
	       if (degree(LR of def) + degree(LR of op)) <= # avail regs
	         if both LRs do not have suggested colors
		    merge2IGNodes(def, op) // i.e., merge 2 LRs 

*/
//---------------------------------------------------------------------------
void LiveRangeInfo::coalesceLRs()  
{
  if(DEBUG_RA >= RA_DEBUG_LiveRanges) 
    cerr << "\nCoalescing LRs ...\n";

  for(Function::const_iterator BBI = Meth->begin(), BBE = Meth->end();
      BBI != BBE; ++BBI) {

    // get the iterator for machine instructions
    const MachineCodeForBasicBlock& MIVec = MachineCodeForBasicBlock::get(BBI);
    MachineCodeForBasicBlock::const_iterator MInstIterator = MIVec.begin();

    // iterate over all the machine instructions in BB
    for( ; MInstIterator != MIVec.end(); ++MInstIterator) {  
      
      const MachineInstr * MInst = *MInstIterator; 

      if( DEBUG_RA >= RA_DEBUG_LiveRanges) {
	cerr << " *Iterating over machine instr ";
	MInst->dump();
	cerr << "\n";
      }


      // iterate over  MI operands to find defs
      for(MachineInstr::const_val_op_iterator DefI = MInst->begin(),
            DefE = MInst->end(); DefI != DefE; ++DefI) {
	if (DefI.isDef()) {            // iff this operand is a def
	  LiveRange *LROfDef = getLiveRangeForValue( *DefI );
	  RegClass *RCOfDef = LROfDef->getRegClass();

	  MachineInstr::const_val_op_iterator UseI = MInst->begin(),
            UseE = MInst->end();
	  for( ; UseI != UseE; ++UseI){ // for all uses

 	    LiveRange *LROfUse = getLiveRangeForValue( *UseI );
	    if (!LROfUse) {             // if LR of use is not found
	      //don't warn about labels
	      if (!isa<BasicBlock>(*UseI) && DEBUG_RA >= RA_DEBUG_LiveRanges)
		cerr << " !! Warning: No LR for use " << RAV(*UseI) << "\n";
	      continue;                 // ignore and continue
	    }

	    if (LROfUse == LROfDef)     // nothing to merge if they are same
	      continue;

	    if (MRI.getRegType(LROfDef) == MRI.getRegType(LROfUse)) {

	      // If the two RegTypes are the same
	      if (!RCOfDef->getInterference(LROfDef, LROfUse) ) {

		unsigned CombinedDegree =
		  LROfDef->getUserIGNode()->getNumOfNeighbors() + 
		  LROfUse->getUserIGNode()->getNumOfNeighbors();

                if (CombinedDegree > RCOfDef->getNumOfAvailRegs()) {
                  // get more precise estimate of combined degree
                  CombinedDegree = LROfDef->getUserIGNode()->
                    getCombinedDegree(LROfUse->getUserIGNode());
                }

		if (CombinedDegree <= RCOfDef->getNumOfAvailRegs()) {
		  // if both LRs do not have suggested colors
		  if (!(LROfDef->hasSuggestedColor() &&  
                        LROfUse->hasSuggestedColor())) {
		    
		    RCOfDef->mergeIGNodesOfLRs(LROfDef, LROfUse);
		    unionAndUpdateLRs(LROfDef, LROfUse);
		  }

		} // if combined degree is less than # of regs
	      } // if def and use do not interfere
	    }// if reg classes are the same
	  } // for all uses
	} // if def
      } // for all defs
    } // for all machine instructions
  } // for all BBs

  if (DEBUG_RA >= RA_DEBUG_LiveRanges) 
    cerr << "\nCoalescing Done!\n";
}





/*--------------------------- Debug code for printing ---------------*/


void LiveRangeInfo::printLiveRanges() {
  LiveRangeMapType::iterator HMI = LiveRangeMap.begin();   // hash map iterator
  cerr << "\nPrinting Live Ranges from Hash Map:\n";
  for( ; HMI != LiveRangeMap.end(); ++HMI) {
    if (HMI->first && HMI->second) {
      cerr << " Value* " << RAV(HMI->first) << "\t: "; 
      if (IGNode* igNode = HMI->second->getUserIGNode())
        cerr << "LR# " << igNode->getIndex();
      else
        cerr << "LR# " << "<no-IGNode>";
      cerr << "\t:Values = "; printSet(*HMI->second); cerr << "\n";
    }
  }
}
