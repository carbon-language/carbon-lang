//===-- LiveRangeInfo.cpp -------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//  Live range construction for coloring-based register allocation for LLVM.
// 
//===----------------------------------------------------------------------===//

#include "IGNode.h"
#include "LiveRangeInfo.h"
#include "RegAllocCommon.h"
#include "RegClass.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegInfo.h"
#include "Support/SetOperations.h"

namespace llvm {

unsigned LiveRange::getRegClassID() const { return getRegClass()->getID(); }

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

      for (LiveRange::iterator LI = LR->begin(); LI != LR->end(); ++LI)
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
  assert(! (L1->hasColor() && L2->hasColor()) ||
         L1->getColor() == L2->getColor());

  set_union(*L1, *L2);                   // add elements of L2 to L1

  for(ValueSet::iterator L2It = L2->begin(); L2It != L2->end(); ++L2It) {
    //assert(( L1->getTypeID() == L2->getTypeID()) && "Merge:Different types");

    L1->insert(*L2It);                  // add the var in L2 to L1
    LiveRangeMap[*L2It] = L1;           // now the elements in L2 should map 
                                        //to L1    
  }
  
  // set call interference for L1 from L2
  if (L2->isCallInterference())
    L1->setCallInterference();
  
  // add the spill costs
  L1->addSpillCost(L2->getSpillCost());

  // If L2 has a color, give L1 that color.  Note that L1 may have had the same
  // color or none, but would not have a different color as asserted above.
  if (L2->hasColor())
    L1->setColor(L2->getColor());

  // Similarly, if LROfUse(L2) has a suggested color, the new range
  // must have the same color.
  if (L2->hasSuggestedColor())
    L1->setSuggestedColor(L2->getSuggestedColor());
  
  delete L2;                        // delete L2 as it is no longer needed
}


//---------------------------------------------------------------------------
// Method for creating a single live range for a definition.
// The definition must be represented by a virtual register (a Value).
// Note: this function does *not* check that no live range exists for def.
//---------------------------------------------------------------------------

LiveRange*
LiveRangeInfo::createNewLiveRange(const Value* Def, bool isCC /* = false*/)
{  
  LiveRange* DefRange = new LiveRange();  // Create a new live range,
  DefRange->insert(Def);                  // add Def to it,
  LiveRangeMap[Def] = DefRange;           // and update the map.

  // set the register class of the new live range
  DefRange->setRegClass(RegClassList[MRI.getRegClassIDOfType(Def->getType(),
                                                             isCC)]);

  if (DEBUG_RA >= RA_DEBUG_LiveRanges) {
    std::cerr << "  Creating a LR for def ";
    if (isCC) std::cerr << " (CC Register!)";
    std::cerr << " : " << RAV(Def) << "\n";
  }
  return DefRange;
}


LiveRange*
LiveRangeInfo::createOrAddToLiveRange(const Value* Def, bool isCC /* = false*/)
{  
  LiveRange *DefRange = LiveRangeMap[Def];

  // check if the LR is already there (because of multiple defs)
  if (!DefRange) { 
    DefRange = createNewLiveRange(Def, isCC);
  } else {                          // live range already exists
    DefRange->insert(Def);          // add the operand to the range
    LiveRangeMap[Def] = DefRange;   // make operand point to merged set
    if (DEBUG_RA >= RA_DEBUG_LiveRanges)
      std::cerr << "   Added to existing LR for def: " << RAV(Def) << "\n";
  }
  return DefRange;
}


//---------------------------------------------------------------------------
// Method for constructing all live ranges in a function. It creates live 
// ranges for all values defined in the instruction stream. Also, it
// creates live ranges for all incoming arguments of the function.
//---------------------------------------------------------------------------
void LiveRangeInfo::constructLiveRanges() {  

  if (DEBUG_RA >= RA_DEBUG_LiveRanges) 
    std::cerr << "Constructing Live Ranges ...\n";

  // first find the live ranges for all incoming args of the function since
  // those LRs start from the start of the function
  for (Function::const_aiterator AI = Meth->abegin(); AI != Meth->aend(); ++AI)
    createNewLiveRange(AI, /*isCC*/ false);

  // Now suggest hardware registers for these function args 
  MRI.suggestRegs4MethodArgs(Meth, *this);

  // Now create LRs for machine instructions.  A new LR will be created 
  // only for defs in the machine instr since, we assume that all Values are
  // defined before they are used. However, there can be multiple defs for
  // the same Value in machine instructions.
  // 
  // Also, find CALL and RETURN instructions, which need extra work.
  //
  MachineFunction &MF = MachineFunction::get(Meth);
  for (MachineFunction::iterator BBI = MF.begin(); BBI != MF.end(); ++BBI) {
    MachineBasicBlock &MBB = *BBI;

    // iterate over all the machine instructions in BB
    for(MachineBasicBlock::iterator MInstIterator = MBB.begin();
        MInstIterator != MBB.end(); ++MInstIterator) {  
      MachineInstr *MInst = MInstIterator; 

      // If the machine instruction is a  call/return instruction, add it to
      // CallRetInstrList for processing its args, ret value, and ret addr.
      // 
      if(TM.getInstrInfo().isReturn(MInst->getOpcode()) ||
	 TM.getInstrInfo().isCall(MInst->getOpcode()))
	CallRetInstrList.push_back(MInst); 
 
      // iterate over explicit MI operands and create a new LR
      // for each operand that is defined by the instruction
      for (MachineInstr::val_op_iterator OpI = MInst->begin(),
             OpE = MInst->end(); OpI != OpE; ++OpI)
	if (OpI.isDef()) {     
	  const Value *Def = *OpI;
          bool isCC = (OpI.getMachineOperand().getType()
                       == MachineOperand::MO_CCRegister);
          LiveRange* LR = createOrAddToLiveRange(Def, isCC);

          // If the operand has a pre-assigned register,
          // set it directly in the LiveRange
          if (OpI.getMachineOperand().hasAllocatedReg()) {
            unsigned getClassId;
            LR->setColor(MRI.getClassRegNum(
                                OpI.getMachineOperand().getAllocatedRegNum(),
                                getClassId));
          }
	}

      // iterate over implicit MI operands and create a new LR
      // for each operand that is defined by the instruction
      for (unsigned i = 0; i < MInst->getNumImplicitRefs(); ++i) 
	if (MInst->getImplicitOp(i).isDef()) {
	  const Value *Def = MInst->getImplicitRef(i);
          LiveRange* LR = createOrAddToLiveRange(Def, /*isCC*/ false);

          // If the implicit operand has a pre-assigned register,
          // set it directly in the LiveRange
          if (MInst->getImplicitOp(i).hasAllocatedReg()) {
            unsigned getClassId;
            LR->setColor(MRI.getClassRegNum(
                                MInst->getImplicitOp(i).getAllocatedRegNum(),
                                getClassId));
          }
	}

    } // for all machine instructions in the BB
  } // for all BBs in function

  // Now we have to suggest clors for call and return arg live ranges.
  // Also, if there are implicit defs (e.g., retun value of a call inst)
  // they must be added to the live range list
  // 
  suggestRegs4CallRets();

  if( DEBUG_RA >= RA_DEBUG_LiveRanges) 
    std::cerr << "Initial Live Ranges constructed!\n";
}


//---------------------------------------------------------------------------
// If some live ranges must be colored with specific hardware registers
// (e.g., for outgoing call args), suggesting of colors for such live
// ranges is done using target specific function. Those functions are called
// from this function. The target specific methods must:
//    1) suggest colors for call and return args. 
//    2) create new LRs for implicit defs in machine instructions
//---------------------------------------------------------------------------
void LiveRangeInfo::suggestRegs4CallRets() {
  std::vector<MachineInstr*>::iterator It = CallRetInstrList.begin();
  for( ; It != CallRetInstrList.end(); ++It) {
    MachineInstr *MInst = *It;
    MachineOpCode OpCode = MInst->getOpcode();

    if ((TM.getInstrInfo()).isReturn(OpCode))
      MRI.suggestReg4RetValue(MInst, *this);
    else if ((TM.getInstrInfo()).isCall(OpCode))
      MRI.suggestRegs4CallArgs(MInst, *this);
    else 
      assert( 0 && "Non call/ret instr in CallRetInstrList" );
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


// Checks if live range LR interferes with any node assigned or suggested to
// be assigned the specified color
// 
inline bool InterferesWithColor(const LiveRange& LR, unsigned color) {
  IGNode* lrNode = LR.getUserIGNode();
  for (unsigned n=0, NN = lrNode->getNumOfNeighbors(); n < NN; n++) {
    LiveRange *neighLR = lrNode->getAdjIGNode(n)->getParentLR();
    if (neighLR->hasColor() && neighLR->getColor() == color)
      return true;
    if (neighLR->hasSuggestedColor() && neighLR->getSuggestedColor() == color)
      return true;
  }
  return false;
}

// Cannot coalesce if any of the following is true:
// (1) Both LRs have suggested colors (should be "different suggested colors"?)
// (2) Both LR1 and LR2 have colors and the colors are different
//    (but if the colors are the same, it is definitely safe to coalesce)
// (3) LR1 has color and LR2 interferes with any LR that has the same color
// (4) LR2 has color and LR1 interferes with any LR that has the same color
// 
inline bool InterfsPreventCoalescing(const LiveRange& LROfDef,
                                     const LiveRange& LROfUse) {
  // (4) if they have different suggested colors, cannot coalesce
  if (LROfDef.hasSuggestedColor() && LROfUse.hasSuggestedColor())
    return true;

  // if neither has a color, nothing more to do.
  if (! LROfDef.hasColor() && ! LROfUse.hasColor())
    return false;

  // (2, 3) if L1 has color...
  if (LROfDef.hasColor()) {
    if (LROfUse.hasColor())
      return (LROfUse.getColor() != LROfDef.getColor());
    return InterferesWithColor(LROfUse, LROfDef.getColor());
  }

  // (4) else only LROfUse has a color: check if that could interfere
  return InterferesWithColor(LROfDef, LROfUse.getColor());
}


void LiveRangeInfo::coalesceLRs()  
{
  if(DEBUG_RA >= RA_DEBUG_LiveRanges) 
    std::cerr << "\nCoalescing LRs ...\n";

  MachineFunction &MF = MachineFunction::get(Meth);
  for (MachineFunction::iterator BBI = MF.begin(); BBI != MF.end(); ++BBI) {
    MachineBasicBlock &MBB = *BBI;

    // iterate over all the machine instructions in BB
    for(MachineBasicBlock::iterator MII = MBB.begin(); MII != MBB.end(); ++MII){
      const MachineInstr *MI = MII;

      if( DEBUG_RA >= RA_DEBUG_LiveRanges) {
	std::cerr << " *Iterating over machine instr ";
	MI->dump();
	std::cerr << "\n";
      }

      // iterate over  MI operands to find defs
      for(MachineInstr::const_val_op_iterator DefI = MI->begin(),
            DefE = MI->end(); DefI != DefE; ++DefI) {
	if (DefI.isDef()) { // this operand is modified
	  LiveRange *LROfDef = getLiveRangeForValue( *DefI );
	  RegClass *RCOfDef = LROfDef->getRegClass();

	  MachineInstr::const_val_op_iterator UseI = MI->begin(),
            UseE = MI->end();
	  for( ; UseI != UseE; ++UseI) { // for all uses
 	    LiveRange *LROfUse = getLiveRangeForValue( *UseI );
	    if (!LROfUse) {             // if LR of use is not found
	      //don't warn about labels
	      if (!isa<BasicBlock>(*UseI) && DEBUG_RA >= RA_DEBUG_LiveRanges)
		std::cerr << " !! Warning: No LR for use " << RAV(*UseI)<< "\n";
	      continue;                 // ignore and continue
	    }

	    if (LROfUse == LROfDef)     // nothing to merge if they are same
	      continue;

	    if (MRI.getRegTypeForLR(LROfDef) ==
                MRI.getRegTypeForLR(LROfUse)) {
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
		  // if both LRs do not have different pre-assigned colors
		  // and both LRs do not have suggested colors
                  if (! InterfsPreventCoalescing(*LROfDef, *LROfUse)) {
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
    std::cerr << "\nCoalescing Done!\n";
}

/*--------------------------- Debug code for printing ---------------*/


void LiveRangeInfo::printLiveRanges() {
  LiveRangeMapType::iterator HMI = LiveRangeMap.begin();   // hash map iterator
  std::cerr << "\nPrinting Live Ranges from Hash Map:\n";
  for( ; HMI != LiveRangeMap.end(); ++HMI) {
    if (HMI->first && HMI->second) {
      std::cerr << " Value* " << RAV(HMI->first) << "\t: "; 
      if (IGNode* igNode = HMI->second->getUserIGNode())
        std::cerr << "LR# " << igNode->getIndex();
      else
        std::cerr << "LR# " << "<no-IGNode>";
      std::cerr << "\t:Values = "; printSet(*HMI->second); std::cerr << "\n";
    }
  }
}

} // End llvm namespace
