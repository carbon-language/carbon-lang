#include "llvm/Target/Sparc.h"
#include "SparcInternals.h"
#include "llvm/Method.h"
#include "llvm/iTerminators.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/InstrSelection.h"

#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/PhyRegAlloc.h"




//---------------------------------------------------------------------------
// UltraSparcRegInfo
//---------------------------------------------------------------------------

/*
Rules for coloring values with sepcial registers:
=================================================

The following are the cases we color values with special regs:

1) Incoming Method Arguements
2) Outgoing Call Arguments
3) Return Value of a call
4) Return Value of a return statement

Both 1 and 3 are defs. Therefore, they can be set directly. For case 1, 
incoming args are colored to %i0-%i5 and %f0 - %fx. For case 3, the return
value of the call must be colored to %o0 or %f0.

For case 2 we can use %o0-%o6 and %f0- %fx and for case 4 we can use %i0 or
%f0. However, we cannot pre-color them directly to those regs
if there are call interferences or they can be already colred by case 1.
(Note that a return value is call is already colored and it is registered
as a call interference as well if it is live after the call). Otherwise, they
can be precolored. In cases where we cannot precolor, we just have to insert 
a copy instruction to copy the LR to the required register.

*/



//---------------------------------------------------------------------------
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






void UltraSparcRegInfo::colorCallArgs(vector<const Instruction *> & 
				      CallInstrList, LiveRangeInfo& LRI,
				      AddedInstrMapType &AddedInstrMap) const
{

  vector<const Instruction *>::const_iterator InstIt;

  // First color the return value of all call instructions. The return value
  // will be in %o0 if the value is an integer type, or in %f0 if the 
  // value is a float type.

  for(InstIt=CallInstrList.begin(); InstIt != CallInstrList.end(); ++InstIt) {

    const Instruction *const CallI = *InstIt;

    // get the live range of return value of this call
    LiveRange *const LR = LRI.getLiveRangeForValue( CallI ); 

    if ( LR ) {

      // Since the call is a def, it cannot be colored by some other instr.
      // Therefore, we can definitely set a color here.
      // However, this def can be used by some other instr like another call
      // or return which places that in a special register. In that case
      // it has to put a copy. Note that, the def will have a call interference
      // with this call instr itself if it is live after this call.

      assert( ! LR->hasColor() && "Can't have a color since this is a def");

      unsigned RegClassID = (LR->getRegClass())->getID();
      
      if( RegClassID == IntRegClassID ) {
	LR->setColor(SparcIntRegOrder::o0);
      }
      else if (RegClassID == FloatRegClassID ) {
	LR->setColor(SparcFloatRegOrder::f0 );
      }
    }
    else {
      cout << "Warning: No Live Range for return value of CALL" << endl;
    }
  }


  for( InstIt=CallInstrList.begin(); InstIt != CallInstrList.end(); ++InstIt) {

    // Inst = LLVM call instruction
    const Instruction *const CallI = *InstIt;

    // find the CALL/JMMPL machine instruction
    MachineCodeForVMInstr &  MInstVec = CallI->getMachineInstrVec();
    MachineCodeForVMInstr::const_iterator MIIt = MInstVec.begin();

    /*
    for( ; MIIt != MInstVec.end() && 
	   ! getUltraSparcInfo().getInstrInfo().isCall((*MIIt)->getOpCode()); 
	 ++MIIt );

    assert( (MIIt != MInstVec.end())  && "CALL/JMPL not found");
    */

    assert(getUltraSparcInfo().getInstrInfo().isCall((*MIIt)->getOpCode()) &&
	   "First machine instruction is not a Call/JMPL Machine Instr");

    // CallMI = CALL/JMPL machine isntruction
    const MachineInstr *const CallMI = *MIIt;

    Instruction::op_const_iterator OpIt = CallI->op_begin();

    unsigned intArgNo=0;


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
	  setCallOrRetArgCol( LR, SparcIntRegOrder::o0 + intArgNo, 
			      CallMI, AddedInstrMap);
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
	      setCallOrRetArgCol(LR, SparcFloatRegOrder::f0 + i,
				 CallMI, AddedInstrMap);	    	    
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
	      setCallOrRetArgCol(LR, SparcFloatRegOrder::f0 + i,
				 CallMI, AddedInstrMap);
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





void UltraSparcRegInfo::colorRetArg(vector<const Instruction *> & 
				    RetInstrList, LiveRangeInfo& LRI,
				    AddedInstrMapType &AddedInstrMap) const
{

  vector<const Instruction *>::const_iterator InstIt;


  for(InstIt=RetInstrList.begin(); InstIt != RetInstrList.end(); ++InstIt) {

    const ReturnInst *const RetI = (ReturnInst *) *InstIt;

    // get the return value of this return instruction
    const Value *RetVal =  (RetI)->getReturnValue();

    if( RetVal ) {

      // find the CALL/JMMPL machine instruction
      MachineCodeForVMInstr &  MInstVec = RetI->getMachineInstrVec();
      MachineCodeForVMInstr::const_iterator MIIt = MInstVec.begin();

      assert(getUltraSparcInfo().getInstrInfo().isReturn((*MIIt)->getOpCode())
	     && "First machine instruction is not a RET Machine Instr");
      // RET machine isntruction
      const MachineInstr *const RetMI = *MIIt;

      LiveRange *const LR = LRI.getLiveRangeForValue( RetVal ); 
      unsigned RegClassID = (LR->getRegClass())->getID();

      if ( LR ) {      
	if( RegClassID == IntRegClassID ) {
	  setCallOrRetArgCol( LR, SparcIntRegOrder::i0,	RetMI, AddedInstrMap);
	}
	else if (RegClassID==FloatRegClassID ) {
	  setCallOrRetArgCol(LR, SparcFloatRegOrder::f0, RetMI, AddedInstrMap);
	}
	
      }
      else {
	cout << "Warning: No LR for return value" << endl;
      }

    }

  }

}



void UltraSparcRegInfo::setCallOrRetArgCol(LiveRange *const LR, 
					   const unsigned RegNo,
					   const MachineInstr *MI,
					   AddedInstrMapType &AIMap) const {

  // if no call interference and LR is NOT previously colored (e.g., as an 
  // incoming arg)
  if( ! LR->getNumOfCallInterferences() && ! LR->hasColor() ) { 
    // we can directly allocate a %o register
    LR->setColor( RegNo);
    if( DEBUG_RA) printReg( LR );
  }
  else {        

    // there are call interferences (e.g., live across a call or produced
    // by a call instr) or this LR is already colored as an incoming argument

    MachineInstr *MI = getCopy2RegMI((*(LR->begin())), RegNo, 
				     (LR->getRegClass())->getID());

    AddedInstrns * AI = AIMap[ MI ];    // get already added instrns for MI
    if( ! AI ) AI = new AddedInstrns(); 

    AI->InstrnsBefore.push_back( MI );  // add the new MI yp AMI
    AIMap[ MI ] = AI;

    
    cout << "Inserted a copy instr for a RET/CALL instr " << endl;

    // We don't color LR here. It's colored as any other normal LR or
    // as an incoming arg or a return value of a call.
  }

}

// Generates a copy machine instruction to copy a value to a given
// register.

MachineInstr * UltraSparcRegInfo::getCopy2RegMI(const Value *SrcVal,
						const unsigned Reg,
 						unsigned RegClassID) const {
  MachineInstr * MI;

  if(  RegClassID == IntRegClassID ) {  // if integer move

    MI = new MachineInstr(ADD, 3);
 
    MI->SetMachineOperand(0, MachineOperand::MO_VirtualRegister, SrcVal);
    MI->SetMachineOperand(1, SparcIntRegOrder::g0, false);
    MI->SetMachineOperand(2, Reg, true);
  }
  else {                                // if FP move

    if(SrcVal->getType()-> getPrimitiveID() == Type::FloatTyID )
      MI = new MachineInstr(FMOVS, 2);
    else if(SrcVal->getType()-> getPrimitiveID() == Type::DoubleTyID)
      MI = new MachineInstr(FMOVD, 2);
    else assert( 0 && "Unknown Type");

    MI->SetMachineOperand(0, MachineOperand::MO_VirtualRegister, SrcVal);
    MI->SetMachineOperand(1, Reg, true);
  }

  return MI;

}


//---------------------------------------------------------------------------
// Print the register assigned to a LR
//---------------------------------------------------------------------------

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
