#include "llvm/Target/Sparc.h"
#include "SparcInternals.h"
#include "llvm/Method.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/InstrSelection.h"

#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/PhyRegAlloc.h"




//---------------------------------------------------------------------------
// UltraSparcRegInfo
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Finds the return value of a call instruction
//---------------------------------------------------------------------------

const Value * 
UltraSparcRegInfo::getCallInstRetVal(const MachineInstr *CallMI) const{

  unsigned OpCode = CallMI->getOpCode();
  unsigned NumOfImpRefs =  CallMI->getNumImplicitRefs();

  if( OpCode == CALL ) {

    // The one before the last implicit operand is the return value of 
    // a CALL instr
    if( NumOfImpRefs > 1 )
      if(  CallMI->implicitRefIsDefined(NumOfImpRefs-2) ) 
	return  CallMI->getImplicitRef(NumOfImpRefs-2); 

  }
  else if( OpCode == JMPLCALL) {

    // The last implicit operand is the return value of a JMPL in   
    if( NumOfImpRefs > 0 )
      if(  CallMI->implicitRefIsDefined(NumOfImpRefs-1) ) 
	return  CallMI->getImplicitRef(NumOfImpRefs-1); 
  }
  else
    assert(0 && "OpCode must be CALL/JMPL for a call instr");

  return NULL;

}

//---------------------------------------------------------------------------
// Finds the return address of a call instruction
//---------------------------------------------------------------------------

const Value *
UltraSparcRegInfo::getCallInstRetAddr(const MachineInstr *CallMI)const {

  unsigned OpCode = CallMI->getOpCode();

  if( OpCode == CALL) {

    unsigned NumOfImpRefs =  CallMI->getNumImplicitRefs();

    assert( NumOfImpRefs && "CALL instr must have at least on ImpRef");
    // The last implicit operand is the return address of a CALL instr
    return  CallMI->getImplicitRef(NumOfImpRefs-1); 

  }
  else if( OpCode == JMPLCALL ) {

    MachineOperand & MO  = ( MachineOperand &) CallMI->getOperand(2);
    return MO.getVRegValue();

  }
  else
    assert(0 && "OpCode must be CALL/JMPL for a call instr");

  assert(0  && "There must be a return addr for a call instr");

  return NULL;

}


//---------------------------------------------------------------------------
// Finds the # of actual arguments of the call instruction
//---------------------------------------------------------------------------

const unsigned 
UltraSparcRegInfo::getCallInstNumArgs(const MachineInstr *CallMI) const {

  unsigned OpCode = CallMI->getOpCode();
  unsigned NumOfImpRefs =  CallMI->getNumImplicitRefs();
  int NumArgs = -1;

  if( OpCode == CALL ) {

    switch( NumOfImpRefs ) {

    case 0: assert(0 && "A CALL inst must have at least one ImpRef (RetAddr)");

    case 1: NumArgs = 0;
            break;
    
    default:  // two or more implicit refs
      if(  CallMI->implicitRefIsDefined(NumOfImpRefs-2) ) 
	NumArgs = NumOfImpRefs - 2;    // i.e., NumOfImpRef-2 is the ret val
      else 
	NumArgs = NumOfImpRefs - 1;
    }

  }
  else if( OpCode == JMPLCALL ) {

    // The last implicit operand is the return value of a JMPL instr
    if( NumOfImpRefs > 0 ) {
      if(  CallMI->implicitRefIsDefined(NumOfImpRefs-1) ) 
	NumArgs = NumOfImpRefs - 1;    // i.e., NumOfImpRef-1 is the ret val
      else 
	NumArgs = NumOfImpRefs;
    }
    else 
      NumArgs = NumOfImpRefs;
  }
  else
    assert(0 && "OpCode must be CALL/JMPL for a call instr");

  assert( (NumArgs != -1)  && "Internal error in getCallInstNumArgs" );
  return (unsigned) NumArgs;
 
  
}


//---------------------------------------------------------------------------
// Suggests a register for the ret address in the RET machine instruction
//---------------------------------------------------------------------------

void UltraSparcRegInfo::suggestReg4RetAddr(const MachineInstr * RetMI, 
					   LiveRangeInfo& LRI) const {

  assert( (RetMI->getNumOperands() >= 2)
          && "JMPL/RETURN must have 3 and 2 operands respectively");
  
  MachineOperand & MO  = ( MachineOperand &) RetMI->getOperand(0);

  MO.setRegForValue( getUnifiedRegNum( IntRegClassID, SparcIntRegOrder::i7) );
  
  // TODO (Optimize): 
  // Instead of setting the color, we can suggest one. In that case,
  // we have to test later whether it received the suggested color.
  // In that case, a LR has to be created at the start of method.
  // It has to be done as follows (remove the setRegVal above):

  /*
  const Value *RetAddrVal = MO.getVRegValue();

  assert( RetAddrVal && "LR for ret address must be created at start");

  LiveRange * RetAddrLR = LRI.getLiveRangeForValue( RetAddrVal);  
  RetAddrLR->setSuggestedColor(getUnifiedRegNum( IntRegClassID, 
  SparcIntRegOrdr::i7) );
  */


}


//---------------------------------------------------------------------------
// Suggests a register for the ret address in the JMPL/CALL machine instr
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestReg4CallAddr(const MachineInstr * CallMI,
					    LiveRangeInfo& LRI,
					    vector<RegClass *> RCList) const {


  const Value *RetAddrVal = getCallInstRetAddr( CallMI );

  // RetAddrVal cannot be NULL (asserted in  getCallInstRetAddr)
  // create a new LR for the return address and color it
  
  LiveRange * RetAddrLR = new LiveRange();  
  RetAddrLR->add( RetAddrVal );
  unsigned RegClassID = getRegClassIDOfValue( RetAddrVal );
  RetAddrLR->setRegClass( RCList[RegClassID] );
  RetAddrLR->setColor(getUnifiedRegNum(IntRegClassID,SparcIntRegOrder::o7));
  LRI.addLRToMap( RetAddrVal, RetAddrLR);
  

  /*  
  assert( (CallMI->getNumOperands() == 3) && "JMPL must have 3 operands");

  // directly set color since the LR of ret address (if there were one) 
  // will not extend after the call instr

  MachineOperand & MO  = ( MachineOperand &) CallMI->getOperand(2);
  MO.setRegForValue( getUnifiedRegNum( IntRegClassID,SparcIntRegOrder::o7) );

  */

}




//---------------------------------------------------------------------------
//  This method will suggest colors to incoming args to a method. 
//  If the arg is passed on stack due to the lack of regs, NOTHING will be
//  done - it will be colored (or spilled) as a normal value.
//---------------------------------------------------------------------------

void UltraSparcRegInfo::suggestRegs4MethodArgs(const Method *const Meth, 
					       LiveRangeInfo& LRI) const 
{

                                                 // get the argument list
  const Method::ArgumentListType& ArgList = Meth->getArgumentList();           
                                                 // get an iterator to arg list
  Method::ArgumentListType::const_iterator ArgIt = ArgList.begin(); 

  // for each argument
  for( unsigned argNo=0; ArgIt != ArgList.end() ; ++ArgIt, ++argNo) {    

    // get the LR of arg
    LiveRange *const LR = LRI.getLiveRangeForValue((const Value *) *ArgIt); 
    assert( LR && "No live range found for method arg");

    unsigned RegType = getRegType( LR );


    // if the arg is in int class - allocate a reg for an int arg
    if( RegType == IntRegType ) {

      if( argNo < NumOfIntArgRegs) {
	LR->setSuggestedColor( SparcIntRegOrder::i0 + argNo );

      }
  
      else {
	// Do NOTHING as this will be colored as a normal value.
	if (DEBUG_RA) cerr << " Int Regr not suggested for method arg\n";
      }
     
    }
    else if( RegType==FPSingleRegType && (argNo*2+1) < NumOfFloatArgRegs) 
      LR->setSuggestedColor( SparcFloatRegOrder::f0 + (argNo * 2 + 1) );
    
 
    else if( RegType == FPDoubleRegType && (argNo*2) < NumOfFloatArgRegs) 
      LR->setSuggestedColor( SparcFloatRegOrder::f0 + (argNo * 2) ); 
    

  }
  
}

//---------------------------------------------------------------------------
// 
//---------------------------------------------------------------------------

void UltraSparcRegInfo::colorMethodArgs(const Method *const Meth, 
					LiveRangeInfo& LRI,
					AddedInstrns *const FirstAI) const {

                                                 // get the argument list
  const Method::ArgumentListType& ArgList = Meth->getArgumentList();           
                                                 // get an iterator to arg list
  Method::ArgumentListType::const_iterator ArgIt = ArgList.begin(); 

  MachineInstr *AdMI;


  // for each argument
  for( unsigned argNo=0; ArgIt != ArgList.end() ; ++ArgIt, ++argNo) {    

    // get the LR of arg
    LiveRange *const LR = LRI.getLiveRangeForValue((const Value *) *ArgIt); 
    assert( LR && "No live range found for method arg");


    unsigned RegType = getRegType( LR );
    unsigned RegClassID = (LR->getRegClass())->getID();


    // find whether this argument is coming in a register (if not, on stack)

    bool isArgInReg = false;
    unsigned UniArgReg = InvalidRegNum;	 // reg that LR MUST be colored with

    if( (RegType== IntRegType && argNo <  NumOfIntArgRegs)) {
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum( RegClassID, SparcIntRegOrder::i0 + argNo );
    }
    else if(RegType == FPSingleRegType && argNo < NumOfFloatArgRegs)  { 
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum( RegClassID, 
				    SparcFloatRegOrder::f0 + argNo*2 + 1 ) ;
    }
    else if(RegType == FPDoubleRegType && argNo < NumOfFloatArgRegs)  { 
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum(RegClassID, SparcFloatRegOrder::f0+argNo*2);
    }

    
    if( LR->hasColor() ) {

      unsigned UniLRReg = getUnifiedRegNum(  RegClassID, LR->getColor() );

      // if LR received the correct color, nothing to do
      if( UniLRReg == UniArgReg )
	continue;

      // We are here because the LR did not have a suggested 
      // color or did not receive the suggested color but LR got a register.
      // Now we have to copy %ix reg (or stack pos of arg) 
      // to the register it was colored with.
      
      // if the arg is coming in UniArgReg register MUST go into
      // the UniLRReg register
      if( isArgInReg ) 
	AdMI = cpReg2RegMI( UniArgReg, UniLRReg, RegType );

      else {

	// Now the arg is coming on stack. Since the LR recieved a register,
	// we just have to load the arg on stack into that register
        const MachineFrameInfo& frameInfo = target.getFrameInfo();
        assert(frameInfo.argsOnStackHaveFixedSize()); 
        
        bool growUp;
	int firstArg =
          frameInfo.getFirstIncomingArgOffset(MachineCodeForMethod::get(Meth), growUp);
	int offsetFromFP =
          growUp? firstArg + argNo * frameInfo.getSizeOfEachArgOnStack()
                : firstArg - argNo * frameInfo.getSizeOfEachArgOnStack();
        
	AdMI = cpMem2RegMI(getFramePointer(), offsetFromFP, 
			   UniLRReg, RegType );
      }

      FirstAI->InstrnsBefore.push_back( AdMI );   
      
    } // if LR received a color

    else {                             

      // Now, the LR did not receive a color. But it has a stack offset for
      // spilling.

      // So, if the arg is coming in UniArgReg register,  we can just move
      // that on to the stack pos of LR


      if( isArgInReg ) {

	MachineInstr *AdIBef = 
	  cpReg2MemMI(UniArgReg, getFramePointer(), 
		      LR->getSpillOffFromFP(), RegType );

	FirstAI->InstrnsBefore.push_back( AdMI );   
      }

      else {

	// Now the arg is coming on stack. Since the LR did NOT 
	// recieved a register as well, it is allocated a stack position. We
	// can simply change the stack poistion of the LR. We can do this,
	// since this method is called before any other method that makes
	// uses of the stack pos of the LR (e.g., updateMachineInstr)

        const MachineFrameInfo& frameInfo = target.getFrameInfo();
        assert(frameInfo.argsOnStackHaveFixedSize()); 
        
        bool growUp;
	int firstArg = frameInfo.getFirstIncomingArgOffset(MachineCodeForMethod::get(Meth), growUp);
	int offsetFromFP =
          growUp? firstArg + argNo * frameInfo.getSizeOfEachArgOnStack()
                : firstArg - argNo * frameInfo.getSizeOfEachArgOnStack();
        
	LR->modifySpillOffFromFP( offsetFromFP );
      }

    }

  }  // for each incoming argument

}




//---------------------------------------------------------------------------
// This method is called before graph coloring to suggest colors to the
// outgoing call args and the return value of the call.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestRegs4CallArgs(const MachineInstr *const CallMI, 
					     LiveRangeInfo& LRI,
					     vector<RegClass *> RCList) const {

  assert ( (UltraSparcInfo->getInstrInfo()).isCall(CallMI->getOpCode()) );

  suggestReg4CallAddr(CallMI, LRI, RCList);


  // First color the return value of the call instruction. The return value
  // will be in %o0 if the value is an integer type, or in %f0 if the 
  // value is a float type.

  // the return value cannot have a LR in machine instruction since it is
  // only defined by the call instruction

  // if type is not void,  create a new live range and set its 
  // register class and add to LRI


  const Value *RetVal = getCallInstRetVal( CallMI );


  if( RetVal ) {

    assert( (! LRI.getLiveRangeForValue( RetVal ) ) && 
	    "LR for ret Value of call already definded!");


      // create a new LR for the return value

    LiveRange * RetValLR = new LiveRange();  
    RetValLR->add( RetVal );
    unsigned RegClassID = getRegClassIDOfValue( RetVal );
    RetValLR->setRegClass( RCList[RegClassID] );
    LRI.addLRToMap( RetVal, RetValLR);
    
    // now suggest a register depending on the register class of ret arg

    if( RegClassID == IntRegClassID ) 
      RetValLR->setSuggestedColor(SparcIntRegOrder::o0);
    else if (RegClassID == FloatRegClassID ) 
      RetValLR->setSuggestedColor(SparcFloatRegOrder::f0 );
    else assert( 0 && "Unknown reg class for return value of call\n");

  }

  
  // Now suggest colors for arguments (operands) of the call instruction.
  // Colors are suggested only if the arg number is smaller than the
  // the number of registers allocated for argument passing.
  // Now, go thru call args - implicit operands of the call MI

  unsigned NumOfCallArgs =  getCallInstNumArgs( CallMI );
  
  for(unsigned argNo=0, i=0; i < NumOfCallArgs; ++i, ++argNo ) {

    const Value *CallArg = CallMI->getImplicitRef(i);
    
    // get the LR of call operand (parameter)
    LiveRange *const LR = LRI.getLiveRangeForValue(CallArg); 

    // not possible to have a null LR since all args (even consts)  
    // must be defined before
    if( !LR ) {          
      if( DEBUG_RA) {
	cerr << " ERROR: In call instr, no LR for arg:  " ;
	printValue(CallArg); cerr << endl;
      }
      assert(0 && "NO LR for call arg");  
      // continue;
    }
    
    unsigned RegType = getRegType( LR );

    // if the arg is in int class - allocate a reg for an int arg
    if( RegType == IntRegType ) {

      if( argNo < NumOfIntArgRegs) 
	LR->setSuggestedColor( SparcIntRegOrder::o0 + argNo );

      else if (DEBUG_RA) 
	// Do NOTHING as this will be colored as a normal value.
	cerr << " Regr not suggested for int call arg" << endl;
      
    }
    else if( RegType == FPSingleRegType &&  (argNo*2 +1)< NumOfFloatArgRegs) 
      LR->setSuggestedColor( SparcFloatRegOrder::f0 + (argNo * 2 + 1) );
    
 
    else if( RegType == FPDoubleRegType && (argNo*2) < NumOfFloatArgRegs) 
      LR->setSuggestedColor( SparcFloatRegOrder::f0 + (argNo * 2) ); 
    

  } // for all call arguments

}


//---------------------------------------------------------------------------
// After graph coloring, we have call this method to see whehter the return
// value and the call args received the correct colors. If not, we have
// to instert copy instructions.
//---------------------------------------------------------------------------

void UltraSparcRegInfo::colorCallArgs(const MachineInstr *const CallMI,
				      LiveRangeInfo& LRI,
				      AddedInstrns *const CallAI,
				      PhyRegAlloc &PRA) const {

  assert ( (UltraSparcInfo->getInstrInfo()).isCall(CallMI->getOpCode()) );

  // Reset the optional args area in the stack frame
  // since that is reused for each call
  // 
  PRA.mcInfo.resetOptionalArgs(target);
  
  // First color the return value of the call.
  // If there is a LR for the return value, it means this
  // method returns a value
  
  MachineInstr *AdMI;

  const Value *RetVal = getCallInstRetVal( CallMI );

  if( RetVal ) {

    LiveRange * RetValLR = LRI.getLiveRangeForValue( RetVal );

    if( !RetValLR ) {
      cerr << "\nNo LR for:";
      printValue( RetVal );
      cerr << endl;
      assert( RetValLR && "ERR:No LR for non-void return value");
      //return;
    }

    unsigned RegClassID = (RetValLR->getRegClass())->getID();    
    bool recvCorrectColor = false;

    unsigned CorrectCol;                // correct color for ret value
    if(RegClassID == IntRegClassID)
      CorrectCol = SparcIntRegOrder::o0;
    else if(RegClassID == FloatRegClassID)
      CorrectCol = SparcFloatRegOrder::f0;
    else {
      assert( 0 && "Unknown RegClass");
      return;
    }

    // if the LR received the correct color, NOTHING to do

    if(  RetValLR->hasColor() )
      if( RetValLR->getColor() == CorrectCol )
	recvCorrectColor = true;


    // if we didn't receive the correct color for some reason, 
    // put copy instruction
    
    if( !recvCorrectColor ) {

      unsigned RegType = getRegType( RetValLR );

      // the  reg that LR must be colored with 
      unsigned UniRetReg = getUnifiedRegNum( RegClassID, CorrectCol);	
      
      if( RetValLR->hasColor() ) {
	
	unsigned 
	  UniRetLRReg=getUnifiedRegNum(RegClassID,RetValLR->getColor());
	
	// the return value is coming in UniRetReg but has to go into
	// the UniRetLRReg

	AdMI = cpReg2RegMI( UniRetReg, UniRetLRReg, RegType ); 	

      } // if LR has color
      else {

	// if the LR did NOT receive a color, we have to move the return
	// value coming in UniRetReg to the stack pos of spilled LR
	
	AdMI = 	cpReg2MemMI(UniRetReg, getFramePointer(), 
			    RetValLR->getSpillOffFromFP(), RegType );
      }

      CallAI->InstrnsAfter.push_back( AdMI );
      
    } // the LR didn't receive the suggested color  
    
  } // if there a return value
  

  // Now color all args of the call instruction

  vector <MachineInstr *> AddedInstrnsBefore;

  unsigned NumOfCallArgs =  getCallInstNumArgs( CallMI );

  for(unsigned argNo=0, i=0; i < NumOfCallArgs; ++i, ++argNo ) {

    const Value *CallArg = CallMI->getImplicitRef(i);

    // get the LR of call operand (parameter)
    LiveRange *const LR = LRI.getLiveRangeForValue(CallArg); 

    unsigned RegType = getRegType( CallArg );
    unsigned RegClassID =  getRegClassIDOfValue( CallArg);
    
    // find whether this argument is coming in a register (if not, on stack)

    bool isArgInReg = false;
    unsigned UniArgReg = InvalidRegNum;  // reg that LR must be colored with

    if( (RegType== IntRegType && argNo <  NumOfIntArgRegs)) {
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum(RegClassID, SparcIntRegOrder::o0 + argNo );
    }
    else if(RegType == FPSingleRegType && argNo < NumOfFloatArgRegs)  { 
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum(RegClassID, 
				   SparcFloatRegOrder::f0 + (argNo*2 + 1) );
    }
    else if(RegType == FPDoubleRegType && argNo < NumOfFloatArgRegs)  { 
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum(RegClassID, SparcFloatRegOrder::f0+argNo*2);
    }


    // not possible to have a null LR since all args (even consts)  
    // must be defined before
    if( !LR ) {          
      if( DEBUG_RA) {
	cerr << " ERROR: In call instr, no LR for arg:  " ;
	printValue(CallArg); cerr << endl;
      }
      assert(0 && "NO LR for call arg");  
      // continue;
    }


    // if the LR received the suggested color, NOTHING to do


    if( LR->hasColor() ) {


      unsigned UniLRReg = getUnifiedRegNum( RegClassID,  LR->getColor() );

      // if LR received the correct color, nothing to do
      if( UniLRReg == UniArgReg )
	continue;

      // We are here because though the LR is allocated a register, it
      // was not allocated the suggested register. So, we have to copy %ix reg 
      // (or stack pos of arg) to the register it was colored with

      // the LR is colored with UniLRReg but has to go into  UniArgReg
      // to pass it as an argument

      if( isArgInReg ) {
	AdMI = cpReg2RegMI(UniLRReg, UniArgReg, RegType );
	AddedInstrnsBefore.push_back( AdMI ); 
      }
      
      else {
	// Now, we have to pass the arg on stack. Since LR received a register
	// we just have to move that register to the stack position where
	// the argument must be passed

	int argOffset = PRA.mcInfo.allocateOptionalArg(target, LR->getType()); 

	AdMI = cpReg2MemMI(UniLRReg, getStackPointer(), argOffset, RegType );

	// Now add the instruction. We can directly add to
	// CallAI->InstrnsBefore since we are just saving a reg on stack
	//
	CallAI->InstrnsBefore.push_back( AdMI ); 

	//cerr << "\nCaution: Passing a reg on stack";
      }


    }

    else {                          // LR is not colored (i.e., spilled)      
      
      if( isArgInReg ) {

	// Now the LR did NOT recieve a register but has a stack poistion.
	// Since, the outgoing arg goes in a register we just have to insert
	// a load instruction to load the LR to outgoing register


	AdMI = cpMem2RegMI(getFramePointer(), LR->getSpillOffFromFP(),
			   UniArgReg, RegType );
        
	cerr << "\nCaution: Loading a spilled val to a reg as a call arg";
	AddedInstrnsBefore.push_back( AdMI );  // Now add the instruction
      }
      
      else {
	// Now, we have to pass the arg on stack. Since LR  also did NOT
	// receive a register we have to move an argument in memory to 
	// outgoing parameter on stack.
	
	// Optoimize: Optimize when reverse pointers in MahineInstr are
	// introduced. 
	// call PRA.getUnusedRegAtMI(....) to get an unused reg. Only if this
	// fails, then use the following code. Currently, we cannot call the
	// above method since we cannot find LVSetBefore without the BB 
	
	int TReg = PRA.getUniRegNotUsedByThisInst( LR->getRegClass(), CallMI );

    /**** NOTE: THIS SHOULD USE THE RIGHT SIZE FOR THE REG BEING PUSHED ****/
	int TmpOff = PRA.mcInfo.pushTempValue(target, 8);
                          // target.findOptimalStorageSize(LR->getType()));
        
	int argOffset = PRA.mcInfo.allocateOptionalArg(target, LR->getType()); 
        
	MachineInstr *Ad1, *Ad2, *Ad3, *Ad4;
        
	// Sequence:
	// (1) Save TReg on stack    
	// (2) Load LR value into TReg from stack pos of LR
	// (3) Store Treg on outgoing Arg pos on stack
	// (4) Load the old value of TReg from stack to TReg (restore it)

	Ad1 = cpReg2MemMI(TReg, getFramePointer(), TmpOff, RegType );
	Ad2 = cpMem2RegMI(getFramePointer(), LR->getSpillOffFromFP(), 
			  TReg, RegType ); 
	Ad3 = cpReg2MemMI(TReg, getStackPointer(), argOffset, RegType );
	Ad4 = cpMem2RegMI(getFramePointer(), TmpOff, TReg, RegType ); 

	// We directly add to CallAI->InstrnsBefore instead of adding to
	// AddedInstrnsBefore since these instructions must not be
	// reordered.
        
	CallAI->InstrnsBefore.push_back( Ad1 );  
	CallAI->InstrnsBefore.push_back( Ad2 );  
	CallAI->InstrnsBefore.push_back( Ad3 );  
	CallAI->InstrnsBefore.push_back( Ad4 );  

	cerr << "\nCaution: Call arg moved from stack2stack for: " << *CallMI ;
      }

    }

  }  // for each parameter in call instruction


  // if we added any instruction before the call instruction, verify
  // that they are in the proper order and if not, reorder them

  if( ! AddedInstrnsBefore.empty() ) {

    if( DEBUG_RA  ) {
      cerr << "\nCalling reorder with instrns: \n";
      for(unsigned i=0; i < AddedInstrnsBefore.size(); i++)
	cerr  << *(AddedInstrnsBefore[i]);
    }

    vector <MachineInstr *> TmpVec;
    OrderAddedInstrns(AddedInstrnsBefore, TmpVec, PRA);

    if( DEBUG_RA   ) {
      cerr << "\nAfter reordering instrns: \n";
      for(unsigned i=0; i < TmpVec.size(); i++)
	cerr << *(TmpVec[i]);
    }

    // copy the results back from TmpVec to InstrnsBefore
    for(unsigned i=0; i < TmpVec.size(); i++)
      CallAI->InstrnsBefore.push_back( TmpVec[i] );
  }


  // Reset optional args area again to be safe
  PRA.mcInfo.resetOptionalArgs(target);
  
  
}

//---------------------------------------------------------------------------
// This method is called for an LLVM return instruction to identify which
// values will be returned from this method and to suggest colors.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestReg4RetValue(const MachineInstr *const RetMI, 
					     LiveRangeInfo& LRI) const {

  assert( (UltraSparcInfo->getInstrInfo()).isReturn( RetMI->getOpCode() ) );

    suggestReg4RetAddr(RetMI, LRI);

  // if there is an implicit ref, that has to be the ret value
  if(  RetMI->getNumImplicitRefs() > 0 ) {

    // The first implicit operand is the return value of a return instr
    const Value *RetVal =  RetMI->getImplicitRef(0);

    MachineInstr *AdMI;
    LiveRange *const LR = LRI.getLiveRangeForValue( RetVal ); 

    if( !LR ) {
     cerr << "\nNo LR for:";
     printValue( RetVal );
     cerr << endl;
     assert( LR && "No LR for return value of non-void method");
     //return;
   }

    unsigned RegClassID = (LR->getRegClass())->getID();
      
    if( RegClassID == IntRegClassID ) 
      LR->setSuggestedColor(SparcIntRegOrder::i0);
    
    else if ( RegClassID == FloatRegClassID ) 
      LR->setSuggestedColor(SparcFloatRegOrder::f0);
      
  }

}



//---------------------------------------------------------------------------
// Colors the return value of a method to %i0 or %f0, if possible. If it is
// not possilbe to directly color the LR, insert a copy instruction to move
// the LR to %i0 or %f0. When the LR is spilled, instead of the copy, we 
// have to put a load instruction.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::colorRetValue(const  MachineInstr *const RetMI, 
				      LiveRangeInfo& LRI,
				      AddedInstrns *const RetAI) const {

  assert( (UltraSparcInfo->getInstrInfo()).isReturn( RetMI->getOpCode() ) );

  // if there is an implicit ref, that has to be the ret value
  if(  RetMI->getNumImplicitRefs() > 0 ) {

    // The first implicit operand is the return value of a return instr
    const Value *RetVal =  RetMI->getImplicitRef(0);

    MachineInstr *AdMI;
    LiveRange *const LR = LRI.getLiveRangeForValue( RetVal ); 

    if( ! LR ) {
	cerr << "\nNo LR for:";
	printValue( RetVal );
	cerr << endl;
	// assert( LR && "No LR for return value of non-void method");
	return;
    }

    unsigned RegClassID =  getRegClassIDOfValue(RetVal);
    unsigned RegType = getRegType( RetVal );

    unsigned CorrectCol;
    if(RegClassID == IntRegClassID)
      CorrectCol = SparcIntRegOrder::i0;
    else if(RegClassID == FloatRegClassID)
      CorrectCol = SparcFloatRegOrder::f0;
    else {
      assert( 0 && "Unknown RegClass");
      return;
    }

    // if the LR received the correct color, NOTHING to do

    if(  LR->hasColor() )
      if( LR->getColor() == CorrectCol )
	return;

    unsigned UniRetReg =  getUnifiedRegNum( RegClassID, CorrectCol );

    if( LR->hasColor() ) {

      // We are here because the LR was allocted a regiter
      // It may be the suggested register or not

      // copy the LR of retun value to i0 or f0

      unsigned UniLRReg =getUnifiedRegNum( RegClassID, LR->getColor());

      // the LR received  UniLRReg but must be colored with UniRetReg
      // to pass as the return value

      AdMI = cpReg2RegMI( UniLRReg, UniRetReg, RegType); 
      RetAI->InstrnsBefore.push_back( AdMI );
    }
    else {                              // if the LR is spilled

      AdMI = cpMem2RegMI(getFramePointer(), LR->getSpillOffFromFP(), 
			 UniRetReg, RegType); 
      RetAI->InstrnsBefore.push_back( AdMI );
      cout << "\nCopied the return value from stack";
    }
  
  } // if there is a return value

}


//---------------------------------------------------------------------------
// Copy from a register to register. Register number must be the unified
// register number
//---------------------------------------------------------------------------

MachineInstr * UltraSparcRegInfo::cpReg2RegMI(const unsigned SrcReg, 
					      const unsigned DestReg,
					      const int RegType) const {

  assert( ((int)SrcReg != InvalidRegNum) && ((int)DestReg != InvalidRegNum) &&
	  "Invalid Register");
  
  MachineInstr * MI = NULL;

  switch( RegType ) {
    
  case IntRegType:
  case IntCCRegType:
  case FloatCCRegType: 
    MI = new MachineInstr(ADD, 3);
    MI->SetMachineOperand(0, SrcReg, false);
    MI->SetMachineOperand(1, SparcIntRegOrder::g0, false);
    MI->SetMachineOperand(2, DestReg, true);
    break;

  case FPSingleRegType:
    MI = new MachineInstr(FMOVS, 2);
    MI->SetMachineOperand(0, SrcReg, false);
    MI->SetMachineOperand(1, DestReg, true);
    break;

  case FPDoubleRegType:
    MI = new MachineInstr(FMOVD, 2);
    MI->SetMachineOperand(0, SrcReg, false);    
    MI->SetMachineOperand(1, DestReg, true);
    break;

  default:
    assert(0 && "Unknow RegType");
  }

  return MI;
}


//---------------------------------------------------------------------------
// Copy from a register to memory (i.e., Store). Register number must 
// be the unified register number
//---------------------------------------------------------------------------


MachineInstr * UltraSparcRegInfo::cpReg2MemMI(const unsigned SrcReg, 
					      const unsigned DestPtrReg,
					      const int Offset,
					      const int RegType) const {


  MachineInstr * MI = NULL;

  switch( RegType ) {
    
  case IntRegType:
  case FloatCCRegType: 
    MI = new MachineInstr(STX, 3);
    MI->SetMachineOperand(0, SrcReg, false);
    MI->SetMachineOperand(1, DestPtrReg, false);
    MI->SetMachineOperand(2, MachineOperand:: MO_SignExtendedImmed, 
			  (int64_t) Offset, false);
    break;

  case FPSingleRegType:
    MI = new MachineInstr(ST, 3);
    MI->SetMachineOperand(0, SrcReg, false);
    MI->SetMachineOperand(1, DestPtrReg, false);
    MI->SetMachineOperand(2, MachineOperand:: MO_SignExtendedImmed, 
			  (int64_t) Offset, false);
    break;

  case FPDoubleRegType:
    MI = new MachineInstr(STD, 3);
    MI->SetMachineOperand(0, SrcReg, false);
    MI->SetMachineOperand(1, DestPtrReg, false);
    MI->SetMachineOperand(2, MachineOperand:: MO_SignExtendedImmed, 
			  (int64_t) Offset, false);
    break;

  case IntCCRegType:
    assert( 0 && "Cannot directly store %ccr to memory");
    
  default:
    assert(0 && "Unknow RegType in cpReg2MemMI");
  }

  return MI;
}


//---------------------------------------------------------------------------
// Copy from memory to a reg (i.e., Load) Register number must be the unified
// register number
//---------------------------------------------------------------------------


MachineInstr * UltraSparcRegInfo::cpMem2RegMI(const unsigned SrcPtrReg,	
					      const int Offset,
					      const unsigned DestReg,
					      const int RegType) const {
  
  MachineInstr * MI = NULL;

  switch( RegType ) {
    
  case IntRegType:
  case FloatCCRegType: 
    MI = new MachineInstr(LDX, 3);
    MI->SetMachineOperand(0, SrcPtrReg, false);
    MI->SetMachineOperand(1, MachineOperand:: MO_SignExtendedImmed, 
			  (int64_t) Offset, false);
    MI->SetMachineOperand(2, DestReg, true);
    break;

  case FPSingleRegType:
    MI = new MachineInstr(LD, 3);
    MI->SetMachineOperand(0, SrcPtrReg, false);
    MI->SetMachineOperand(1, MachineOperand:: MO_SignExtendedImmed, 
			  (int64_t) Offset, false);
    MI->SetMachineOperand(2, DestReg, true);

    break;

  case FPDoubleRegType:
    MI = new MachineInstr(LDD, 3);
    MI->SetMachineOperand(0, SrcPtrReg, false);
    MI->SetMachineOperand(1, MachineOperand:: MO_SignExtendedImmed, 
			  (int64_t) Offset, false);
    MI->SetMachineOperand(2, DestReg, true);
    break;

  case IntCCRegType:
    assert( 0 && "Cannot directly load into %ccr from memory");

  default:
    assert(0 && "Unknown RegType in cpMem2RegMI");
  }

  return MI;
}





//---------------------------------------------------------------------------
// Generate a copy instruction to copy a value to another. Temporarily
// used by PhiElimination code.
//---------------------------------------------------------------------------


MachineInstr * UltraSparcRegInfo::cpValue2Value(Value *Src, Value *Dest) const{

  int RegType = getRegType( Src );

  assert( (RegType==getRegType(Src))  && "Src & Dest are diff types");

  MachineInstr * MI = NULL;

  switch( RegType ) {
    
  case IntRegType:

    MI = new MachineInstr(ADD, 3);
    MI->SetMachineOperand(0, MachineOperand:: MO_VirtualRegister, Src, false);
    MI->SetMachineOperand(1, SparcIntRegOrder::g0, false);
    MI->SetMachineOperand(2, MachineOperand:: MO_VirtualRegister, Dest, true);
    break;

  case FPSingleRegType:
    MI = new MachineInstr(FMOVS, 2);
    MI->SetMachineOperand(0, MachineOperand:: MO_VirtualRegister, Src, false);
    MI->SetMachineOperand(1, MachineOperand:: MO_VirtualRegister, Dest, true);
    break;


  case FPDoubleRegType:
    MI = new MachineInstr(FMOVD, 2);
    MI->SetMachineOperand(0, MachineOperand:: MO_VirtualRegister, Src, false);
    MI->SetMachineOperand(1, MachineOperand:: MO_VirtualRegister, Dest, true);
    break;

  default:
    assert(0 && "Unknow RegType in CpValu2Value");
  }

  return MI;
}






//----------------------------------------------------------------------------
// This method inserts caller saving/restoring instructons before/after
// a call machine instruction.
//----------------------------------------------------------------------------


void UltraSparcRegInfo::insertCallerSavingCode(const MachineInstr *MInst, 
					       const BasicBlock *BB,
					       PhyRegAlloc &PRA) const {
  // assert( (getInstrInfo()).isCall( MInst->getOpCode() ) );

  // Clear the temp area of the stack
  PRA.mcInfo.popAllTempValues(target);

  hash_set<unsigned> PushedRegSet;

  // Now find the LR of the return value of the call
  // The last *implicit operand* is the return value of a call
  // Insert it to to he PushedRegSet since we must not save that register
  // and restore it after the call.
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but we must not save/restore it.


  const Value *RetVal = getCallInstRetVal( MInst );

  if( RetVal ) {

    LiveRange *RetValLR = PRA.LRI.getLiveRangeForValue( RetVal );
    assert( RetValLR && "No LR for RetValue of call");

    if(  RetValLR->hasColor())
      PushedRegSet.insert(
	 getUnifiedRegNum((RetValLR->getRegClass())->getID(), 
				      RetValLR->getColor() ) );
  }


  const LiveVarSet *LVSetAft =  PRA.LVI->getLiveVarSetAfterMInst(MInst, BB);

  LiveVarSet::const_iterator LIt = LVSetAft->begin();

  // for each live var in live variable set after machine inst
  for( ; LIt != LVSetAft->end(); ++LIt) {

   //  get the live range corresponding to live var
    LiveRange *const LR = PRA.LRI.getLiveRangeForValue(*LIt );    

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if( LR )   {  
      
      if( LR->hasColor() ) {

	unsigned RCID = (LR->getRegClass())->getID();
	unsigned Color = LR->getColor();

	if ( isRegVolatile(RCID, Color) ) {

	  // if the value is in both LV sets (i.e., live before and after 
	  // the call machine instruction)

	  unsigned Reg = getUnifiedRegNum(RCID, Color);
	  
	  if( PushedRegSet.find(Reg) == PushedRegSet.end() ) {
	    
	    // if we haven't already pushed that register

	    unsigned RegType = getRegType( LR );

	    // Now get two instructions - to push on stack and pop from stack
	    // and add them to InstrnsBefore and InstrnsAfter of the
	    // call instruction

    /**** NOTE: THIS SHOULD USE THE RIGHT SIZE FOR THE REG BEING PUSHED ****/
	    int StackOff =  PRA.mcInfo.pushTempValue(target, 8);
                // target.findOptimalStorageSize(LR->getType()));
            
	    MachineInstr *AdIBefCC, *AdIAftCC, *AdICpCC;
	    MachineInstr *AdIBef, *AdIAft;


	    //---- Insert code for pushing the reg on stack ----------
		  
	    if( RegType == IntCCRegType ) {

	      // Handle IntCCRegType specially since we cannot directly 
	      // push %ccr on to the stack

	      const LiveVarSet *LVSetBef = 
		PRA.LVI->getLiveVarSetBeforeMInst(MInst, BB);

	      // get a free INTEGER register
	      int FreeIntReg = 
		PRA.getUsableUniRegAtMI(LR->getRegClass(), IntRegType, MInst, 
				     LVSetBef, AdIBefCC, AdIAftCC);

	      // insert the instructions in reverse order since we are
	      // adding them to the front of InstrnsBefore

	      if(AdIAftCC)
		(PRA.AddedInstrMap[MInst]->InstrnsBefore).push_front(AdIAftCC);

	      AdICpCC = cpCCR2IntMI(FreeIntReg);
	      (PRA.AddedInstrMap[MInst]->InstrnsBefore).push_front(AdICpCC);

	      if(AdIBefCC)
		(PRA.AddedInstrMap[MInst]->InstrnsBefore).push_front(AdIBefCC);

	      if(DEBUG_RA) {
		cerr << "\n!! Inserted caller saving (push) inst for %ccr:";
		if(AdIBefCC) cerr << "\t" <<  *(AdIBefCC);
		cerr  << "\t" << *AdICpCC;
		if(AdIAftCC) cerr  << "\t" << *(AdIAftCC);
	      }

	    } else  {  
	      // for any other register type, just add the push inst
	      AdIBef = cpReg2MemMI(Reg, getFramePointer(), StackOff, RegType );
	      ((PRA.AddedInstrMap[MInst])->InstrnsBefore).push_front(AdIBef);
	    }


	    //---- Insert code for popping the reg from the stack ----------

	    if( RegType == IntCCRegType ) {

	      // Handle IntCCRegType specially since we cannot directly 
	      // pop %ccr on from the stack
	      
	      // get a free INT register
	      int FreeIntReg = 
		PRA.getUsableUniRegAtMI(LR->getRegClass(), IntRegType, MInst, 
				     LVSetAft, AdIBefCC, AdIAftCC);
	      
	      if(AdIBefCC)
		(PRA.AddedInstrMap[MInst]->InstrnsAfter).push_back(AdIBefCC);

	      AdICpCC = cpInt2CCRMI(FreeIntReg);
	      (PRA.AddedInstrMap[MInst]->InstrnsAfter).push_back(AdICpCC);
	    
	      if(AdIAftCC)
		(PRA.AddedInstrMap[MInst]->InstrnsAfter).push_back(AdIAftCC);

	      if(DEBUG_RA) {

		cerr << "\n!! Inserted caller saving (pop) inst for %ccr:";
		if(AdIBefCC) cerr << "\t" <<  *(AdIBefCC);
		cerr  << "\t" << *AdICpCC;
		if(AdIAftCC) cerr  << "\t" << *(AdIAftCC);
	      }

	    } else {
	      // for any other register type, just add the pop inst
	      AdIAft = cpMem2RegMI(getFramePointer(), StackOff, Reg, RegType );
	      ((PRA.AddedInstrMap[MInst])->InstrnsAfter).push_back(AdIAft);
	    }
	    
	    PushedRegSet.insert( Reg );

	    if(DEBUG_RA) {
	      cerr << "\nFor call inst:" << *MInst;
	      cerr << " -inserted caller saving instrs:\n\t ";
              if( RegType == IntCCRegType )
                cerr << *AdIBefCC << "\t" << *AdIAftCC  ;
              else
                cerr << *AdIBef   << "\t" << *AdIAft    ;
	    }	    
	  } // if not already pushed

	} // if LR has a volatile color
	
      } // if LR has color

    } // if there is a LR for Var
    
  } // for each value in the LV set after instruction
  
  // Clear the temp area of the stack
  PRA.mcInfo.popAllTempValues(target);

}

//---------------------------------------------------------------------------
// Copies %ccr into an integer register. IntReg is the UNIFIED register
// number.
//---------------------------------------------------------------------------

MachineInstr * UltraSparcRegInfo::cpCCR2IntMI(const unsigned IntReg) const {
  MachineInstr * MI = NULL;

  MI = new MachineInstr(RDCCR, 2);
  MI->SetMachineOperand(0, SparcIntCCRegOrder::ccr, false);
  MI->SetMachineOperand(1, IntReg, true);

  return MI;
}

//---------------------------------------------------------------------------
// Copies an integer register into  %ccr. IntReg is the UNIFIED register
// number.
//---------------------------------------------------------------------------

MachineInstr * UltraSparcRegInfo::cpInt2CCRMI(const unsigned IntReg) const {
  MachineInstr * MI = NULL;

  MI = new MachineInstr(WRCCR, 3);
  MI->SetMachineOperand(0, IntReg, false);
  MI->SetMachineOperand(1, SparcIntRegOrder::g0, false);
  MI->SetMachineOperand(2, SparcIntCCRegOrder::ccr, true);

  return MI;
}




//---------------------------------------------------------------------------
// Print the register assigned to a LR
//---------------------------------------------------------------------------

void UltraSparcRegInfo::printReg(const LiveRange *const LR) {

  unsigned RegClassID = (LR->getRegClass())->getID();

  cerr << " *Node " << (LR->getUserIGNode())->getIndex();

  if( ! LR->hasColor() ) {
    cerr << " - could not find a color" << endl;
    return;
  }
  
  // if a color is found

  cerr << " colored with color "<< LR->getColor();

  if( RegClassID == IntRegClassID ) {

    cerr<< " [" << SparcIntRegOrder::getRegName(LR->getColor()) ;
    cerr << "]" << endl;
  }
  else if ( RegClassID == FloatRegClassID) {
    cerr << "[" << SparcFloatRegOrder::getRegName(LR->getColor());
    if( LR->getTypeID() == Type::DoubleTyID )
      cerr << "+" << SparcFloatRegOrder::getRegName(LR->getColor()+1);
    cerr << "]" << endl;
  }
}

//---------------------------------------------------------------------------
// This method examines instructions inserted by RegAlloc code before a
// machine instruction to detect invalid orders that destroy values before
// they are used. If it detects such conditions, it reorders the instructions.
//
// The unordered instructions come in the UnordVec. These instructions are
// instructions inserted by RegAlloc. All such instruction MUST have 
// their USES BEFORE THE DEFS after reordering.

// The UnordVec & OrdVec must be DISTINCT. The OrdVec must be empty when
// this method is called.

// This method uses two vectors for efficiency in accessing

// Since instructions are inserted in RegAlloc, this assumes that the 
// first operand is the source reg and the last operand is the dest reg.

// All the uses are before THE def to a register


//---------------------------------------------------------------------------
void UltraSparcRegInfo::OrderAddedInstrns( vector<MachineInstr *> &UnordVec, 
					   vector<MachineInstr *> &OrdVec,
					   PhyRegAlloc &PRA) const{

  /*
    Problem: We can have instructions inserted by RegAlloc like
    1. add %ox %g0 %oy
    2. add %oy %g0 %oz, where z!=x or z==x

    This is wrong since %oy used by 2 is overwritten by 1
  
    Solution:
    We re-order the instructions so that the uses are before the defs

    Algorithm:
    
    do
      for each instruction 'DefInst' in the UnOrdVec
         for each instruction 'UseInst' that follows the DefInst
           if the reg defined by DefInst is used by UseInst
	     mark DefInst as not movable in this iteration
	 If DefInst is not marked as not-movable, move DefInst to OrdVec
    while all instructions in DefInst are moved to OrdVec
    
    For moving, we call the move2OrdVec(). It checks whether there is a def
    in it for the uses in the instruction to be added to OrdVec. If there
    are no preceding defs, it just appends the instruction. If there is a
    preceding def, it puts two instructions to save the reg on stack before
    the load and puts a restore at use.

  */


  bool CouldMoveAll;
  bool DebugPrint = false;

  do {

    CouldMoveAll = true;

    vector<MachineInstr *>::iterator DefIt = UnordVec.begin();

    for( ; DefIt !=  UnordVec.end(); ++DefIt ) {

      // for each instruction in the UnordVec do ...

      MachineInstr *DefInst = *DefIt;

      if( DefInst == NULL) continue;

      //cerr << "\nInst in UnordVec = " <<  *DefInst;
      
      // last operand is the def (unless for a store which has no def reg)
      MachineOperand& DefOp = DefInst->getOperand(DefInst->getNumOperands()-1);
      
      if( DefOp.opIsDef() &&  
	  DefOp.getOperandType() ==  MachineOperand::MO_MachineRegister) {
	
	// If the operand in DefInst is a def ...
	
	bool DefEqUse = false;
	
	vector<MachineInstr *>::iterator UseIt = DefIt;
	UseIt++;
	
	for( ; UseIt !=  UnordVec.end(); ++UseIt ) {

	  MachineInstr *UseInst = *UseIt;
	  if( UseInst == NULL) continue;
	  
	  // for each inst (UseInst) that is below the DefInst do ...
	  

	  MachineOperand& UseOp = UseInst->getOperand(0);
	  
	  if( ! UseOp.opIsDef() &&  
	      UseOp.getOperandType() == MachineOperand::MO_MachineRegister) {
	    
	    // if use is a register ...
	    
	    if( DefOp.getMachineRegNum() == UseOp.getMachineRegNum() ) {
	      
	      // if Def and this use are the same, it means that this use
	      // is destroyed by a def before it is used
	      
	      // cerr << "\nCouldn't move " << *DefInst;

	      DefEqUse = true;
	      CouldMoveAll = false;	
	      DebugPrint = true;
	      break;
	    } // if two registers are equal
	    
	  } // if use is a register
	  
	}// for all use instructions
	
	if( ! DefEqUse ) {
	  
	  // after examining all the instructions that follow the DefInst
	  // if there are no dependencies, we can move it to the OrdVec

	  // cerr << "Moved to Ord: " << *DefInst;

	  moveInst2OrdVec(OrdVec, DefInst, PRA);

	  //OrdVec.push_back(DefInst);

	  // mark the pos of DefInst with NULL to indicate that it is
	  // empty
	  *DefIt = NULL;
	}
    
      } // if Def is a machine register
      
    } // for all instructions in the UnordVec
    

  } while( !CouldMoveAll);


  if(DebugPrint) {
    cerr << "\nAdded instructions were reordered to:\n";
    for(unsigned int i=0; i < OrdVec.size(); i++)
      cerr << *(OrdVec[i]);
  }

}








void UltraSparcRegInfo::moveInst2OrdVec(vector<MachineInstr *> &OrdVec,
					MachineInstr *UnordInst,
					PhyRegAlloc &PRA ) const {

  MachineOperand& UseOp = UnordInst->getOperand(0);

  if( ! UseOp.opIsDef() &&  
      UseOp.getOperandType() ==  MachineOperand::MO_MachineRegister) {

    // for the use of UnordInst, see whether there is a defining instr
    // before in the OrdVec
    bool DefEqUse = false;

    vector<MachineInstr *>::iterator OrdIt = OrdVec.begin();
  
    for( ; OrdIt !=  OrdVec.end(); ++OrdIt ) {

      MachineInstr *OrdInst = *OrdIt ;

      MachineOperand& DefOp = 
	OrdInst->getOperand(OrdInst->getNumOperands()-1);

      if( DefOp.opIsDef() &&  
	  DefOp.getOperandType() == MachineOperand::MO_MachineRegister) {

	//cerr << "\nDefining Ord Inst: " <<  *OrdInst;
	  
	if( DefOp.getMachineRegNum() == UseOp.getMachineRegNum() ) {

	  // we are here because there is a preceding def in the OrdVec 
	  // for the use in this intr we are going to insert. This
	  // happened because the original code was like:
	  // 1. add %ox %g0 %oy
	  // 2. add %oy %g0 %ox
	  // In Round1, we added 2 to OrdVec but 1 remained in UnordVec
	  // Now we are processing %ox of 1.
	  // We have to 
	      
	  const int UReg = DefOp.getMachineRegNum();
	  const int RegType = getRegType(UReg);
	  MachineInstr *AdIBef, *AdIAft;
	      
	  // TODO: Change 8 below
    /**** NOTE: THIS SHOULD USE THE RIGHT SIZE FOR THE REG BEING PUSHED ****/
	  const int StackOff =  PRA.mcInfo.pushTempValue(target, 8);
	  
	  // Save the UReg (%ox) on stack before it's destroyed
	  AdIBef=cpReg2MemMI(UReg, getFramePointer(), StackOff, RegType);
	  OrdIt = OrdVec.insert( OrdIt, AdIBef);
	  OrdIt++;  // points to current instr we processed
	  
	  // Load directly into DReg (%oy)
	  MachineOperand&  DOp=
	    (UnordInst->getOperand(UnordInst->getNumOperands()-1));
	  assert(DOp.opIsDef() && "Last operand is not the def");
	  const int DReg = DOp.getMachineRegNum();
	  
	  AdIAft=cpMem2RegMI(getFramePointer(), StackOff, DReg, RegType);
	  OrdVec.push_back(AdIAft);
	    
	  cerr << "\nFixed CIRCULAR references by reordering";

	  if( DEBUG_RA ) {
	    cerr << "\nBefore CIRCULAR Reordering:\n";
	    cerr << *UnordInst;
	    cerr << *OrdInst;
	  
	    cerr << "\nAfter CIRCULAR Reordering - All Inst so far:\n";
	    for(unsigned i=0; i < OrdVec.size(); i++)
	      cerr << *(OrdVec[i]);
	  }
	  
	  // Do not copy the UseInst to OrdVec
	  DefEqUse = true;
	  break;  
	  
	}// if two registers are equal

      } // if Def is a register

    } // for each instr in OrdVec

    if( !DefEqUse ) {  

      // We didn't find a def in the OrdVec, so just append this inst
      OrdVec.push_back( UnordInst );  
      //cerr << "Reordered Inst (Moved Dn): " <<  *UnordInst;
    }
    
  }// if the operand in UnordInst is a use
 
}






