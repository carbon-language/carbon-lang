//===-- SparcRegInfo.cpp - Sparc Target Register Information --------------===//
//
// This file contains implementation of Sparc specific helper methods
// used for register allocation.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "SparcRegClassInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/PhyRegAlloc.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/FunctionLiveVarInfo.h"   // FIXME: Remove
#include "../../CodeGen/RegAlloc/RegAllocCommon.h"   // FIXME!
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"

enum {
  BadRegClass = ~0
};

UltraSparcRegInfo::UltraSparcRegInfo(const UltraSparc &tgt)
  : TargetRegInfo(tgt), NumOfIntArgRegs(6), NumOfFloatArgRegs(32)
{
  MachineRegClassArr.push_back(new SparcIntRegClass(IntRegClassID));
  MachineRegClassArr.push_back(new SparcFloatRegClass(FloatRegClassID));
  MachineRegClassArr.push_back(new SparcIntCCRegClass(IntCCRegClassID));
  MachineRegClassArr.push_back(new SparcFloatCCRegClass(FloatCCRegClassID));
  MachineRegClassArr.push_back(new SparcSpecialRegClass(SpecialRegClassID));
  
  assert(SparcFloatRegClass::StartOfNonVolatileRegs == 32 && 
         "32 Float regs are used for float arg passing");
}


// getZeroRegNum - returns the register that contains always zero.
// this is the unified register number
//
int UltraSparcRegInfo::getZeroRegNum() const {
  return getUnifiedRegNum(UltraSparcRegInfo::IntRegClassID,
                          SparcIntRegClass::g0);
}

// getCallAddressReg - returns the reg used for pushing the address when a
// method is called. This can be used for other purposes between calls
//
unsigned UltraSparcRegInfo::getCallAddressReg() const {
  return getUnifiedRegNum(UltraSparcRegInfo::IntRegClassID,
                          SparcIntRegClass::o7);
}

// Returns the register containing the return address.
// It should be made sure that this  register contains the return 
// value when a return instruction is reached.
//
unsigned UltraSparcRegInfo::getReturnAddressReg() const {
  return getUnifiedRegNum(UltraSparcRegInfo::IntRegClassID,
                          SparcIntRegClass::i7);
}

// Register get name implementations...

// Int register names in same order as enum in class SparcIntRegClass
static const char * const IntRegNames[] = {
  "o0", "o1", "o2", "o3", "o4", "o5",       "o7",
  "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
  "i0", "i1", "i2", "i3", "i4", "i5",  
  "i6", "i7",
  "g0", "g1", "g2", "g3", "g4", "g5",  "g6", "g7", 
  "o6"
}; 

const char * const SparcIntRegClass::getRegName(unsigned reg) const {
  assert(reg < NumOfAllRegs);
  return IntRegNames[reg];
}

static const char * const FloatRegNames[] = {    
  "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",  "f8",  "f9", 
  "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19",
  "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29",
  "f30", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39",
  "f40", "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49",
  "f50", "f51", "f52", "f53", "f54", "f55", "f56", "f57", "f58", "f59",
  "f60", "f61", "f62", "f63"
};

const char * const SparcFloatRegClass::getRegName(unsigned reg) const {
  assert (reg < NumOfAllRegs);
  return FloatRegNames[reg];
}


static const char * const IntCCRegNames[] = {    
  "xcc",  "icc",  "ccr"
};

const char * const SparcIntCCRegClass::getRegName(unsigned reg) const {
  assert(reg < 3);
  return IntCCRegNames[reg];
}

static const char * const FloatCCRegNames[] = {    
  "fcc0", "fcc1",  "fcc2",  "fcc3"
};

const char * const SparcFloatCCRegClass::getRegName(unsigned reg) const {
  assert (reg < 5);
  return FloatCCRegNames[reg];
}

static const char * const SpecialRegNames[] = {    
  "fsr"
};

const char * const SparcSpecialRegClass::getRegName(unsigned reg) const {
  assert (reg < 1);
  return SpecialRegNames[reg];
}

// Get unified reg number for frame pointer
unsigned UltraSparcRegInfo::getFramePointer() const {
  return getUnifiedRegNum(UltraSparcRegInfo::IntRegClassID,
                          SparcIntRegClass::i6);
}

// Get unified reg number for stack pointer
unsigned UltraSparcRegInfo::getStackPointer() const {
  return getUnifiedRegNum(UltraSparcRegInfo::IntRegClassID,
                          SparcIntRegClass::o6);
}


//---------------------------------------------------------------------------
// Finds whether a call is an indirect call
//---------------------------------------------------------------------------

inline bool
isVarArgsFunction(const Type *funcType) {
  return cast<FunctionType>(cast<PointerType>(funcType)
                            ->getElementType())->isVarArg();
}

inline bool
isVarArgsCall(const MachineInstr *CallMI) {
  Value* callee = CallMI->getOperand(0).getVRegValue();
  // const Type* funcType = isa<Function>(callee)? callee->getType()
  //   : cast<PointerType>(callee->getType())->getElementType();
  const Type* funcType = callee->getType();
  return isVarArgsFunction(funcType);
}


// Get the register number for the specified argument #argNo,
// 
// Return value:
//      getInvalidRegNum(),  if there is no int register available for the arg. 
//      regNum,              otherwise (this is NOT the unified reg. num).
//                           regClassId is set to the register class ID.
// 
int
UltraSparcRegInfo::regNumForIntArg(bool inCallee, bool isVarArgsCall,
                                   unsigned argNo, unsigned& regClassId) const
{
  regClassId = IntRegClassID;
  if (argNo >= NumOfIntArgRegs)
    return getInvalidRegNum();
  else
    return argNo + (inCallee? SparcIntRegClass::i0 : SparcIntRegClass::o0);
}

// Get the register number for the specified FP argument #argNo,
// Use INT regs for FP args if this is a varargs call.
// 
// Return value:
//      getInvalidRegNum(),  if there is no int register available for the arg. 
//      regNum,              otherwise (this is NOT the unified reg. num).
//                           regClassId is set to the register class ID.
// 
int
UltraSparcRegInfo::regNumForFPArg(unsigned regType,
                                  bool inCallee, bool isVarArgsCall,
                                  unsigned argNo, unsigned& regClassId) const
{
  if (isVarArgsCall)
    return regNumForIntArg(inCallee, isVarArgsCall, argNo, regClassId);
  else
    {
      regClassId = FloatRegClassID;
      if (regType == FPSingleRegType)
        return (argNo*2+1 >= NumOfFloatArgRegs)?
          getInvalidRegNum() : SparcFloatRegClass::f0 + (argNo * 2 + 1);
      else if (regType == FPDoubleRegType)
        return (argNo*2 >= NumOfFloatArgRegs)?
          getInvalidRegNum() : SparcFloatRegClass::f0 + (argNo * 2);
      else
        assert(0 && "Illegal FP register type");
	return 0;
    }
}


//---------------------------------------------------------------------------
// Finds the return address of a call sparc specific call instruction
//---------------------------------------------------------------------------

// The following 4  methods are used to find the RegType (SparcInternals.h)
// of a LiveRange, a Value, and for a given register unified reg number.
//
int UltraSparcRegInfo::getRegTypeForClassAndType(unsigned regClassID,
                                                 const Type* type) const
{
  switch (regClassID) {
  case IntRegClassID:                   return IntRegType; 
  case FloatRegClassID:
    if (type == Type::FloatTy)          return FPSingleRegType;
    else if (type == Type::DoubleTy)    return FPDoubleRegType;
    assert(0 && "Unknown type in FloatRegClass"); return 0;
  case IntCCRegClassID:                 return IntCCRegType; 
  case FloatCCRegClassID:               return FloatCCRegType; 
  case SpecialRegClassID:               return SpecialRegType; 
  default: assert( 0 && "Unknown reg class ID"); return 0;
  }
}

int UltraSparcRegInfo::getRegTypeForDataType(const Type* type) const
{
  return getRegTypeForClassAndType(getRegClassIDOfType(type), type);
}

int UltraSparcRegInfo::getRegTypeForLR(const LiveRange *LR) const
{
  return getRegTypeForClassAndType(LR->getRegClassID(), LR->getType());
}

int UltraSparcRegInfo::getRegType(int unifiedRegNum) const
{
  if (unifiedRegNum < 32) 
    return IntRegType;
  else if (unifiedRegNum < (32 + 32))
    return FPSingleRegType;
  else if (unifiedRegNum < (64 + 32))
    return FPDoubleRegType;
  else if (unifiedRegNum < (64+32+4))
    return FloatCCRegType;
  else if (unifiedRegNum < (64+32+4+2))  
    return IntCCRegType;             
  else 
    assert(0 && "Invalid unified register number in getRegType");
  return 0;
}


// To find the register class used for a specified Type
//
unsigned UltraSparcRegInfo::getRegClassIDOfType(const Type *type,
                                                bool isCCReg) const {
  Type::PrimitiveID ty = type->getPrimitiveID();
  unsigned res;
    
  // FIXME: Comparing types like this isn't very safe...
  if ((ty && ty <= Type::LongTyID) || (ty == Type::LabelTyID) ||
      (ty == Type::FunctionTyID) ||  (ty == Type::PointerTyID) )
    res = IntRegClassID;             // sparc int reg (ty=0: void)
  else if (ty <= Type::DoubleTyID)
    res = FloatRegClassID;           // sparc float reg class
  else { 
    //std::cerr << "TypeID: " << ty << "\n";
    assert(0 && "Cannot resolve register class for type");
    return 0;
  }
  
  if (isCCReg)
    return res + 2;      // corresponding condition code register 
  else 
    return res;
}

unsigned UltraSparcRegInfo::getRegClassIDOfRegType(int regType) const {
  switch(regType) {
  case IntRegType:      return IntRegClassID;
  case FPSingleRegType:
  case FPDoubleRegType: return FloatRegClassID;
  case IntCCRegType:    return IntCCRegClassID;
  case FloatCCRegType:  return FloatCCRegClassID;
  default:
    assert(0 && "Invalid register type in getRegClassIDOfRegType");
    return 0;
  }
}

//---------------------------------------------------------------------------
// Suggests a register for the ret address in the RET machine instruction.
// We always suggest %i7 by convention.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestReg4RetAddr(MachineInstr *RetMI, 
					   LiveRangeInfo& LRI) const {

  assert(target.getInstrInfo().isReturn(RetMI->getOpCode()));
  
  // return address is always mapped to i7 so set it immediately
  RetMI->SetRegForOperand(0, getUnifiedRegNum(IntRegClassID,
                                              SparcIntRegClass::i7));
  
  // Possible Optimization: 
  // Instead of setting the color, we can suggest one. In that case,
  // we have to test later whether it received the suggested color.
  // In that case, a LR has to be created at the start of method.
  // It has to be done as follows (remove the setRegVal above):

  // MachineOperand & MO  = RetMI->getOperand(0);
  // const Value *RetAddrVal = MO.getVRegValue();
  // assert( RetAddrVal && "LR for ret address must be created at start");
  // LiveRange * RetAddrLR = LRI.getLiveRangeForValue( RetAddrVal);  
  // RetAddrLR->setSuggestedColor(getUnifiedRegNum( IntRegClassID, 
  //                              SparcIntRegOrdr::i7) );
}


//---------------------------------------------------------------------------
// Suggests a register for the ret address in the JMPL/CALL machine instr.
// Sparc ABI dictates that %o7 be used for this purpose.
//---------------------------------------------------------------------------
void
UltraSparcRegInfo::suggestReg4CallAddr(MachineInstr * CallMI,
                                       LiveRangeInfo& LRI) const
{
  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(CallMI); 
  const Value *RetAddrVal = argDesc->getReturnAddrReg();
  assert(RetAddrVal && "INTERNAL ERROR: Return address value is required");

  // A LR must already exist for the return address.
  LiveRange *RetAddrLR = LRI.getLiveRangeForValue(RetAddrVal);
  assert(RetAddrLR && "INTERNAL ERROR: No LR for return address of call!");

  unsigned RegClassID = RetAddrLR->getRegClassID();
  RetAddrLR->setColor(getUnifiedRegNum(IntRegClassID, SparcIntRegClass::o7));
}



//---------------------------------------------------------------------------
//  This method will suggest colors to incoming args to a method. 
//  According to the Sparc ABI, the first 6 incoming args are in 
//  %i0 - %i5 (if they are integer) OR in %f0 - %f31 (if they are float).
//  If the arg is passed on stack due to the lack of regs, NOTHING will be
//  done - it will be colored (or spilled) as a normal live range.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestRegs4MethodArgs(const Function *Meth, 
					       LiveRangeInfo& LRI) const 
{
  // Check if this is a varArgs function. needed for choosing regs.
  bool isVarArgs = isVarArgsFunction(Meth->getType());
  
  // Count the arguments, *ignoring* whether they are int or FP args.
  // Use this common arg numbering to pick the right int or fp register.
  unsigned argNo=0;
  for(Function::const_aiterator I = Meth->abegin(), E = Meth->aend();
      I != E; ++I, ++argNo) {
    LiveRange *LR = LRI.getLiveRangeForValue(I);
    assert(LR && "No live range found for method arg");
    
    unsigned regType = getRegTypeForLR(LR);
    unsigned regClassIDOfArgReg = BadRegClass; // for chosen reg (unused)
    
    int regNum = (regType == IntRegType)
      ? regNumForIntArg(/*inCallee*/ true, isVarArgs, argNo, regClassIDOfArgReg)
      : regNumForFPArg(regType, /*inCallee*/ true, isVarArgs, argNo,
                       regClassIDOfArgReg); 
    
    if (regNum != getInvalidRegNum())
      LR->setSuggestedColor(regNum);
  }
}


//---------------------------------------------------------------------------
// This method is called after graph coloring to move incoming args to
// the correct hardware registers if they did not receive the correct
// (suggested) color through graph coloring.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::colorMethodArgs(const Function *Meth, 
					LiveRangeInfo &LRI,
					AddedInstrns *FirstAI) const {

  // check if this is a varArgs function. needed for choosing regs.
  bool isVarArgs = isVarArgsFunction(Meth->getType());
  MachineInstr *AdMI;

  // for each argument
  // for each argument.  count INT and FP arguments separately.
  unsigned argNo=0, intArgNo=0, fpArgNo=0;
  for(Function::const_aiterator I = Meth->abegin(), E = Meth->aend();
      I != E; ++I, ++argNo) {
    // get the LR of arg
    LiveRange *LR = LRI.getLiveRangeForValue(I);
    assert( LR && "No live range found for method arg");

    unsigned regType = getRegTypeForLR(LR);
    unsigned RegClassID = LR->getRegClassID();
    
    // Find whether this argument is coming in a register (if not, on stack)
    // Also find the correct register the argument must use (UniArgReg)
    //
    bool isArgInReg = false;
    unsigned UniArgReg = getInvalidRegNum(); // reg that LR MUST be colored with
    unsigned regClassIDOfArgReg = BadRegClass; // reg class of chosen reg
    
    int regNum = (regType == IntRegType)
      ? regNumForIntArg(/*inCallee*/ true, isVarArgs,
                        argNo, regClassIDOfArgReg)
      : regNumForFPArg(regType, /*inCallee*/ true, isVarArgs,
                       argNo, regClassIDOfArgReg);
    
    if(regNum != getInvalidRegNum()) {
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum( regClassIDOfArgReg, regNum);
    }
    
    if( ! LR->isMarkedForSpill() ) {    // if this arg received a register

      unsigned UniLRReg = getUnifiedRegNum(  RegClassID, LR->getColor() );

      // if LR received the correct color, nothing to do
      //
      if( UniLRReg == UniArgReg )
	continue;

      // We are here because the LR did not receive the suggested 
      // but LR received another register.
      // Now we have to copy the %i reg (or stack pos of arg) 
      // to the register the LR was colored with.
      
      // if the arg is coming in UniArgReg register, it MUST go into
      // the UniLRReg register
      //
      if( isArgInReg ) {
	if( regClassIDOfArgReg != RegClassID ) {
          assert(0 && "This could should work but it is not tested yet");
          
	  // It is a variable argument call: the float reg must go in a %o reg.
	  // We have to move an int reg to a float reg via memory.
          // 
          assert(isVarArgs &&
                 RegClassID == FloatRegClassID && 
                 regClassIDOfArgReg == IntRegClassID &&
                 "This should only be an Int register for an FP argument");
          
 	  int TmpOff = MachineFunction::get(Meth).getInfo()->pushTempValue(
                                                getSpilledRegSize(regType));
	  cpReg2MemMI(FirstAI->InstrnsBefore,
                      UniArgReg, getFramePointer(), TmpOff, IntRegType);
          
	  cpMem2RegMI(FirstAI->InstrnsBefore,
                      getFramePointer(), TmpOff, UniLRReg, regType);
	}
	else {	
	  cpReg2RegMI(FirstAI->InstrnsBefore, UniArgReg, UniLRReg, regType);
	}
      }
      else {

	// Now the arg is coming on stack. Since the LR recieved a register,
	// we just have to load the arg on stack into that register
	//
        const TargetFrameInfo& frameInfo = target.getFrameInfo();
	int offsetFromFP =
          frameInfo.getIncomingArgOffset(MachineFunction::get(Meth),
                                         argNo);

        // float arguments on stack are right justified so adjust the offset!
        // int arguments are also right justified but they are always loaded as
        // a full double-word so the offset does not need to be adjusted.
        if (regType == FPSingleRegType) {
          unsigned argSize = target.getTargetData().getTypeSize(LR->getType());
          unsigned slotSize = frameInfo.getSizeOfEachArgOnStack();
          assert(argSize <= slotSize && "Insufficient slot size!");
          offsetFromFP += slotSize - argSize;
        }

	cpMem2RegMI(FirstAI->InstrnsBefore,
                    getFramePointer(), offsetFromFP, UniLRReg, regType);
      }
      
    } // if LR received a color

    else {                             

      // Now, the LR did not receive a color. But it has a stack offset for
      // spilling.
      // So, if the arg is coming in UniArgReg register,  we can just move
      // that on to the stack pos of LR

      if( isArgInReg ) {
        
	if( regClassIDOfArgReg != RegClassID ) {
          assert(0 &&
                 "FP arguments to a varargs function should be explicitly "
                 "copied to/from int registers by instruction selection!");
          
	  // It must be a float arg for a variable argument call, which
          // must come in a %o reg.  Move the int reg to the stack.
          // 
          assert(isVarArgs && regClassIDOfArgReg == IntRegClassID &&
                 "This should only be an Int register for an FP argument");
          
          cpReg2MemMI(FirstAI->InstrnsBefore, UniArgReg,
                      getFramePointer(), LR->getSpillOffFromFP(), IntRegType);
        }
        else {
           cpReg2MemMI(FirstAI->InstrnsBefore, UniArgReg,
                       getFramePointer(), LR->getSpillOffFromFP(), regType);
        }
      }

      else {

	// Now the arg is coming on stack. Since the LR did NOT 
	// recieved a register as well, it is allocated a stack position. We
	// can simply change the stack position of the LR. We can do this,
	// since this method is called before any other method that makes
	// uses of the stack pos of the LR (e.g., updateMachineInstr)
        // 
        const TargetFrameInfo& frameInfo = target.getFrameInfo();
	int offsetFromFP =
          frameInfo.getIncomingArgOffset(MachineFunction::get(Meth),
                                         argNo);

        // FP arguments on stack are right justified so adjust offset!
        // int arguments are also right justified but they are always loaded as
        // a full double-word so the offset does not need to be adjusted.
        if (regType == FPSingleRegType) {
          unsigned argSize = target.getTargetData().getTypeSize(LR->getType());
          unsigned slotSize = frameInfo.getSizeOfEachArgOnStack();
          assert(argSize <= slotSize && "Insufficient slot size!");
          offsetFromFP += slotSize - argSize;
        }
        
	LR->modifySpillOffFromFP( offsetFromFP );
      }

    }

  }  // for each incoming argument

}



//---------------------------------------------------------------------------
// This method is called before graph coloring to suggest colors to the
// outgoing call args and the return value of the call.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestRegs4CallArgs(MachineInstr *CallMI, 
					     LiveRangeInfo& LRI) const {
  assert ( (target.getInstrInfo()).isCall(CallMI->getOpCode()) );

  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(CallMI); 
  
  suggestReg4CallAddr(CallMI, LRI);

  // First color the return value of the call instruction, if any.
  // The return value will be in %o0 if the value is an integer type,
  // or in %f0 if the value is a float type.
  // 
  if (const Value *RetVal = argDesc->getReturnValue()) {
    LiveRange *RetValLR = LRI.getLiveRangeForValue(RetVal);
    assert(RetValLR && "No LR for return Value of call!");

    unsigned RegClassID = RetValLR->getRegClassID();

    // now suggest a register depending on the register class of ret arg
    if( RegClassID == IntRegClassID ) 
      RetValLR->setSuggestedColor(SparcIntRegClass::o0);
    else if (RegClassID == FloatRegClassID ) 
      RetValLR->setSuggestedColor(SparcFloatRegClass::f0 );
    else assert( 0 && "Unknown reg class for return value of call\n");
  }

  // Now suggest colors for arguments (operands) of the call instruction.
  // Colors are suggested only if the arg number is smaller than the
  // the number of registers allocated for argument passing.
  // Now, go thru call args - implicit operands of the call MI

  unsigned NumOfCallArgs = argDesc->getNumArgs();
  
  for(unsigned argNo=0, i=0, intArgNo=0, fpArgNo=0;
       i < NumOfCallArgs; ++i, ++argNo) {    

    const Value *CallArg = argDesc->getArgInfo(i).getArgVal();
    
    // get the LR of call operand (parameter)
    LiveRange *const LR = LRI.getLiveRangeForValue(CallArg); 
    if (!LR)
      continue;                    // no live ranges for constants and labels

    unsigned regType = getRegTypeForLR(LR);
    unsigned regClassIDOfArgReg = BadRegClass; // chosen reg class (unused)

    // Choose a register for this arg depending on whether it is
    // an INT or FP value.  Here we ignore whether or not it is a
    // varargs calls, because FP arguments will be explicitly copied
    // to an integer Value and handled under (argCopy != NULL) below.
    int regNum = (regType == IntRegType)
      ? regNumForIntArg(/*inCallee*/ false, /*isVarArgs*/ false,
                        argNo, regClassIDOfArgReg)
      : regNumForFPArg(regType, /*inCallee*/ false, /*isVarArgs*/ false,
                       argNo, regClassIDOfArgReg); 
    
    // If a register could be allocated, use it.
    // If not, do NOTHING as this will be colored as a normal value.
    if(regNum != getInvalidRegNum())
      LR->setSuggestedColor(regNum);
  } // for all call arguments
}


//---------------------------------------------------------------------------
// this method is called for an LLVM return instruction to identify which
// values will be returned from this method and to suggest colors.
//---------------------------------------------------------------------------
void UltraSparcRegInfo::suggestReg4RetValue(MachineInstr *RetMI, 
                                            LiveRangeInfo &LRI) const {

  assert( (target.getInstrInfo()).isReturn( RetMI->getOpCode() ) );

  suggestReg4RetAddr(RetMI, LRI);

  // To find the return value (if any), we can get the LLVM return instr.
  // from the return address register, which is the first operand
  Value* tmpI = RetMI->getOperand(0).getVRegValue();
  ReturnInst* retI=cast<ReturnInst>(cast<TmpInstruction>(tmpI)->getOperand(0));
  if (const Value *RetVal = retI->getReturnValue())
    if (LiveRange *const LR = LRI.getLiveRangeForValue(RetVal))
      LR->setSuggestedColor(LR->getRegClassID() == IntRegClassID
                            ? (unsigned) SparcIntRegClass::i0
                            : (unsigned) SparcFloatRegClass::f0);
}

//---------------------------------------------------------------------------
// Check if a specified register type needs a scratch register to be
// copied to/from memory.  If it does, the reg. type that must be used
// for scratch registers is returned in scratchRegType.
//
// Only the int CC register needs such a scratch register.
// The FP CC registers can (and must) be copied directly to/from memory.
//---------------------------------------------------------------------------

bool
UltraSparcRegInfo::regTypeNeedsScratchReg(int RegType,
                                          int& scratchRegType) const
{
  if (RegType == IntCCRegType)
    {
      scratchRegType = IntRegType;
      return true;
    }
  return false;
}

//---------------------------------------------------------------------------
// Copy from a register to register. Register number must be the unified
// register number.
//---------------------------------------------------------------------------

void
UltraSparcRegInfo::cpReg2RegMI(std::vector<MachineInstr*>& mvec,
                               unsigned SrcReg,
                               unsigned DestReg,
                               int RegType) const {
  assert( ((int)SrcReg != getInvalidRegNum()) && 
          ((int)DestReg != getInvalidRegNum()) &&
	  "Invalid Register");
  
  MachineInstr * MI = NULL;
  
  switch( RegType ) {
    
  case IntCCRegType:
    if (getRegType(DestReg) == IntRegType) {
      // copy intCC reg to int reg
      MI = (BuildMI(V9::RDCCR, 2)
            .addMReg(getUnifiedRegNum(UltraSparcRegInfo::IntCCRegClassID,
                                      SparcIntCCRegClass::ccr))
            .addMReg(DestReg,MOTy::Def));
    } else {
      // copy int reg to intCC reg
      assert(getRegType(SrcReg) == IntRegType
             && "Can only copy CC reg to/from integer reg");
      MI = (BuildMI(V9::WRCCRr, 3)
            .addMReg(SrcReg)
            .addMReg(SparcIntRegClass::g0)
            .addMReg(getUnifiedRegNum(UltraSparcRegInfo::IntCCRegClassID,
                                      SparcIntCCRegClass::ccr), MOTy::Def));
    }
    break;
    
  case FloatCCRegType: 
    assert(0 && "Cannot copy FPCC register to any other register");
    break;
    
  case IntRegType:
    MI = BuildMI(V9::ADDr, 3).addMReg(SrcReg).addMReg(getZeroRegNum())
      .addMReg(DestReg, MOTy::Def);
    break;
    
  case FPSingleRegType:
    MI = BuildMI(V9::FMOVS, 2).addMReg(SrcReg).addMReg(DestReg, MOTy::Def);
    break;

  case FPDoubleRegType:
    MI = BuildMI(V9::FMOVD, 2).addMReg(SrcReg).addMReg(DestReg, MOTy::Def);
    break;

  default:
    assert(0 && "Unknown RegType");
    break;
  }
  
  if (MI)
    mvec.push_back(MI);
}

//---------------------------------------------------------------------------
// Copy from a register to memory (i.e., Store). Register number must 
// be the unified register number
//---------------------------------------------------------------------------


void
UltraSparcRegInfo::cpReg2MemMI(std::vector<MachineInstr*>& mvec,
                               unsigned SrcReg, 
                               unsigned DestPtrReg,
                               int Offset, int RegType,
                               int scratchReg) const {
  MachineInstr * MI = NULL;
  switch (RegType) {
  case IntRegType:
    assert(target.getInstrInfo().constantFitsInImmedField(V9::STXi, Offset));
    MI = BuildMI(V9::STXi,3).addMReg(SrcReg).addMReg(DestPtrReg)
      .addSImm(Offset);
    break;

  case FPSingleRegType:
    assert(target.getInstrInfo().constantFitsInImmedField(V9::STFi, Offset));
    MI = BuildMI(V9::STFi, 3).addMReg(SrcReg).addMReg(DestPtrReg)
      .addSImm(Offset);
    break;

  case FPDoubleRegType:
    assert(target.getInstrInfo().constantFitsInImmedField(V9::STDFi, Offset));
    MI = BuildMI(V9::STDFi,3).addMReg(SrcReg).addMReg(DestPtrReg)
      .addSImm(Offset);
    break;

  case IntCCRegType:
    assert(scratchReg >= 0 && "Need scratch reg to store %ccr to memory");
    assert(getRegType(scratchReg) ==IntRegType && "Invalid scratch reg");
    MI = (BuildMI(V9::RDCCR, 2)
          .addMReg(getUnifiedRegNum(UltraSparcRegInfo::IntCCRegClassID,
                                    SparcIntCCRegClass::ccr))
          .addMReg(scratchReg, MOTy::Def));
    mvec.push_back(MI);
    
    cpReg2MemMI(mvec, scratchReg, DestPtrReg, Offset, IntRegType);
    return;
    
  case FloatCCRegType: {
    assert(target.getInstrInfo().constantFitsInImmedField(V9::STXFSRi, Offset));
    unsigned fsrRegNum =  getUnifiedRegNum(UltraSparcRegInfo::SpecialRegClassID,
                                           SparcSpecialRegClass::fsr);
    MI = BuildMI(V9::STXFSRi, 3)
      .addMReg(fsrRegNum).addMReg(DestPtrReg).addSImm(Offset);
    break;
  }
  default:
    assert(0 && "Unknown RegType in cpReg2MemMI");
  }
  mvec.push_back(MI);
}


//---------------------------------------------------------------------------
// Copy from memory to a reg (i.e., Load) Register number must be the unified
// register number
//---------------------------------------------------------------------------


void
UltraSparcRegInfo::cpMem2RegMI(std::vector<MachineInstr*>& mvec,
                               unsigned SrcPtrReg,	
                               int Offset,
                               unsigned DestReg,
                               int RegType,
                               int scratchReg) const {
  MachineInstr * MI = NULL;
  switch (RegType) {
  case IntRegType:
    assert(target.getInstrInfo().constantFitsInImmedField(V9::LDXi, Offset));
    MI = BuildMI(V9::LDXi, 3).addMReg(SrcPtrReg).addSImm(Offset)
      .addMReg(DestReg, MOTy::Def);
    break;

  case FPSingleRegType:
    assert(target.getInstrInfo().constantFitsInImmedField(V9::LDFi, Offset));
    MI = BuildMI(V9::LDFi, 3).addMReg(SrcPtrReg).addSImm(Offset)
      .addMReg(DestReg, MOTy::Def);
    break;

  case FPDoubleRegType:
    assert(target.getInstrInfo().constantFitsInImmedField(V9::LDDFi, Offset));
    MI = BuildMI(V9::LDDFi, 3).addMReg(SrcPtrReg).addSImm(Offset)
      .addMReg(DestReg, MOTy::Def);
    break;

  case IntCCRegType:
    assert(scratchReg >= 0 && "Need scratch reg to load %ccr from memory");
    assert(getRegType(scratchReg) ==IntRegType && "Invalid scratch reg");
    cpMem2RegMI(mvec, SrcPtrReg, Offset, scratchReg, IntRegType);
    MI = (BuildMI(V9::WRCCRr, 3)
          .addMReg(scratchReg)
          .addMReg(SparcIntRegClass::g0)
          .addMReg(getUnifiedRegNum(UltraSparcRegInfo::IntCCRegClassID,
                                    SparcIntCCRegClass::ccr), MOTy::Def));
    break;
    
  case FloatCCRegType: {
    assert(target.getInstrInfo().constantFitsInImmedField(V9::LDXFSRi, Offset));
    unsigned fsrRegNum =  getUnifiedRegNum(UltraSparcRegInfo::SpecialRegClassID,
                                           SparcSpecialRegClass::fsr);
    MI = BuildMI(V9::LDXFSRi, 3).addMReg(SrcPtrReg).addSImm(Offset)
      .addMReg(fsrRegNum, MOTy::UseAndDef);
    break;
  }
  default:
    assert(0 && "Unknown RegType in cpMem2RegMI");
  }
  mvec.push_back(MI);
}


//---------------------------------------------------------------------------
// Generate a copy instruction to copy a value to another. Temporarily
// used by PhiElimination code.
//---------------------------------------------------------------------------


void
UltraSparcRegInfo::cpValue2Value(Value *Src, Value *Dest,
                                 std::vector<MachineInstr*>& mvec) const {
  int RegType = getRegTypeForDataType(Src->getType());
  MachineInstr * MI = NULL;

  switch( RegType ) {
  case IntRegType:
    MI = BuildMI(V9::ADDr, 3).addReg(Src).addMReg(getZeroRegNum())
      .addRegDef(Dest);
    break;
  case FPSingleRegType:
    MI = BuildMI(V9::FMOVS, 2).addReg(Src).addRegDef(Dest);
    break;
  case FPDoubleRegType:
    MI = BuildMI(V9::FMOVD, 2).addReg(Src).addRegDef(Dest);
    break;
  default:
    assert(0 && "Unknow RegType in CpValu2Value");
  }

  mvec.push_back(MI);
}






//----------------------------------------------------------------------------
// This method inserts caller saving/restoring instructons before/after
// a call machine instruction. The caller saving/restoring instructions are
// inserted like:
//
//    ** caller saving instructions
//    other instructions inserted for the call by ColorCallArg
//    CALL instruction
//    other instructions inserted for the call ColorCallArg
//    ** caller restoring instructions
//
//----------------------------------------------------------------------------


void
UltraSparcRegInfo::insertCallerSavingCode
(std::vector<MachineInstr*> &instrnsBefore,
 std::vector<MachineInstr*> &instrnsAfter,
 MachineInstr *CallMI, 
 const BasicBlock *BB,
 PhyRegAlloc &PRA) const
{
  assert(target.getInstrInfo().isCall(CallMI->getOpCode()));
  
  // has set to record which registers were saved/restored
  //
  hash_set<unsigned> PushedRegSet;

  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(CallMI);
  
  // if the call is to a instrumentation function, do not insert save and
  // restore instructions the instrumentation function takes care of save
  // restore for volatile regs.
  //
  // FIXME: this should be made general, not specific to the reoptimizer!
  //
  const Function *Callee = argDesc->getCallInst()->getCalledFunction();
  bool isLLVMFirstTrigger = Callee && Callee->getName() == "llvm_first_trigger";

  // Now check if the call has a return value (using argDesc) and if so,
  // find the LR of the TmpInstruction representing the return value register.
  // (using the last or second-last *implicit operand* of the call MI).
  // Insert it to to the PushedRegSet since we must not save that register
  // and restore it after the call.
  // We do this because, we look at the LV set *after* the instruction
  // to determine, which LRs must be saved across calls. The return value
  // of the call is live in this set - but we must not save/restore it.
  // 
  if (const Value *origRetVal = argDesc->getReturnValue()) {
    unsigned retValRefNum = (CallMI->getNumImplicitRefs() -
                             (argDesc->getIndirectFuncPtr()? 1 : 2));
    const TmpInstruction* tmpRetVal =
      cast<TmpInstruction>(CallMI->getImplicitRef(retValRefNum));
    assert(tmpRetVal->getOperand(0) == origRetVal &&
           tmpRetVal->getType() == origRetVal->getType() &&
           "Wrong implicit ref?");
    LiveRange *RetValLR = PRA.LRI.getLiveRangeForValue( tmpRetVal );
    assert(RetValLR && "No LR for RetValue of call");

    if (! RetValLR->isMarkedForSpill())
      PushedRegSet.insert(getUnifiedRegNum(RetValLR->getRegClassID(),
                                           RetValLR->getColor()));
  }

  const ValueSet &LVSetAft =  PRA.LVI->getLiveVarSetAfterMInst(CallMI, BB);
  ValueSet::const_iterator LIt = LVSetAft.begin();

  // for each live var in live variable set after machine inst
  for( ; LIt != LVSetAft.end(); ++LIt) {

   //  get the live range corresponding to live var
    LiveRange *const LR = PRA.LRI.getLiveRangeForValue(*LIt );    

    // LR can be null if it is a const since a const 
    // doesn't have a dominating def - see Assumptions above
    if( LR )   {  
      
      if(! LR->isMarkedForSpill()) {

        assert(LR->hasColor() && "LR is neither spilled nor colored?");
	unsigned RCID = LR->getRegClassID();
	unsigned Color = LR->getColor();

	if ( isRegVolatile(RCID, Color) ) {

	  //if the function is special LLVM function,
	  //And the register is not modified by call, don't save and restore
	  if(isLLVMFirstTrigger && !modifiedByCall(RCID, Color))
	    continue;

	  // if the value is in both LV sets (i.e., live before and after 
	  // the call machine instruction)
          
	  unsigned Reg = getUnifiedRegNum(RCID, Color);
	  
	  if( PushedRegSet.find(Reg) == PushedRegSet.end() ) {
	    
	    // if we haven't already pushed that register

	    unsigned RegType = getRegTypeForLR(LR);

	    // Now get two instructions - to push on stack and pop from stack
	    // and add them to InstrnsBefore and InstrnsAfter of the
	    // call instruction
            // 
	    int StackOff =
              PRA.MF.getInfo()->pushTempValue(getSpilledRegSize(RegType));
            
	    //---- Insert code for pushing the reg on stack ----------
            
	    std::vector<MachineInstr*> AdIBef, AdIAft;
            
            // We may need a scratch register to copy the saved value
            // to/from memory.  This may itself have to insert code to
            // free up a scratch register.  Any such code should go before
            // the save code.  The scratch register, if any, is by default
            // temporary and not "used" by the instruction unless the
            // copy code itself decides to keep the value in the scratch reg.
            int scratchRegType = -1;
            int scratchReg = -1;
            if (regTypeNeedsScratchReg(RegType, scratchRegType))
              { // Find a register not live in the LVSet before CallMI
                const ValueSet &LVSetBef =
                  PRA.LVI->getLiveVarSetBeforeMInst(CallMI, BB);
                scratchReg = PRA.getUsableUniRegAtMI(scratchRegType, &LVSetBef,
                                                   CallMI, AdIBef, AdIAft);
                assert(scratchReg != getInvalidRegNum());
              }
            
            if (AdIBef.size() > 0)
              instrnsBefore.insert(instrnsBefore.end(),
                                   AdIBef.begin(), AdIBef.end());
            
            cpReg2MemMI(instrnsBefore, Reg,getFramePointer(),StackOff,RegType,
                        scratchReg);
            
            if (AdIAft.size() > 0)
              instrnsBefore.insert(instrnsBefore.end(),
                                   AdIAft.begin(), AdIAft.end());
            
	    //---- Insert code for popping the reg from the stack ----------

	    AdIBef.clear();
            AdIAft.clear();
            
            // We may need a scratch register to copy the saved value
            // from memory.  This may itself have to insert code to
            // free up a scratch register.  Any such code should go
            // after the save code.  As above, scratch is not marked "used".
            // 
            scratchRegType = -1;
            scratchReg = -1;
            if (regTypeNeedsScratchReg(RegType, scratchRegType))
              { // Find a register not live in the LVSet after CallMI
                scratchReg = PRA.getUsableUniRegAtMI(scratchRegType, &LVSetAft,
                                                 CallMI, AdIBef, AdIAft);
                assert(scratchReg != getInvalidRegNum());
              }
            
            if (AdIBef.size() > 0)
              instrnsAfter.insert(instrnsAfter.end(),
                                  AdIBef.begin(), AdIBef.end());
            
	    cpMem2RegMI(instrnsAfter, getFramePointer(), StackOff,Reg,RegType,
                        scratchReg);
            
            if (AdIAft.size() > 0)
              instrnsAfter.insert(instrnsAfter.end(),
                                  AdIAft.begin(), AdIAft.end());
	    
	    PushedRegSet.insert(Reg);
            
	    if(DEBUG_RA) {
	      std::cerr << "\nFor call inst:" << *CallMI;
	      std::cerr << " -inserted caller saving instrs: Before:\n\t ";
              for_each(instrnsBefore.begin(), instrnsBefore.end(),
                       std::mem_fun(&MachineInstr::dump));
	      std::cerr << " -and After:\n\t ";
              for_each(instrnsAfter.begin(), instrnsAfter.end(),
                       std::mem_fun(&MachineInstr::dump));
	    }	    
	  } // if not already pushed

	} // if LR has a volatile color
	
      } // if LR has color

    } // if there is a LR for Var
    
  } // for each value in the LV set after instruction
}


//---------------------------------------------------------------------------
// Print the register assigned to a LR
//---------------------------------------------------------------------------

void UltraSparcRegInfo::printReg(const LiveRange *LR) const {
  unsigned RegClassID = LR->getRegClassID();
  std::cerr << " *Node " << (LR->getUserIGNode())->getIndex();

  if (!LR->hasColor()) {
    std::cerr << " - could not find a color\n";
    return;
  }
  
  // if a color is found

  std::cerr << " colored with color "<< LR->getColor();

  unsigned uRegName = getUnifiedRegNum(RegClassID, LR->getColor());
  
  std::cerr << "[";
  std::cerr<< getUnifiedRegName(uRegName);
  if (RegClassID == FloatRegClassID && LR->getType() == Type::DoubleTy)
    std::cerr << "+" << getUnifiedRegName(uRegName+1);
  std::cerr << "]\n";
}
