//===-- SparcRegInfo.cpp - Sparc Target Register Information --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains implementation of Sparc specific helper methods
// used for register allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "RegAlloc/LiveRangeInfo.h"
#include "RegAlloc/LiveRange.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "SparcInternals.h"
#include "SparcRegClassInfo.h"
#include "SparcRegInfo.h"
#include "SparcTargetMachine.h"

namespace llvm {

enum {
  BadRegClass = ~0
};

SparcRegInfo::SparcRegInfo(const SparcTargetMachine &tgt)
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
int SparcRegInfo::getZeroRegNum() const {
  return getUnifiedRegNum(SparcRegInfo::IntRegClassID,
                          SparcIntRegClass::g0);
}

// getCallAddressReg - returns the reg used for pushing the address when a
// method is called. This can be used for other purposes between calls
//
unsigned SparcRegInfo::getCallAddressReg() const {
  return getUnifiedRegNum(SparcRegInfo::IntRegClassID,
                          SparcIntRegClass::o7);
}

// Returns the register containing the return address.
// It should be made sure that this  register contains the return 
// value when a return instruction is reached.
//
unsigned SparcRegInfo::getReturnAddressReg() const {
  return getUnifiedRegNum(SparcRegInfo::IntRegClassID,
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
unsigned SparcRegInfo::getFramePointer() const {
  return getUnifiedRegNum(SparcRegInfo::IntRegClassID,
                          SparcIntRegClass::i6);
}

// Get unified reg number for stack pointer
unsigned SparcRegInfo::getStackPointer() const {
  return getUnifiedRegNum(SparcRegInfo::IntRegClassID,
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
SparcRegInfo::regNumForIntArg(bool inCallee, bool isVarArgsCall,
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
SparcRegInfo::regNumForFPArg(unsigned regType,
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
int SparcRegInfo::getRegTypeForClassAndType(unsigned regClassID,
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

int SparcRegInfo::getRegTypeForDataType(const Type* type) const
{
  return getRegTypeForClassAndType(getRegClassIDOfType(type), type);
}

int SparcRegInfo::getRegTypeForLR(const LiveRange *LR) const
{
  return getRegTypeForClassAndType(LR->getRegClassID(), LR->getType());
}

int SparcRegInfo::getRegType(int unifiedRegNum) const
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
unsigned SparcRegInfo::getRegClassIDOfType(const Type *type,
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

unsigned SparcRegInfo::getRegClassIDOfRegType(int regType) const {
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
void SparcRegInfo::suggestReg4RetAddr(MachineInstr *RetMI, 
					   LiveRangeInfo& LRI) const {

  assert(target.getInstrInfo().isReturn(RetMI->getOpcode()));
  
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
SparcRegInfo::suggestReg4CallAddr(MachineInstr * CallMI,
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
void SparcRegInfo::suggestRegs4MethodArgs(const Function *Meth, 
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
void SparcRegInfo::colorMethodArgs(const Function *Meth, 
                            LiveRangeInfo &LRI,
                            std::vector<MachineInstr*>& InstrnsBefore,
                            std::vector<MachineInstr*>& InstrnsAfter) const {

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
	  cpReg2MemMI(InstrnsBefore,
                      UniArgReg, getFramePointer(), TmpOff, IntRegType);
          
	  cpMem2RegMI(InstrnsBefore,
                      getFramePointer(), TmpOff, UniLRReg, regType);
	}
	else {	
	  cpReg2RegMI(InstrnsBefore, UniArgReg, UniLRReg, regType);
	}
      }
      else {

	// Now the arg is coming on stack. Since the LR received a register,
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

	cpMem2RegMI(InstrnsBefore,
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
          
          cpReg2MemMI(InstrnsBefore, UniArgReg,
                      getFramePointer(), LR->getSpillOffFromFP(), IntRegType);
        }
        else {
           cpReg2MemMI(InstrnsBefore, UniArgReg,
                       getFramePointer(), LR->getSpillOffFromFP(), regType);
        }
      }

      else {

	// Now the arg is coming on stack. Since the LR did NOT 
	// received a register as well, it is allocated a stack position. We
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
void SparcRegInfo::suggestRegs4CallArgs(MachineInstr *CallMI, 
					     LiveRangeInfo& LRI) const {
  assert ( (target.getInstrInfo()).isCall(CallMI->getOpcode()) );

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
void SparcRegInfo::suggestReg4RetValue(MachineInstr *RetMI, 
                                            LiveRangeInfo& LRI) const {

  assert( (target.getInstrInfo()).isReturn( RetMI->getOpcode() ) );

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
SparcRegInfo::regTypeNeedsScratchReg(int RegType,
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
SparcRegInfo::cpReg2RegMI(std::vector<MachineInstr*>& mvec,
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
            .addMReg(getUnifiedRegNum(SparcRegInfo::IntCCRegClassID,
                                      SparcIntCCRegClass::ccr))
            .addMReg(DestReg,MOTy::Def));
    } else {
      // copy int reg to intCC reg
      assert(getRegType(SrcReg) == IntRegType
             && "Can only copy CC reg to/from integer reg");
      MI = (BuildMI(V9::WRCCRr, 3)
            .addMReg(SrcReg)
            .addMReg(SparcIntRegClass::g0)
            .addMReg(getUnifiedRegNum(SparcRegInfo::IntCCRegClassID,
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
SparcRegInfo::cpReg2MemMI(std::vector<MachineInstr*>& mvec,
                               unsigned SrcReg, 
                               unsigned PtrReg,
                               int Offset, int RegType,
                               int scratchReg) const {
  MachineInstr * MI = NULL;
  int OffReg = -1;

  // If the Offset will not fit in the signed-immediate field, find an
  // unused register to hold the offset value.  This takes advantage of
  // the fact that all the opcodes used below have the same size immed. field.
  // Use the register allocator, PRA, to find an unused reg. at this MI.
  // 
  if (RegType != IntCCRegType)          // does not use offset below
    if (! target.getInstrInfo().constantFitsInImmedField(V9::LDXi, Offset)) {
#ifdef CAN_FIND_FREE_REGISTER_TRANSPARENTLY
      RegClass* RC = PRA.getRegClassByID(this->getRegClassIDOfRegType(RegType));
      OffReg = PRA.getUnusedUniRegAtMI(RC, RegType, MInst, LVSetBef);
#else
      // Default to using register g4 for holding large offsets
      OffReg = getUnifiedRegNum(SparcRegInfo::IntRegClassID,
                                SparcIntRegClass::g4);
#endif
      assert(OffReg >= 0 && "FIXME: cpReg2MemMI cannot find an unused reg.");
      mvec.push_back(BuildMI(V9::SETSW, 2).addZImm(Offset).addReg(OffReg));
    }

  switch (RegType) {
  case IntRegType:
    if (target.getInstrInfo().constantFitsInImmedField(V9::STXi, Offset))
      MI = BuildMI(V9::STXi,3).addMReg(SrcReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI = BuildMI(V9::STXr,3).addMReg(SrcReg).addMReg(PtrReg).addMReg(OffReg);
    break;

  case FPSingleRegType:
    if (target.getInstrInfo().constantFitsInImmedField(V9::STFi, Offset))
      MI = BuildMI(V9::STFi, 3).addMReg(SrcReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI = BuildMI(V9::STFr, 3).addMReg(SrcReg).addMReg(PtrReg).addMReg(OffReg);
    break;

  case FPDoubleRegType:
    if (target.getInstrInfo().constantFitsInImmedField(V9::STDFi, Offset))
      MI = BuildMI(V9::STDFi,3).addMReg(SrcReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI = BuildMI(V9::STDFr,3).addMReg(SrcReg).addMReg(PtrReg).addSImm(OffReg);
    break;

  case IntCCRegType:
    assert(scratchReg >= 0 && "Need scratch reg to store %ccr to memory");
    assert(getRegType(scratchReg) ==IntRegType && "Invalid scratch reg");
    MI = (BuildMI(V9::RDCCR, 2)
          .addMReg(getUnifiedRegNum(SparcRegInfo::IntCCRegClassID,
                                    SparcIntCCRegClass::ccr))
          .addMReg(scratchReg, MOTy::Def));
    mvec.push_back(MI);
    
    cpReg2MemMI(mvec, scratchReg, PtrReg, Offset, IntRegType);
    return;

  case FloatCCRegType: {
    unsigned fsrReg =  getUnifiedRegNum(SparcRegInfo::SpecialRegClassID,
                                           SparcSpecialRegClass::fsr);
    if (target.getInstrInfo().constantFitsInImmedField(V9::STXFSRi, Offset))
      MI=BuildMI(V9::STXFSRi,3).addMReg(fsrReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI=BuildMI(V9::STXFSRr,3).addMReg(fsrReg).addMReg(PtrReg).addMReg(OffReg);
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
SparcRegInfo::cpMem2RegMI(std::vector<MachineInstr*>& mvec,
                               unsigned PtrReg,	
                               int Offset,
                               unsigned DestReg,
                               int RegType,
                               int scratchReg) const {
  MachineInstr * MI = NULL;
  int OffReg = -1;

  // If the Offset will not fit in the signed-immediate field, find an
  // unused register to hold the offset value.  This takes advantage of
  // the fact that all the opcodes used below have the same size immed. field.
  // Use the register allocator, PRA, to find an unused reg. at this MI.
  // 
  if (RegType != IntCCRegType)          // does not use offset below
    if (! target.getInstrInfo().constantFitsInImmedField(V9::LDXi, Offset)) {
#ifdef CAN_FIND_FREE_REGISTER_TRANSPARENTLY
      RegClass* RC = PRA.getRegClassByID(this->getRegClassIDOfRegType(RegType));
      OffReg = PRA.getUnusedUniRegAtMI(RC, RegType, MInst, LVSetBef);
#else
      // Default to using register g4 for holding large offsets
      OffReg = getUnifiedRegNum(SparcRegInfo::IntRegClassID,
                                SparcIntRegClass::g4);
#endif
      assert(OffReg >= 0 && "FIXME: cpReg2MemMI cannot find an unused reg.");
      mvec.push_back(BuildMI(V9::SETSW, 2).addZImm(Offset).addReg(OffReg));
    }

  switch (RegType) {
  case IntRegType:
    if (target.getInstrInfo().constantFitsInImmedField(V9::LDXi, Offset))
      MI = BuildMI(V9::LDXi, 3).addMReg(PtrReg).addSImm(Offset).addMReg(DestReg,
                                                                    MOTy::Def);
    else
      MI = BuildMI(V9::LDXr, 3).addMReg(PtrReg).addMReg(OffReg).addMReg(DestReg,
                                                                    MOTy::Def);
    break;

  case FPSingleRegType:
    if (target.getInstrInfo().constantFitsInImmedField(V9::LDFi, Offset))
      MI = BuildMI(V9::LDFi, 3).addMReg(PtrReg).addSImm(Offset).addMReg(DestReg,
                                                                    MOTy::Def);
    else
      MI = BuildMI(V9::LDFr, 3).addMReg(PtrReg).addMReg(OffReg).addMReg(DestReg,
                                                                    MOTy::Def);
    break;

  case FPDoubleRegType:
    if (target.getInstrInfo().constantFitsInImmedField(V9::LDDFi, Offset))
      MI= BuildMI(V9::LDDFi, 3).addMReg(PtrReg).addSImm(Offset).addMReg(DestReg,
                                                                    MOTy::Def);
    else
      MI= BuildMI(V9::LDDFr, 3).addMReg(PtrReg).addMReg(OffReg).addMReg(DestReg,
                                                                    MOTy::Def);
    break;

  case IntCCRegType:
    assert(scratchReg >= 0 && "Need scratch reg to load %ccr from memory");
    assert(getRegType(scratchReg) ==IntRegType && "Invalid scratch reg");
    cpMem2RegMI(mvec, PtrReg, Offset, scratchReg, IntRegType);
    MI = (BuildMI(V9::WRCCRr, 3)
          .addMReg(scratchReg)
          .addMReg(SparcIntRegClass::g0)
          .addMReg(getUnifiedRegNum(SparcRegInfo::IntCCRegClassID,
                                    SparcIntCCRegClass::ccr), MOTy::Def));
    break;
    
  case FloatCCRegType: {
    unsigned fsrRegNum =  getUnifiedRegNum(SparcRegInfo::SpecialRegClassID,
                                           SparcSpecialRegClass::fsr);
    if (target.getInstrInfo().constantFitsInImmedField(V9::LDXFSRi, Offset))
      MI = BuildMI(V9::LDXFSRi, 3).addMReg(PtrReg).addSImm(Offset)
        .addMReg(fsrRegNum, MOTy::UseAndDef);
    else
      MI = BuildMI(V9::LDXFSRr, 3).addMReg(PtrReg).addMReg(OffReg)
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
SparcRegInfo::cpValue2Value(Value *Src, Value *Dest,
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



//---------------------------------------------------------------------------
// Print the register assigned to a LR
//---------------------------------------------------------------------------

void SparcRegInfo::printReg(const LiveRange *LR) const {
  unsigned RegClassID = LR->getRegClassID();
  std::cerr << " Node ";

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

} // End llvm namespace
