//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
// This file also implements MachineInstrInfo and MachineCacheInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"

// External object describing the machine instructions
// Initialized only when the TargetMachine class is created
// and reset when that class is destroyed.
// 
const MachineInstrDescriptor* TargetInstrDescriptors = 0;

//---------------------------------------------------------------------------
// class MachineInstructionInfo
//	Interface to description of machine instructions
//---------------------------------------------------------------------------


MachineInstrInfo::MachineInstrInfo(const MachineInstrDescriptor* Desc,
				   unsigned DescSize,
				   unsigned NumRealOpCodes)
  : desc(Desc), descSize(DescSize), numRealOpCodes(NumRealOpCodes) {
  // FIXME: TargetInstrDescriptors should not be global
  assert(TargetInstrDescriptors == NULL && desc != NULL);
  TargetInstrDescriptors = desc;	// initialize global variable
}

MachineInstrInfo::~MachineInstrInfo() {
  TargetInstrDescriptors = NULL;	// reset global variable
}


bool MachineInstrInfo::constantFitsInImmedField(MachineOpCode opCode,
                                                int64_t intValue) const {
  // First, check if opCode has an immed field.
  bool isSignExtended;
  uint64_t maxImmedValue = maxImmedConstant(opCode, isSignExtended);
  if (maxImmedValue != 0)
    {
      // NEED TO HANDLE UNSIGNED VALUES SINCE THEY MAY BECOME MUCH
      // SMALLER AFTER CASTING TO SIGN-EXTENDED int, short, or char.
      // See CreateUIntSetInstruction in SparcInstrInfo.cpp.
      
      // Now check if the constant fits
      if (intValue <= (int64_t) maxImmedValue &&
	  intValue >= -((int64_t) maxImmedValue+1))
	return true;
    }
  
  return false;
}

bool MachineInstrInfo::ConstantTypeMustBeLoaded(const Constant* CV) const {
  assert(CV->getType()->isPrimitiveType() || isa<PointerType>(CV->getType()));
  return !(CV->getType()->isIntegral() || isa<PointerType>(CV->getType()));
}
