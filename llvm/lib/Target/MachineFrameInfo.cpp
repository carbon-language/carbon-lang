//===-- MachineFrameInfo.cpp-----------------------------------------------===//
// 
// Interface to layout of stack frame on target machine.  Most functions of
// class MachineFrameInfo have to be machine-specific so there is little code
// here.
// 
//===----------------------------------------------------------------------===//

#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"

int
MachineFrameInfo::getIncomingArgOffset(MachineCodeForMethod& mcInfo,
                                       unsigned argNum) const
{
  assert(argsOnStackHaveFixedSize()); 
  
  unsigned relativeOffset = argNum * getSizeOfEachArgOnStack();
  bool growUp;                          // do args grow up or down
  int firstArg = getFirstIncomingArgOffset(mcInfo, growUp);
  int offset = growUp? firstArg + relativeOffset 
                     : firstArg - relativeOffset; 
  return offset; 
}


int
MachineFrameInfo::getOutgoingArgOffset(MachineCodeForMethod& mcInfo,
                                       unsigned argNum) const
{
  assert(argsOnStackHaveFixedSize()); 
  assert(((int) argNum - this->getNumFixedOutgoingArgs())
         <= (int) mcInfo.getMaxOptionalNumArgs());
  
  unsigned relativeOffset = argNum * getSizeOfEachArgOnStack();
  bool growUp;                          // do args grow up or down
  int firstArg = getFirstOutgoingArgOffset(mcInfo, growUp);
  int offset = growUp? firstArg + relativeOffset 
                     : firstArg - relativeOffset; 
  
  return offset; 
}
