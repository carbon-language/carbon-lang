//===-- llvm/CodeGen/MachineFrameInfo.h -------------------------*- C++ -*-===//
//
// Interface to layout of stack frame on target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_FRAMEINFO_H
#define LLVM_CODEGEN_FRAMEINFO_H

#include "Support/NonCopyable.h"
#include <vector>

class MachineCodeForMethod;
class TargetMachine;

struct MachineFrameInfo : public NonCopyableV {
  const TargetMachine &target;
  
public:
  MachineFrameInfo(const TargetMachine& tgt) : target(tgt) {}
  
  // These methods provide constant parameters of the frame layout.
  // 
  virtual int  getStackFrameSizeAlignment       () const = 0;
  virtual int  getMinStackFrameSize             () const = 0;
  virtual int  getNumFixedOutgoingArgs          () const = 0;
  virtual int  getSizeOfEachArgOnStack          () const = 0;
  virtual bool argsOnStackHaveFixedSize         () const = 0;

  // This method adjusts a stack offset to meet alignment rules of target.
  // 
  virtual int  adjustAlignment                  (int unalignedOffset,
                                                 bool growUp,
                                                 unsigned int align) const {
    return unalignedOffset + (growUp? +1:-1)*(unalignedOffset % align);
  }

  // These methods compute offsets using the frame contents for a
  // particular method.  The frame contents are obtained from the
  // MachineCodeInfoForMethod object for the given method.
  // The first few methods have default machine-independent implementations.
  // The rest must be implemented by the machine-specific subclass.
  // 
  virtual int getIncomingArgOffset              (MachineCodeForMethod& mcInfo,
                                                 unsigned argNum) const;
  virtual int getOutgoingArgOffset              (MachineCodeForMethod& mcInfo,
                                                 unsigned argNum) const;
  
  virtual int getFirstIncomingArgOffset         (MachineCodeForMethod& mcInfo,
                                                 bool& growUp) const=0;
  virtual int getFirstOutgoingArgOffset         (MachineCodeForMethod& mcInfo,
                                                 bool& growUp) const=0;
  virtual int getFirstOptionalOutgoingArgOffset (MachineCodeForMethod&,
                                                 bool& growUp) const=0;
  virtual int getFirstAutomaticVarOffset        (MachineCodeForMethod& mcInfo,
                                                 bool& growUp) const=0;
  virtual int getRegSpillAreaOffset             (MachineCodeForMethod& mcInfo,
                                                 bool& growUp) const=0;
  virtual int getTmpAreaOffset                  (MachineCodeForMethod& mcInfo,
                                                 bool& growUp) const=0;
  virtual int getDynamicAreaOffset              (MachineCodeForMethod& mcInfo,
                                                 bool& growUp) const=0;

  //
  // These methods specify the base register used for each stack area
  // (generally FP or SP)
  // 
  virtual int getIncomingArgBaseRegNum()               const=0;
  virtual int getOutgoingArgBaseRegNum()               const=0;
  virtual int getOptionalOutgoingArgBaseRegNum()       const=0;
  virtual int getAutomaticVarBaseRegNum()              const=0;
  virtual int getRegSpillAreaBaseRegNum()              const=0;
  virtual int getDynamicAreaBaseRegNum()               const=0;
};

#endif
