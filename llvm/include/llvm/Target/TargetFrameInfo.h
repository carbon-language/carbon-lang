// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	MachineFrameInfo.h
// 
// Purpose:
//	Interface to layout of stack frame on target machine.
// 
// History:
//	11/6/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_CODEGEN_FRAMEINFO_H
#define LLVM_CODEGEN_FRAMEINFO_H

#include "Support/NonCopyable.h"
#include <vector>


//************************* Forward Declarations **************************/

class MachineCodeForMethod;


//*************************** External Classes ****************************/


class MachineFrameInfo : public NonCopyableV {
public:
  const TargetMachine& target;
  
public:
  /*ctor*/ MachineFrameInfo(const TargetMachine& tgt) : target(tgt) {}
  
  //
  // These methods provide constant parameters of the frame layout.
  // 
  virtual int  getStackFrameSizeAlignment       () const = 0;
  virtual int  getMinStackFrameSize             () const = 0;
  virtual int  getNumFixedOutgoingArgs          () const = 0;
  virtual int  getSizeOfEachArgOnStack          () const = 0;
  virtual bool argsOnStackHaveFixedSize         () const = 0;
  
  //
  // These methods compute offsets using the frame contents for a
  // particular method.  The frame contents are obtained from the
  // MachineCodeInfoForMethod object for the given method.
  // 
  virtual int getFirstIncomingArgOffset         (MachineCodeForMethod& mcInfo,
                                                 bool& pos) const=0;
  virtual int getFirstOutgoingArgOffset         (MachineCodeForMethod& mcInfo,
                                                 bool& pos) const=0;
  virtual int getFirstOptionalOutgoingArgOffset (MachineCodeForMethod&,
                                                 bool& pos) const=0;
  virtual int getFirstAutomaticVarOffset        (MachineCodeForMethod& mcInfo,
                                                 bool& pos) const=0;
  virtual int getRegSpillAreaOffset             (MachineCodeForMethod& mcInfo,
                                                 bool& pos) const=0;
  virtual int getTmpAreaOffset                  (MachineCodeForMethod& mcInfo,
                                                 bool& pos) const=0;
  virtual int getDynamicAreaOffset              (MachineCodeForMethod& mcInfo,
                                                 bool& pos) const=0;

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
