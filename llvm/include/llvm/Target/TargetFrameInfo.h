//===-- llvm/Target/TargetFrameInfo.h ---------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Interface to describe the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETFRAMEINFO_H
#define LLVM_TARGET_TARGETFRAMEINFO_H

namespace llvm {

class MachineFunction;

struct TargetFrameInfo {
  enum StackDirection {
    StackGrowsUp,        // Adding to the stack increases the stack address
    StackGrowsDown       // Adding to the stack decreases the stack address
  };
private:
  StackDirection StackDir;
  unsigned StackAlignment;
  int LocalAreaOffset;
public:
  TargetFrameInfo(StackDirection D, unsigned StackAl, int LAO)
    : StackDir(D), StackAlignment(StackAl), LocalAreaOffset(LAO) {}

  // These methods return information that describes the abstract stack layout
  // of the target machine.

  /// getStackGrowthDirection - Return the direction the stack grows
  ///
  StackDirection getStackGrowthDirection() const { return StackDir; }

  /// getStackAlignment - This method returns the number of bytes that the stack
  /// pointer must be aligned to.  Typically, this is the largest alignment for
  /// any data object in the target.
  ///
  unsigned getStackAlignment() const { return StackAlignment; }

  /// getOffsetOfLocalArea - This method returns the offset of the local area
  /// from the stack pointer on entrance to a function.
  ///
  int getOffsetOfLocalArea() const { return LocalAreaOffset; }

  //===--------------------------------------------------------------------===//
  // These methods provide details of the stack frame used by Sparc, thus they
  // are Sparc specific.
  //===--------------------------------------------------------------------===//

  virtual int  getStackFrameSizeAlignment       () const { abort(); }
  virtual int  getMinStackFrameSize             () const { abort(); }
  virtual int  getNumFixedOutgoingArgs          () const { abort(); }
  virtual int  getSizeOfEachArgOnStack          () const { abort(); }
  virtual bool argsOnStackHaveFixedSize         () const { abort(); }

  // This method adjusts a stack offset to meet alignment rules of target.
  virtual int adjustAlignment(int unalignedOffset, bool growUp,
			      unsigned align) const { abort(); }

  // These methods compute offsets using the frame contents for a particular
  // function.  The frame contents are obtained from the MachineFunction object
  // for the given function.  The rest must be implemented by the
  // machine-specific subclass.
  // 
  virtual int getIncomingArgOffset              (MachineFunction& mcInfo,
						 unsigned argNum)const{abort();}
  virtual int getOutgoingArgOffset              (MachineFunction& mcInfo,
						 unsigned argNum)const{abort();}
  
  virtual int getFirstIncomingArgOffset         (MachineFunction& mcInfo,
						 bool& growUp) const { abort();}
  virtual int getFirstOutgoingArgOffset         (MachineFunction& mcInfo,
						 bool& growUp) const {abort();}
  virtual int getFirstOptionalOutgoingArgOffset (MachineFunction&,
                                                 bool& growUp) const {abort();}
  virtual int getFirstAutomaticVarOffset        (MachineFunction& mcInfo,
                                                 bool& growUp) const {abort();}
  virtual int getRegSpillAreaOffset             (MachineFunction& mcInfo,
                                                 bool& growUp) const {abort();}
  virtual int getTmpAreaOffset                  (MachineFunction& mcInfo,
                                                 bool& growUp) const {abort();}
  virtual int getDynamicAreaOffset              (MachineFunction& mcInfo,
                                                 bool& growUp) const {abort();}

  //
  // These methods specify the base register used for each stack area
  // (generally FP or SP)
  // 
  virtual int getIncomingArgBaseRegNum()               const { abort(); }
  virtual int getOutgoingArgBaseRegNum()               const { abort(); }
  virtual int getOptionalOutgoingArgBaseRegNum()       const { abort(); }
  virtual int getAutomaticVarBaseRegNum()              const { abort(); }
  virtual int getRegSpillAreaBaseRegNum()              const { abort(); }
  virtual int getDynamicAreaBaseRegNum()               const { abort(); }
};

} // End llvm namespace

#endif
