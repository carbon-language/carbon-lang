//=====-- PowerPCSubtarget.h - Define Subtarget for the PPC ---*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPCSUBTARGET_H
#define POWERPCSUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"

namespace llvm {
class Module;

class PPCSubtarget : public TargetSubtarget {
protected:
  /// stackAlignment - The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned StackAlignment;

  /// Used by the ISel to turn in optimizations for POWER4-derived architectures
  bool IsGigaProcessor;
  bool IsAIX;
  bool IsDarwin;
public:
  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  PPCSubtarget(const Module &M);

  /// getStackAlignment - Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return StackAlignment; }

  bool isAIX() const { return IsAIX; }
  bool isDarwin() const { return IsDarwin; }
  
  bool isGigaProcessor() const { return IsGigaProcessor; }
};
} // End llvm namespace

#endif
