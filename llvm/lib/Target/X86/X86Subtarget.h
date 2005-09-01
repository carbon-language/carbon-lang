//=====---- X86Subtarget.h - Define Subtarget for the X86 -----*- C++ -*--====//
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

#ifndef X86SUBTARGET_H
#define X86SUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"

#include <string>

namespace llvm {
class Module;

class X86Subtarget : public TargetSubtarget {
protected:
  /// stackAlignment - The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned stackAlignment;

  /// Used by instruction selector
  bool indirectExternAndWeakGlobals;

  /// Used by the asm printer
  bool asmDarwinLinkerStubs;
  bool asmLeadingUnderscore;
  bool asmAlignmentIsInBytes;
  bool asmPrintDotLocalConstants;
  bool asmPrintDotLCommConstants;
  bool asmPrintConstantAlignment;
public:
  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  X86Subtarget(const Module &M, const std::string &FS);

  /// getStackAlignment - Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return stackAlignment; }

  /// Returns true if the instruction selector should treat global values
  /// referencing external or weak symbols as indirect rather than direct
  /// references.
  bool getIndirectExternAndWeakGlobals() const {
    return indirectExternAndWeakGlobals;
  }
};
} // End llvm namespace

#endif
