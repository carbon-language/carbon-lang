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
public:
  enum AsmWriterFlavorTy {
    att, intel, unset
  };

protected:
  enum X86SSEEnum {
    NoMMXSSE, MMX, SSE1, SSE2, SSE3
  };

  /// AsmFlavor - Which x86 asm dialect to use.
  AsmWriterFlavorTy AsmFlavor;

  /// X86SSELevel - MMX, SSE1, SSE2, SSE3, or none supported.
  X86SSEEnum X86SSELevel;

  /// HasX86_64 - True if the processor supports X86-64 instructions.
  bool HasX86_64;

  /// stackAlignment - The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned stackAlignment;

  /// Min. memset / memcpy size that is turned into rep/movs, rep/stos ops.
  unsigned MinRepStrSizeThreshold;

private:
  /// Is64Bit - True if the processor supports 64-bit instructions and module
  /// pointer size is 64 bit.
  bool Is64Bit;

public:
  enum {
    isELF, isCygwin, isDarwin, isWindows
  } TargetType;
    
  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  X86Subtarget(const Module &M, const std::string &FS, bool is64Bit);

  /// getStackAlignment - Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return stackAlignment; }

  /// getMinRepStrSizeThreshold - Returns the minimum memset / memcpy size
  /// required to turn the operation into a X86 rep/movs or rep/stos
  /// instruction. This is only used if the src / dst alignment is not DWORD
  /// aligned.
  unsigned getMinRepStrSizeThreshold() const { return MinRepStrSizeThreshold; }
 
  /// DetectSubtargetFeatures - Auto-detect CPU features using CPUID instruction.
  ///
  void DetectSubtargetFeatures();

  bool is64Bit() const { return Is64Bit; }

  bool hasMMX() const { return X86SSELevel >= MMX; }
  bool hasSSE1() const { return X86SSELevel >= SSE1; }
  bool hasSSE2() const { return X86SSELevel >= SSE2; }
  bool hasSSE3() const { return X86SSELevel >= SSE3; }
  
  bool isFlavorAtt() const { return AsmFlavor == att; }
  bool isFlavorIntel() const { return AsmFlavor == intel; }

  bool isTargetDarwin() const { return TargetType == isDarwin; }
  bool isTargetELF() const { return TargetType == isELF; }
  bool isTargetWindows() const { return TargetType == isWindows; }
  bool isTargetCygwin() const { return TargetType == isCygwin; }  
};
} // End llvm namespace

#endif
