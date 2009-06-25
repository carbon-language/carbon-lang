//===-- IntrinsicLowering.h - Intrinsic Function Lowering -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IntrinsicLowering interface.  This interface allows
// addition of domain-specific or front-end specific intrinsics to LLVM without
// having to modify all of the C backend or interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INTRINSICLOWERING_H
#define LLVM_CODEGEN_INTRINSICLOWERING_H

#include "llvm/Intrinsics.h"

namespace llvm {
  class CallInst;
  class Module;
  class TargetData;

  class IntrinsicLowering {
    const TargetData& TD;
    
    Constant *SetjmpFCache;
    Constant *LongjmpFCache;
    Constant *AbortFCache;
    Constant *MemcpyFCache;
    Constant *MemmoveFCache;
    Constant *MemsetFCache;
    Constant *sqrtFCache;
    Constant *sqrtDCache;
    Constant *sqrtLDCache;
    Constant *logFCache;
    Constant *logDCache;
    Constant *logLDCache;
    Constant *log2FCache;
    Constant *log2DCache;
    Constant *log2LDCache;
    Constant *log10FCache;
    Constant *log10DCache;
    Constant *log10LDCache;
    Constant *expFCache;
    Constant *expDCache;
    Constant *expLDCache;
    Constant *exp2FCache;
    Constant *exp2DCache;
    Constant *exp2LDCache;
    Constant *powFCache;
    Constant *powDCache;
    Constant *powLDCache;
    
    bool Warned;
  public:
    explicit IntrinsicLowering(const TargetData &td) :
      TD(td), SetjmpFCache(0), LongjmpFCache(0), AbortFCache(0),
      MemcpyFCache(0), MemmoveFCache(0), MemsetFCache(0), sqrtFCache(0),
      sqrtDCache(0), sqrtLDCache(0), logFCache(0), logDCache(0), logLDCache(0), 
      log2FCache(0), log2DCache(0), log2LDCache(0), log10FCache(0), 
      log10DCache(0), log10LDCache(0), expFCache(0), expDCache(0), 
      expLDCache(0), exp2FCache(0), exp2DCache(0), exp2LDCache(0), powFCache(0),
      powDCache(0), powLDCache(0), Warned(false) {}

    /// AddPrototypes - This method, if called, causes all of the prototypes
    /// that might be needed by an intrinsic lowering implementation to be
    /// inserted into the module specified.
    void AddPrototypes(Module &M);

    /// LowerIntrinsicCall - This method replaces a call with the LLVM function
    /// which should be used to implement the specified intrinsic function call.
    /// If an intrinsic function must be implemented by the code generator 
    /// (such as va_start), this function should print a message and abort.
    ///
    /// Otherwise, if an intrinsic function call can be lowered, the code to
    /// implement it (often a call to a non-intrinsic function) is inserted
    /// _after_ the call instruction and the call is deleted.  The caller must
    /// be capable of handling this kind of change.
    ///
    void LowerIntrinsicCall(CallInst *CI);
  };
}

#endif
