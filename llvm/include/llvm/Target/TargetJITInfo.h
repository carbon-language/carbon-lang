//===- Target/TargetJITInfo.h - Target Information for JIT ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file exposes an abstract interface used by the Just-In-Time code
// generator to perform target-specific activities, such as emitting stubs.  If
// a TargetMachine supports JIT code generation, it should provide one of these
// objects through the getJITInfo() method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETJITINFO_H
#define LLVM_TARGET_TARGETJITINFO_H

namespace llvm {
  class Function;
  class FunctionPassManager;
  class MachineCodeEmitter;

  /// TargetJITInfo - Target specific information required by the Just-In-Time
  /// code generator.
  struct TargetJITInfo {
    virtual ~TargetJITInfo() {}
    
    /// addPassesToJITCompile - Add passes to the specified pass manager to
    /// implement a fast code generator for this target.
    ///
    virtual void addPassesToJITCompile(FunctionPassManager &PM) = 0;
    
    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction (void *Old, void *New) = 0;
    
    /// getJITStubForFunction - Create or return a stub for the specified
    /// function.  This stub acts just like the specified function, except that
    /// it allows the "address" of the function to be taken without having to
    /// generate code for it.  Targets do not need to implement this method, but
    /// doing so will allow for faster startup of the JIT.
    ///
    virtual void *getJITStubForFunction(Function *F, MachineCodeEmitter &MCE) {
      return 0;
    }
  };
} // End llvm namespace

#endif
