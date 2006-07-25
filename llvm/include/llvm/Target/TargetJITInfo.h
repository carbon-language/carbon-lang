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

#include <cassert>
#include <vector>

namespace llvm {
  class Function;
  class FunctionPassManager;
  class MachineBasicBlock;
  class MachineCodeEmitter;
  class MachineRelocation;

  /// TargetJITInfo - Target specific information required by the Just-In-Time
  /// code generator.
  class TargetJITInfo {
  public:
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
    virtual void replaceMachineCodeForFunction(void *Old, void *New) = 0;

    /// emitFunctionStub - Use the specified MachineCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.  Return the address of the resultant function.
    virtual void *emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
      assert(0 && "This target doesn't implement emitFunctionStub!");
      return 0;
    }

    /// LazyResolverFn - This typedef is used to represent the function that
    /// unresolved call points should invoke.  This is a target specific
    /// function that knows how to walk the stack and find out which stub the
    /// call is coming from.
    typedef void (*LazyResolverFn)();

    /// JITCompilerFn - This typedef is used to represent the JIT function that
    /// lazily compiles the function corresponding to a stub.  The JIT keeps
    /// track of the mapping between stubs and LLVM Functions, the target
    /// provides the ability to figure out the address of a stub that is called
    /// by the LazyResolverFn.
    typedef void* (*JITCompilerFn)(void *);

    /// getLazyResolverFunction - This method is used to initialize the JIT,
    /// giving the target the function that should be used to compile a
    /// function, and giving the JIT the target function used to do the lazy
    /// resolving.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn) {
      assert(0 && "Not implemented for this target!");
      return 0;
    }

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
      assert(NumRelocs == 0 && "This target does not have relocations!");
    }

    /// resolveBBRefs - Resolve branches to BasicBlocks for the JIT emitted
    /// function.
    virtual void resolveBBRefs(MachineCodeEmitter &MCE) {}

    /// synchronizeICache - On some targets, the JIT emitted code must be
    /// explicitly refetched to ensure correct execution.
    virtual void synchronizeICache(const void *Addr, size_t len) {}

    /// addBBRef - Add a BasicBlock reference to be resolved after the function
    /// is emitted.
    void addBBRef(MachineBasicBlock *BB, intptr_t PC) {
      BBRefs.push_back(std::make_pair(BB, PC));
    }

    /// needsGOT - Allows a target to specify that it would like the
    // JIT to manage a GOT for it.
    bool needsGOT() const { return useGOT; }

  protected:
    bool useGOT;

    // Tracks which instruction references which BasicBlock
    std::vector<std::pair<MachineBasicBlock*, intptr_t> > BBRefs;
    
  };
} // End llvm namespace

#endif
