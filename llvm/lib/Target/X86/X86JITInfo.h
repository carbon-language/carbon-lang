//===-- X86JITInfo.h - X86 implementation of the JIT interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86JITINFO_H
#define X86JITINFO_H

#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class X86Subtarget;

  class X86JITInfo : public TargetJITInfo {
    uintptr_t PICBase;
    char *TLSOffset;
    bool useSSE;
  public:
    explicit X86JITInfo(bool UseSSE);

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    void replaceMachineCodeForFunction(void *Old, void *New) override;

    /// emitGlobalValueIndirectSym - Use the specified JITCodeEmitter object
    /// to emit an indirect symbol which contains the address of the specified
    /// ptr.
    void *emitGlobalValueIndirectSym(const GlobalValue* GV, void *ptr,
                                     JITCodeEmitter &JCE) override;

    // getStubLayout - Returns the size and alignment of the largest call stub
    // on X86.
    StubLayout getStubLayout() override;

    /// emitFunctionStub - Use the specified JITCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.
    void *emitFunctionStub(const Function* F, void *Target,
                           JITCodeEmitter &JCE) override;

    /// getPICJumpTableEntry - Returns the value of the jumptable entry for the
    /// specific basic block.
    uintptr_t getPICJumpTableEntry(uintptr_t BB, uintptr_t JTBase) override;

    /// getLazyResolverFunction - Expose the lazy resolver to the JIT.
    LazyResolverFn getLazyResolverFunction(JITCompilerFn) override;

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    void relocate(void *Function, MachineRelocation *MR,
                  unsigned NumRelocs, unsigned char* GOTBase) override;

    /// allocateThreadLocalMemory - Each target has its own way of
    /// handling thread local variables. This method returns a value only
    /// meaningful to the target.
    char* allocateThreadLocalMemory(size_t size) override;

    /// setPICBase / getPICBase - Getter / setter of PICBase, used to compute
    /// PIC jumptable entry.
    void setPICBase(uintptr_t Base) { PICBase = Base; }
    uintptr_t getPICBase() const { return PICBase; }
  };
}

#endif
