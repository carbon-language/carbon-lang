//===- ARMJITInfo.h - ARM implementation of the JIT interface  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMJITINFO_H
#define ARMJITINFO_H

#include "llvm/Target/TargetJITInfo.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  class ARMTargetMachine;

  class ARMJITInfo : public TargetJITInfo {
    ARMTargetMachine &TM;

    // ConstPoolId2AddrMap - A map from constant pool ids to the corresponding
    // CONSTPOOL_ENTRY addresses.
    SmallVector<intptr_t, 32> ConstPoolId2AddrMap;

  public:
    explicit ARMJITInfo(ARMTargetMachine &tm) : TM(tm) { useGOT = false; }

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);

    /// emitFunctionStub - Use the specified MachineCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.
    virtual void *emitFunctionStub(const Function* F, void *Fn,
                                   MachineCodeEmitter &MCE);

    /// getLazyResolverFunction - Expose the lazy resolver to the JIT.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);
  
    /// hasCustomConstantPool - Allows a target to specify that constant
    /// pool address resolution is handled by the target.
    virtual bool hasCustomConstantPool() const { return true; }

    /// ResizeConstPoolMap - Resize constant pool ids to CONSTPOOL_ENTRY
    /// addresses map.
    void ResizeConstPoolMap(unsigned Size) {
      ConstPoolId2AddrMap.resize(Size);
    }

    /// getConstantPoolEntryAddr - The ARM target puts all constant
    /// pool entries into constant islands. Resolve the constant pool index
    /// into the address where the constant is stored.
    intptr_t getConstantPoolEntryAddr(unsigned CPI) const {
      assert(CPI < ConstPoolId2AddrMap.size());
      return ConstPoolId2AddrMap[CPI];
    }

    /// addConstantPoolEntryAddr - Map a Constant Pool Index (CPI) to the address
    /// where its associated value is stored. When relocations are processed,
    /// this value will be used to resolve references to the constant.
    void addConstantPoolEntryAddr(unsigned CPI, intptr_t Addr) {
      assert(CPI < ConstPoolId2AddrMap.size());
      ConstPoolId2AddrMap[CPI] = Addr;
    }
  };
}

#endif
