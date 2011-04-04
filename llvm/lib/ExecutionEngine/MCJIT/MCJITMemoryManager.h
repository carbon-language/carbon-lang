//===-- MCJITMemoryManager.h - Definition for the Memory Manager ---C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_MCJITMEMORYMANAGER_H
#define LLVM_LIB_EXECUTIONENGINE_MCJITMEMORYMANAGER_H

#include "llvm/Module.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include <assert.h>

namespace llvm {

// The MCJIT memory manager is a layer between the standard JITMemoryManager
// and the RuntimeDyld interface that maps objects, by name, onto their
// matching LLVM IR counterparts in the module(s) being compiled.
class MCJITMemoryManager : public RTDyldMemoryManager {
  JITMemoryManager *JMM;

  // FIXME: Multiple modules.
  Module *M;
public:
  MCJITMemoryManager(JITMemoryManager *jmm) : JMM(jmm) {}

  // Allocate ActualSize bytes, or more, for the named function. Return
  // a pointer to the allocated memory and update Size to reflect how much
  // memory was acutally allocated.
  uint64_t startFunctionBody(const char *Name, uintptr_t &Size) {
    Function *F = M->getFunction(Name);
    assert(F && "No matching function in JIT IR Module!");
    return (uint64_t)JMM->startFunctionBody(F, Size);
  }

  // Mark the end of the function, including how much of the allocated
  // memory was actually used.
  void endFunctionBody(const char *Name, uint64_t FunctionStart,
                       uint64_t FunctionEnd) {
    Function *F = M->getFunction(Name);
    assert(F && "No matching function in JIT IR Module!");
    // The JITMemoryManager interface makes the unfortunate assumption that
    // the address space/sizes we're compiling on are the same as what we're
    // compiling for, so it uses pointer types for its addresses. Explicit
    // casts between them to deal with that.
    return JMM->endFunctionBody(F, (uint8_t*)FunctionStart,
                                (uint8_t*)FunctionEnd);
  }

};

} // End llvm namespace

#endif
