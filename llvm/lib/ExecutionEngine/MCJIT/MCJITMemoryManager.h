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
  virtual void anchor();
  JITMemoryManager *JMM;

  // FIXME: Multiple modules.
  Module *M;
public:
  MCJITMemoryManager(JITMemoryManager *jmm, Module *m) :
    JMM(jmm?jmm:JITMemoryManager::CreateDefaultMemManager()), M(m) {}
  // We own the JMM, so make sure to delete it.
  ~MCJITMemoryManager() { delete JMM; }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID) {
    return JMM->allocateDataSection(Size, Alignment, SectionID);
  }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID) {
    return JMM->allocateCodeSection(Size, Alignment, SectionID);
  }

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true) {
    return JMM->getPointerToNamedFunction(Name, AbortOnFailure);
  }

  // Allocate ActualSize bytes, or more, for the named function. Return
  // a pointer to the allocated memory and update Size to reflect how much
  // memory was acutally allocated.
  uint8_t *startFunctionBody(const char *Name, uintptr_t &Size) {
    // FIXME: This should really reference the MCAsmInfo to get the global
    //        prefix.
    if (Name[0] == '_') ++Name;
    Function *F = M->getFunction(Name);
    // Some ObjC names have a prefixed \01 in the IR. If we failed to find
    // the symbol and it's of the ObjC conventions (starts with "-" or
    // "+"), try prepending a \01 and see if we can find it that way.
    if (!F && (Name[0] == '-' || Name[0] == '+'))
      F = M->getFunction((Twine("\1") + Name).str());
    assert(F && "No matching function in JIT IR Module!");
    return JMM->startFunctionBody(F, Size);
  }

  // Mark the end of the function, including how much of the allocated
  // memory was actually used.
  void endFunctionBody(const char *Name, uint8_t *FunctionStart,
                       uint8_t *FunctionEnd) {
    // FIXME: This should really reference the MCAsmInfo to get the global
    //        prefix.
    if (Name[0] == '_') ++Name;
    Function *F = M->getFunction(Name);
    // Some ObjC names have a prefixed \01 in the IR. If we failed to find
    // the symbol and it's of the ObjC conventions (starts with "-" or
    // "+"), try prepending a \01 and see if we can find it that way.
    if (!F && (Name[0] == '-' || Name[0] == '+'))
      F = M->getFunction((Twine("\1") + Name).str());
    assert(F && "No matching function in JIT IR Module!");
    JMM->endFunctionBody(F, FunctionStart, FunctionEnd);
  }

};

} // End llvm namespace

#endif
