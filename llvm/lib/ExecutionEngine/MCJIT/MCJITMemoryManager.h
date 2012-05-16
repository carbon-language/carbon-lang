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

};

} // End llvm namespace

#endif
