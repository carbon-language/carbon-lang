//===-- LeakDetector.cpp - Implement LeakDetector interface ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the LeakDetector class.
//
//===----------------------------------------------------------------------===//

#include "Support/LeakDetector.h"
#include "llvm/Value.h"
#include <set>

namespace llvm {

// Lazily allocate set so that release build doesn't have to do anything.
static std::set<const void*> *Objects = 0;
static std::set<const Value*> *LLVMObjects = 0;

// Because the most common usage pattern, by far, is to add a garbage object,
// then remove it immediately, we optimize this case.  When an object is added,
// it is not added to the set immediately, it is added to the CachedValue Value.
// If it is immediately removed, no set search need be performed.
//
static const Value *CachedValue;

void LeakDetector::addGarbageObjectImpl(void *Object) {
  if (Objects == 0)
    Objects = new std::set<const void*>();
  assert(Objects->count(Object) == 0 && "Object already in set!");
  Objects->insert(Object);
}

void LeakDetector::removeGarbageObjectImpl(void *Object) {
  if (Objects)
    Objects->erase(Object);
}

void LeakDetector::addGarbageObjectImpl(const Value *Object) {
  if (CachedValue) {
    if (LLVMObjects == 0)
      LLVMObjects = new std::set<const Value*>();
    assert(LLVMObjects->count(CachedValue) == 0 && "Object already in set!");
    LLVMObjects->insert(CachedValue);
  }
  CachedValue = Object;
}

void LeakDetector::removeGarbageObjectImpl(const Value *Object) {
  if (Object == CachedValue)
    CachedValue = 0;             // Cache hit!
  else if (LLVMObjects)
    LLVMObjects->erase(Object);
}

void LeakDetector::checkForGarbageImpl(const std::string &Message) {
  if (CachedValue)  // Flush the cache to the set...
    addGarbageObjectImpl((Value*)0);

  assert(CachedValue == 0 && "No value should be cached anymore!");

  if ((Objects && !Objects->empty()) || (LLVMObjects && !LLVMObjects->empty())){
    std::cerr << "Leaked objects found: " << Message << "\n";

    if (Objects && !Objects->empty()) {
      std::cerr << "  Non-Value objects leaked:";
      for (std::set<const void*>::iterator I = Objects->begin(),
             E = Objects->end(); I != E; ++I)
        std::cerr << " " << *I;
    }

    if (LLVMObjects && !LLVMObjects->empty()) {
      std::cerr << "  LLVM Value subclasses leaked:";
      for (std::set<const Value*>::iterator I = LLVMObjects->begin(),
             E = LLVMObjects->end(); I != E; ++I)
        std::cerr << **I << "\n";
    }

    std::cerr << "This is probably because you removed an LLVM value "
              << "(Instruction, BasicBlock, \netc), but didn't delete it.  "
              << "Please check your code for memory leaks.\n";

    // Clear out results so we don't get duplicate warnings on next call...
    delete Objects; delete LLVMObjects;
    Objects = 0; LLVMObjects = 0;
  }
}

} // End llvm namespace
