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

#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/Visibility.h"
#include "llvm/Value.h"
#include <iostream>
#include <set>
using namespace llvm;

namespace {
  template <class T>
  struct VISIBILITY_HIDDEN PrinterTrait {
    static void print(const T* P) { std::cerr << P; }
  };

  template<>
  struct VISIBILITY_HIDDEN PrinterTrait<Value> {
    static void print(const Value* P) { std::cerr << *P; }
  };

  template <typename T>
  struct VISIBILITY_HIDDEN LeakDetectorImpl {
    LeakDetectorImpl(const char* const name) : Cache(0), Name(name) { }

    // Because the most common usage pattern, by far, is to add a
    // garbage object, then remove it immediately, we optimize this
    // case.  When an object is added, it is not added to the set
    // immediately, it is added to the CachedValue Value.  If it is
    // immediately removed, no set search need be performed.
    void addGarbage(const T* o) {
      if (Cache) {
        assert(Ts.count(Cache) == 0 && "Object already in set!");
        Ts.insert(Cache);
      }
      Cache = o;
    }

    void removeGarbage(const T* o) {
      if (o == Cache)
        Cache = 0; // Cache hit
      else
        Ts.erase(o);
    }

    bool hasGarbage(const std::string& Message) {
      addGarbage(0); // Flush the Cache

      assert(Cache == 0 && "No value should be cached anymore!");

      if (!Ts.empty()) {
        std::cerr
            << "Leaked " << Name << " objects found: " << Message << ":\n";
        for (typename std::set<const T*>::iterator I = Ts.begin(),
               E = Ts.end(); I != E; ++I) {
          std::cerr << "\t";
          PrinterTrait<T>::print(*I);
          std::cerr << "\n";
        }
        std::cerr << '\n';

        return true;
      }
      return false;
    }

  private:
    std::set<const T*> Ts;
    const T* Cache;
    const char* const Name;
  };

  LeakDetectorImpl<void>  *Objects;
  LeakDetectorImpl<Value> *LLVMObjects;

  LeakDetectorImpl<void> &getObjects() {
    if (Objects == 0)
      Objects = new LeakDetectorImpl<void>("GENERIC");
    return *Objects;
  }

  LeakDetectorImpl<Value> &getLLVMObjects() {
    if (LLVMObjects == 0)
      LLVMObjects = new LeakDetectorImpl<Value>("LLVM");
    return *LLVMObjects;
  }

  void clearGarbage() {
    delete Objects;
    delete LLVMObjects;
    Objects = 0;
    LLVMObjects = 0;
  }
}

void LeakDetector::addGarbageObjectImpl(void *Object) {
  getObjects().addGarbage(Object);
}

void LeakDetector::addGarbageObjectImpl(const Value *Object) {
  getLLVMObjects().addGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(void *Object) {
  getObjects().removeGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(const Value *Object) {
  getLLVMObjects().removeGarbage(Object);
}

void LeakDetector::checkForGarbageImpl(const std::string &Message) {
  // use non-short-circuit version so that both checks are performed
  if (getObjects().hasGarbage(Message) |
      getLLVMObjects().hasGarbage(Message))
    std::cerr << "\nThis is probably because you removed an object, but didn't "
                 "delete it.  Please check your code for memory leaks.\n";

  // Clear out results so we don't get duplicate warnings on
  // next call...
  clearGarbage();
}
