//===-- LeakDetector.cpp - Implement LeakDetector interface ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LeakDetector class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LeakDetector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/RWMutex.h"
#include "llvm/System/Threading.h"
#include "llvm/Value.h"
using namespace llvm;

namespace {
  template <class T>
  struct VISIBILITY_HIDDEN PrinterTrait {
    static void print(const T* P) { cerr << P; }
  };

  template<>
  struct VISIBILITY_HIDDEN PrinterTrait<Value> {
    static void print(const Value* P) { cerr << *P; }
  };

  ManagedStatic<sys::SmartRWMutex<true> > LeakDetectorLock;

  template <typename T>
  struct VISIBILITY_HIDDEN LeakDetectorImpl {
    explicit LeakDetectorImpl(const char* const name = "") : 
      Cache(0), Name(name) { }

    void clear() {
      Cache = 0;
      Ts.clear();
    }
    
    void setName(const char* n) { 
      Name = n;
    }
    
    // Because the most common usage pattern, by far, is to add a
    // garbage object, then remove it immediately, we optimize this
    // case.  When an object is added, it is not added to the set
    // immediately, it is added to the CachedValue Value.  If it is
    // immediately removed, no set search need be performed.
    void addGarbage(const T* o) {
      sys::SmartScopedWriter<true> Writer(*LeakDetectorLock);
      if (Cache) {
        assert(Ts.count(Cache) == 0 && "Object already in set!");
        Ts.insert(Cache);
      }
      Cache = o;
    }

    void removeGarbage(const T* o) {
      sys::SmartScopedWriter<true> Writer(*LeakDetectorLock);
      if (o == Cache)
        Cache = 0; // Cache hit
      else
        Ts.erase(o);
    }

    bool hasGarbage(const std::string& Message) {
      addGarbage(0); // Flush the Cache

      sys::SmartScopedReader<true> Reader(*LeakDetectorLock);
      assert(Cache == 0 && "No value should be cached anymore!");

      if (!Ts.empty()) {
        cerr << "Leaked " << Name << " objects found: " << Message << ":\n";
        for (typename SmallPtrSet<const T*, 8>::iterator I = Ts.begin(),
               E = Ts.end(); I != E; ++I) {
          cerr << "\t";
          PrinterTrait<T>::print(*I);
          cerr << "\n";
        }
        cerr << '\n';

        return true;
      }
      
      return false;
    }

  private:
    SmallPtrSet<const T*, 8> Ts;
    const T* Cache;
    const char* Name;
  };

  static ManagedStatic<LeakDetectorImpl<void> > Objects;
  static ManagedStatic<LeakDetectorImpl<Value> > LLVMObjects;

  static void clearGarbage() {
    Objects->clear();
    LLVMObjects->clear();
  }
}

void LeakDetector::addGarbageObjectImpl(void *Object) {
  Objects->addGarbage(Object);
}

void LeakDetector::addGarbageObjectImpl(const Value *Object) {
  LLVMObjects->addGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(void *Object) {
  Objects->removeGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(const Value *Object) {
  LLVMObjects->removeGarbage(Object);
}

void LeakDetector::checkForGarbageImpl(const std::string &Message) {
  Objects->setName("GENERIC");
  LLVMObjects->setName("LLVM");
  
  // use non-short-circuit version so that both checks are performed
  if (Objects->hasGarbage(Message) |
      LLVMObjects->hasGarbage(Message))
    cerr << "\nThis is probably because you removed an object, but didn't "
         << "delete it.  Please check your code for memory leaks.\n";

  // Clear out results so we don't get duplicate warnings on
  // next call...
  clearGarbage();
}
