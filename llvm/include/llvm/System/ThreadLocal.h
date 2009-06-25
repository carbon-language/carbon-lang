//===- llvm/System/ThreadLocal.h - Thread Local Data ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::ThreadLocal class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_THREAD_LOCAL_H
#define LLVM_SYSTEM_THREAD_LOCAL_H

#include "llvm/System/Threading.h"
#include <cassert>

namespace llvm {
  namespace sys {
    class ThreadLocalImpl {
      void* data;
    public:
      ThreadLocalImpl();
      virtual ~ThreadLocalImpl();
      void setInstance(const void* d);
      const void* getInstance();
    };
    
    template<class T>
    class ThreadLocal : public ThreadLocalImpl {
    public:
      ThreadLocal() : ThreadLocalImpl() { }
      T* get() { return static_cast<T*>(getInstance()); }
      void set(T* d) { setInstance(d); }
    };
  }
}

#endif
