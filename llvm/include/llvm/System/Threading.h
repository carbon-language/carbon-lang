//===-- llvm/System/Threading.h - Control multithreading mode --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TThis file defines llvm_start_multithreaded() and friends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_THREADING_H
#define LLVM_SYSTEM_THREADING_H

namespace llvm {
  /// llvm_start_multithreaded - Allocate and initialize structures needed to
  /// make LLVM safe for multithreading.  The return value indicates whether
  /// multithreaded initialization succeeded.  LLVM will still be operational
  /// on "failed" return, and will still be safe for hosting threading 
  /// applications in the JIT, but will not be safe for concurrent calls to the
  /// LLVM APIs.
  /// THIS MUST EXECUTE IN ISOLATION FROM ALL OTHER LLVM API CALLS.
  bool llvm_start_multithreaded();
  
  /// llvm_stop_multithreaded - Deallocate structures necessary to make LLVM
  /// safe for multithreading.
  /// THIS MUST EXECUTE IN ISOLATION FROM ALL OTHER LLVM API CALLS.
  void llvm_stop_multithreaded();
  
  /// llvm_is_multithreaded - Check whether LLVM is executing in thread-safe
  /// mode or not.
  bool llvm_is_multithreaded();
  
  /// acquire_global_lock - Acquire the global lock.  This is a no-op if called
  /// before llvm_start_multithreaded().
  void llvm_acquire_global_lock();
  
  /// release_global_lock - Release the global lock.  This is a no-op if called
  /// before llvm_start_multithreaded().
  void llvm_release_global_lock();
}

#endif
