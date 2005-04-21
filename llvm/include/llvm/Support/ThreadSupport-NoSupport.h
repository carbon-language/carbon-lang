//===-- llvm/Support/ThreadSupport-NoSupport.h - Generic Impl ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic ThreadSupport implementation used when there is
// no supported threading mechanism on the current system.  Users should never
// #include this file directly!
//
//===----------------------------------------------------------------------===//

// Users should never #include this file directly!  As such, no include guards
// are needed.

#ifndef LLVM_SUPPORT_THREADSUPPORT_H
#error "Code should not #include Support/ThreadSupport-NoSupport.h directly!"
#endif

namespace llvm {
  /// Mutex - This class allows user code to protect variables shared between
  /// threads.  It implements a "recursive" mutex, to simplify user code.
  ///
  /// Since there is no platform support for _creating threads_, the non-thread
  /// implementation of this class is a noop.
  ///
  struct Mutex {
    void acquire () {}
    void release () {}
  };
}
