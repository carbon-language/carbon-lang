//===- lib/Debugger/FDHandle.h - File Descriptor Handle ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DEBUGGER_FDHANDLE_H
#define LIB_DEBUGGER_FDHANDLE_H

#include "llvm/System/Path.h"

namespace llvm {

/// FDHandle - Simple handle class to make sure a file descriptor gets closed
/// when the object is destroyed.  This handle acts similarly to an
/// std::auto_ptr, in that the copy constructor and assignment operators
/// transfer ownership of the handle.  This means that FDHandle's do not have
/// value semantics.
///
class FDHandle {
  int FD;
public:
  FDHandle() : FD(-1) {}
  FDHandle(int fd) : FD(fd) {}
  FDHandle(FDHandle &RHS) : FD(RHS.FD) {
    RHS.FD = -1;       // Transfer ownership
  }

  ~FDHandle() throw();

  /// get - Get the current file descriptor, without releasing ownership of it.
  int get() const { return FD; }
  operator int() const { return FD; }

  FDHandle &operator=(int fd) throw();

  FDHandle &operator=(FDHandle &RHS) {
    int fd = RHS.FD;
    RHS.FD = -1;       // Transfer ownership
    return operator=(fd);
  }

  /// release - Take ownership of the file descriptor away from the FDHandle
  /// object, so that the file is not closed when the FDHandle is destroyed.
  int release() {
    int Ret = FD;
    FD = -1;
    return Ret;
  }
};

} // End llvm namespace

#endif
