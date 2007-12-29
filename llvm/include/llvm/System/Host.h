//===- llvm/System/Host.h - Host machine characteristics --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods for querying the nature of the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_HOST_H
#define LLVM_SYSTEM_HOST_H

namespace llvm {
namespace sys {

  inline bool littleEndianHost() {
    union {
      int i;
      char c;
    };
    i = 1;
    return c;
  }

  inline bool bigEndianHost() {
    return !littleEndianHost();
  }

}
}

#endif
