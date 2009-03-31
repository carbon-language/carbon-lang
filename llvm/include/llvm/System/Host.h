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

#include <string>

namespace llvm {
namespace sys {

  inline bool isLittleEndianHost() {
    union {
      int i;
      char c;
    };
    i = 1;
    return c;
  }

  inline bool isBigEndianHost() {
    return !isLittleEndianHost();
  }

  /// getHostTriple() - Return the target triple of the running
  /// system.
  ///
  /// The target triple is a string in the format of:
  ///   CPU_TYPE-VENDOR-OPERATING_SYSTEM
  /// or
  ///   CPU_TYPE-VENDOR-KERNEL-OPERATING_SYSTEM
  std::string getHostTriple();

}
}

#endif
