// WebAssemblyRegisterInfo.h - WebAssembly Register Information Impl -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the WebAssembly implementation of the
/// WebAssemblyRegisterInfo class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYREGISTERINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYREGISTERINFO_H

namespace llvm {

class MachineFunction;
class RegScavenger;
class TargetRegisterClass;
class Triple;

class WebAssemblyRegisterInfo final {
  const Triple &TT;

public:
  explicit WebAssemblyRegisterInfo(const Triple &TT);
};

} // end namespace llvm

#endif
