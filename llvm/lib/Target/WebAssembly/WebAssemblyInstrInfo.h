//=- WebAssemblyInstrInfo.h - WebAssembly Instruction Information -*- C++ -*-=//
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
/// TargetInstrInfo class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYINSTRINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYINSTRINFO_H

#include "WebAssemblyRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {

class WebAssemblySubtarget;

class WebAssemblyInstrInfo final {
  const WebAssemblyRegisterInfo RI;

public:
  explicit WebAssemblyInstrInfo(const WebAssemblySubtarget &STI);

  const WebAssemblyRegisterInfo &getRegisterInfo() const { return RI; }
};

} // end namespace llvm

#endif
