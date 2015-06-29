//- WebAssemblyISelLowering.h - WebAssembly DAG Lowering Interface -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the interfaces that WebAssembly uses to lower LLVM
/// code into a selection DAG.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYISELLOWERING_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYISELLOWERING_H

#include "llvm/Target/TargetLowering.h"

namespace llvm {

namespace WebAssemblyISD {

enum {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // add memory opcodes starting at ISD::FIRST_TARGET_MEMORY_OPCODE here...
};

} // end namespace WebAssemblyISD

class WebAssemblySubtarget;
class WebAssemblyTargetMachine;

class WebAssemblyTargetLowering final : public TargetLowering {
public:
  WebAssemblyTargetLowering(const TargetMachine &TM,
                            const WebAssemblySubtarget &STI);

private:
  /// Keep a pointer to the WebAssemblySubtarget around so that we can make the
  /// right decision when generating code for different targets.
  const WebAssemblySubtarget *Subtarget;
};

} // end namespace llvm

#endif
