//===-- OrcTargetSupport.h - Code to support specific targets  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Target specific code for Orc, e.g. callback assembly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCTARGETSUPPORT_H
#define LLVM_EXECUTIONENGINE_ORC_ORCTARGETSUPPORT_H

#include "IndirectionUtils.h"

namespace llvm {

/// @brief Insert callback asm into module M for the symbols managed by
///        JITResolveCallbackHandler J.
void insertX86CallbackAsm(Module &M, JITResolveCallbackHandler &J);
}

#endif // LLVM_EXECUTIONENGINE_ORC_ORCTARGETSUPPORT_H
