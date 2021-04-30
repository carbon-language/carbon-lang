//===--- BinaryEmitter.h - collection of functions to emit code and data --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_EMITTER_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_EMITTER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class MCStreamer;

namespace bolt {
class BinaryContext;
class BinaryFunction;

/// Emit all code and data in the BinaryContext \p BC.
///
/// \p OrgSecPrefix is used to modify name of emitted original sections
/// contained in \p BC. This is done to distinguish them from sections emitted
/// by LLVM backend.
void emitBinaryContext(MCStreamer &Streamer, BinaryContext &BC,
                       StringRef OrgSecPrefix = "");

/// Emit \p BF function code. The caller is responsible for emitting function
/// symbol(s) and setting the section to emit the code to.
void emitFunctionBody(MCStreamer &Streamer, BinaryFunction &BF,
                      bool EmitColdPart, bool EmitCodeOnly = false);

} // namespace bolt
} // namespace llvm

#endif
