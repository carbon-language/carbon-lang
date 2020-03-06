//===--- BinaryEmitter.h - collection of functions to emit code and data --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_EMITTER_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_EMITTER_H

#include "BinaryContext.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
namespace bolt {

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
