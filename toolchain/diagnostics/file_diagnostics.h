// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_FILE_DIAGNOSTICS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_FILE_DIAGNOSTICS_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// We frequently want a `DiagnosticEmitter` that directly uses a filename. Note
// that an empty string can be used for a diagnostic that has no particular
// location.
//
// Note this provides no way to set a line or column on diagnostics. More
// specific emitters must be used for that.
class FileDiagnosticEmitter : public DiagnosticEmitter<llvm::StringRef> {
 public:
  using DiagnosticEmitter::DiagnosticEmitter;

 protected:
  // Converts a filename directly to the diagnostic location.
  auto ConvertLoc(llvm::StringRef filename, ContextFnT /*context_fn*/) const
      -> ConvertedDiagnosticLoc override {
    return {.loc = {.filename = filename}, .last_byte_offset = -1};
  }
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_FILE_DIAGNOSTICS_H_
