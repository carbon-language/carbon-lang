// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_FILENAME_DIAGNOSTICS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_FILENAME_DIAGNOSTICS_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// We frequently want a `DiagnosticEmitter` that directly uses a filename. Note
// that an empty string can be used for a diagnostic that has no particular
// location.
class FilenameDiagnosticEmitter : public DiagnosticEmitter<llvm::StringRef> {
 public:
  explicit FilenameDiagnosticEmitter(DiagnosticConsumer* consumer)
      : DiagnosticEmitter<llvm::StringRef>(converter_, *consumer) {}

 private:
  // Converts a filename directly to the diagnostic location.
  struct FilenameDiagnosticConverter : DiagnosticConverter<llvm::StringRef> {
    auto ConvertLoc(llvm::StringRef filename, ContextFnT /*context_fn*/) const
        -> ConvertedDiagnosticLoc override {
      return {.loc = {.filename = filename}, .last_byte_offset = -1};
    }
  };

  FilenameDiagnosticConverter converter_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_FILENAME_DIAGNOSTICS_H_
