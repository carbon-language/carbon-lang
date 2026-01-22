// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_CLANG_INVOCATION_H_
#define CARBON_TOOLCHAIN_BASE_CLANG_INVOCATION_H_

#include <string>

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Converts diagnostics from the Clang driver to Carbon diagnostics.
class ClangDriverDiagnosticConsumer : public clang::DiagnosticConsumer {
 public:
  // Creates an instance with the location that triggers calling Clang.
  // `context` must not be null.
  explicit ClangDriverDiagnosticConsumer(Diagnostics::NoLocEmitter* emitter)
      : emitter_(emitter) {}

  // Generates a Carbon warning for each Clang warning and a Carbon error for
  // each Clang error or fatal.
  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override;

 private:
  // Diagnostic emitter. Note that driver diagnostics don't have meaningful
  // locations attached.
  Diagnostics::NoLocEmitter* emitter_;
};

// Builds and returns a clang `CompilerInvocation` to use when building code for
// interop, from a list of clang driver arguments. Emits diagnostics to
// `consumer` if the arguments are invalid.
auto BuildClangInvocation(Diagnostics::Consumer& consumer,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                          const InstallPaths& install_paths,
                          llvm::StringRef target_str,
                          llvm::ArrayRef<llvm::StringRef> extra_args = {})
    -> std::unique_ptr<clang::CompilerInvocation>;

// Appends the default Clang command line arguments used when building a
// Carbon-compatible Clang invocation.
//
// Where possible, code should use `BuildClangInvocation` above. However, when
// invoking Clang directly, this can be used to get the core compatible flags.
auto AppendDefaultClangArgs(const InstallPaths& install_paths,
                            llvm::StringRef target_str,
                            llvm::SmallVectorImpl<std::string>& args) -> void;

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_CLANG_INVOCATION_H_
