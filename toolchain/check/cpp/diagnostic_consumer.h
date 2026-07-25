// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_CONSUMER_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_CONSUMER_H_

#include <memory>

#include "clang/Basic/Diagnostic.h"

namespace clang {
class CompilerInvocation;
class DiagnosticConsumer;
}  // namespace clang

namespace Carbon::Diagnostics {
class Consumer;
}  // namespace Carbon::Diagnostics

namespace Carbon::Check {

class Context;
class CppDiagnosticListener;

// Creates a Clang DiagnosticConsumer that adapts Clang diagnostics to Carbon.
auto MakeDiagnosticConsumer(
    Diagnostics::Consumer& consumer,
    std::shared_ptr<clang::CompilerInvocation> invocation)
    -> std::unique_ptr<clang::DiagnosticConsumer>;

// Creates a diagnostic listener attached to the given Carbon context. The
// returned listener must not outlive the context.
auto MakeContextDiagnosticListener(clang::DiagnosticConsumer& consumer,
                                   Context& context)
    -> std::unique_ptr<CppDiagnosticListener>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_CONSUMER_H_
