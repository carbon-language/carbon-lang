// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_

#include <string>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class SourceManager;
}  // namespace clang

namespace Carbon::Check {

class CarbonClangDiagnosticConsumer;

// Interface for listening to Clang diagnostics from
// CarbonClangDiagnosticConsumer.
//
// Pushes itself onto the consumer's listener stack on construction and pops
// itself on destruction.
class CppDiagnosticListener {
 public:
  // A change Clang suggests: what to put in `range`, which is empty for an
  // insertion and where `text` is empty for a deletion.
  struct FixIt {
    clang::CharSourceRange range;
    std::string text;
  };

  struct Diagnostic {
    clang::DiagnosticsEngine::Level level;
    // Where the diagnostic points, reaching across the token it names so that a
    // note marks the token rather than a single column of it.
    clang::CharSourceRange location;
    const clang::SourceManager* source_manager = nullptr;
    std::string message;
    // What a note says about `reason_range` rather than about the declaration
    // it names, split off from `message`, or empty for a note that says only
    // one thing. Clang words a candidate note as "<which candidate>: <what is
    // wrong with it>" and marks the source the second half is about, which is
    // two labels here rather than one sentence hung off the name.
    std::string reason;
    clang::CharSourceRange reason_range;
    // What Clang would underline, which becomes the ranges the diagnostic
    // marks. Clang's own rendering of them is not kept: the toolchain draws
    // these the way it draws its own. `reason_range` is not among them, since
    // it carries words instead.
    llvm::SmallVector<clang::CharSourceRange, 2> ranges;
    llvm::SmallVector<FixIt, 0> fix_its;
  };

  explicit CppDiagnosticListener(CarbonClangDiagnosticConsumer& consumer);
  virtual ~CppDiagnosticListener();

  // Emits a group of buffered diagnostics, which will comprise a single leading
  // diagnostic -- typically an error / warning / remark -- followed by a
  // sequence of attached notes. The first diagnostic can also be a note if the
  // diagnostics engine is flushed in the middle of Clang emitting a diagnostic,
  // or if Clang emits a "stray" note not attached to any diagnostic. Both would
  // generally indicate Clang bugs.
  virtual auto EmitDiagnostics(llvm::ArrayRef<Diagnostic> diags) -> void = 0;

 protected:
  auto consumer() -> CarbonClangDiagnosticConsumer& { return *consumer_; }

 private:
  CarbonClangDiagnosticConsumer* consumer_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_
