// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DIAGNOSTIC_LOC_CONVERTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_DIAGNOSTIC_LOC_CONVERTER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/absolute_node_id.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Converter from compact location information into a diagnostic location
// describing a filename, line location, and potentially a sequence of imports.
// Such diagnostics locations are used to render user-facing diagnostics and
// also locations for stack trace in crash diagnostics.
class DiagnosticLocConverter {
 public:
  // Information about an import within which the location was found.
  struct ImportLoc {
    Diagnostics::Loc loc;
    // TODO: Include the name of the imported library in this information so it
    // can be included in the diagnostic.
  };

  // Information about a location that has been converted from a LocId to a
  // diagnostic location.
  struct LocAndImports {
    llvm::SmallVector<ImportLoc> imports;
    Diagnostics::ConvertedLoc loc;
  };

  // `tree_and_subtrees_getters` and `sem_ir` must not be null.
  explicit DiagnosticLocConverter(
      const Parse::GetTreeAndSubtreesStore* tree_and_subtrees_getters,
      const File* sem_ir)
      : tree_and_subtrees_getters_(tree_and_subtrees_getters),
        sem_ir_(sem_ir) {}

  // Converts the given location into a sequence of import locations and a final
  // diagnostic location.
  auto ConvertWithImports(LocId loc_id, bool token_only) const -> LocAndImports;

  // Converts the given location into a diagnostic location.
  auto Convert(LocId loc_id, bool token_only) const
      -> Diagnostics::ConvertedLoc;

 private:
  // Converts an `absolute_node_id` in either a Carbon file or C++ import to a
  // diagnostic location.
  auto ConvertImpl(AbsoluteNodeId absolute_node_id, bool token_only) const
      -> Diagnostics::ConvertedLoc;

  // Converts a `node_id` corresponding to a specific check IR to a diagnostic
  // location.
  auto ConvertImpl(CheckIRId check_ir_id, Parse::NodeId node_id,
                   bool token_only) const -> Diagnostics::ConvertedLoc;

  // Converts a location pointing into C++ code to a diagnostic location.
  auto ConvertImpl(ClangSourceLocId clang_source_loc_id) const
      -> Diagnostics::ConvertedLoc;

  // Converters for each SemIR.
  const Parse::GetTreeAndSubtreesStore* tree_and_subtrees_getters_;

  // The current SemIR being processed.
  const File* sem_ir_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_DIAGNOSTIC_LOC_CONVERTER_H_
