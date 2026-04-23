// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_REF_H_
#define CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_REF_H_

#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A specific node location in a file. Usually refers to a NodeId in a Carbon
// source file, but can also refer to a Clang source location within imported
// C++ code.
class AbsoluteNodeRef {
 public:
  // A specific node location in a file.
  explicit AbsoluteNodeRef(const File* file, Parse::NodeId node_id)
      : file_(file), is_cpp_(false), node_id_(node_id) {}

  // A Clang source location within imported C++ code.
  explicit AbsoluteNodeRef(const File* file,
                           ClangSourceLocId clang_source_loc_id)
      : file_(file), is_cpp_(true), clang_source_loc_id_(clang_source_loc_id) {}

  // The file containing the location.
  auto file() const -> const File* { return file_; }

  // The ID of the IR.
  auto check_ir_id() const -> CheckIRId { return file_->check_ir_id(); }

  // Returns true if this is a C++ location.
  auto is_cpp() const -> bool { return is_cpp_; }

  // The specific node location in a file. Must be called only if
  // `is_cpp()` is false.
  auto node_id() const -> Parse::NodeId {
    CARBON_CHECK(!is_cpp());
    return node_id_;
  }

  // The Clang source location. Must be called only if `is_cpp()` is true.
  auto clang_source_loc_id() const -> ClangSourceLocId {
    CARBON_CHECK(is_cpp());
    return clang_source_loc_id_;
  }

 private:
  // The file containing the location.
  const File* file_;

  // True if this is a C++ location.
  bool is_cpp_;

  union {
    // See `node_id()`.
    Parse::NodeId node_id_;
    // See `clang_source_loc_id()`.
    ClangSourceLocId clang_source_loc_id_;
  };
};

// Resolves the `LocId` to a series of `AbsoluteNodeRef`s, which may be in
// different files. The vector will have one entry if there were no imports, and
// multiple entries when imports are traversed. The final entry is the actual
// declaration.
//
// Note that the `LocId` here is typically not canonical, and it uses that fact
// for non-canonical locations built from an `ExportDecl` instruction.
auto GetAbsoluteNodeRef(const File* sem_ir, LocId loc_id)
    -> llvm::SmallVector<AbsoluteNodeRef>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_REF_H_
