// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_ID_H_
#define CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_ID_H_

#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A specific node location in a file. Can refer to a Clang source location
// within imported C++ code.
class AbsoluteNodeId {
 public:
  // A specific node location in a file.
  explicit AbsoluteNodeId(CheckIRId check_ir_id, Parse::NodeId node_id)
      : check_ir_id_(check_ir_id), is_cpp_(false), node_id_(node_id) {}

  // A Clang source location within imported C++ code.
  explicit AbsoluteNodeId(CheckIRId check_ir_id, ClangSourceLocId clang_source_loc_id)
      : check_ir_id_(check_ir_id),
        is_cpp_(true),
        clang_source_loc_id_(clang_source_loc_id) {}

  // The ID of the IR.
  auto check_ir_id() const -> CheckIRId { return check_ir_id_; }

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
  // See `check_ir_id()`.
  CheckIRId check_ir_id_;

  // True if this is a C++ location.
  bool is_cpp_;

  union {
    // See `node_id()`.
    Parse::NodeId node_id_;
    // See `clang_source_loc_id()`.
    ClangSourceLocId clang_source_loc_id_;
  };
};

// Resolves the `LocId` to a series of `NodeId`s, which may be in different
// files. The vector will have one entry if there were no imports, and multiple
// entries when imports are traversed. The final entry is the actual
// declaration.
//
// Note that the `LocId` here is typically not canonical, and it uses that fact
// for non-canonical locations built from an `ExportDecl` instruction.
auto GetAbsoluteNodeId(const File* sem_ir, LocId loc_id)
    -> llvm::SmallVector<AbsoluteNodeId>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_ID_H_
