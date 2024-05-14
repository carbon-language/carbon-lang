// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PARAM_AND_ARG_REFS_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_PARAM_AND_ARG_REFS_STACK_H_

#include "common/check.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/node_stack.h"

namespace Carbon::Check {

// The stack of instruction blocks being used for per-element tracking of
// instructions in parameter and argument instruction blocks. Versus
// InstBlockStack, an element will have 1 or more instructions in blocks in
// InstBlockStack, but only ever 1 instruction in blocks here. The result is
// typically referred to as "param_refs" or "arg_refs".
class ParamAndArgRefsStack {
 public:
  explicit ParamAndArgRefsStack(SemIR::File& sem_ir,
                                llvm::raw_ostream* vlog_stream,
                                NodeStack& node_stack)
      : node_stack_(&node_stack),
        stack_("param_and_arg_refs_stack", sem_ir, vlog_stream) {}

  // Starts handling parameters or arguments.
  auto Push() -> void { stack_.Push(); }

  // On a comma, pushes the most recent instruction, becoming param or arg ref.
  // This also pops the NodeStack, meaning its top will remain start_kind.
  auto ApplyComma() -> void {
    // Support expressions, parameters, and other nodes like `StructField`
    // that produce InstIds.
    stack_.AddInstId(node_stack_->Pop<SemIR::InstId>());
  }

  // Detects whether there's an entry to push from the end of a parameter or
  // argument list, and if so, moves it to the current parameter or argument
  // list. Does not pop the list. `start_kind` is the node kind at the start
  // of the parameter or argument list, and will be at the top of the parse node
  // stack when this function returns.
  auto EndNoPop(Parse::NodeKind start_kind) -> void {
    if (!node_stack_->PeekIs(start_kind)) {
      // Support expressions, parameters, and other nodes like
      // `StructField` that produce InstIds.
      stack_.AddInstId(node_stack_->Pop<SemIR::InstId>());
    }
  }

  // Pops the current parameter or argument list. Should only be called after
  // `EndNoPop`.
  auto Pop() -> SemIR::InstBlockId { return stack_.Pop(); }

  // Detects whether there's an entry to push. Pops and returns the argument
  // list. This is the same as `EndNoPop` followed by `Pop`.
  auto EndAndPop(Parse::NodeKind start_kind) -> SemIR::InstBlockId {
    EndNoPop(start_kind);
    return Pop();
  }

  // Pops the top instruction block, and discards it if it hasn't had an ID
  // allocated.
  auto PopAndDiscard() -> void { stack_.PopAndDiscard(); }

  // Returns a view of the contents of the top instruction block on the stack.
  auto PeekCurrentBlockContents() -> llvm::ArrayRef<SemIR::InstId> {
    return stack_.PeekCurrentBlockContents();
  }

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void { stack_.VerifyOnFinish(); }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void {
    return stack_.PrintForStackDump(output);
  }

 private:
  // The node stack is manipulated when adding refs.
  NodeStack* node_stack_;

  // The refs stack.
  InstBlockStack stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PARAM_AND_ARG_REFS_STACK_H_
