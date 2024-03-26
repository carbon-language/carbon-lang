// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_

#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// A function.
struct Function : public Printable<Function> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", enclosing_scope: " << enclosing_scope_id
        << ", param_refs: " << param_refs_id;
    if (return_type_id.is_valid()) {
      out << ", return_type: " << return_type_id;
    }
    if (return_slot_id.is_valid()) {
      out << ", return_slot: " << return_slot_id;
    }
    if (!body_block_ids.empty()) {
      out << llvm::formatv(
          ", body: [{0}]",
          llvm::make_range(body_block_ids.begin(), body_block_ids.end()));
    }
    out << "}";
  }

  // Given a parameter reference instruction from `param_refs_id` or
  // `implicit_param_refs_id`, returns the corresponding `Param` instruction
  // and its ID.
  static auto GetParamFromParamRefId(const File& sem_ir, InstId param_ref_id)
      -> std::pair<InstId, Param>;

  // The following members always have values, and do not change throughout the
  // lifetime of the function.

  // The function name.
  NameId name_id;
  // The enclosing scope.
  NameScopeId enclosing_scope_id;
  // The first declaration of the function. This is a FunctionDecl.
  InstId decl_id;
  // A block containing a single reference instruction per implicit parameter.
  InstBlockId implicit_param_refs_id;
  // A block containing a single reference instruction per parameter.
  InstBlockId param_refs_id;
  // The return type. This will be invalid if the return type wasn't specified.
  TypeId return_type_id;
  // The storage for the return value, which is a reference expression whose
  // type is the return type of the function. Will be invalid if the function
  // doesn't have a return slot. If this is valid, a call to the function is
  // expected to have an additional final argument corresponding to the return
  // slot.
  InstId return_slot_id;
  // Whether the declaration is extern.
  bool is_extern;

  // The following members are set at the end of a builtin function definition.

  // If this is a builtin function, the corresponding builtin kind.
  BuiltinFunctionKind builtin_kind = BuiltinFunctionKind::None;

  // The following members are set at the `{` of the function definition.

  // The definition, if the function has been defined or is currently being
  // defined. This is a FunctionDecl.
  InstId definition_id = InstId::Invalid;

  // The following members are accumulated throughout the function definition.

  // A list of the statically reachable code blocks in the body of the
  // function, in lexical order. The first block is the entry block. This will
  // be empty for declarations that don't have a visible definition.
  llvm::SmallVector<InstBlockId> body_block_ids = {};
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
