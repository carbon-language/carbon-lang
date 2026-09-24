// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DOMINANCE_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_SEM_IR_DOMINANCE_TEST_HELPERS_H_

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Builds a `SemIR::File` containing synthetic function bodies for dominance
// tests and benchmarks.
//
// The generated IR only has the properties that `VerifyDominance` looks at: it
// has no locations or real types, and instructions may be shared between
// functions, neither of which the verifier examines.
class DominanceTestFile {
 public:
  DominanceTestFile()
      : file_(/*parse_tree=*/nullptr, CheckIRId(0),
              /*packaging_decl=*/std::nullopt, value_stores_,
              "dominance_test.carbon"),
        cond_id_(AddConstant()) {}

  auto file() -> File& { return file_; }

  // Adds a block whose contents are filled in later by `SetBlock`.
  auto AddBlock() -> InstBlockId {
    return file_.inst_blocks().AddPlaceholder();
  }

  // Adds a block containing `insts` followed by `terminators`.
  auto AddBlock(llvm::ArrayRef<InstId> insts,
                llvm::ArrayRef<InstId> terminators = {}) -> InstBlockId {
    auto block_id = AddBlock();
    SetBlock(block_id, insts, terminators);
    return block_id;
  }

  // Fills a block created by `AddBlock()` with `insts` followed by
  // `terminators`.
  auto SetBlock(InstBlockId block_id, llvm::ArrayRef<InstId> insts,
                llvm::ArrayRef<InstId> terminators = {}) -> void {
    if (terminators.empty()) {
      file_.inst_blocks().ReplacePlaceholder(block_id, insts);
      return;
    }
    llvm::SmallVector<InstId> all_insts;
    all_insts.reserve(insts.size() + terminators.size());
    all_insts.append(insts.begin(), insts.end());
    all_insts.append(terminators.begin(), terminators.end());
    file_.inst_blocks().ReplacePlaceholder(block_id, all_insts);
  }

  // Adds an instruction that is not in any block.
  template <typename InstT>
  auto AddInst(InstT inst) -> InstId {
    return file_.insts().AddInNoBlock(LocIdAndInst::NoLoc(inst));
  }

  // Adds an instruction with no constant value, so that uses of it must be
  // dominated by its evaluation.
  template <typename InstT>
  auto AddNonConstInst(InstT inst) -> InstId {
    auto inst_id = AddInst(inst);
    file_.constant_values().Set(inst_id, ConstantId::NotConstant);
    return inst_id;
  }

  // Adds an instruction producing a non-constant value.
  auto AddValue() -> InstId {
    return AddNonConstInst(
        BoolLiteral{.type_id = TypeType::TypeId, .value = BoolValue(false)});
  }

  // Adds an instruction producing a constant value.
  auto AddConstant() -> InstId {
    auto inst_id = AddInst(
        BoolLiteral{.type_id = TypeType::TypeId, .value = BoolValue(true)});
    file_.constant_values().Set(inst_id,
                                ConstantId::ForConcreteConstant(inst_id));
    return inst_id;
  }

  // Adds an instruction that uses the value of `value_id`.
  auto AddUse(InstId value_id) -> InstId {
    return AddNonConstInst(
        ValueAsRef{.type_id = TypeType::TypeId, .value_id = value_id});
  }

  auto AddReturn() -> InstId { return AddInst(Return{}); }

  auto AddBranch(InstBlockId target_id) -> InstId {
    return AddInst(Branch{.target_id = LabelId(target_id)});
  }

  auto AddBranchIf(InstBlockId target_id) -> InstId {
    return AddInst(
        BranchIf{.target_id = LabelId(target_id), .cond_id = cond_id_});
  }

  // Adds a generic, along with a specific for it that a function can be
  // attached to. `VerifyDominance` verifies a generic function's body once for
  // the generic itself and once for each resolved specific. The specific is
  // resolved, with `value_block_id` as the value block for its declaration.
  auto AddGeneric(InstBlockId value_block_id = InstBlockId::Empty)
      -> GenericId {
    auto decl_id =
        AddInst(FunctionDecl{.type_id = TypeType::TypeId,
                             .function_id = FunctionId(0),
                             .decl_block_id = DeclInstBlockId::None});
    auto generic_id =
        file_.generics().Add(Generic{.decl_id = decl_id,
                                     .bindings_id = InstBlockId::Empty,
                                     .self_specific_id = SpecificId::None});
    auto specific_id =
        file_.specifics().GetOrAdd(generic_id, InstBlockId::Empty);
    file_.specifics()
        .Get(specific_id)
        .SetValueBlock(GenericInstIndex::Declaration, value_block_id);
    return generic_id;
  }

  auto AddFunction(llvm::ArrayRef<InstBlockId> body_block_ids = {},
                   GenericId generic_id = GenericId::None) -> FunctionId {
    auto name_id = file_.identifiers().Add("F");
    return file_.functions().Add(
        {{.name_id = NameId::ForIdentifier(name_id),
          .parent_scope_id = NameScopeId::Package,
          .generic_id = generic_id,
          .first_param_node_id = Parse::NodeId::None,
          .last_param_node_id = Parse::NodeId::None,
          .pattern_block_id = InstBlockId::Empty,
          .implicit_param_patterns_id = InstBlockId::None,
          .param_patterns_id = InstBlockId::Empty,
          .is_extern = false,
          .extern_library_id = LibraryNameId::None,
          .non_owning_decl_id = InstId::None,
          .first_owning_decl_id = InstId::None},
         {.call_param_patterns_id = InstBlockId::Empty,
          .call_params_id = InstBlockId::Empty,
          .call_param_default_values_id = InstBlockId::Empty,
          .call_param_ranges = Function::CallParamIndexRanges::Empty,
          .return_type_inst_id = TypeInstId::None,
          .return_form_inst_id = InstId::None,
          .return_pattern_id = InstId::None,
          .body_block_ids = llvm::SmallVector<InstBlockId>(
              body_block_ids.begin(), body_block_ids.end())}});
  }

 protected:
  SharedValueStores value_stores_;
  File file_;
  // The condition used by conditional branches. It's constant, so it needs no
  // dominating evaluation.
  InstId cond_id_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_DOMINANCE_TEST_HELPERS_H_
