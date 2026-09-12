// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/dominance.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>

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
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;

class DominanceTest : public ::testing::Test {
 protected:
  DominanceTest()
      : file_(/*parse_tree=*/nullptr, CheckIRId(0),
              /*packaging_decl=*/std::nullopt, value_stores_,
              "test_file.carbon") {}

  // Runs the dominance check, and returns the error message it produced, or an
  // empty string if it succeeded.
  auto Verify() -> std::string {
    ErrorOr<Success> result = VerifyDominance(file_);
    return result.ok() ? "" : result.error().message();
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

  // Adds an instruction that refers to `inst_id` as a `MetaInstId`, which names
  // the instruction rather than its value.
  auto AddMetaUse(InstId inst_id) -> InstId {
    return AddNonConstInst(AccessMemberAction{.type_id = TypeType::TypeId,
                                              .base_id = MetaInstId(inst_id),
                                              .name_id = NameId(0)});
  }

  auto AddReturn() -> InstId { return AddInst(Return{}); }

  auto AddBranch(InstBlockId target_id) -> InstId {
    return AddInst(Branch{.target_id = LabelId(target_id)});
  }

  auto AddBranchIf(InstBlockId target_id, InstId cond_id) -> InstId {
    return AddInst(
        BranchIf{.target_id = LabelId(target_id), .cond_id = cond_id});
  }

  // Adds a generic, along with a resolved specific for it, that a function can
  // be attached to.
  auto AddGeneric() -> GenericId {
    auto decl_id =
        AddInst(FunctionDecl{.type_id = TypeType::TypeId,
                             .function_id = FunctionId(0),
                             .decl_block_id = DeclInstBlockId::None});
    auto generic_id =
        file_.generics().Add(Generic{.decl_id = decl_id,
                                     .bindings_id = InstBlockId::Empty,
                                     .self_specific_id = SpecificId::None});
    file_.specifics().GetOrAdd(generic_id, InstBlockId::Empty);
    return generic_id;
  }

  // Adds an instruction whose constant value is an `inst_value` naming
  // `target_id`. This is the form that the operand of a `splice_inst` takes.
  auto AddInstValue(InstId target_id) -> InstId {
    auto value_id = AddInst(InstValue{.type_id = TypeType::TypeId,
                                      .inst_id = MetaInstId(target_id)});
    file_.constant_values().Set(value_id,
                                ConstantId::ForConcreteConstant(value_id));

    auto use_id = AddInst(InstValue{.type_id = TypeType::TypeId,
                                    .inst_id = MetaInstId(target_id)});
    file_.constant_values().Set(use_id,
                                ConstantId::ForConcreteConstant(value_id));
    return use_id;
  }

  auto AddFunction(llvm::ArrayRef<InstBlockId> body_block_ids = {},
                   llvm::StringRef name = "F",
                   GenericId generic_id = GenericId::None) -> FunctionId {
    auto name_id = file_.identifiers().Add(name);
    return file_.functions().Add(
        {{.name_id = SemIR::NameId::ForIdentifier(name_id),
          .parent_scope_id = SemIR::NameScopeId::Package,
          .generic_id = generic_id,
          .first_param_node_id = Parse::NodeId::None,
          .last_param_node_id = Parse::NodeId::None,
          .pattern_block_id = SemIR::InstBlockId::Empty,
          .implicit_param_patterns_id = SemIR::InstBlockId::None,
          .param_patterns_id = SemIR::InstBlockId::Empty,
          .is_extern = false,
          .extern_library_id = SemIR::LibraryNameId::None,
          .non_owning_decl_id = SemIR::InstId::None,
          .first_owning_decl_id = SemIR::InstId::None},
         {.call_param_patterns_id = SemIR::InstBlockId::Empty,
          .call_params_id = SemIR::InstBlockId::Empty,
          .call_param_default_values_id = SemIR::InstBlockId::Empty,
          .call_param_ranges = SemIR::Function::CallParamIndexRanges::Empty,
          .return_type_inst_id = SemIR::TypeInstId::None,
          .return_form_inst_id = SemIR::InstId::None,
          .return_pattern_id = SemIR::InstId::None,
          .body_block_ids = llvm::SmallVector<InstBlockId>(
              body_block_ids.begin(), body_block_ids.end())}});
  }

  SharedValueStores value_stores_;
  File file_;
};

TEST_F(DominanceTest, FunctionWithNoBody) {
  AddFunction();
  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, StraightLineUseAfterEvaluation) {
  auto value_id = AddValue();
  auto use_id = AddUse(value_id);
  AddFunction({file_.inst_blocks().Add({value_id, use_id, AddReturn()})});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, StraightLineUseBeforeEvaluation) {
  auto value_id = AddValue();
  auto use_id = AddUse(value_id);
  AddFunction({file_.inst_blocks().Add({use_id, value_id, AddReturn()})});

  EXPECT_THAT(Verify(), HasSubstr("not dominated by any evaluation"));
}

TEST_F(DominanceTest, NeverEvaluatedValue) {
  // A non-constant value that is not evaluated anywhere in the body, as would
  // happen for a non-constant global or import, doesn't dominate its uses.
  auto global_id = AddValue();
  auto use_id = AddUse(global_id);
  AddFunction({file_.inst_blocks().Add({use_id, AddReturn()})});

  EXPECT_THAT(Verify(),
              HasSubstr("not dominated by any evaluation and is not constant"));
}

TEST_F(DominanceTest, ConstantIsExempt) {
  auto constant_id = AddConstant();
  auto use_id = AddUse(constant_id);
  AddFunction({file_.inst_blocks().Add({use_id, AddReturn()})});

  EXPECT_THAT(Verify(), IsEmpty());
}

// The following three tests pin the allowlists for pre-existing dominance
// violations. Each of them fails if the corresponding allowlist is removed, so
// they should be removed, and the violations diagnosed, together with it.
// See `CollectDeclInsts` and `DominanceVerifier::VerifyOperand`.

TEST_F(DominanceTest, FileScopeInstIsAllowlisted) {
  // A file-scope instruction is evaluated in `__global_init`, if at all, so it
  // doesn't dominate uses in any other function.
  auto global_id = AddValue();
  file_.set_top_inst_block_id(file_.inst_blocks().Add({global_id}));

  auto use_id = AddUse(global_id);
  AddFunction({file_.inst_blocks().Add({use_id, AddReturn()})});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, ClassBodyInstIsAllowlisted) {
  // A `let` in a class body produces a `wrapper_binding` that isn't evaluated
  // in any function, but `A.x` can name it from one. This mirrors the
  // `public_global_access` case in
  // `check/testdata/class/access/access_modifiers.carbon`:
  //
  //     class A { let x: i32 = 5; }
  //     let x: i32 = A.x;
  auto binding_id =
      AddNonConstInst(WrapperBinding{.type_id = TypeType::TypeId,
                                     .entity_name_id = EntityNameId::None,
                                     .value_id = AddConstant()});
  auto name_id = file_.identifiers().Add("A");
  file_.classes().Add(
      {{.name_id = NameId::ForIdentifier(name_id),
        .parent_scope_id = NameScopeId::Package,
        .generic_id = GenericId::None,
        .first_param_node_id = Parse::NodeId::None,
        .last_param_node_id = Parse::NodeId::None,
        .pattern_block_id = InstBlockId::Empty,
        .implicit_param_patterns_id = InstBlockId::None,
        .param_patterns_id = InstBlockId::Empty,
        .is_extern = false,
        .extern_library_id = LibraryNameId::None,
        .non_owning_decl_id = InstId::None,
        .first_owning_decl_id = InstId::None},
       {.self_type_id = TypeType::TypeId,
        .inheritance_kind = Class::Final,
        .body_block_id = file_.inst_blocks().Add({binding_id})}});

  auto use_id = AddUse(binding_id);
  AddFunction({file_.inst_blocks().Add({use_id, AddReturn()})});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, ImportRefIsAllowlisted) {
  // A non-constant import isn't evaluated in the importing file at all.
  auto import_id =
      AddNonConstInst(ImportRefLoaded{.type_id = TypeType::TypeId,
                                      .import_ir_inst_id = ImportIRInstId::None,
                                      .entity_name_id = EntityNameId::None});
  auto use_id = AddUse(import_id);
  AddFunction({file_.inst_blocks().Add({use_id, AddReturn()})});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, MetaInstIdOperandIsExempt) {
  // A `MetaInstId` names the identity of an instruction rather than its value,
  // so it needn't be dominated even though it's evaluated later.
  auto value_id = AddValue();
  auto use_id = AddMetaUse(value_id);
  AddFunction({file_.inst_blocks().Add({use_id, value_id, AddReturn()})});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, ErroneousFileIsNotChecked) {
  auto value_id = AddValue();
  auto use_id = AddUse(value_id);
  AddFunction({file_.inst_blocks().Add({use_id, value_id, AddReturn()})});
  file_.set_has_errors(true);

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, FileVerifyChecksDominance) {
  // `File::Verify` should run the dominance check as well as its other checks.
  auto value_id = AddValue();
  auto use_id = AddUse(value_id);
  AddFunction({file_.inst_blocks().Add({use_id, value_id, AddReturn()})});

  auto result = file_.Verify();
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.error().message(),
              HasSubstr("not dominated by any evaluation"));
}

// A fixture for tests over a diamond:
//
//     entry
//     /   \
//   then  else
//     \   /
//      exit
class DominanceDiamondTest : public DominanceTest {
 protected:
  DominanceDiamondTest()
      : entry_id_(file_.inst_blocks().AddPlaceholder()),
        then_id_(file_.inst_blocks().AddPlaceholder()),
        else_id_(file_.inst_blocks().AddPlaceholder()),
        exit_id_(file_.inst_blocks().AddPlaceholder()) {}

  // Fills in the diamond, prefixing each block with the given instructions.
  auto BuildDiamond(llvm::ArrayRef<InstId> entry_insts,
                    llvm::ArrayRef<InstId> then_insts,
                    llvm::ArrayRef<InstId> else_insts,
                    llvm::ArrayRef<InstId> exit_insts) -> void {
    auto cond_id = AddConstant();

    llvm::SmallVector<InstId> entry(entry_insts.begin(), entry_insts.end());
    entry.push_back(cond_id);
    entry.push_back(AddBranchIf(then_id_, cond_id));
    entry.push_back(AddBranch(else_id_));
    file_.inst_blocks().ReplacePlaceholder(entry_id_, entry);

    llvm::SmallVector<InstId> then(then_insts.begin(), then_insts.end());
    then.push_back(AddBranch(exit_id_));
    file_.inst_blocks().ReplacePlaceholder(then_id_, then);

    llvm::SmallVector<InstId> otherwise(else_insts.begin(), else_insts.end());
    otherwise.push_back(AddBranch(exit_id_));
    file_.inst_blocks().ReplacePlaceholder(else_id_, otherwise);

    llvm::SmallVector<InstId> exit(exit_insts.begin(), exit_insts.end());
    exit.push_back(AddReturn());
    file_.inst_blocks().ReplacePlaceholder(exit_id_, exit);

    AddFunction({entry_id_, then_id_, else_id_, exit_id_});
  }

  InstBlockId entry_id_;
  InstBlockId then_id_;
  InstBlockId else_id_;
  InstBlockId exit_id_;
};

TEST_F(DominanceDiamondTest, EvaluationInEntryDominatesAllBranches) {
  auto value_id = AddValue();
  BuildDiamond(/*entry_insts=*/{value_id}, /*then_insts=*/{AddUse(value_id)},
               /*else_insts=*/{AddUse(value_id)},
               /*exit_insts=*/{AddUse(value_id)});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceDiamondTest, EvaluationInOneBranchDoesNotDominateTheOther) {
  auto value_id = AddValue();
  BuildDiamond(/*entry_insts=*/{}, /*then_insts=*/{value_id},
               /*else_insts=*/{AddUse(value_id)}, /*exit_insts=*/{});

  EXPECT_THAT(Verify(), HasSubstr("not dominated by any evaluation"));
}

TEST_F(DominanceDiamondTest, EvaluationInOneBranchDoesNotDominateJoin) {
  auto value_id = AddValue();
  BuildDiamond(/*entry_insts=*/{}, /*then_insts=*/{value_id},
               /*else_insts=*/{}, /*exit_insts=*/{AddUse(value_id)});

  EXPECT_THAT(Verify(), HasSubstr("not dominated by any evaluation"));
}

TEST_F(DominanceDiamondTest, EvaluationInBothBranchesDoesNotDominateJoin) {
  // The same instruction can be evaluated in more than one block; a use is
  // dominated if it's dominated by any of the evaluations. Here neither
  // evaluation dominates the join, so this is still an error.
  auto value_id = AddValue();
  BuildDiamond(/*entry_insts=*/{}, /*then_insts=*/{value_id},
               /*else_insts=*/{value_id}, /*exit_insts=*/{AddUse(value_id)});

  EXPECT_THAT(Verify(), HasSubstr("not dominated by any evaluation"));
}

TEST_F(DominanceDiamondTest, MultipleEvaluationsEachDominateTheirOwnUse) {
  auto value_id = AddValue();
  BuildDiamond(/*entry_insts=*/{},
               /*then_insts=*/{value_id, AddUse(value_id)},
               /*else_insts=*/{value_id, AddUse(value_id)},
               /*exit_insts=*/{});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, UnreachableBlock) {
  auto entry_id = file_.inst_blocks().AddPlaceholder();
  auto unreachable_id = file_.inst_blocks().Add({AddReturn()});
  file_.inst_blocks().ReplacePlaceholder(entry_id, {AddReturn()});
  AddFunction({entry_id, unreachable_id});

  EXPECT_THAT(Verify(), HasSubstr("is unreachable from entry block"));
}

TEST_F(DominanceTest, BranchOutsideFunctionBody) {
  auto outside_id = file_.inst_blocks().Add({AddReturn()});
  auto entry_id = file_.inst_blocks().Add({AddBranch(outside_id)});
  AddFunction({entry_id});

  EXPECT_THAT(Verify(), HasSubstr("which is not in function body"));
}

TEST_F(DominanceTest, LongChainOfBlocks) {
  // A function body long enough that walking it recursively would overflow the
  // stack. Each block uses a value evaluated in the block before it, so the
  // dominator tree is a single chain.
  constexpr int NumBlocks = 100'000;

  llvm::SmallVector<InstBlockId> block_ids;
  block_ids.reserve(NumBlocks);
  for (int i = 0; i != NumBlocks; ++i) {
    block_ids.push_back(file_.inst_blocks().AddPlaceholder());
  }

  auto value_id = AddValue();
  for (int i = 0; i != NumBlocks; ++i) {
    llvm::SmallVector<InstId> insts;
    if (i == 0) {
      insts.push_back(value_id);
    } else {
      insts.push_back(AddUse(value_id));
    }
    insts.push_back(i + 1 == NumBlocks ? AddReturn()
                                       : AddBranch(block_ids[i + 1]));
    file_.inst_blocks().ReplacePlaceholder(block_ids[i], insts);
  }
  AddFunction(block_ids);

  EXPECT_THAT(Verify(), IsEmpty());
}

// A loop:
//
//   entry -> header -> body -> header
//                   -> exit
class DominanceLoopTest : public DominanceTest {
 protected:
  // Fills in the loop, prefixing the header and body blocks with the given
  // instructions.
  auto BuildLoop(llvm::ArrayRef<InstId> header_insts,
                 llvm::ArrayRef<InstId> body_insts) -> void {
    auto entry_id = file_.inst_blocks().AddPlaceholder();
    auto header_id = file_.inst_blocks().AddPlaceholder();
    auto body_id = file_.inst_blocks().AddPlaceholder();
    auto exit_id = file_.inst_blocks().AddPlaceholder();

    auto cond_id = AddConstant();
    file_.inst_blocks().ReplacePlaceholder(entry_id,
                                           {cond_id, AddBranch(header_id)});

    llvm::SmallVector<InstId> header(header_insts);
    header.push_back(AddBranchIf(body_id, cond_id));
    header.push_back(AddBranch(exit_id));
    file_.inst_blocks().ReplacePlaceholder(header_id, header);

    llvm::SmallVector<InstId> body(body_insts);
    body.push_back(AddBranch(header_id));
    file_.inst_blocks().ReplacePlaceholder(body_id, body);

    file_.inst_blocks().ReplacePlaceholder(exit_id, {AddReturn()});

    AddFunction({entry_id, header_id, body_id, exit_id});
  }
};

TEST_F(DominanceLoopTest, HeaderEvaluationDominatesBody) {
  auto value_id = AddValue();
  BuildLoop(/*header_insts=*/{value_id}, /*body_insts=*/{AddUse(value_id)});

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceLoopTest, BodyEvaluationDoesNotDominateHeader) {
  // The back edge means the body is executed before the header on some paths,
  // but not on the path that first reaches the header.
  auto value_id = AddValue();
  BuildLoop(/*header_insts=*/{AddUse(value_id)}, /*body_insts=*/{value_id});

  EXPECT_THAT(Verify(), HasSubstr("not dominated by any evaluation"));
}

TEST_F(DominanceTest, SpliceBlockEvaluatesItsContents) {
  auto generic_id = AddGeneric();

  auto value_id = AddValue();

  // A spliced block that evaluates a use of `value_id` and produces it.
  auto inner_id = AddUse(value_id);
  auto splice_block_id = AddInst(SpliceBlock{
      .type_id = TypeType::TypeId,
      .block_id = AbsoluteInstBlockId(file_.inst_blocks().Add({inner_id})),
      .result_id = inner_id});

  auto action_id = AddInstValue(splice_block_id);
  auto splice_id =
      AddInst(SpliceInst{.type_id = TypeType::TypeId, .inst_id = action_id});

  // The use after the splice sees the instructions the splice evaluated.
  auto late_use_id = AddUse(inner_id);
  AddFunction({file_.inst_blocks().Add(
                  {value_id, action_id, splice_id, late_use_id, AddReturn()})},
              "GenericFn", generic_id);

  EXPECT_THAT(Verify(), IsEmpty());
}

TEST_F(DominanceTest, SpliceBlockUsingLaterValue) {
  auto generic_id = AddGeneric();

  // A spliced block that uses a value evaluated after the splice.
  auto value_id = AddValue();
  auto inner_id = AddUse(value_id);
  auto splice_block_id = AddInst(SpliceBlock{
      .type_id = TypeType::TypeId,
      .block_id = AbsoluteInstBlockId(file_.inst_blocks().Add({inner_id})),
      .result_id = inner_id});

  auto action_id = AddInstValue(splice_block_id);
  auto splice_id =
      AddInst(SpliceInst{.type_id = TypeType::TypeId, .inst_id = action_id});

  AddFunction(
      {file_.inst_blocks().Add({action_id, splice_id, value_id, AddReturn()})},
      "GenericFn", generic_id);

  EXPECT_THAT(Verify(), HasSubstr("not dominated by any evaluation"));
}

}  // namespace
}  // namespace Carbon::SemIR
