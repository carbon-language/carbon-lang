// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/check/declaration_name_stack.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Check {

Context::Context(const Lex::TokenizedBuffer& tokens, DiagnosticEmitter& emitter,
                 const Parse::Tree& parse_tree, SemIR::File& sem_ir,
                 llvm::raw_ostream* vlog_stream)
    : tokens_(&tokens),
      emitter_(&emitter),
      parse_tree_(&parse_tree),
      sem_ir_(&sem_ir),
      vlog_stream_(vlog_stream),
      node_stack_(parse_tree, vlog_stream),
      inst_block_stack_("inst_block_stack_", sem_ir, vlog_stream),
      params_or_args_stack_("params_or_args_stack_", sem_ir, vlog_stream),
      args_type_info_stack_("args_type_info_stack_", sem_ir, vlog_stream),
      declaration_name_stack_(this) {
  // Inserts the "Error" and "Type" types as "used types" so that
  // canonicalization can skip them. We don't emit either for lowering.
  canonical_types_.insert({SemIR::InstId::BuiltinError, SemIR::TypeId::Error});
  canonical_types_.insert(
      {SemIR::InstId::BuiltinTypeType, SemIR::TypeId::TypeType});
}

auto Context::TODO(Parse::Node parse_node, std::string label) -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "Semantics TODO: `{0}`.",
                    std::string);
  emitter_->Emit(parse_node, SemanticsTodo, std::move(label));
  return false;
}

auto Context::VerifyOnFinish() -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  CARBON_CHECK(name_lookup_.empty()) << name_lookup_.size();
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
  CARBON_CHECK(inst_block_stack_.empty()) << inst_block_stack_.size();
  CARBON_CHECK(params_or_args_stack_.empty()) << params_or_args_stack_.size();
}

auto Context::AddInst(SemIR::Inst inst) -> SemIR::InstId {
  auto inst_id = inst_block_stack_.AddInst(inst);
  CARBON_VLOG() << "AddInst: " << inst << "\n";
  return inst_id;
}

auto Context::AddInstAndPush(Parse::Node parse_node, SemIR::Inst inst) -> void {
  auto inst_id = AddInst(inst);
  node_stack_.Push(parse_node, inst_id);
}

auto Context::DiagnoseDuplicateName(Parse::Node parse_node,
                                    SemIR::InstId prev_def_id) -> void {
  CARBON_DIAGNOSTIC(NameDeclarationDuplicate, Error,
                    "Duplicate name being declared in the same scope.");
  CARBON_DIAGNOSTIC(NameDeclarationPrevious, Note,
                    "Name is previously declared here.");
  auto prev_def = insts().Get(prev_def_id);
  emitter_->Build(parse_node, NameDeclarationDuplicate)
      .Note(prev_def.parse_node(), NameDeclarationPrevious)
      .Emit();
}

auto Context::DiagnoseNameNotFound(Parse::Node parse_node, IdentifierId name_id)
    -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name `{0}` not found.",
                    llvm::StringRef);
  emitter_->Emit(parse_node, NameNotFound, identifiers().Get(name_id));
}

auto Context::NoteIncompleteClass(SemIR::ClassId class_id,
                                  DiagnosticBuilder& builder) -> void {
  CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                    "Class was forward declared here.");
  CARBON_DIAGNOSTIC(ClassIncompleteWithinDefinition, Note,
                    "Class is incomplete within its definition.");
  const auto& class_info = classes().Get(class_id);
  CARBON_CHECK(!class_info.is_defined()) << "Class is not incomplete";
  if (class_info.definition_id.is_valid()) {
    builder.Note(insts().Get(class_info.definition_id).parse_node(),
                 ClassIncompleteWithinDefinition);
  } else {
    builder.Note(insts().Get(class_info.declaration_id).parse_node(),
                 ClassForwardDeclaredHere);
  }
}

auto Context::AddNameToLookup(Parse::Node name_node, IdentifierId name_id,
                              SemIR::InstId target_id) -> void {
  if (current_scope().names.insert(name_id).second) {
    name_lookup_[name_id].push_back(target_id);
  } else {
    DiagnoseDuplicateName(name_node, name_lookup_[name_id].back());
  }
}

auto Context::LookupName(Parse::Node parse_node, IdentifierId name_id,
                         SemIR::NameScopeId scope_id, bool print_diagnostics)
    -> SemIR::InstId {
  if (scope_id == SemIR::NameScopeId::Invalid) {
    auto it = name_lookup_.find(name_id);
    if (it == name_lookup_.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemIR::InstId::BuiltinError;
    }
    CARBON_CHECK(!it->second.empty())
        << "Should have been erased: " << identifiers().Get(name_id);

    // TODO: Check for ambiguous lookups.
    return it->second.back();
  } else {
    const auto& scope = name_scopes().Get(scope_id);
    auto it = scope.find(name_id);
    if (it == scope.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemIR::InstId::BuiltinError;
    }

    return it->second;
  }
}

auto Context::PushScope(SemIR::InstId scope_inst_id,
                        SemIR::NameScopeId scope_id) -> void {
  scope_stack_.push_back(
      {.scope_inst_id = scope_inst_id, .scope_id = scope_id});
}

auto Context::PopScope() -> void {
  auto scope = scope_stack_.pop_back_val();
  for (const auto& str_id : scope.names) {
    auto it = name_lookup_.find(str_id);
    if (it->second.size() == 1) {
      // Erase names that no longer resolve.
      name_lookup_.erase(it);
    } else {
      it->second.pop_back();
    }
  }
}

auto Context::FollowNameReferences(SemIR::InstId inst_id) -> SemIR::InstId {
  while (auto name_ref = insts().Get(inst_id).TryAs<SemIR::NameReference>()) {
    inst_id = name_ref->value_id;
  }
  return inst_id;
}

auto Context::GetConstantValue(SemIR::InstId inst_id) -> SemIR::InstId {
  // TODO: The constant value of an instruction should be computed as we build
  // the instruction, or at least cached once computed.
  while (true) {
    auto inst = insts().Get(inst_id);
    switch (inst.kind()) {
      case SemIR::NameReference::Kind:
        inst_id = inst.As<SemIR::NameReference>().value_id;
        break;

      case SemIR::BindName::Kind:
        inst_id = inst.As<SemIR::BindName>().value_id;
        break;

      case SemIR::Field::Kind:
      case SemIR::FunctionDeclaration::Kind:
        return inst_id;

      default:
        // TODO: Handle the remaining cases.
        return SemIR::InstId::Invalid;
    }
  }
}

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::Node parse_node, Args... args)
    -> SemIR::InstBlockId {
  if (!context.inst_block_stack().is_current_block_reachable()) {
    return SemIR::InstBlockId::Unreachable;
  }
  auto block_id = context.inst_blocks().AddDefaultValue();
  context.AddInst(BranchNode{parse_node, block_id, args...});
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::Node parse_node)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Branch>(*this, parse_node);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::Node parse_node,
                                                SemIR::InstId arg_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchWithArg>(*this, parse_node,
                                                              arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::Node parse_node,
                                           SemIR::InstId cond_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchIf>(*this, parse_node,
                                                         cond_id);
}

auto Context::AddConvergenceBlockAndPush(Parse::Node parse_node, int num_blocks)
    -> void {
  CARBON_CHECK(num_blocks >= 2) << "no convergence";

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst(SemIR::Branch{parse_node, new_block_id});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::Node parse_node, std::initializer_list<SemIR::InstId> block_args)
    -> SemIR::InstId {
  CARBON_CHECK(block_args.size() >= 2) << "no convergence";

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst(SemIR::BranchWithArg{parse_node, new_block_id, arg_id});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id = insts().Get(*block_args.begin()).type_id();
  return AddInst(SemIR::BlockArg{parse_node, result_type_id, new_block_id});
}

// Add the current code block to the enclosing function.
auto Context::AddCurrentCodeBlockToFunction() -> void {
  CARBON_CHECK(!inst_block_stack().empty()) << "no current code block";
  CARBON_CHECK(!return_scope_stack().empty()) << "no current function";

  if (!inst_block_stack().is_current_block_reachable()) {
    // Don't include unreachable blocks in the function.
    return;
  }

  auto function_id =
      insts()
          .GetAs<SemIR::FunctionDeclaration>(return_scope_stack().back())
          .function_id;
  functions()
      .Get(function_id)
      .body_block_ids.push_back(inst_block_stack().PeekOrAdd());
}

auto Context::is_current_position_reachable() -> bool {
  if (!inst_block_stack().is_current_block_reachable()) {
    return false;
  }

  // Our current position is at the end of a reachable block. That position is
  // reachable unless the previous instruction is a terminator instruction.
  auto block_contents = inst_block_stack().PeekCurrentBlockContents();
  if (block_contents.empty()) {
    return true;
  }
  const auto& last_inst = insts().Get(block_contents.back());
  return last_inst.kind().terminator_kind() !=
         SemIR::TerminatorKind::Terminator;
}

auto Context::ParamOrArgStart() -> void { params_or_args_stack_.Push(); }

auto Context::ParamOrArgComma() -> void {
  ParamOrArgSave(node_stack_.PopExpression());
}

auto Context::ParamOrArgEndNoPop(Parse::NodeKind start_kind) -> void {
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) != start_kind) {
    ParamOrArgSave(node_stack_.PopExpression());
  }
}

auto Context::ParamOrArgPop() -> SemIR::InstBlockId {
  return params_or_args_stack_.Pop();
}

auto Context::ParamOrArgEnd(Parse::NodeKind start_kind) -> SemIR::InstBlockId {
  ParamOrArgEndNoPop(start_kind);
  return ParamOrArgPop();
}

namespace {
// Worklist-based type completion mechanism.
//
// When attempting to complete a type, we may find other types that also need to
// be completed: types nested within that type, and the value representation of
// the type. In order to complete a type without recursing arbitrarily deeply,
// we use a worklist of tasks:
//
// - An `AddNestedIncompleteTypes` step adds a task for all incomplete types
//   nested within a type to the work list.
// - A `BuildValueRepresentation` step computes the value representation for a
//   type, once all of its nested types are complete, and marks the type as
//   complete.
class TypeCompleter {
 public:
  TypeCompleter(
      Context& context,
      std::optional<llvm::function_ref<auto()->Context::DiagnosticBuilder>>
          diagnoser)
      : context_(context), diagnoser_(diagnoser) {}

  // Attempts to complete the given type. Returns true if it is now complete,
  // false if it could not be completed.
  auto Complete(SemIR::TypeId type_id) -> bool {
    Push(type_id);
    while (!work_list_.empty()) {
      if (!ProcessStep()) {
        return false;
      }
    }
    return true;
  }

 private:
  // Adds `type_id` to the work list, if it's not already complete.
  auto Push(SemIR::TypeId type_id) -> void {
    if (!context_.sem_ir().IsTypeComplete(type_id)) {
      work_list_.push_back({type_id, Phase::AddNestedIncompleteTypes});
    }
  }

  // Runs the next step.
  auto ProcessStep() -> bool {
    auto [type_id, phase] = work_list_.back();

    // We might have enqueued the same type more than once. Just skip the
    // type if it's already complete.
    if (context_.sem_ir().IsTypeComplete(type_id)) {
      work_list_.pop_back();
      return true;
    }

    auto inst_id = context_.sem_ir().GetTypeAllowBuiltinTypes(type_id);
    auto inst = context_.insts().Get(inst_id);

    auto old_work_list_size = work_list_.size();

    switch (phase) {
      case Phase::AddNestedIncompleteTypes:
        if (!AddNestedIncompleteTypes(inst)) {
          return false;
        }
        CARBON_CHECK(work_list_.size() >= old_work_list_size)
            << "AddNestedIncompleteTypes should not remove work items";
        work_list_[old_work_list_size - 1].phase =
            Phase::BuildValueRepresentation;
        break;

      case Phase::BuildValueRepresentation: {
        auto value_rep = BuildValueRepresentation(type_id, inst);
        context_.sem_ir().CompleteType(type_id, value_rep);
        CARBON_CHECK(old_work_list_size == work_list_.size())
            << "BuildValueRepresentation should not change work items";
        work_list_.pop_back();

        // Also complete the value representation type, if necessary. This
        // should never fail: the value representation shouldn't require any
        // additional nested types to be complete.
        if (!context_.sem_ir().IsTypeComplete(value_rep.type_id)) {
          work_list_.push_back(
              {value_rep.type_id, Phase::BuildValueRepresentation});
        }
        // For a pointer representation, the pointee also needs to be complete.
        if (value_rep.kind == SemIR::ValueRepresentation::Pointer) {
          auto pointee_type_id =
              context_.sem_ir().GetPointeeType(value_rep.type_id);
          if (!context_.sem_ir().IsTypeComplete(pointee_type_id)) {
            work_list_.push_back(
                {pointee_type_id, Phase::BuildValueRepresentation});
          }
        }
        break;
      }
    }

    return true;
  }

  // Adds any types nested within `type_inst` that need to be complete for
  // `type_inst` to be complete to our work list.
  auto AddNestedIncompleteTypes(SemIR::Inst type_inst) -> bool {
    switch (type_inst.kind()) {
      case SemIR::ArrayType::Kind:
        Push(type_inst.As<SemIR::ArrayType>().element_type_id);
        break;

      case SemIR::StructType::Kind:
        for (auto field_id : context_.inst_blocks().Get(
                 type_inst.As<SemIR::StructType>().fields_id)) {
          Push(context_.insts()
                   .GetAs<SemIR::StructTypeField>(field_id)
                   .field_type_id);
        }
        break;

      case SemIR::TupleType::Kind:
        for (auto element_type_id : context_.type_blocks().Get(
                 type_inst.As<SemIR::TupleType>().elements_id)) {
          Push(element_type_id);
        }
        break;

      case SemIR::ClassType::Kind: {
        auto class_type = type_inst.As<SemIR::ClassType>();
        auto& class_info = context_.classes().Get(class_type.class_id);
        if (!class_info.is_defined()) {
          if (diagnoser_) {
            auto builder = (*diagnoser_)();
            context_.NoteIncompleteClass(class_type.class_id, builder);
            builder.Emit();
          }
          return false;
        }
        Push(class_info.object_representation_id);
        break;
      }

      case SemIR::ConstType::Kind:
        Push(type_inst.As<SemIR::ConstType>().inner_id);
        break;

      default:
        break;
    }

    return true;
  }

  // Makes an empty value representation, which is used for types that have no
  // state, such as empty structs and tuples.
  auto MakeEmptyRepresentation(Parse::Node parse_node) const
      -> SemIR::ValueRepresentation {
    return {.kind = SemIR::ValueRepresentation::None,
            .type_id = context_.CanonicalizeTupleType(parse_node, {})};
  }

  // Makes a value representation that uses pass-by-copy, copying the given
  // type.
  auto MakeCopyRepresentation(
      SemIR::TypeId rep_id,
      SemIR::ValueRepresentation::AggregateKind aggregate_kind =
          SemIR::ValueRepresentation::NotAggregate) const
      -> SemIR::ValueRepresentation {
    return {.kind = SemIR::ValueRepresentation::Copy,
            .aggregate_kind = aggregate_kind,
            .type_id = rep_id};
  }

  // Makes a value representation that uses pass-by-address with the given
  // pointee type.
  auto MakePointerRepresentation(
      Parse::Node parse_node, SemIR::TypeId pointee_id,
      SemIR::ValueRepresentation::AggregateKind aggregate_kind =
          SemIR::ValueRepresentation::NotAggregate) const
      -> SemIR::ValueRepresentation {
    // TODO: Should we add `const` qualification to `pointee_id`?
    return {.kind = SemIR::ValueRepresentation::Pointer,
            .aggregate_kind = aggregate_kind,
            .type_id = context_.GetPointerType(parse_node, pointee_id)};
  }

  // Gets the value representation of a nested type, which should already be
  // complete.
  auto GetNestedValueRepresentation(SemIR::TypeId nested_type_id) const {
    CARBON_CHECK(context_.sem_ir().IsTypeComplete(nested_type_id))
        << "Nested type should already be complete";
    auto value_rep = context_.sem_ir().GetValueRepresentation(nested_type_id);
    CARBON_CHECK(value_rep.kind != SemIR::ValueRepresentation::Unknown)
        << "Complete type should have a value representation";
    return value_rep;
  };

  auto BuildCrossReferenceValueRepresentation(SemIR::TypeId type_id,
                                              SemIR::CrossReference xref) const
      -> SemIR::ValueRepresentation {
    auto xref_inst = context_.sem_ir()
                         .GetCrossReferenceIR(xref.ir_id)
                         .insts()
                         .Get(xref.inst_id);

    // The canonical description of a type should only have cross-references
    // for entities owned by another File, such as builtins, which are owned
    // by the prelude, and named entities like classes and interfaces, which
    // we don't support yet.
    CARBON_CHECK(xref_inst.kind() == SemIR::Builtin::Kind)
        << "TODO: Handle other kinds of inst cross-references";

    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (xref_inst.As<SemIR::Builtin>().builtin_kind) {
      case SemIR::BuiltinKind::TypeType:
      case SemIR::BuiltinKind::Error:
      case SemIR::BuiltinKind::Invalid:
      case SemIR::BuiltinKind::BoolType:
      case SemIR::BuiltinKind::IntegerType:
      case SemIR::BuiltinKind::FloatingPointType:
      case SemIR::BuiltinKind::NamespaceType:
      case SemIR::BuiltinKind::FunctionType:
      case SemIR::BuiltinKind::BoundMethodType:
        return MakeCopyRepresentation(type_id);

      case SemIR::BuiltinKind::StringType:
        // TODO: Decide on string value semantics. This should probably be a
        // custom value representation carrying a pointer and size or
        // similar.
        return MakePointerRepresentation(Parse::Node::Invalid, type_id);
    }
    llvm_unreachable("All builtin kinds were handled above");
  }

  auto BuildStructOrTupleValueRepresentation(Parse::Node parse_node,
                                             std::size_t num_elements,
                                             SemIR::TypeId elementwise_rep,
                                             bool same_as_object_rep) const
      -> SemIR::ValueRepresentation {
    SemIR::ValueRepresentation::AggregateKind aggregate_kind =
        same_as_object_rep ? SemIR::ValueRepresentation::ValueAndObjectAggregate
                           : SemIR::ValueRepresentation::ValueAggregate;

    if (num_elements == 1) {
      // The value representation for a struct or tuple with a single element
      // is a struct or tuple containing the value representation of the
      // element.
      // TODO: Consider doing the same whenever `elementwise_rep` is
      // sufficiently small.
      return MakeCopyRepresentation(elementwise_rep, aggregate_kind);
    }
    // For a struct or tuple with multiple fields, we use a pointer
    // to the elementwise value representation.
    return MakePointerRepresentation(parse_node, elementwise_rep,
                                     aggregate_kind);
  }

  auto BuildStructTypeValueRepresentation(SemIR::TypeId type_id,
                                          SemIR::StructType struct_type) const
      -> SemIR::ValueRepresentation {
    // TODO: Share more code with tuples.
    auto fields = context_.inst_blocks().Get(struct_type.fields_id);
    if (fields.empty()) {
      return MakeEmptyRepresentation(struct_type.parse_node);
    }

    // Find the value representation for each field, and construct a struct
    // of value representations.
    llvm::SmallVector<SemIR::InstId> value_rep_fields;
    value_rep_fields.reserve(fields.size());
    bool same_as_object_rep = true;
    for (auto field_id : fields) {
      auto field = context_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto field_value_rep = GetNestedValueRepresentation(field.field_type_id);
      if (field_value_rep.type_id != field.field_type_id) {
        same_as_object_rep = false;
        field.field_type_id = field_value_rep.type_id;
        field_id = context_.AddInst(field);
      }
      value_rep_fields.push_back(field_id);
    }

    auto value_rep = same_as_object_rep
                         ? type_id
                         : context_.CanonicalizeStructType(
                               struct_type.parse_node,
                               context_.inst_blocks().Add(value_rep_fields));
    return BuildStructOrTupleValueRepresentation(
        struct_type.parse_node, fields.size(), value_rep, same_as_object_rep);
  }

  auto BuildTupleTypeValueRepresentation(SemIR::TypeId type_id,
                                         SemIR::TupleType tuple_type) const
      -> SemIR::ValueRepresentation {
    // TODO: Share more code with structs.
    auto elements = context_.type_blocks().Get(tuple_type.elements_id);
    if (elements.empty()) {
      return MakeEmptyRepresentation(tuple_type.parse_node);
    }

    // Find the value representation for each element, and construct a tuple
    // of value representations.
    llvm::SmallVector<SemIR::TypeId> value_rep_elements;
    value_rep_elements.reserve(elements.size());
    bool same_as_object_rep = true;
    for (auto element_type_id : elements) {
      auto element_value_rep = GetNestedValueRepresentation(element_type_id);
      if (element_value_rep.type_id != element_type_id) {
        same_as_object_rep = false;
      }
      value_rep_elements.push_back(element_value_rep.type_id);
    }

    auto value_rep = same_as_object_rep
                         ? type_id
                         : context_.CanonicalizeTupleType(tuple_type.parse_node,
                                                          value_rep_elements);
    return BuildStructOrTupleValueRepresentation(
        tuple_type.parse_node, elements.size(), value_rep, same_as_object_rep);
  }

  // Builds and returns the value representation for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildValueRepresentation(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::ValueRepresentation {
    // TODO: This can emit new SemIR instructions. Consider emitting them into a
    // dedicated file-scope instruction block where possible, or somewhere else
    // that better reflects the definition of the type, rather than wherever the
    // type happens to first be required to be complete.

    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (inst.kind()) {
      case SemIR::AddressOf::Kind:
      case SemIR::ArrayIndex::Kind:
      case SemIR::ArrayInit::Kind:
      case SemIR::Assign::Kind:
      case SemIR::BinaryOperatorAdd::Kind:
      case SemIR::BindName::Kind:
      case SemIR::BindValue::Kind:
      case SemIR::BlockArg::Kind:
      case SemIR::BoolLiteral::Kind:
      case SemIR::BoundMethod::Kind:
      case SemIR::Branch::Kind:
      case SemIR::BranchIf::Kind:
      case SemIR::BranchWithArg::Kind:
      case SemIR::Call::Kind:
      case SemIR::ClassDeclaration::Kind:
      case SemIR::ClassFieldAccess::Kind:
      case SemIR::ClassInit::Kind:
      case SemIR::Dereference::Kind:
      case SemIR::Field::Kind:
      case SemIR::FunctionDeclaration::Kind:
      case SemIR::InitializeFrom::Kind:
      case SemIR::IntegerLiteral::Kind:
      case SemIR::NameReference::Kind:
      case SemIR::Namespace::Kind:
      case SemIR::NoOp::Kind:
      case SemIR::Parameter::Kind:
      case SemIR::RealLiteral::Kind:
      case SemIR::Return::Kind:
      case SemIR::ReturnExpression::Kind:
      case SemIR::SelfParameter::Kind:
      case SemIR::SpliceBlock::Kind:
      case SemIR::StringLiteral::Kind:
      case SemIR::StructAccess::Kind:
      case SemIR::StructTypeField::Kind:
      case SemIR::StructLiteral::Kind:
      case SemIR::StructInit::Kind:
      case SemIR::StructValue::Kind:
      case SemIR::Temporary::Kind:
      case SemIR::TemporaryStorage::Kind:
      case SemIR::TupleAccess::Kind:
      case SemIR::TupleIndex::Kind:
      case SemIR::TupleLiteral::Kind:
      case SemIR::TupleInit::Kind:
      case SemIR::TupleValue::Kind:
      case SemIR::UnaryOperatorNot::Kind:
      case SemIR::ValueAsReference::Kind:
      case SemIR::ValueOfInitializer::Kind:
      case SemIR::VarStorage::Kind:
        CARBON_FATAL() << "Type refers to non-type inst " << inst;

      case SemIR::CrossReference::Kind:
        return BuildCrossReferenceValueRepresentation(
            type_id, inst.As<SemIR::CrossReference>());

      case SemIR::ArrayType::Kind: {
        // For arrays, it's convenient to always use a pointer representation,
        // even when the array has zero or one element, in order to support
        // indexing.
        return MakePointerRepresentation(
            inst.parse_node(), type_id,
            SemIR::ValueRepresentation::ObjectAggregate);
      }

      case SemIR::StructType::Kind:
        return BuildStructTypeValueRepresentation(type_id,
                                                  inst.As<SemIR::StructType>());

      case SemIR::TupleType::Kind:
        return BuildTupleTypeValueRepresentation(type_id,
                                                 inst.As<SemIR::TupleType>());

      case SemIR::ClassType::Kind:
        // The value representation for a class is a pointer to the object
        // representation.
        // TODO: Support customized value representations for classes.
        // TODO: Pick a better value representation when possible.
        return MakePointerRepresentation(
            inst.parse_node(),
            context_.classes()
                .Get(inst.As<SemIR::ClassType>().class_id)
                .object_representation_id,
            SemIR::ValueRepresentation::ObjectAggregate);

      case SemIR::Builtin::Kind:
        CARBON_FATAL() << "Builtins should be named as cross-references";

      case SemIR::PointerType::Kind:
      case SemIR::UnboundFieldType::Kind:
        return MakeCopyRepresentation(type_id);

      case SemIR::ConstType::Kind:
        // The value representation of `const T` is the same as that of `T`.
        // Objects are not modifiable through their value representations.
        return GetNestedValueRepresentation(
            inst.As<SemIR::ConstType>().inner_id);
    }
  }

  enum class Phase : int8_t {
    // The next step is to add nested types to the list of types to complete.
    AddNestedIncompleteTypes,
    // The next step is to build the value representation for the type.
    BuildValueRepresentation,
  };

  struct WorkItem {
    SemIR::TypeId type_id;
    Phase phase;
  };

  Context& context_;
  llvm::SmallVector<WorkItem> work_list_;
  std::optional<llvm::function_ref<auto()->Context::DiagnosticBuilder>>
      diagnoser_;
};
}  // namespace

auto Context::TryToCompleteType(
    SemIR::TypeId type_id,
    std::optional<llvm::function_ref<auto()->DiagnosticBuilder>> diagnoser)
    -> bool {
  return TypeCompleter(*this, diagnoser).Complete(type_id);
}

auto Context::CanonicalizeTypeImpl(
    SemIR::InstKind kind,
    llvm::function_ref<void(llvm::FoldingSetNodeID& canonical_id)> profile_type,
    llvm::function_ref<SemIR::InstId()> make_inst) -> SemIR::TypeId {
  llvm::FoldingSetNodeID canonical_id;
  kind.Profile(canonical_id);
  profile_type(canonical_id);

  void* insert_pos;
  auto* node =
      canonical_type_nodes_.FindNodeOrInsertPos(canonical_id, insert_pos);
  if (node != nullptr) {
    return node->type_id();
  }

  auto inst_id = make_inst();
  auto type_id = types().Add({.inst_id = inst_id});
  CARBON_CHECK(canonical_types_.insert({inst_id, type_id}).second);
  type_node_storage_.push_back(
      std::make_unique<TypeNode>(canonical_id, type_id));

  // In a debug build, check that our insertion position is still valid. It
  // could have been invalidated by a misbehaving `make_inst`.
  CARBON_DCHECK([&] {
    void* check_insert_pos;
    auto* check_node = canonical_type_nodes_.FindNodeOrInsertPos(
        canonical_id, check_insert_pos);
    return !check_node && insert_pos == check_insert_pos;
  }()) << "Type was created recursively during canonicalization";

  canonical_type_nodes_.InsertNode(type_node_storage_.back().get(), insert_pos);
  return type_id;
}

// Compute a fingerprint for a tuple type, for use as a key in a folding set.
static auto ProfileTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids,
                             llvm::FoldingSetNodeID& canonical_id) -> void {
  for (auto type_id : type_ids) {
    canonical_id.AddInteger(type_id.index);
  }
}

// Compute a fingerprint for a type, for use as a key in a folding set.
static auto ProfileType(Context& semantics_context, SemIR::Inst inst,
                        llvm::FoldingSetNodeID& canonical_id) -> void {
  switch (inst.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = inst.As<SemIR::ArrayType>();
      canonical_id.AddInteger(
          semantics_context.sem_ir().GetArrayBoundValue(array_type.bound_id));
      canonical_id.AddInteger(array_type.element_type_id.index);
      break;
    }
    case SemIR::Builtin::Kind:
      canonical_id.AddInteger(inst.As<SemIR::Builtin>().builtin_kind.AsInt());
      break;
    case SemIR::ClassType::Kind:
      canonical_id.AddInteger(inst.As<SemIR::ClassType>().class_id.index);
      break;
    case SemIR::CrossReference::Kind: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto xref = inst.As<SemIR::CrossReference>();
      canonical_id.AddInteger(xref.ir_id.index);
      canonical_id.AddInteger(xref.inst_id.index);
      break;
    }
    case SemIR::ConstType::Kind:
      canonical_id.AddInteger(
          semantics_context
              .GetUnqualifiedType(inst.As<SemIR::ConstType>().inner_id)
              .index);
      break;
    case SemIR::PointerType::Kind:
      canonical_id.AddInteger(inst.As<SemIR::PointerType>().pointee_id.index);
      break;
    case SemIR::StructType::Kind: {
      auto fields = semantics_context.inst_blocks().Get(
          inst.As<SemIR::StructType>().fields_id);
      for (const auto& field_id : fields) {
        auto field =
            semantics_context.insts().GetAs<SemIR::StructTypeField>(field_id);
        canonical_id.AddInteger(field.name_id.index);
        canonical_id.AddInteger(field.field_type_id.index);
      }
      break;
    }
    case SemIR::TupleType::Kind:
      ProfileTupleType(semantics_context.type_blocks().Get(
                           inst.As<SemIR::TupleType>().elements_id),
                       canonical_id);
      break;
    case SemIR::UnboundFieldType::Kind: {
      auto unbound_field_type = inst.As<SemIR::UnboundFieldType>();
      canonical_id.AddInteger(unbound_field_type.class_type_id.index);
      canonical_id.AddInteger(unbound_field_type.field_type_id.index);
      break;
    }
    default:
      CARBON_FATAL() << "Unexpected type inst " << inst;
  }
}

auto Context::CanonicalizeTypeAndAddInstIfNew(SemIR::Inst inst)
    -> SemIR::TypeId {
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileType(*this, inst, canonical_id);
  };
  auto make_inst = [&] { return AddInst(inst); };
  return CanonicalizeTypeImpl(inst.kind(), profile_node, make_inst);
}

auto Context::CanonicalizeType(SemIR::InstId inst_id) -> SemIR::TypeId {
  inst_id = FollowNameReferences(inst_id);

  auto it = canonical_types_.find(inst_id);
  if (it != canonical_types_.end()) {
    return it->second;
  }

  auto inst = insts().Get(inst_id);
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileType(*this, inst, canonical_id);
  };
  auto make_inst = [&] { return inst_id; };
  return CanonicalizeTypeImpl(inst.kind(), profile_node, make_inst);
}

auto Context::CanonicalizeStructType(Parse::Node parse_node,
                                     SemIR::InstBlockId refs_id)
    -> SemIR::TypeId {
  return CanonicalizeTypeAndAddInstIfNew(
      SemIR::StructType{parse_node, SemIR::TypeId::TypeType, refs_id});
}

auto Context::CanonicalizeTupleType(Parse::Node parse_node,
                                    llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  // Defer allocating a SemIR::TypeBlockId until we know this is a new type.
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileTupleType(type_ids, canonical_id);
  };
  auto make_tuple_inst = [&] {
    return AddInst(SemIR::TupleType{parse_node, SemIR::TypeId::TypeType,
                                    type_blocks().Add(type_ids)});
  };
  return CanonicalizeTypeImpl(SemIR::TupleType::Kind, profile_tuple,
                              make_tuple_inst);
}

auto Context::GetBuiltinType(SemIR::BuiltinKind kind) -> SemIR::TypeId {
  CARBON_CHECK(kind != SemIR::BuiltinKind::Invalid);
  auto type_id = CanonicalizeType(SemIR::InstId::ForBuiltin(kind));
  // To keep client code simpler, complete builtin types before returning them.
  bool complete = TryToCompleteType(type_id);
  CARBON_CHECK(complete) << "Failed to complete builtin type";
  return type_id;
}

auto Context::GetPointerType(Parse::Node parse_node,
                             SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return CanonicalizeTypeAndAddInstIfNew(
      SemIR::PointerType{parse_node, SemIR::TypeId::TypeType, pointee_type_id});
}

auto Context::GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId {
  SemIR::Inst type_inst =
      insts().Get(sem_ir_->GetTypeAllowBuiltinTypes(type_id));
  if (auto const_type = type_inst.TryAs<SemIR::ConstType>()) {
    return const_type->inner_id;
  }
  return type_id;
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  node_stack_.PrintForStackDump(output);
  inst_block_stack_.PrintForStackDump(output);
  params_or_args_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon::Check
