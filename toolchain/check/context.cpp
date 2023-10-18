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
#include "toolchain/check/node_block_stack.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Check {

Context::Context(const Lex::TokenizedBuffer& tokens, DiagnosticEmitter& emitter,
                 const Parse::Tree& parse_tree, SemIR::File& semantics_ir,
                 llvm::raw_ostream* vlog_stream)
    : tokens_(&tokens),
      emitter_(&emitter),
      parse_tree_(&parse_tree),
      semantics_ir_(&semantics_ir),
      vlog_stream_(vlog_stream),
      node_stack_(parse_tree, vlog_stream),
      node_block_stack_("node_block_stack_", semantics_ir, vlog_stream),
      params_or_args_stack_("params_or_args_stack_", semantics_ir, vlog_stream),
      args_type_info_stack_("args_type_info_stack_", semantics_ir, vlog_stream),
      declaration_name_stack_(this) {
  // Inserts the "Error" and "Type" types as "used types" so that
  // canonicalization can skip them. We don't emit either for lowering.
  canonical_types_.insert({SemIR::NodeId::BuiltinError, SemIR::TypeId::Error});
  canonical_types_.insert(
      {SemIR::NodeId::BuiltinTypeType, SemIR::TypeId::TypeType});
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
  CARBON_CHECK(node_block_stack_.empty()) << node_block_stack_.size();
  CARBON_CHECK(params_or_args_stack_.empty()) << params_or_args_stack_.size();
}

auto Context::AddNode(SemIR::Node node) -> SemIR::NodeId {
  auto node_id = node_block_stack_.AddNode(node);
  CARBON_VLOG() << "AddNode: " << node << "\n";
  return node_id;
}

auto Context::AddNodeAndPush(Parse::Node parse_node, SemIR::Node node) -> void {
  auto node_id = AddNode(node);
  node_stack_.Push(parse_node, node_id);
}

auto Context::DiagnoseDuplicateName(Parse::Node parse_node,
                                    SemIR::NodeId prev_def_id) -> void {
  CARBON_DIAGNOSTIC(NameDeclarationDuplicate, Error,
                    "Duplicate name being declared in the same scope.");
  CARBON_DIAGNOSTIC(NameDeclarationPrevious, Note,
                    "Name is previously declared here.");
  auto prev_def = semantics_ir_->GetNode(prev_def_id);
  emitter_->Build(parse_node, NameDeclarationDuplicate)
      .Note(prev_def.parse_node(), NameDeclarationPrevious)
      .Emit();
}

auto Context::DiagnoseNameNotFound(Parse::Node parse_node,
                                   SemIR::StringId name_id) -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name `{0}` not found.",
                    llvm::StringRef);
  emitter_->Emit(parse_node, NameNotFound, semantics_ir_->GetString(name_id));
}

auto Context::NoteIncompleteClass(SemIR::ClassDeclaration class_decl,
                                  DiagnosticBuilder& builder) -> void {
  CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                    "Class was forward declared here.");
  builder.Note(class_decl.parse_node, ClassForwardDeclaredHere);
}

auto Context::AddNameToLookup(Parse::Node name_node, SemIR::StringId name_id,
                              SemIR::NodeId target_id) -> void {
  if (current_scope().names.insert(name_id).second) {
    name_lookup_[name_id].push_back(target_id);
  } else {
    DiagnoseDuplicateName(name_node, name_lookup_[name_id].back());
  }
}

auto Context::LookupName(Parse::Node parse_node, SemIR::StringId name_id,
                         SemIR::NameScopeId scope_id, bool print_diagnostics)
    -> SemIR::NodeId {
  if (scope_id == SemIR::NameScopeId::Invalid) {
    auto it = name_lookup_.find(name_id);
    if (it == name_lookup_.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemIR::NodeId::BuiltinError;
    }
    CARBON_CHECK(!it->second.empty())
        << "Should have been erased: " << semantics_ir_->GetString(name_id);

    // TODO: Check for ambiguous lookups.
    return it->second.back();
  } else {
    const auto& scope = semantics_ir_->GetNameScope(scope_id);
    auto it = scope.find(name_id);
    if (it == scope.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemIR::NodeId::BuiltinError;
    }

    return it->second;
  }
}

auto Context::PushScope(SemIR::NameScopeId scope_id) -> void {
  scope_stack_.push_back({.scope_id = scope_id});
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

auto Context::FollowNameReferences(SemIR::NodeId node_id) -> SemIR::NodeId {
  while (auto name_ref =
             semantics_ir().GetNode(node_id).TryAs<SemIR::NameReference>()) {
    node_id = name_ref->value_id;
  }
  return node_id;
}

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::Node parse_node, Args... args)
    -> SemIR::NodeBlockId {
  if (!context.node_block_stack().is_current_block_reachable()) {
    return SemIR::NodeBlockId::Unreachable;
  }
  auto block_id = context.semantics_ir().AddNodeBlockId();
  context.AddNode(BranchNode(parse_node, block_id, args...));
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::Node parse_node)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Branch>(*this, parse_node);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::Node parse_node,
                                                SemIR::NodeId arg_id)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchWithArg>(*this, parse_node,
                                                              arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::Node parse_node,
                                           SemIR::NodeId cond_id)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchIf>(*this, parse_node,
                                                         cond_id);
}

auto Context::AddConvergenceBlockAndPush(Parse::Node parse_node, int num_blocks)
    -> void {
  CARBON_CHECK(num_blocks >= 2) << "no convergence";

  SemIR::NodeBlockId new_block_id = SemIR::NodeBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (node_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::NodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlockId();
      }
      AddNode(SemIR::Branch(parse_node, new_block_id));
    }
    node_block_stack().Pop();
  }
  node_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::Node parse_node, std::initializer_list<SemIR::NodeId> block_args)
    -> SemIR::NodeId {
  CARBON_CHECK(block_args.size() >= 2) << "no convergence";

  SemIR::NodeBlockId new_block_id = SemIR::NodeBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (node_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::NodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlockId();
      }
      AddNode(SemIR::BranchWithArg(parse_node, new_block_id, arg_id));
    }
    node_block_stack().Pop();
  }
  node_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id =
      semantics_ir().GetNode(*block_args.begin()).type_id();
  return AddNode(SemIR::BlockArg(parse_node, result_type_id, new_block_id));
}

// Add the current code block to the enclosing function.
auto Context::AddCurrentCodeBlockToFunction() -> void {
  CARBON_CHECK(!node_block_stack().empty()) << "no current code block";
  CARBON_CHECK(!return_scope_stack().empty()) << "no current function";

  if (!node_block_stack().is_current_block_reachable()) {
    // Don't include unreachable blocks in the function.
    return;
  }

  auto function_id =
      semantics_ir()
          .GetNodeAs<SemIR::FunctionDeclaration>(return_scope_stack().back())
          .function_id;
  semantics_ir()
      .GetFunction(function_id)
      .body_block_ids.push_back(node_block_stack().PeekOrAdd());
}

auto Context::is_current_position_reachable() -> bool {
  if (!node_block_stack().is_current_block_reachable()) {
    return false;
  }

  // Our current position is at the end of a reachable block. That position is
  // reachable unless the previous instruction is a terminator instruction.
  auto block_contents = node_block_stack().PeekCurrentBlockContents();
  if (block_contents.empty()) {
    return true;
  }
  const auto& last_node = semantics_ir().GetNode(block_contents.back());
  return last_node.kind().terminator_kind() !=
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

auto Context::ParamOrArgPop() -> SemIR::NodeBlockId {
  return params_or_args_stack_.Pop();
}

auto Context::ParamOrArgEnd(Parse::NodeKind start_kind) -> SemIR::NodeBlockId {
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
    if (!context_.semantics_ir().IsTypeComplete(type_id)) {
      work_list_.push_back({type_id, Phase::AddNestedIncompleteTypes});
    }
  }

  // Runs the next step.
  auto ProcessStep() -> bool {
    auto [type_id, phase] = work_list_.back();

    // We might have enqueued the same type more than once. Just skip the
    // type if it's already complete.
    if (context_.semantics_ir().IsTypeComplete(type_id)) {
      work_list_.pop_back();
      return true;
    }

    auto node_id = context_.semantics_ir().GetTypeAllowBuiltinTypes(type_id);
    auto node = context_.semantics_ir().GetNode(node_id);

    auto old_work_list_size = work_list_.size();

    switch (phase) {
      case Phase::AddNestedIncompleteTypes:
        if (!AddNestedIncompleteTypes(node)) {
          return false;
        }
        CARBON_CHECK(work_list_.size() >= old_work_list_size)
            << "AddNestedIncompleteTypes should not remove work items";
        work_list_[old_work_list_size - 1].phase =
            Phase::BuildValueRepresentation;
        break;

      case Phase::BuildValueRepresentation: {
        auto value_rep = BuildValueRepresentation(type_id, node);
        context_.semantics_ir().CompleteType(type_id, value_rep);
        CARBON_CHECK(old_work_list_size == work_list_.size())
            << "BuildValueRepresentation should not change work items";
        work_list_.pop_back();

        // Also complete the value representation type, if necessary. This
        // should never fail: the value representation shouldn't require any
        // additional nested types to be complete.
        if (!context_.semantics_ir().IsTypeComplete(value_rep.type_id)) {
          work_list_.push_back(
              {value_rep.type_id, Phase::BuildValueRepresentation});
        }
        break;
      }
    }

    return true;
  }

  // Adds any types nested within `type_node` that need to be complete for
  // `type_node` to be complete to our work list.
  auto AddNestedIncompleteTypes(SemIR::Node type_node) -> bool {
    switch (type_node.kind()) {
      case SemIR::ArrayType::Kind:
        Push(type_node.As<SemIR::ArrayType>().element_type_id);
        break;

      case SemIR::StructType::Kind:
        for (auto field_id : context_.semantics_ir().GetNodeBlock(
                 type_node.As<SemIR::StructType>().fields_id)) {
          Push(context_.semantics_ir()
                   .GetNodeAs<SemIR::StructTypeField>(field_id)
                   .type_id);
        }
        break;

      case SemIR::TupleType::Kind:
        for (auto element_type_id : context_.semantics_ir().GetTypeBlock(
                 type_node.As<SemIR::TupleType>().elements_id)) {
          Push(element_type_id);
        }
        break;

      case SemIR::ClassDeclaration::Kind:
        // TODO: Support class definitions and complete class types.
        if (diagnoser_) {
          auto builder = (*diagnoser_)();
          context_.NoteIncompleteClass(type_node.As<SemIR::ClassDeclaration>(),
                                       builder);
          builder.Emit();
        }
        return false;

      case SemIR::ConstType::Kind:
        Push(type_node.As<SemIR::ConstType>().inner_id);
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
  auto MakeCopyRepresentation(SemIR::TypeId rep_id) const
      -> SemIR::ValueRepresentation {
    return {.kind = SemIR::ValueRepresentation::Copy, .type_id = rep_id};
  }

  // Makes a value representation that uses pass-by-address with the given
  // pointee type.
  auto MakePointerRepresentation(Parse::Node parse_node,
                                 SemIR::TypeId pointee_id) const
      -> SemIR::ValueRepresentation {
    // TODO: Should we add `const` qualification to `pointee_id`?
    return {.kind = SemIR::ValueRepresentation::Pointer,
            .type_id = context_.GetPointerType(parse_node, pointee_id)};
  }

  // Gets the value representation of a nested type, which should already be
  // complete.
  auto GetNestedValueRepresentation(SemIR::TypeId nested_type_id) const {
    CARBON_CHECK(context_.semantics_ir().IsTypeComplete(nested_type_id))
        << "Nested type should already be complete";
    auto value_rep =
        context_.semantics_ir().GetValueRepresentation(nested_type_id);
    CARBON_CHECK(value_rep.kind != SemIR::ValueRepresentation::Unknown)
        << "Complete type should have a value representation";
    return value_rep;
  };

  auto BuildCrossReferenceValueRepresentation(SemIR::TypeId type_id,
                                              SemIR::CrossReference xref) const
      -> SemIR::ValueRepresentation {
    auto xref_node = context_.semantics_ir()
                         .GetCrossReferenceIR(xref.ir_id)
                         .GetNode(xref.node_id);

    // The canonical description of a type should only have cross-references
    // for entities owned by another File, such as builtins, which are owned
    // by the prelude, and named entities like classes and interfaces, which
    // we don't support yet.
    CARBON_CHECK(xref_node.kind() == SemIR::Builtin::Kind)
        << "TODO: Handle other kinds of node cross-references";

    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (xref_node.As<SemIR::Builtin>().builtin_kind) {
      case SemIR::BuiltinKind::TypeType:
      case SemIR::BuiltinKind::Error:
      case SemIR::BuiltinKind::Invalid:
      case SemIR::BuiltinKind::BoolType:
      case SemIR::BuiltinKind::IntegerType:
      case SemIR::BuiltinKind::FloatingPointType:
      case SemIR::BuiltinKind::NamespaceType:
      case SemIR::BuiltinKind::FunctionType:
        return MakeCopyRepresentation(type_id);

      case SemIR::BuiltinKind::StringType:
        // TODO: Decide on string value semantics. This should probably be a
        // custom value representation carrying a pointer and size or
        // similar.
        return MakePointerRepresentation(Parse::Node::Invalid, type_id);
    }
    llvm_unreachable("All builtin kinds were handled above");
  }

  auto BuildStructTypeValueRepresentation(SemIR::TypeId type_id,
                                          SemIR::StructType struct_type) const
      -> SemIR::ValueRepresentation {
    // TODO: Share code with tuples.
    auto fields = context_.semantics_ir().GetNodeBlock(struct_type.fields_id);
    if (fields.empty()) {
      return MakeEmptyRepresentation(struct_type.parse_node);
    }

    // Find the value representation for each field, and construct a struct
    // of value representations.
    llvm::SmallVector<SemIR::NodeId> value_rep_fields;
    value_rep_fields.reserve(fields.size());
    bool same_as_object_rep = true;
    for (auto field_id : fields) {
      auto field =
          context_.semantics_ir().GetNodeAs<SemIR::StructTypeField>(field_id);
      auto field_value_rep = GetNestedValueRepresentation(field.type_id);
      if (field_value_rep.type_id != field.type_id) {
        same_as_object_rep = false;
        field.type_id = field_value_rep.type_id;
        field_id = context_.AddNode(field);
      }
      value_rep_fields.push_back(field_id);
    }

    auto value_rep =
        same_as_object_rep
            ? type_id
            : context_.CanonicalizeStructType(
                  struct_type.parse_node,
                  context_.semantics_ir().AddNodeBlock(value_rep_fields));
    if (fields.size() == 1) {
      // The value representation for a struct with a single field is a
      // struct containing the value representation of the field.
      // TODO: Consider doing the same for structs with multiple small
      // fields.
      return MakeCopyRepresentation(value_rep);
    }
    // For a struct with multiple fields, we use a pointer representation.
    return MakePointerRepresentation(struct_type.parse_node, value_rep);
  }

  auto BuildTupleTypeValueRepresentation(SemIR::TypeId type_id,
                                         SemIR::TupleType tuple_type) const
      -> SemIR::ValueRepresentation {
    // TODO: Share code with structs.
    auto elements =
        context_.semantics_ir().GetTypeBlock(tuple_type.elements_id);
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
    if (elements.size() == 1) {
      // The value representation for a tuple with a single element is a
      // tuple containing the value representation of that element.
      // TODO: Consider doing the same for tuples with multiple small
      // elements.
      return MakeCopyRepresentation(value_rep);
    }
    // For a tuple with multiple elements, we use a pointer representation.
    return MakePointerRepresentation(tuple_type.parse_node, value_rep);
  }

  // Builds and returns the value representation for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildValueRepresentation(SemIR::TypeId type_id, SemIR::Node node) const
      -> SemIR::ValueRepresentation {
    // TODO: This can emit new SemIR nodes. Consider emitting them into a
    // dedicated file-scope node block where possible, or somewhere else that
    // better reflects the definition of the type, rather than wherever the
    // type happens to first be required to be complete.

    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (node.kind()) {
      case SemIR::AddressOf::Kind:
      case SemIR::ArrayIndex::Kind:
      case SemIR::ArrayInit::Kind:
      case SemIR::Assign::Kind:
      case SemIR::BinaryOperatorAdd::Kind:
      case SemIR::BindName::Kind:
      case SemIR::BindValue::Kind:
      case SemIR::BlockArg::Kind:
      case SemIR::BoolLiteral::Kind:
      case SemIR::Branch::Kind:
      case SemIR::BranchIf::Kind:
      case SemIR::BranchWithArg::Kind:
      case SemIR::Call::Kind:
      case SemIR::Dereference::Kind:
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
      case SemIR::VarStorage::Kind:
        CARBON_FATAL() << "Type refers to non-type node " << node;

      case SemIR::CrossReference::Kind:
        return BuildCrossReferenceValueRepresentation(
            type_id, node.As<SemIR::CrossReference>());

      case SemIR::ArrayType::Kind: {
        // For arrays, it's convenient to always use a pointer representation,
        // even when the array has zero or one element, in order to support
        // indexing.
        return MakePointerRepresentation(node.parse_node(), type_id);
      }

      case SemIR::StructType::Kind:
        return BuildStructTypeValueRepresentation(type_id,
                                                  node.As<SemIR::StructType>());

      case SemIR::TupleType::Kind:
        return BuildTupleTypeValueRepresentation(type_id,
                                                 node.As<SemIR::TupleType>());

      case SemIR::ClassDeclaration::Kind:
        // TODO: Support class definitions and complete class types.
        CARBON_FATAL() << "Class types are currently never complete";

      case SemIR::Builtin::Kind:
        CARBON_FATAL() << "Builtins should be named as cross-references";

      case SemIR::PointerType::Kind:
        return MakeCopyRepresentation(type_id);

      case SemIR::ConstType::Kind:
        // The value representation of `const T` is the same as that of `T`.
        // Objects are not modifiable through their value representations.
        return GetNestedValueRepresentation(
            node.As<SemIR::ConstType>().inner_id);
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
    SemIR::NodeKind kind,
    llvm::function_ref<void(llvm::FoldingSetNodeID& canonical_id)> profile_type,
    llvm::function_ref<SemIR::NodeId()> make_node) -> SemIR::TypeId {
  llvm::FoldingSetNodeID canonical_id;
  kind.Profile(canonical_id);
  profile_type(canonical_id);

  void* insert_pos;
  auto* node =
      canonical_type_nodes_.FindNodeOrInsertPos(canonical_id, insert_pos);
  if (node != nullptr) {
    return node->type_id();
  }

  auto node_id = make_node();
  auto type_id = semantics_ir_->AddType(node_id);
  CARBON_CHECK(canonical_types_.insert({node_id, type_id}).second);
  type_node_storage_.push_back(
      std::make_unique<TypeNode>(canonical_id, type_id));

  // In a debug build, check that our insertion position is still valid. It
  // could have been invalidated by a misbehaving `make_node`.
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
static auto ProfileType(Context& semantics_context, SemIR::Node node,
                        llvm::FoldingSetNodeID& canonical_id) -> void {
  switch (node.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = node.As<SemIR::ArrayType>();
      canonical_id.AddInteger(
          semantics_context.semantics_ir().GetArrayBoundValue(
              array_type.bound_id));
      canonical_id.AddInteger(array_type.element_type_id.index);
      break;
    }
    case SemIR::Builtin::Kind:
      canonical_id.AddInteger(node.As<SemIR::Builtin>().builtin_kind.AsInt());
      break;
    case SemIR::ClassDeclaration::Kind:
      canonical_id.AddInteger(
          node.As<SemIR::ClassDeclaration>().class_id.index);
      break;
    case SemIR::CrossReference::Kind: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto xref = node.As<SemIR::CrossReference>();
      canonical_id.AddInteger(xref.ir_id.index);
      canonical_id.AddInteger(xref.node_id.index);
      break;
    }
    case SemIR::ConstType::Kind:
      canonical_id.AddInteger(
          semantics_context
              .GetUnqualifiedType(node.As<SemIR::ConstType>().inner_id)
              .index);
      break;
    case SemIR::PointerType::Kind:
      canonical_id.AddInteger(node.As<SemIR::PointerType>().pointee_id.index);
      break;
    case SemIR::StructType::Kind: {
      auto fields = semantics_context.semantics_ir().GetNodeBlock(
          node.As<SemIR::StructType>().fields_id);
      for (const auto& field_id : fields) {
        auto field =
            semantics_context.semantics_ir().GetNodeAs<SemIR::StructTypeField>(
                field_id);
        canonical_id.AddInteger(field.name_id.index);
        canonical_id.AddInteger(field.type_id.index);
      }
      break;
    }
    case SemIR::TupleType::Kind:
      ProfileTupleType(semantics_context.semantics_ir().GetTypeBlock(
                           node.As<SemIR::TupleType>().elements_id),
                       canonical_id);
      break;
    default:
      CARBON_FATAL() << "Unexpected type node " << node;
  }
}

auto Context::CanonicalizeTypeAndAddNodeIfNew(SemIR::Node node)
    -> SemIR::TypeId {
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileType(*this, node, canonical_id);
  };
  auto make_node = [&] { return AddNode(node); };
  return CanonicalizeTypeImpl(node.kind(), profile_node, make_node);
}

auto Context::CanonicalizeType(SemIR::NodeId node_id) -> SemIR::TypeId {
  node_id = FollowNameReferences(node_id);

  auto it = canonical_types_.find(node_id);
  if (it != canonical_types_.end()) {
    return it->second;
  }

  auto node = semantics_ir_->GetNode(node_id);
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileType(*this, node, canonical_id);
  };
  auto make_node = [&] { return node_id; };
  return CanonicalizeTypeImpl(node.kind(), profile_node, make_node);
}

auto Context::CanonicalizeStructType(Parse::Node parse_node,
                                     SemIR::NodeBlockId refs_id)
    -> SemIR::TypeId {
  return CanonicalizeTypeAndAddNodeIfNew(
      SemIR::StructType(parse_node, SemIR::TypeId::TypeType, refs_id));
}

auto Context::CanonicalizeTupleType(Parse::Node parse_node,
                                    llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  // Defer allocating a SemIR::TypeBlockId until we know this is a new type.
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileTupleType(type_ids, canonical_id);
  };
  auto make_tuple_node = [&] {
    return AddNode(SemIR::TupleType(parse_node, SemIR::TypeId::TypeType,
                                    semantics_ir_->AddTypeBlock(type_ids)));
  };
  return CanonicalizeTypeImpl(SemIR::TupleType::Kind, profile_tuple,
                              make_tuple_node);
}

auto Context::GetBuiltinType(SemIR::BuiltinKind kind) -> SemIR::TypeId {
  CARBON_CHECK(kind != SemIR::BuiltinKind::Invalid);
  auto type_id = CanonicalizeType(SemIR::NodeId::ForBuiltin(kind));
  // To keep client code simpler, complete builtin types before returning them.
  bool complete = TryToCompleteType(type_id);
  CARBON_CHECK(complete) << "Failed to complete builtin type";
  return type_id;
}

auto Context::GetPointerType(Parse::Node parse_node,
                             SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return CanonicalizeTypeAndAddNodeIfNew(
      SemIR::PointerType(parse_node, SemIR::TypeId::TypeType, pointee_type_id));
}

auto Context::GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId {
  SemIR::Node type_node =
      semantics_ir_->GetNode(semantics_ir_->GetTypeAllowBuiltinTypes(type_id));
  if (auto const_type = type_node.TryAs<SemIR::ConstType>()) {
    return const_type->inner_id;
  }
  return type_id;
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  node_stack_.PrintForStackDump(output);
  node_block_stack_.PrintForStackDump(output);
  params_or_args_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon::Check
