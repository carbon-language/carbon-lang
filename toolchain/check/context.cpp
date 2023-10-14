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

auto Context::PushScope() -> void { scope_stack_.push_back({}); }

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

// Attempts to complete the given type.
auto Context::TryToCompleteType(
    SemIR::TypeId type_id,
    std::optional<llvm::function_ref<auto()->DiagnosticBuilder>> diagnoser)
    -> bool {
  if (semantics_ir().IsTypeComplete(type_id)) {
    return true;
  }

  auto node_id = semantics_ir().GetTypeAllowBuiltinTypes(type_id);
  auto node = semantics_ir().GetNode(node_id);

  auto set_empty_representation = [&]() {
    semantics_ir().CompleteType(
        type_id, {.kind = SemIR::ValueRepresentation::None,
                  .type_id = CanonicalizeTupleType(node.parse_node(), {})});
    return true;
  };

  auto set_copy_representation = [&](SemIR::TypeId rep_id) {
    semantics_ir().CompleteType(
        type_id, {.kind = SemIR::ValueRepresentation::Copy, .type_id = rep_id});
    return true;
  };

  auto set_pointer_representation = [&](SemIR::TypeId pointee_id) {
    // TODO: Should we add `const` qualification to `pointee_id`?
    semantics_ir().CompleteType(
        type_id, {.kind = SemIR::ValueRepresentation::Pointer,
                  .type_id = GetPointerType(node.parse_node(), pointee_id)});
    return true;
  };

  // Requires a type nested within this one to be complete.
  auto try_to_complete_nested_type = [&](SemIR::TypeId nested_type_id,
                                         auto diagnostic_annotator) -> bool {
    decltype(diagnoser) nested_diagnoser;
    if (diagnoser) {
      nested_diagnoser = [&]() -> DiagnosticBuilder {
        auto builder = (*diagnoser)();
        diagnostic_annotator(builder);
        return builder;
      };
    }
    // TODO: Make this non-recursive.
    return TryToCompleteType(nested_type_id, nested_diagnoser);
  };

  // Requires a type nested within this one to be complete, and returns its
  // value representation. If the type is not complete, returns std::nullopt.
  auto get_nested_value_representation = [&](SemIR::TypeId nested_type_id,
                                             auto diagnostic_annotator)
      -> std::optional<SemIR::ValueRepresentation> {
    if (!try_to_complete_nested_type(nested_type_id, diagnostic_annotator)) {
      return std::nullopt;
    }
    auto value_rep = semantics_ir().GetValueRepresentation(nested_type_id);
    CARBON_CHECK(value_rep.kind != SemIR::ValueRepresentation::Unknown)
        << "Complete type should have a value representation";
    return value_rep;
  };

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

    case SemIR::CrossReference::Kind: {
      auto xref = node.As<SemIR::CrossReference>();
      auto xref_node =
          semantics_ir().GetCrossReferenceIR(xref.ir_id).GetNode(xref.node_id);

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
          return set_copy_representation(type_id);

        case SemIR::BuiltinKind::StringType:
          // TODO: Decide on string value semantics. This should probably be a
          // custom value representation carrying a pointer and size or
          // similar.
          return set_pointer_representation(type_id);
      }
      llvm_unreachable("All builtin kinds were handled above");
    }

    case SemIR::ArrayType::Kind: {
      if (!try_to_complete_nested_type(
              node.As<SemIR::ArrayType>().element_type_id,
              [&](DiagnosticBuilder& /*builder*/) {
                // TODO: Add a note mentioning the array.
              })) {
        return false;
      }
      // For arrays, it's convenient to always use a pointer representation,
      // even when the array has zero or one element, in order to support
      // indexing.
      return set_pointer_representation(type_id);
    }

    case SemIR::StructType::Kind: {
      auto fields =
          semantics_ir().GetNodeBlock(node.As<SemIR::StructType>().fields_id);
      if (fields.empty()) {
        return set_empty_representation();
      }

      // Find the value representation for each field, and construct a struct
      // of value representations.
      llvm::SmallVector<SemIR::NodeId> value_rep_fields;
      value_rep_fields.reserve(fields.size());
      bool same_as_object_rep = true;
      for (auto field_id : fields) {
        auto field = semantics_ir().GetNodeAs<SemIR::StructTypeField>(field_id);

        // A struct is complete if and only if all its fields are complete.
        auto field_value_rep = get_nested_value_representation(
            field.type_id, [&](DiagnosticBuilder& /*builder*/) {
              // TODO: Add a note mentioning the field.
            });
        if (!field_value_rep) {
          return false;
        }
        if (field_value_rep->type_id != field.type_id) {
          same_as_object_rep = false;
          field.type_id = field_value_rep->type_id;
          field_id = AddNode(field);
        }
        value_rep_fields.push_back(field_id);
      }

      auto value_rep = same_as_object_rep
                           ? type_id
                           : CanonicalizeStructType(
                                 node.parse_node(),
                                 semantics_ir().AddNodeBlock(value_rep_fields));
      if (fields.size() == 1) {
        // The value representation for a struct with a single field is a struct
        // containing the value representation of the field.
        // TODO: Consider doing the same for structs with multiple small fields.
        return set_copy_representation(value_rep);
      }
      // For a struct with multiple fields, we use a pointer representation.
      return set_pointer_representation(value_rep);
    }

    case SemIR::TupleType::Kind: {
      // TODO: Extract and share code with structs.
      auto elements =
          semantics_ir().GetTypeBlock(node.As<SemIR::TupleType>().elements_id);
      if (elements.empty()) {
        return set_empty_representation();
      }

      // Find the value representation for each element, and construct a tuple
      // of value representations.
      llvm::SmallVector<SemIR::TypeId> value_rep_elements;
      value_rep_elements.reserve(elements.size());
      bool same_as_object_rep = true;
      for (auto element_type_id : elements) {
        // A tuple is complete if and only if all its elements are complete.
        auto element_value_rep = get_nested_value_representation(
            element_type_id, [&](DiagnosticBuilder& /*builder*/) {
              // TODO: Add a note mentioning the tuple element.
            });
        if (!element_value_rep) {
          return false;
        }
        if (element_value_rep->type_id != element_type_id) {
          same_as_object_rep = false;
        }
        value_rep_elements.push_back(element_value_rep->type_id);
      }

      auto value_rep =
          same_as_object_rep
              ? type_id
              : CanonicalizeTupleType(node.parse_node(), value_rep_elements);
      if (elements.size() == 1) {
        // The value representation for a tuple with a single element is a tuple
        // containing the value representation of that element.
        // TODO: Consider doing the same for tuples with multiple small
        // elements.
        return set_copy_representation(value_rep);
      }
      // For a tuple with multiple elements, we use a pointer representation.
      return set_pointer_representation(value_rep);
    }

    case SemIR::ClassDeclaration::Kind: {
      // TODO: Support class definitions and complete class types.
      if (diagnoser) {
        CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                          "Class was forward declared here.");
        (*diagnoser)().Note(node.parse_node(), ClassForwardDeclaredHere).Emit();
      }
      return false;
    }

    case SemIR::Builtin::Kind:
      CARBON_FATAL() << "Builtins should be named as cross-references";

    case SemIR::PointerType::Kind:
      return set_copy_representation(type_id);

    case SemIR::ConstType::Kind: {
      // The value representation of `const T` is the same as that of `T`.
      // Objects are not modifiable through their value representations.
      auto inner_value_rep = get_nested_value_representation(
          node.As<SemIR::ConstType>().inner_id,
          [&](DiagnosticBuilder& /*builder*/) {});
      if (!inner_value_rep) {
        return false;
      }
      semantics_ir().CompleteType(type_id, *inner_value_rep);
      return true;
    }
  }

  llvm_unreachable("All node kinds were handled above");
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

  // Now we've formed the type, try to complete it and build its value
  // representation.
  // TODO: Delay doing this until a complete type is required, and issue a
  // diagnostic if it fails.
  // TODO: Consider emitting this into the file's global node block
  // (or somewhere else that better reflects the definition of the type
  // rather than the coincidental first use).
  TryToCompleteType(type_id);
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
