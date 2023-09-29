// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
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

Context::Context(const Lex::TokenizedBuffer& tokens,
                 DiagnosticEmitter<Parse::Node>& emitter,
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

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::Node parse_node, Args... args)
    -> SemIR::NodeBlockId {
  if (!context.node_block_stack().is_current_block_reachable()) {
    return SemIR::NodeBlockId::Unreachable;
  }
  auto block_id = context.semantics_ir().AddNodeBlockId();
  context.AddNode(BranchNode::Make(parse_node, block_id, args...));
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::Node parse_node)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Node::Branch>(*this, parse_node);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::Node parse_node,
                                                SemIR::NodeId arg_id)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Node::BranchWithArg>(
      *this, parse_node, arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::Node parse_node,
                                           SemIR::NodeId cond_id)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Node::BranchIf>(
      *this, parse_node, cond_id);
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
      AddNode(SemIR::Node::Branch::Make(parse_node, new_block_id));
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
      AddNode(
          SemIR::Node::BranchWithArg::Make(parse_node, new_block_id, arg_id));
    }
    node_block_stack().Pop();
  }
  node_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id =
      semantics_ir().GetNode(*block_args.begin()).type_id();
  return AddNode(
      SemIR::Node::BlockArg::Make(parse_node, result_type_id, new_block_id));
}

// Add the current code block to the enclosing function.
auto Context::AddCurrentCodeBlockToFunction() -> void {
  CARBON_CHECK(!node_block_stack().empty()) << "no current code block";
  CARBON_CHECK(!return_scope_stack().empty()) << "no current function";

  if (!node_block_stack().is_current_block_reachable()) {
    // Don't include unreachable blocks in the function.
    return;
  }

  auto function_id = semantics_ir()
                         .GetNode(return_scope_stack().back())
                         .GetAsFunctionDeclaration();
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

namespace {
// A handle to a new block that may be modified, with copy-on-write semantics.
//
// The constructor is given the ID of an existing block that provides the
// initial contents of the new block. The new block is lazily allocated; if no
// modifications have been made, the `id()` function will return the original
// block ID.
//
// This is intended to avoid an unnecessary block allocation in the case where
// the new block ends up being exactly the same as the original block.
class CopyOnWriteBlock {
 public:
  // Constructs the block. If `source_id` is valid, it is used as the initial
  // value of the block. Otherwise, uninitialized storage for `size` elements
  // is allocated.
  CopyOnWriteBlock(SemIR::File& file, SemIR::NodeBlockId source_id, size_t size)
      : file_(file), source_id_(source_id) {
    if (!source_id_.is_valid()) {
      id_ = file_.AddUninitializedNodeBlock(size);
    }
  }

  auto id() -> SemIR::NodeBlockId const { return id_; }

  auto Set(int i, SemIR::NodeId value) -> void {
    if (source_id_.is_valid() && file_.GetNodeBlock(id_)[i] == value) {
      return;
    }
    if (id_ == source_id_) {
      id_ = file_.AddNodeBlock(file_.GetNodeBlock(source_id_));
    }
    file_.GetNodeBlock(id_)[i] = value;
  }

 private:
  SemIR::File& file_;
  SemIR::NodeBlockId source_id_;
  SemIR::NodeBlockId id_ = source_id_;
};
}  // namespace

// A block of code that contains pending instructions that might be needed but
// that haven't been inserted yet.
class Context::PendingBlock {
 public:
  PendingBlock(Context& context) : context_(context) {}

  PendingBlock(const PendingBlock&) = delete;
  PendingBlock& operator=(const PendingBlock&) = delete;

  // A scope in which we will tentatively add nodes to a pending block. If we
  // leave the scope without inserting or merging the block, nodes added after
  // this point will be removed again.
  class DiscardUnusedNodesScope {
   public:
    // If `block` is not null, enters the scope. If `block` is null, this object
    // has no effect.
    DiscardUnusedNodesScope(PendingBlock* block)
        : block_(block), size_(block ? block->nodes_.size() : 0) {}
    ~DiscardUnusedNodesScope() {
      if (block_ && block_->nodes_.size() > size_) {
        block_->nodes_.truncate(size_);
      }
    }

   private:
    PendingBlock* block_;
    size_t size_;
  };

  auto AddNode(SemIR::Node node) -> SemIR::NodeId {
    auto node_id = context_.semantics_ir().AddNodeInNoBlock(node);
    nodes_.push_back(node_id);
    return node_id;
  }

  // Insert the pending block of code at the current position.
  auto InsertHere() -> void {
    for (auto id : nodes_) {
      context_.node_block_stack().AddNodeId(id);
    }
    nodes_.clear();
  }

  // Replace the node at target_id with the nodes in this block. The new value
  // for target_id should be value_id.
  auto MergeReplacing(SemIR::NodeId target_id, SemIR::NodeId value_id) -> void {
    auto value = context_.semantics_ir().GetNode(value_id);

    // There are three cases here:

    if (nodes_.empty()) {
      // 1) The block is empty. Replace `target_id` with an empty splice
      // pointing at `value_id`.
      context_.semantics_ir().ReplaceNode(
          target_id,
          SemIR::Node::SpliceBlock::Make(value.parse_node(), value.type_id(),
                                         SemIR::NodeBlockId::Empty, value_id));
    } else if (nodes_.size() == 1 && nodes_[0] == value_id) {
      // 2) The block is {value_id}. Replace `target_id` with the node referred
      // to by `value_id`. This is intended to be the common case.
      context_.semantics_ir().ReplaceNode(target_id, value);
    } else {
      // 3) Anything else: splice it into the IR, replacing `target_id`.
      context_.semantics_ir().ReplaceNode(
          target_id,
          SemIR::Node::SpliceBlock::Make(
              value.parse_node(), value.type_id(),
              context_.semantics_ir().AddNodeBlock(nodes_), value_id));
    }

    // Prepare to stash more pending instructions.
    nodes_.clear();
  }

 private:
  Context& context_;
  llvm::SmallVector<SemIR::NodeId> nodes_;
};

// Performs a conversion from a tuple to an array type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToArray(Context& context, SemIR::Node tuple_type,
                                SemIR::Node array_type, SemIR::NodeId value_id,
                                Context::ConversionTarget target)
    -> SemIR::NodeId {
  auto [array_bound_id, element_type_id] = array_type.GetAsArrayType();
  auto tuple_elem_types_id = tuple_type.GetAsTupleType();
  const auto& tuple_elem_types =
      context.semantics_ir().GetTypeBlock(tuple_elem_types_id);

  auto value = context.semantics_ir().GetNode(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (value.kind() == SemIR::NodeKind::TupleLiteral) {
    literal_elems =
        context.semantics_ir().GetNodeBlock(value.GetAsTupleLiteral());
  } else {
    value_id = context.MaterializeIfInitializing(value_id);
  }


  // Check that the tuple is the right size.
  uint64_t array_bound =
      context.semantics_ir().GetArrayBoundValue(array_bound_id);
  if (tuple_elem_types.size() != array_bound) {
    CARBON_DIAGNOSTIC(
        ArrayInitFromLiteralArgCountMismatch, Error,
        "Cannot initialize array of {0} element(s) from {1} initializer(s).",
        uint64_t, size_t);
    CARBON_DIAGNOSTIC(ArrayInitFromExpressionArgCountMismatch, Error,
                      "Cannot initialize array of {0} element(s) from tuple "
                      "with {1} element(s).",
                      uint64_t, size_t);
    context.emitter().Emit(value.parse_node(),
                           literal_elems.empty()
                               ? ArrayInitFromExpressionArgCountMismatch
                               : ArrayInitFromLiteralArgCountMismatch,
                           array_bound, tuple_elem_types.size());
    return SemIR::NodeId::BuiltinError;
  }

  Context::PendingBlock target_block_storage(context);
  Context::PendingBlock* target_block =
      target.init_block ? target.init_block : &target_block_storage;

  // Arrays are always initialized in-place. Allocate a temporary as the
  // destination for the array initialization if we weren't given one.
  SemIR::NodeId return_slot_id = target.init_id;
  if (!target.init_id.is_valid()) {
    return_slot_id = target_block->AddNode(SemIR::Node::TemporaryStorage::Make(
        value.parse_node(), target.type_id));
  }

  // Initialize each element of the array from the corresponding element of the
  // tuple.
  // TODO: Annotate diagnostics coming from here with the array element index,
  // if initializing from a tuple literal.
  llvm::SmallVector<SemIR::NodeId> inits;
  inits.reserve(array_bound + 1);
  for (auto [i, src_type_id] : llvm::enumerate(tuple_elem_types)) {
    Context::PendingBlock::DiscardUnusedNodesScope scope(target_block);
    // TODO: Add a new node kind for indexing an array at a constant index
    // so that we don't need an integer literal node here.
    auto index_id = target_block->AddNode(SemIR::Node::IntegerLiteral::Make(
        value.parse_node(),
        context.CanonicalizeType(SemIR::NodeId::BuiltinIntegerType),
        context.semantics_ir().AddIntegerLiteral(llvm::APInt(32, i))));
    auto target_id = target_block->AddNode(SemIR::Node::ArrayIndex::Make(
        value.parse_node(), element_type_id, return_slot_id, index_id));
    // Note, this is computing the source location not the destination, so it
    // goes into the current code block, not into the target block.
    // TODO: Ideally we would also discard this node if it's unused.
    auto src_id = !literal_elems.empty()
                      ? literal_elems[i]
                      : context.AddNode(SemIR::Node::TupleAccess::Make(
                            value.parse_node(), src_type_id, value_id,
                            SemIR::MemberIndex(i)));
    auto init_id =
        context.Convert(value.parse_node(), src_id,
                        {.kind = Context::ConversionTarget::FullInitializer,
                         .type_id = element_type_id,
                         .init_id = target_id,
                         .init_block = target_block});
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    inits.push_back(init_id);
  }

  // The last element of the refs block contains the return slot for the array
  // initialization. Flush the temporary here if we didn't insert it earlier.
  target_block->InsertHere();
  inits.push_back(return_slot_id);

  return context.AddNode(
      SemIR::Node::ArrayInit::Make(value.parse_node(), target.type_id, value_id,
                                   context.semantics_ir().AddNodeBlock(inits)));
}

// Performs a conversion from a tuple to a tuple type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToTuple(Context& context, SemIR::Node src_type,
                                SemIR::Node dest_type, SemIR::NodeId value_id,
                                Context::ConversionTarget target)
    -> SemIR::NodeId {
  auto src_elem_types =
      context.semantics_ir().GetTypeBlock(src_type.GetAsTupleType());
  auto dest_elem_types =
      context.semantics_ir().GetTypeBlock(dest_type.GetAsTupleType());

  auto value = context.semantics_ir().GetNode(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (value.kind() == SemIR::NodeKind::TupleLiteral) {
    literal_elems =
        context.semantics_ir().GetNodeBlock(value.GetAsTupleLiteral());
  } else {
    value_id = context.MaterializeIfInitializing(value_id);
  }

  // Check that the tuples are the same size.
  if (src_elem_types.size() != dest_elem_types.size()) {
    CARBON_DIAGNOSTIC(TupleInitElementCountMismatch, Error,
                      "Cannot initialize tuple of {0} element(s) from tuple "
                      "with {1} element(s).",
                      size_t, size_t);
    context.emitter().Emit(value.parse_node(), TupleInitElementCountMismatch,
                           dest_elem_types.size(), src_elem_types.size());
    return SemIR::NodeId::BuiltinError;
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  Context::ConversionTarget::Kind inner_kind =
      !is_init ? Context::ConversionTarget::Value
      : SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                             target.type_id)
                  .kind == SemIR::InitializingRepresentation::InPlace
          ? Context::ConversionTarget::FullInitializer
          : Context::ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(context.semantics_ir(),
                             value.kind() == SemIR::NodeKind::TupleLiteral
                                 ? value.GetAsTupleLiteral()
                                 : SemIR::NodeBlockId::Invalid,
                             src_elem_types.size());
  for (auto [i, src_type_id, dest_type_id] :
       llvm::enumerate(src_elem_types, dest_elem_types)) {
    Context::PendingBlock::DiscardUnusedNodesScope scope(target.init_block);
    auto target_id =
        is_init ? target.init_block->AddNode(SemIR::Node::TupleAccess::Make(
                      value.parse_node(), dest_type_id, target.init_id,
                      SemIR::MemberIndex(i)))
                : SemIR::NodeId::Invalid;
    // Note, this is computing the source location not the destination, so it
    // goes into the current code block, not into the target block.
    // TODO: Ideally we would also discard this node if it's unused.
    auto src_id = !literal_elems.empty()
                      ? literal_elems[i]
                      : context.AddNode(SemIR::Node::TupleAccess::Make(
                            value.parse_node(), src_type_id, value_id,
                            SemIR::MemberIndex(i)));
    auto init_id = context.Convert(value.parse_node(), src_id,
                                   {.kind = inner_kind,
                                    .type_id = dest_type_id,
                                    .init_id = target_id,
                                    .init_block = target.init_block});
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return context.AddNode(
      is_init
          ? SemIR::Node::TupleInit::Make(value.parse_node(), target.type_id,
                                         value_id, new_block.id())
          : SemIR::Node::TupleValue::Make(value.parse_node(), target.type_id,
                                          value_id, new_block.id()));
}

// Performs a conversion from a struct to a struct type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertStructToStruct(Context& context, SemIR::Node src_type,
                                SemIR::Node dest_type, SemIR::NodeId value_id,
                                Context::ConversionTarget target)
    -> SemIR::NodeId {
  auto src_elem_fields =
      context.semantics_ir().GetNodeBlock(src_type.GetAsStructType());
  auto dest_elem_fields =
      context.semantics_ir().GetNodeBlock(dest_type.GetAsStructType());

  auto value = context.semantics_ir().GetNode(value_id);

  // If we're initializing from a struct literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (value.kind() == SemIR::NodeKind::StructLiteral) {
    literal_elems =
        context.semantics_ir().GetNodeBlock(value.GetAsStructLiteral());
  } else {
    value_id = context.MaterializeIfInitializing(value_id);
  }

  // Check that the structs are the same size.
  // TODO: Check the field names are the same up to permutation, compute the
  // permutation, and use it below.
  if (src_elem_fields.size() != dest_elem_fields.size()) {
    CARBON_DIAGNOSTIC(StructInitElementCountMismatch, Error,
                      "Cannot initialize struct of {0} element(s) from struct "
                      "with {1} element(s).",
                      size_t, size_t);
    context.emitter().Emit(value.parse_node(), StructInitElementCountMismatch,
                           dest_elem_fields.size(), src_elem_fields.size());
    return SemIR::NodeId::BuiltinError;
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  Context::ConversionTarget::Kind inner_kind =
      !is_init ? Context::ConversionTarget::Value
      : SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                             target.type_id)
                  .kind == SemIR::InitializingRepresentation::InPlace
          ? Context::ConversionTarget::FullInitializer
          : Context::ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(context.semantics_ir(),
                             value.kind() == SemIR::NodeKind::StructLiteral
                                 ? value.GetAsStructLiteral()
                                 : SemIR::NodeBlockId::Invalid,
                             src_elem_fields.size());
  for (auto [i, src_field_id, dest_field_id] :
       llvm::enumerate(src_elem_fields, dest_elem_fields)) {
    auto [src_name_id, src_type_id] =
        context.semantics_ir().GetNode(src_field_id).GetAsStructTypeField();
    auto [dest_name_id, dest_type_id] =
        context.semantics_ir().GetNode(dest_field_id).GetAsStructTypeField();
    if (src_name_id != dest_name_id) {
      CARBON_DIAGNOSTIC(
          StructInitFieldNameMismatch, Error,
          "Mismatched names for field {0} in struct initialization: "
          "source has field name `{1}`, destination has field name `{2}`.",
          size_t, llvm::StringRef, llvm::StringRef);
      context.emitter().Emit(value.parse_node(), StructInitFieldNameMismatch,
                             i + 1,
                             context.semantics_ir().GetString(src_name_id),
                             context.semantics_ir().GetString(dest_name_id));
      return SemIR::NodeId::BuiltinError;
    }
    Context::PendingBlock::DiscardUnusedNodesScope scope(target.init_block);
    auto target_id =
        is_init ? target.init_block->AddNode(SemIR::Node::StructAccess::Make(
                      value.parse_node(), dest_type_id, target.init_id,
                      SemIR::MemberIndex(i)))
                : SemIR::NodeId::Invalid;
    // Note, this is computing the source location not the destination, so it
    // goes into the current code block, not into the target block.
    // TODO: Ideally we would also discard this node if it's unused.
    auto src_id = !literal_elems.empty()
                      ? literal_elems[i]
                      : context.AddNode(SemIR::Node::StructAccess::Make(
                            value.parse_node(), src_type_id, value_id,
                            SemIR::MemberIndex(i)));
    auto init_id = context.Convert(value.parse_node(), src_id,
                                   {.kind = inner_kind,
                                    .type_id = dest_type_id,
                                    .init_id = target_id,
                                    .init_block = target.init_block});
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return context.AddNode(
      is_init
          ? SemIR::Node::StructInit::Make(value.parse_node(), target.type_id,
                                          value_id, new_block.id())
          : SemIR::Node::StructValue::Make(value.parse_node(), target.type_id,
                                           value_id, new_block.id()));
}

// Returns whether `category` is a valid expression category to produce as a
// result of a conversion with kind `target_kind`, or at most needs a temporary
// to be materialized.
static bool IsValidExpressionCategoryForConversionTarget(
    SemIR::ExpressionCategory category,
    Context::ConversionTarget::Kind target_kind) {
  switch (target_kind) {
    case Context::ConversionTarget::Value:
      return category == SemIR::ExpressionCategory::Value;
    case Context::ConversionTarget::ValueOrReference:
    case Context::ConversionTarget::Discarded:
      return category == SemIR::ExpressionCategory::Value ||
             category == SemIR::ExpressionCategory::DurableReference ||
             category == SemIR::ExpressionCategory::EphemeralReference ||
             category == SemIR::ExpressionCategory::Initializing;
    case Context::ConversionTarget::Initializer:
    case Context::ConversionTarget::FullInitializer:
      return category == SemIR::ExpressionCategory::Initializing;
  }
}

auto Context::PerformBuiltinConversion(Parse::Node parse_node,
                                       SemIR::NodeId value_id,
                                       ConversionTarget target)
    -> SemIR::NodeId {
  auto value = semantics_ir().GetNode(value_id);
  auto value_type_id = value.type_id();
  auto target_type_node = semantics_ir_->GetNode(
      semantics_ir().GetTypeAllowBuiltinTypes(target.type_id));

  // Various forms of implicit conversion are supported as builtin conversions
  // rather than being implemented purely in the Carbon prelude as `impl`s of
  // `ImplicitAs`. There are a few reasons we need to perform some of these
  // conversions as builtins:
  //
  // 1) Conversions from struct and tuple *literals* have special rules, and
  //    cannot be performed in the library. Specifically, we must recurse into
  //    the elements of the literal before performing initialization in order
  //    to avoid unnecessary conversions between expression categories.
  // 2) (Not implemented yet) Conversion of a facet to a facet type depends on
  //    the value of the facet, not only its type.
  // 3) Some of these conversions are used while checking the library
  //    definition of `ImplicitAs` itself or implementations of it.
  //
  // We also expect to see better performance by avoiding an `impl` lookup for
  // common conversions.
  //
  // TODO: We should provide a debugging flag to turn off as many of these
  // builtin conversions as we can so that we can test that they do the same
  // thing as the library implementations.
  //
  // These builtin conversions all correspond to `final impl`s in the library,
  // so we don't need to worry about `ImplicitAs` being specialized in any of
  // these cases.

  // If the value is already of the right kind and expression category, there's
  // nothing to do. Performing a conversion would decompose and rebuild tuples
  // and structs, so it's important that we bail out early in this case.
  if (value_type_id == target.type_id &&
      IsValidExpressionCategoryForConversionTarget(
          SemIR::GetExpressionCategory(semantics_ir(), value_id),
          target.kind)) {
    return value_id;
  }

  // A tuple (T1, T2, ..., Tn) converts to (U1, U2, ..., Un) if each Ti
  // converts to Ui.
  if (target_type_node.kind() == SemIR::NodeKind::TupleType) {
    auto value_type_node = semantics_ir_->GetNode(
        semantics_ir_->GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::TupleType) {
      return ConvertTupleToTuple(*this, value_type_node, target_type_node,
                                 value_id, target);
    }
  }

  // A struct {.f_1: T_1, .f_2: T_2, ..., .f_n: T_n} converts to
  // {.f_p(1): U_p(1), .f_p(2): U_p(2), ..., .f_p(n): U_p(n)} if
  // (p(1), ..., p(n)) is a permutation of (1, ..., n) and each Ti converts
  // to Ui.
  if (target_type_node.kind() == SemIR::NodeKind::StructType) {
    auto value_type_node = semantics_ir_->GetNode(
        semantics_ir_->GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::StructType) {
      return ConvertStructToStruct(*this, value_type_node, target_type_node,
                                   value_id, target);
    }
  }

  // A tuple (T1, T2, ..., Tn) converts to [T; n] if each Ti converts to T.
  if (target_type_node.kind() == SemIR::NodeKind::ArrayType) {
    auto value_type_node = semantics_ir_->GetNode(
        semantics_ir_->GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::TupleType) {
      return ConvertTupleToArray(*this, value_type_node, target_type_node,
                                 value_id, target);
    }
  }

  if (target.type_id == SemIR::TypeId::TypeType) {
    // A tuple of types converts to type `type`.
    // TODO: This should apply even for non-literal tuples.
    if (value.kind() == SemIR::NodeKind::TupleLiteral) {
      auto tuple_block_id = value.GetAsTupleLiteral();
      llvm::SmallVector<SemIR::TypeId> type_ids;
      // If it is empty tuple type, we don't fetch anything.
      if (tuple_block_id != SemIR::NodeBlockId::Empty) {
        const auto& tuple_block = semantics_ir_->GetNodeBlock(tuple_block_id);
        for (auto tuple_node_id : tuple_block) {
          // TODO: This call recurses back to this function. Switch to an
          // iterative approach.
          type_ids.push_back(ExpressionAsType(parse_node, tuple_node_id));
        }
      }
      auto tuple_type_id =
          CanonicalizeTupleType(parse_node, std::move(type_ids));
      return semantics_ir_->GetTypeAllowBuiltinTypes(tuple_type_id);
    }

    // `{}` converts to `{} as type`.
    // TODO: This conversion should also be performed for a non-literal value
    // of type `{}`.
    if (value.kind() == SemIR::NodeKind::StructLiteral &&
        value.GetAsStructLiteral() == SemIR::NodeBlockId::Empty) {
      value_id = semantics_ir_->GetTypeAllowBuiltinTypes(value_type_id);
    }
  }

  // No builtin conversion applies.
  return value_id;
}

auto Context::Convert(Parse::Node parse_node, SemIR::NodeId expr_id,
                      ConversionTarget target) -> SemIR::NodeId {
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (expr_id == SemIR::NodeId::BuiltinError) {
    return expr_id;
  }
  if (semantics_ir().GetNode(expr_id).type_id() == SemIR::TypeId::Error ||
      target.type_id == SemIR::TypeId::Error) {
    return SemIR::NodeId::BuiltinError;
  }

  if (SemIR::GetExpressionCategory(semantics_ir(), expr_id) ==
      SemIR::ExpressionCategory::NotExpression) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    CARBON_DIAGNOSTIC(UseOfNonExpressionAsValue, Error,
                      "Expression cannot be used as a value.");
    emitter().Emit(semantics_ir().GetNode(expr_id).parse_node(),
                   UseOfNonExpressionAsValue);
    return SemIR::NodeId::BuiltinError;
  }

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(parse_node, expr_id, target);
  if (expr_id == SemIR::NodeId::BuiltinError) {
    return expr_id;
  }

  // If the types don't match at this point, we can't perform the conversion.
  // TODO: Look for an ImplicitAs impl.
  SemIR::Node expr = semantics_ir().GetNode(expr_id);
  if (expr.type_id() != target.type_id) {
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      std::string, std::string);
    emitter_
        ->Build(parse_node, ImplicitAsConversionFailure,
                semantics_ir_->StringifyType(expr.type_id()),
                semantics_ir_->StringifyType(target.type_id))
        .Emit();
    return SemIR::NodeId::BuiltinError;
  }

  // Now perform any necessary value category conversions.
  switch (SemIR::GetExpressionCategory(semantics_ir(), expr_id)) {
    case SemIR::ExpressionCategory::NotExpression:
    case SemIR::ExpressionCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression " << expr
                     << " after builtin conversions";

    case SemIR::ExpressionCategory::Initializing:
      if (target.is_initializer()) {
        if (orig_expr_id == expr_id) {
          // Don't fill in the return slot if we created the expression through
          // a conversion. In that case, we will have created it with the
          // target already set.
          // TODO: Find a better way to track whether we need to do this.
          MarkInitializerFor(expr_id, target.init_id, *target.init_block);
        }
        break;
      }

      // Commit to using a temporary for this initializing expression.
      // TODO: Don't create a temporary if the initializing representation
      // is already a value representation.
      expr_id = FinalizeTemporary(expr_id,
                                  target.kind == ConversionTarget::Discarded);
      // We now have an ephemeral reference.
      [[fallthrough]];

    case SemIR::ExpressionCategory::DurableReference:
    case SemIR::ExpressionCategory::EphemeralReference: {
      // If we have a reference and don't want one, form a value binding.
      if (target.kind != ConversionTarget::ValueOrReference &&
          target.kind != ConversionTarget::Discarded) {
        // TODO: Support types with custom value representations.
        expr_id = AddNode(SemIR::Node::BindValue::Make(
            expr.parse_node(), expr.type_id(), expr_id));
      }
      break;
    }

    case SemIR::ExpressionCategory::Value:
      break;
  }

  // Perform a final destination store, if necessary.
  if (target.kind == ConversionTarget::FullInitializer) {
    if (auto init_rep = SemIR::GetInitializingRepresentation(semantics_ir(),
                                                             target.type_id);
        init_rep.kind == SemIR::InitializingRepresentation::ByCopy) {
      target.init_block->InsertHere();
      expr_id = AddNode(SemIR::Node::InitializeFrom::Make(
          parse_node, target.type_id, expr_id, target.init_id));
    }
  }

  return expr_id;
}

auto Context::Initialize(Parse::Node parse_node, SemIR::NodeId target_id,
                         SemIR::NodeId value_id) -> SemIR::NodeId {
  PendingBlock target_block(*this);
  return Convert(parse_node, value_id,
                 {.kind = ConversionTarget::Initializer,
                  .type_id = semantics_ir().GetNode(target_id).type_id(),
                  .init_id = target_id,
                  .init_block = &target_block});
}

// Given an initializing expression, find its return slot. Returns `Invalid` if
// there is no return slot, because the initialization is not performed in
// place.
static auto FindReturnSlotForInitializer(SemIR::File& semantics_ir,
                                         SemIR::NodeId init_id)
    -> SemIR::NodeId {
  SemIR::Node init = semantics_ir.GetNode(init_id);
  switch (init.kind()) {
    default:
      CARBON_FATAL() << "Initialization from unexpected node " << init;

    case SemIR::NodeKind::StructInit:
    case SemIR::NodeKind::TupleInit:
      // TODO: Track a return slot for these initializers.
      CARBON_FATAL() << init
                     << " should be created with its return slot already "
                        "filled in properly";

    case SemIR::NodeKind::InitializeFrom: {
      auto [src_id, dest_id] = init.GetAsInitializeFrom();
      return dest_id;
    }

    case SemIR::NodeKind::Call: {
      auto [refs_id, callee_id] = init.GetAsCall();
      if (!semantics_ir.GetFunction(callee_id).return_slot_id.is_valid()) {
        return SemIR::NodeId::Invalid;
      }
      return semantics_ir.GetNodeBlock(refs_id).back();
    }

    case SemIR::NodeKind::ArrayInit: {
      auto [src_id, refs_id] = init.GetAsArrayInit();
      return semantics_ir.GetNodeBlock(refs_id).back();
    }
  }
}

auto Context::MarkInitializerFor(SemIR::NodeId init_id, SemIR::NodeId target_id,
                                 PendingBlock& target_block) -> void {
  auto return_slot_id = FindReturnSlotForInitializer(semantics_ir(), init_id);
  if (return_slot_id.is_valid()) {
    // Replace the temporary in the return slot with a reference to our target.
    CARBON_CHECK(semantics_ir().GetNode(return_slot_id).kind() ==
                 SemIR::NodeKind::TemporaryStorage)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << semantics_ir().GetNode(return_slot_id);
    target_block.MergeReplacing(return_slot_id, target_id);
  }
}

auto Context::FinalizeTemporary(SemIR::NodeId init_id, bool discarded)
    -> SemIR::NodeId {
  auto return_slot_id = FindReturnSlotForInitializer(semantics_ir(), init_id);
  if (return_slot_id.is_valid()) {
    // The return slot should already have a materialized temporary in it.
    CARBON_CHECK(semantics_ir().GetNode(return_slot_id).kind() ==
                 SemIR::NodeKind::TemporaryStorage)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << semantics_ir().GetNode(return_slot_id);
    auto init = semantics_ir().GetNode(init_id);
    return AddNode(SemIR::Node::Temporary::Make(
        init.parse_node(), init.type_id(), return_slot_id, init_id));
  }

  if (discarded) {
    // Don't invent a temporary that we're going to discard.
    return SemIR::NodeId::Invalid;
  }

  // The initializer has no return slot, but we want to produce a temporary
  // object. Materialize one now.
  // TODO: Consider using an invalid ID to mean that we immediately
  // materialize and initialize a temporary, rather than two separate
  // nodes.
  auto init = semantics_ir().GetNode(init_id);
  auto temporary_id = AddNode(
      SemIR::Node::TemporaryStorage::Make(init.parse_node(), init.type_id()));
  return AddNode(SemIR::Node::Temporary::Make(init.parse_node(), init.type_id(),
                                              temporary_id, init_id));
}

auto Context::HandleDiscardedExpression(SemIR::NodeId expr_id) -> void {
  // If we discard an initializing expression, convert it to a value or
  // reference so that it has something to initialize.
  auto expr = semantics_ir().GetNode(expr_id);
  Convert(expr.parse_node(), expr_id,
          {.kind = ConversionTarget::Discarded, .type_id = expr.type_id()});

  // TODO: This will eventually need to do some "do not discard" analysis.
}

auto Context::ImplicitAsForArgs(Parse::Node call_parse_node,
                                SemIR::NodeBlockId arg_refs_id,
                                Parse::Node param_parse_node,
                                SemIR::NodeBlockId param_refs_id,
                                bool has_return_slot) -> bool {
  // If both arguments and parameters are empty, return quickly. Otherwise,
  // we'll fetch both so that errors are consistent.
  if (arg_refs_id == SemIR::NodeBlockId::Empty &&
      param_refs_id == SemIR::NodeBlockId::Empty) {
    return true;
  }

  auto arg_refs = semantics_ir_->GetNodeBlock(arg_refs_id);
  auto param_refs = semantics_ir_->GetNodeBlock(param_refs_id);

  if (has_return_slot) {
    // There's no entry in the parameter block for the return slot, so ignore
    // the corresponding entry in the argument block.
    // TODO: Consider adding the return slot to the parameter list.
    CARBON_CHECK(!arg_refs.empty()) << "missing return slot";
    arg_refs = arg_refs.drop_back();
  }

  // If sizes mismatch, fail early.
  if (arg_refs.size() != param_refs.size()) {
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                      "{0} argument(s) passed to function expecting "
                      "{1} argument(s).",
                      int, int);
    CARBON_DIAGNOSTIC(InCallToFunction, Note,
                      "Calling function declared here.");
    emitter_
        ->Build(call_parse_node, CallArgCountMismatch, arg_refs.size(),
                param_refs.size())
        .Note(param_parse_node, InCallToFunction)
        .Emit();
    return false;
  }

  if (param_refs.empty()) {
    return true;
  }

  int diag_param_index;
  DiagnosticAnnotationScope annotate_diagnostics(emitter_, [&](auto& builder) {
    CARBON_DIAGNOSTIC(InCallToFunctionParam, Note,
                      "Initializing parameter {0} of function declared here.",
                      int);
    builder.Note(param_parse_node, InCallToFunctionParam, diag_param_index + 1);
  });

  // Check type conversions per-element.
  for (auto [i, value_id, param_ref] : llvm::enumerate(arg_refs, param_refs)) {
    diag_param_index = i;

    auto as_type_id = semantics_ir_->GetNode(param_ref).type_id();
    // TODO: Convert to the proper expression category. For now, we assume
    // parameters are all `let` bindings.
    value_id = ConvertToValueOfType(call_parse_node, value_id, as_type_id);
    if (value_id == SemIR::NodeId::BuiltinError) {
      return false;
    }
    arg_refs[i] = value_id;
  }

  return true;
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
    case SemIR::NodeKind::ArrayType: {
      auto [bound_id, element_type_id] = node.GetAsArrayType();
      canonical_id.AddInteger(
          semantics_context.semantics_ir().GetArrayBoundValue(bound_id));
      canonical_id.AddInteger(element_type_id.index);
      break;
    }
    case SemIR::NodeKind::Builtin:
      canonical_id.AddInteger(node.GetAsBuiltin().AsInt());
      break;
    case SemIR::NodeKind::CrossReference: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto [xref_id, node_id] = node.GetAsCrossReference();
      canonical_id.AddInteger(xref_id.index);
      canonical_id.AddInteger(node_id.index);
      break;
    }
    case SemIR::NodeKind::ConstType:
      canonical_id.AddInteger(
          semantics_context.GetUnqualifiedType(node.GetAsConstType()).index);
      break;
    case SemIR::NodeKind::PointerType:
      canonical_id.AddInteger(node.GetAsPointerType().index);
      break;
    case SemIR::NodeKind::StructType: {
      auto refs =
          semantics_context.semantics_ir().GetNodeBlock(node.GetAsStructType());
      for (const auto& ref_id : refs) {
        auto ref = semantics_context.semantics_ir().GetNode(ref_id);
        auto [name_id, type_id] = ref.GetAsStructTypeField();
        canonical_id.AddInteger(name_id.index);
        canonical_id.AddInteger(type_id.index);
      }
      break;
    }
    case SemIR::NodeKind::TupleType:
      ProfileTupleType(
          semantics_context.semantics_ir().GetTypeBlock(node.GetAsTupleType()),
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
  return CanonicalizeTypeAndAddNodeIfNew(SemIR::Node::StructType::Make(
      parse_node, SemIR::TypeId::TypeType, refs_id));
}

auto Context::CanonicalizeTupleType(Parse::Node parse_node,
                                    llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  // Defer allocating a SemIR::TypeBlockId until we know this is a new type.
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileTupleType(type_ids, canonical_id);
  };
  auto make_tuple_node = [&] {
    return AddNode(
        SemIR::Node::TupleType::Make(parse_node, SemIR::TypeId::TypeType,
                                     semantics_ir_->AddTypeBlock(type_ids)));
  };
  return CanonicalizeTypeImpl(SemIR::NodeKind::TupleType, profile_tuple,
                              make_tuple_node);
}

auto Context::GetPointerType(Parse::Node parse_node,
                             SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return CanonicalizeTypeAndAddNodeIfNew(SemIR::Node::PointerType::Make(
      parse_node, SemIR::TypeId::TypeType, pointee_type_id));
}

auto Context::GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId {
  SemIR::Node type_node =
      semantics_ir_->GetNode(semantics_ir_->GetTypeAllowBuiltinTypes(type_id));
  if (type_node.kind() == SemIR::NodeKind::ConstType) {
    return type_node.GetAsConstType();
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
