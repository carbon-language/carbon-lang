// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check.h"

#include <variant>

#include "common/check.h"
#include "common/error.h"
#include "common/variant_helpers.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/function.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/import.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_node_diagnostic_converter.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Handles the transformation of a SemIRLocation to a DiagnosticLocation.
//
// TODO: Move this to diagnostic_helpers.cpp.
class SemIRDiagnosticConverter : public DiagnosticConverter<SemIRLocation> {
 public:
  explicit SemIRDiagnosticConverter(
      const llvm::DenseMap<const SemIR::File*, Parse::NodeLocationConverter*>*
          node_converters,
      const SemIR::File* sem_ir)
      : node_converters_(node_converters), sem_ir_(sem_ir) {}

  auto ConvertLocation(SemIRLocation loc) const -> DiagnosticLocation override {
    // Parse nodes always refer to the current IR.
    if (!loc.is_inst_id) {
      return ConvertLocationInFile(sem_ir_, loc.node_location);
    }

    const auto* cursor_ir = sem_ir_;
    auto cursor_inst_id = loc.inst_id;
    while (true) {
      // If the parse node is valid, use it for the location.
      if (auto node_id = cursor_ir->insts().GetNodeId(cursor_inst_id);
          node_id.is_valid()) {
        return ConvertLocationInFile(cursor_ir, node_id);
      }

      // If the parse node was invalid, recurse through import references when
      // possible.
      if (auto import_ref = cursor_ir->insts().TryGetAs<SemIR::AnyImportRef>(
              cursor_inst_id)) {
        cursor_ir = cursor_ir->import_irs().Get(import_ref->ir_id);
        cursor_inst_id = import_ref->inst_id;
        continue;
      }

      // If a namespace has an instruction for an import, switch to looking at
      // it.
      if (auto ns =
              cursor_ir->insts().TryGetAs<SemIR::Namespace>(cursor_inst_id)) {
        if (ns->import_id.is_valid()) {
          cursor_inst_id = ns->import_id;
          continue;
        }
      }

      // Invalid parse node but not an import; just nothing to point at.
      return ConvertLocationInFile(cursor_ir, Parse::NodeId::Invalid);
    }
  }

  auto ConvertArg(llvm::Any arg) const -> llvm::Any override {
    if (auto* name_id = llvm::any_cast<SemIR::NameId>(&arg)) {
      return sem_ir_->names().GetFormatted(*name_id).str();
    }
    if (auto* type_id = llvm::any_cast<SemIR::TypeId>(&arg)) {
      return sem_ir_->StringifyType(*type_id);
    }
    if (auto* typed_int = llvm::any_cast<TypedInt>(&arg)) {
      // TODO: Once unsigned integers are supported, compute the signedness
      // here.
      constexpr bool IsUnsigned = false;
      return llvm::APSInt(typed_int->value, IsUnsigned);
    }
    return DiagnosticConverter<SemIRLocation>::ConvertArg(arg);
  }

 private:
  auto ConvertLocationInFile(const SemIR::File* sem_ir,
                             Parse::NodeLocation node_location) const
      -> DiagnosticLocation {
    auto it = node_converters_->find(sem_ir);
    CARBON_CHECK(it != node_converters_->end());
    return it->second->ConvertLocation(node_location);
  }

  const llvm::DenseMap<const SemIR::File*, Parse::NodeLocationConverter*>*
      node_converters_;
  const SemIR::File* sem_ir_;
};

struct UnitInfo {
  // A given import within the file, with its destination.
  struct Import {
    Parse::Tree::PackagingNames names;
    UnitInfo* unit_info;
  };
  // A file's imports corresponding to a single package, for the map.
  struct PackageImports {
    // Use the constructor so that the SmallVector is only constructed
    // as-needed.
    explicit PackageImports(Parse::ImportDirectiveId node_id)
        : node_id(node_id) {}

    // The first `import` directive in the file, which declared the package's
    // identifier (even if the import failed). Used for associating diagnostics
    // not specific to a single import.
    Parse::ImportDirectiveId node_id;
    // Whether there's an import that failed to load.
    bool has_load_error = false;
    // The list of valid imports.
    llvm::SmallVector<Import> imports;
  };

  explicit UnitInfo(Unit& unit)
      : unit(&unit),
        converter(unit.tokens, unit.tokens->source().filename(),
                  unit.parse_tree),
        err_tracker(*unit.consumer),
        emitter(converter, err_tracker) {}

  Unit* unit;

  // Emitter information.
  Parse::NodeLocationConverter converter;
  ErrorTrackingDiagnosticConsumer err_tracker;
  DiagnosticEmitter<Parse::NodeLocation> emitter;

  // A map of package names to outgoing imports. If the
  // import's target isn't available, the unit will be nullptr to assist with
  // name lookup. Invalid imports (for example, `import Main;`) aren't added
  // because they won't add identifiers to name lookup.
  llvm::DenseMap<IdentifierId, PackageImports> package_imports_map;

  // The remaining number of imports which must be checked before this unit can
  // be processed.
  int32_t imports_remaining = 0;

  // A list of incoming imports. This will be empty for `impl` files, because
  // imports only touch `api` files.
  llvm::SmallVector<UnitInfo*> incoming_imports;
};

// Add imports to the root block.
static auto InitPackageScopeAndImports(Context& context, UnitInfo& unit_info)
    -> void {
  // First create the constant values map for all imported IRs. We'll populate
  // these with mappings for namespaces as we go.
  size_t num_irs = context.import_irs().size();
  for (auto& [_, package_imports] : unit_info.package_imports_map) {
    num_irs += package_imports.imports.size();
  }
  context.import_ir_constant_values().resize(
      num_irs, SemIR::ConstantValueStore(SemIR::ConstantId::Invalid));

  // Importing makes many namespaces, so only canonicalize the type once.
  auto namespace_type_id =
      context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType);

  // Define the package scope, with an instruction for `package` expressions to
  // reference.
  auto package_scope_id = context.name_scopes().Add(
      SemIR::InstId::PackageNamespace, SemIR::NameId::PackageNamespace,
      SemIR::NameScopeId::Invalid);
  CARBON_CHECK(package_scope_id == SemIR::NameScopeId::Package);

  auto package_inst_id = context.AddInst(
      {Parse::NodeId::Invalid,
       SemIR::Namespace{namespace_type_id, SemIR::NameScopeId::Package,
                        SemIR::InstId::Invalid}});
  CARBON_CHECK(package_inst_id == SemIR::InstId::PackageNamespace);

  // Add imports from the current package.
  auto self_import = unit_info.package_imports_map.find(IdentifierId::Invalid);
  if (self_import != unit_info.package_imports_map.end()) {
    bool error_in_import = self_import->second.has_load_error;
    for (const auto& import : self_import->second.imports) {
      const auto& import_sem_ir = **import.unit_info->unit->sem_ir;
      ImportLibraryFromCurrentPackage(context, namespace_type_id,
                                      import_sem_ir);
      error_in_import |= import_sem_ir.name_scopes()
                             .Get(SemIR::NameScopeId::Package)
                             .has_error;
    }

    // If an import of the current package caused an error for the imported
    // file, it transitively affects the current file too.
    if (error_in_import) {
      context.name_scopes().Get(SemIR::NameScopeId::Package).has_error = true;
    }
    context.scope_stack().Push(package_inst_id, SemIR::NameScopeId::Package,
                               error_in_import);
  } else {
    // Push the scope; there are no names to add.
    context.scope_stack().Push(package_inst_id, SemIR::NameScopeId::Package);
  }
  CARBON_CHECK(context.scope_stack().PeekIndex() == ScopeIndex::Package);

  for (auto& [package_id, package_imports] : unit_info.package_imports_map) {
    if (!package_id.is_valid()) {
      // Current package is handled above.
      continue;
    }

    llvm::SmallVector<const SemIR::File*> sem_irs;
    for (auto import : package_imports.imports) {
      sem_irs.push_back(&**import.unit_info->unit->sem_ir);
    }
    ImportLibrariesFromOtherPackage(context, namespace_type_id,
                                    package_imports.node_id, package_id,
                                    sem_irs, package_imports.has_load_error);
  }

  CARBON_CHECK(context.import_irs().size() == num_irs)
      << "Created an unexpected number of IRs";
}

namespace {
// State used to track the next inline method body that we will encounter and
// need to reorder.
class NextInlineMethodCache {
 public:
  explicit NextInlineMethodCache(const Parse::Tree* tree) : tree_(tree) {
    SkipTo(Parse::InlineMethodIndex(0));
  }

  // Set the specified inline method index as being the next one to parse.
  auto SkipTo(Parse::InlineMethodIndex next_index) -> void {
    index_ = next_index;
    if (static_cast<std::size_t>(index_.index) ==
        tree_->inline_methods().size()) {
      start_id_ = Parse::NodeId::Invalid;
    } else {
      start_id_ = tree_->inline_methods().Get(index_).start_id;
    }
  }

  // Returns the index of the next inline method to be parsed.
  auto index() const -> Parse::InlineMethodIndex { return index_; }

  // Returns the ID of the start node of the next inline method.
  auto start_id() const -> Parse::NodeId { return start_id_; }

 private:
  const Parse::Tree* tree_;
  Parse::InlineMethodIndex index_ = Parse::InlineMethodIndex::Invalid;
  Parse::NodeId start_id_ = Parse::NodeId::Invalid;
};
}  // namespace

// Determines whether we are currently declaring a name in a scope in which
// method definitions are delayed. When entering another delayed method scope,
// the inner declaration's methods are parsed at the end of the outer
// declaration, not the inner one.  For example:
//
// ```
// class A {
//   class B {
//     fn F() -> A { return {}; }
//   }
// } // A.B.F is parsed here, with A complete.
//
// fn F() {
//   class C {
//     fn G() {}
//   } // C.G is parsed here.
// }
// ```
static auto IsInDelayedMethodScope(Context& context) -> bool {
  auto inst_id = context.name_scopes().GetInstIdIfValid(
      context.decl_name_stack().PeekTargetScope());
  if (!inst_id.is_valid()) {
    return false;
  }
  switch (context.insts().Get(inst_id).kind()) {
    case SemIR::ClassDecl::Kind:
    case SemIR::ImplDecl::Kind:
    case SemIR::InterfaceDecl::Kind:
      // TODO: Named constraints, mixins.
      return true;

    default:
      return false;
  }
}

// Determines whether this node kind is the start of a delayed method scope.
static auto IsStartOfDelayedMethodScope(Parse::NodeKind kind) -> bool {
  switch (kind) {
    case Parse::NodeKind::ClassDefinitionStart:
    case Parse::NodeKind::ImplDefinitionStart:
    case Parse::NodeKind::InterfaceDefinitionStart:
    case Parse::NodeKind::NamedConstraintDefinitionStart:
      // TODO: Mixins.
      return true;
    default:
      return false;
  }
}

// Determines whether this node kind is the end of a delayed method scope.
static auto IsEndOfDelayedMethodScope(Parse::NodeKind kind) -> bool {
  switch (kind) {
    case Parse::NodeKind::ClassDefinition:
    case Parse::NodeKind::ImplDefinition:
    case Parse::NodeKind::InterfaceDefinition:
    case Parse::NodeKind::NamedConstraintDefinition:
      // TODO: Mixins.
      return true;
    default:
      return false;
  }
}

namespace {
// A worklist of pending tasks to perform to parse skipped inline method
// bodies.
class InlineMethodWorklist {
 public:
  // A worklist task that indicates we should parse a skipped method.
  struct ParseSkippedMethod {
    // The method that we skipped.
    Parse::InlineMethodIndex method_index;
    // The suspended function. Only present when `method_index` is valid.
    std::optional<SuspendedFunction> sus_fn;
  };

  // A worklist task that indicates we should enter a nested method scope, which
  // is a scope like a class or interface that can have skipped method bodies.
  struct EnterMethodScope {
    // The suspended scope. This is only set once we reach the end of the scope.
    std::optional<DeclNameStack::SuspendedName> sus_name;
    // Whether this scope is itself within a method scope. If so, we'll delay
    // processing its contents until we reach the end of the enclosing scope.
    bool in_method_scope;
  };

  // A worklist task that indicates we should leave a method scope.
  struct LeaveMethodScope {
    // Whether this scope is itself within a method scope.
    bool in_method_scope;
  };

  // A pending parsing task.
  using Task =
      std::variant<ParseSkippedMethod, EnterMethodScope, LeaveMethodScope>;

  InlineMethodWorklist() { worklist_.reserve(64); }

  // Suspend the current function definition and push a task onto the worklist
  // to finish it later.
  auto SuspendFunctionAndPush(Context& context, Parse::InlineMethodIndex index,
                              Parse::FunctionDefinitionStartId node_id)
      -> void {
    worklist_.push_back(ParseSkippedMethod{
        index, HandleFunctionDefinitionSuspend(context, node_id)});
  }

  // Push a task to re-enter a method scope, so that functions defined within it
  // are parsed in the right context.
  auto PushEnterMethodScope(Context& context) -> void {
    method_scopes_.push_back(worklist_.size());
    worklist_.push_back(
        EnterMethodScope{std::nullopt, IsInDelayedMethodScope(context)});
  }

  // Suspend the current method scope, which is finished but still on the
  // decl_name_stack, and push a task to leave the scope when we're parsing
  // inline methods. Returns `true` if inline methods should be parsed
  // immediately.
  auto SuspendFinishedScopeAndPush(Context& context) -> bool {
    auto method_scope_index = method_scopes_.pop_back_val();

    // If we've not found any inline method definitions in this scope, clean up
    // the stack.
    if (method_scope_index == worklist_.size() - 1) {
      context.decl_name_stack().PopScope();
      worklist_.pop_back();
      return false;
    }

    // If we're in a nested method scope, keep track of its scope but don't
    // parse methods now.
    auto& enter_scope = get<EnterMethodScope>(worklist_[method_scope_index]);
    if (enter_scope.in_method_scope) {
      // This is a nested method scope. Suspend the inner scope so we can
      // restore it when we come to parse the method bodies.
      enter_scope.sus_name = context.decl_name_stack().Suspend();

      // Enqueue a task to leave the nested scope.
      worklist_.push_back(LeaveMethodScope{.in_method_scope = true});
      return false;
    }

    // We're at the end of a non-nested method scope. Prepare to start parsing
    // inline methods. Enqueue a task to leave this outer scope and end parsing
    // inline methods.
    worklist_.push_back(LeaveMethodScope{.in_method_scope = false});

    // We'll process the worklist in reverse index order, so reverse the part of
    // it we're about to execute so we run our tasks in the order in which they
    // were pushed.
    std::reverse(worklist_.begin() + method_scope_index, worklist_.end());

    // Pop the `EnterMethodScope` that's now on the end of the worklist.
    // We stay in that scope rather than suspending then immediately
    // resuming it.
    CARBON_CHECK(holds_alternative<EnterMethodScope>(worklist_.back()))
        << "Unexpected task in worklist.";
    worklist_.pop_back();
    return true;
  }

  // Pop the next task off the worklist.
  auto Pop() -> Task { return worklist_.pop_back_val(); }

  // CHECK that the work list has no further work.
  auto VerifyEmpty() {
    CARBON_CHECK(worklist_.empty() && method_scopes_.empty())
        << "Tasks left behind on worklist.";
  }

 private:
  // A worklist of parsing tasks we'll need to do later.
  // Don't allocate any inline storage here. A Task is fairly large, so we never
  // want this to live on the stack. Instead, we reserve space in the
  // constructor for a fairly large number of inline method definitions.
  llvm::SmallVector<Task, 0> worklist_;

  // Indexes in `worklist` of method scopes that are currently still open.
  llvm::SmallVector<size_t> method_scopes_;
};
}  // namespace

namespace {
// A traversal of the node IDs in the parse tree, in the order in which we need
// to check them.
class NodeIdTraversal {
 public:
  explicit NodeIdTraversal(Context& context)
      : context_(context), next_inline_method_(&context.parse_tree()) {
    chunks_.push_back({.it = context.parse_tree().postorder().begin(),
                       .end = context.parse_tree().postorder().end(),
                       .next_method = Parse::InlineMethodIndex::Invalid});
  }

  // Finds the next `NodeId` to parse. Returns nullopt if the traversal is
  // complete.
  auto Next() -> std::optional<Parse::NodeId> {
    while (true) {
      // If we're parsing skipped methods, find the next method we're parsing,
      // restore the suspended state, and add a corresponding `Chunk` to the top
      // of the chunk list.
      if (chunks_.back().parsing_skipped_methods) {
        VariantMatch(
            worklist_.Pop(),
            // Entering a nested class.
            [&](InlineMethodWorklist::EnterMethodScope&& enter) {
              CARBON_CHECK(enter.sus_name)
                  << "Entering a scope with no suspension information.";
              context_.decl_name_stack().Restore(std::move(*enter.sus_name));
            },
            // Leaving a nested class or the top-level class.
            [&](InlineMethodWorklist::LeaveMethodScope&& leave) {
              if (!leave.in_method_scope) {
                // We're done with parsing skipped methods.
                chunks_.back().parsing_skipped_methods = false;
              }
              context_.decl_name_stack().PopScope();
            },
            // Resume parsing this method.
            [&](InlineMethodWorklist::ParseSkippedMethod&& parse_method) {
              auto& [method_index, sus_fn] = parse_method;
              const auto& method =
                  context_.parse_tree().inline_methods().Get(method_index);
              HandleFunctionDefinitionResume(context_, method.start_id,
                                             std::move(*sus_fn));
              chunks_.push_back(
                  {.it = context_.parse_tree().postorder(method.start_id).end(),
                   .end = context_.parse_tree()
                              .postorder(method.definition_id)
                              .end(),
                   .next_method = next_inline_method_.index()});
              ++method_index.index;
              next_inline_method_.SkipTo(method_index);
            });
        continue;
      }

      // If we're not parsing skipped methods, produce the next parse node for
      // this chunk. If we've run out of parse nodes, we're done with this chunk
      // of the parse tree.
      if (chunks_.back().it == chunks_.back().end) {
        auto old_chunk = chunks_.pop_back_val();

        // If we're out of chunks, then we're done entirely.
        if (chunks_.empty()) {
          worklist_.VerifyEmpty();
          return std::nullopt;
        }

        next_inline_method_.SkipTo(old_chunk.next_method);
        continue;
      }

      auto node_id = *chunks_.back().it;

      // If we've reached the start of an inline method, skip to the end of it,
      // and track that we need to parse it later.
      if (node_id == next_inline_method_.start_id()) {
        const auto& method = context_.parse_tree().inline_methods().Get(
            next_inline_method_.index());
        worklist_.SuspendFunctionAndPush(context_, next_inline_method_.index(),
                                         method.start_id);

        // Continue parsing after the end of the definition.
        chunks_.back().it =
            context_.parse_tree().postorder(method.definition_id).end();
        next_inline_method_.SkipTo(method.next_method_index);
        continue;
      }

      ++chunks_.back().it;
      return node_id;
    }
  }

  // Performs any processing necessary before handling a node.
  auto BeforeHandle(Parse::NodeKind parse_kind) -> void {
    // When we reach the start of a delayed method scope, add a task to the
    // worklist to parse future skipped methods in the new context.
    if (IsStartOfDelayedMethodScope(parse_kind)) {
      worklist_.PushEnterMethodScope(context_);
    }
  }

  // Performs any processing necessary after handling a node.
  auto AfterHandle(Parse::NodeKind parse_kind) -> void {
    // When we reach the end of a delayed method scope, add a task to the
    // worklist to leave the scope. If this is not a nested scope, start parsing
    // the skipped methods now.
    if (IsEndOfDelayedMethodScope(parse_kind)) {
      if (worklist_.SuspendFinishedScopeAndPush(context_)) {
        chunks_.back().parsing_skipped_methods = true;
      }
    }
  }

 private:
  // A chunk of the parse tree that we need to parse.
  struct Chunk {
    Parse::Tree::PostorderIterator it;
    Parse::Tree::PostorderIterator end;
    // The next method to parse after this chunk completes.
    Parse::InlineMethodIndex next_method;
    // Whether we are currently parsing skipped methods, rather than the tokens
    // of this chunk. If so, we'll pull tasks off `worklist` and execute them
    // until we're done with this batch of skipped methods. Otherwise, we'll
    // pull node IDs from `*it` until it reaches `end`.
    bool parsing_skipped_methods = false;
  };

  Context& context_;
  NextInlineMethodCache next_inline_method_;
  InlineMethodWorklist worklist_;
  llvm::SmallVector<Chunk> chunks_;
};
}  // namespace

// Loops over all nodes in the tree. On some errors, this may return early,
// for example if an unrecoverable state is encountered.
// NOLINTNEXTLINE(readability-function-size)
static auto ProcessNodeIds(Context& context,
                           ErrorTrackingDiagnosticConsumer& err_tracker)
    -> bool {
  NodeIdTraversal traversal(context);

  while (auto maybe_node_id = traversal.Next()) {
    auto node_id = *maybe_node_id;
    auto parse_kind = context.parse_tree().node_kind(node_id);

    traversal.BeforeHandle(parse_kind);

    switch (parse_kind) {
#define CARBON_PARSE_NODE_KIND(Name)                                         \
  case Parse::NodeKind::Name: {                                              \
    if (!Check::Handle##Name(context, Parse::Name##Id(node_id))) {           \
      CARBON_CHECK(err_tracker.seen_error())                                 \
          << "Handle" #Name " returned false without printing a diagnostic"; \
      return false;                                                          \
    }                                                                        \
    break;                                                                   \
  }
#include "toolchain/parse/node_kind.def"
    }

    traversal.AfterHandle(parse_kind);
  }
  return true;
}

// Produces and checks the IR for the provided Parse::Tree.
static auto CheckParseTree(
    llvm::DenseMap<const SemIR::File*, Parse::NodeLocationConverter*>*
        node_converters,
    const SemIR::File& builtin_ir, UnitInfo& unit_info,
    llvm::raw_ostream* vlog_stream) -> void {
  unit_info.unit->sem_ir->emplace(
      *unit_info.unit->value_stores,
      unit_info.unit->tokens->source().filename().str(), &builtin_ir);

  // For ease-of-access.
  SemIR::File& sem_ir = **unit_info.unit->sem_ir;
  CARBON_CHECK(node_converters->insert({&sem_ir, &unit_info.converter}).second);

  SemIRDiagnosticConverter converter(node_converters, &sem_ir);
  Context::DiagnosticEmitter emitter(converter, unit_info.err_tracker);
  Context context(*unit_info.unit->tokens, emitter, *unit_info.unit->parse_tree,
                  sem_ir, vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the file.
  context.inst_block_stack().Push();

  InitPackageScopeAndImports(context, unit_info);

  if (!ProcessNodeIds(context, unit_info.err_tracker)) {
    context.sem_ir().set_has_errors(true);
    return;
  }

  // Pop information for the file-level scope.
  sem_ir.set_top_inst_block_id(context.inst_block_stack().Pop());
  context.scope_stack().Pop();
  context.FinalizeExports();
  context.FinalizeGlobalInit();

  context.VerifyOnFinish();

  sem_ir.set_has_errors(unit_info.err_tracker.seen_error());

#ifndef NDEBUG
  if (auto verify = sem_ir.Verify(); !verify.ok()) {
    CARBON_FATAL() << sem_ir << "Built invalid semantics IR: " << verify.error()
                   << "\n";
  }
#endif
}

// The package and library names, used as map keys.
using ImportKey = std::pair<llvm::StringRef, llvm::StringRef>;

// Returns a key form of the package object. file_package_id is only used for
// imports, not the main package directive; as a consequence, it will be invalid
// for the main package directive.
static auto GetImportKey(UnitInfo& unit_info, IdentifierId file_package_id,
                         Parse::Tree::PackagingNames names) -> ImportKey {
  auto* stores = unit_info.unit->value_stores;
  llvm::StringRef package_name =
      names.package_id.is_valid()  ? stores->identifiers().Get(names.package_id)
      : file_package_id.is_valid() ? stores->identifiers().Get(file_package_id)
                                   : "";
  llvm::StringRef library_name =
      names.library_id.is_valid()
          ? stores->string_literal_values().Get(names.library_id)
          : "";
  return {package_name, library_name};
}

static constexpr llvm::StringLiteral ExplicitMainName = "Main";

// Marks an import as required on both the source and target file.
//
// The ID comparisons between the import and unit are okay because they both
// come from the same file.
static auto TrackImport(
    llvm::DenseMap<ImportKey, UnitInfo*>& api_map,
    llvm::DenseMap<ImportKey, Parse::NodeId>* explicit_import_map,
    UnitInfo& unit_info, Parse::Tree::PackagingNames import) -> void {
  const auto& packaging = unit_info.unit->parse_tree->packaging_directive();

  IdentifierId file_package_id =
      packaging ? packaging->names.package_id : IdentifierId::Invalid;
  auto import_key = GetImportKey(unit_info, file_package_id, import);

  // True if the import has `Main` as the package name, even if it comes from
  // the file's packaging (diagnostics may differentiate).
  bool is_explicit_main = import_key.first == ExplicitMainName;

  // Explicit imports need more validation than implicit ones. We try to do
  // these in an order of imports that should be removed, followed by imports
  // that might be valid with syntax fixes.
  if (explicit_import_map) {
    // Diagnose redundant imports.
    if (auto [insert_it, success] =
            explicit_import_map->insert({import_key, import.node_id});
        !success) {
      CARBON_DIAGNOSTIC(RepeatedImport, Error,
                        "Library imported more than once.");
      CARBON_DIAGNOSTIC(FirstImported, Note, "First import here.");
      unit_info.emitter.Build(import.node_id, RepeatedImport)
          .Note(insert_it->second, FirstImported)
          .Emit();
      return;
    }

    // True if the file's package is implicitly `Main` (by omitting an explicit
    // package name).
    bool is_file_implicit_main =
        !packaging || !packaging->names.package_id.is_valid();
    // True if the import is using implicit "current package" syntax (by
    // omitting an explicit package name).
    bool is_import_implicit_current_package = !import.package_id.is_valid();
    // True if the import is using `default` library syntax.
    bool is_import_default_library = !import.library_id.is_valid();
    // True if the import and file point at the same package, even by
    // incorrectly specifying the current package name to `import`.
    bool is_same_package = is_import_implicit_current_package ||
                           import.package_id == file_package_id;
    // True if the import points at the same library as the file's library.
    bool is_same_library =
        is_same_package &&
        (packaging ? import.library_id == packaging->names.library_id
                   : is_import_default_library);

    // Diagnose explicit imports of the same library, whether from `api` or
    // `impl`.
    if (is_same_library) {
      CARBON_DIAGNOSTIC(ExplicitImportApi, Error,
                        "Explicit import of `api` from `impl` file is "
                        "redundant with implicit import.");
      CARBON_DIAGNOSTIC(ImportSelf, Error, "File cannot import itself.");
      bool is_impl =
          !packaging || packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl;
      unit_info.emitter.Emit(import.node_id,
                             is_impl ? ExplicitImportApi : ImportSelf);
      return;
    }

    // Diagnose explicit imports of `Main//default`. There is no `api` for it.
    // This lets other diagnostics handle explicit `Main` package naming.
    if (is_file_implicit_main && is_import_implicit_current_package &&
        is_import_default_library) {
      CARBON_DIAGNOSTIC(ImportMainDefaultLibrary, Error,
                        "Cannot import `Main//default`.");
      unit_info.emitter.Emit(import.node_id, ImportMainDefaultLibrary);

      return;
    }

    if (!is_import_implicit_current_package) {
      // Diagnose explicit imports of the same package that use the package
      // name.
      if (is_same_package || (is_file_implicit_main && is_explicit_main)) {
        CARBON_DIAGNOSTIC(
            ImportCurrentPackageByName, Error,
            "Imports from the current package must omit the package name.");
        unit_info.emitter.Emit(import.node_id, ImportCurrentPackageByName);
        return;
      }

      // Diagnose explicit imports from `Main`.
      if (is_explicit_main) {
        CARBON_DIAGNOSTIC(ImportMainPackage, Error,
                          "Cannot import `Main` from other packages.");
        unit_info.emitter.Emit(import.node_id, ImportMainPackage);
        return;
      }
    }
  } else if (is_explicit_main) {
    // An implicit import with an explicit `Main` occurs when a `package` rule
    // has bad syntax, which will have been diagnosed when building the API map.
    // As a consequence, we return silently.
    return;
  }

  // Get the package imports.
  auto package_imports_it = unit_info.package_imports_map
                                .try_emplace(import.package_id, import.node_id)
                                .first;

  if (auto api = api_map.find(import_key); api != api_map.end()) {
    // Add references between the file and imported api.
    package_imports_it->second.imports.push_back({import, api->second});
    ++unit_info.imports_remaining;
    api->second->incoming_imports.push_back(&unit_info);
  } else {
    // The imported api is missing.
    package_imports_it->second.has_load_error = true;
    CARBON_DIAGNOSTIC(LibraryApiNotFound, Error,
                      "Corresponding API not found.");
    CARBON_DIAGNOSTIC(ImportNotFound, Error, "Imported API not found.");
    unit_info.emitter.Emit(import.node_id, explicit_import_map
                                               ? ImportNotFound
                                               : LibraryApiNotFound);
  }
}

// Builds a map of `api` files which might be imported. Also diagnoses issues
// related to the packaging because the strings are loaded as part of getting
// the ImportKey (which we then do for `impl` files too).
static auto BuildApiMapAndDiagnosePackaging(
    llvm::SmallVector<UnitInfo, 0>& unit_infos)
    -> llvm::DenseMap<ImportKey, UnitInfo*> {
  llvm::DenseMap<ImportKey, UnitInfo*> api_map;
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.unit->parse_tree->packaging_directive();
    // An import key formed from the `package` or `library` directive. Or, for
    // Main//default, a placeholder key.
    auto import_key = packaging ? GetImportKey(unit_info, IdentifierId::Invalid,
                                               packaging->names)
                                // Construct a boring key for Main//default.
                                : ImportKey{"", ""};

    // Diagnose explicit `Main` uses before they become marked as possible
    // APIs.
    if (import_key.first == ExplicitMainName) {
      CARBON_DIAGNOSTIC(ExplicitMainPackage, Error,
                        "`Main//default` must omit `package` directive.");
      CARBON_DIAGNOSTIC(ExplicitMainLibrary, Error,
                        "Use `library` directive in `Main` package libraries.");
      unit_info.emitter.Emit(packaging->names.node_id,
                             import_key.second.empty() ? ExplicitMainPackage
                                                       : ExplicitMainLibrary);
      continue;
    }

    bool is_impl =
        packaging && packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl;

    // Add to the `api` map and diagnose duplicates. This occurs before the
    // file extension check because we might emit both diagnostics in situations
    // where the user forgets (or has syntax errors with) a package line
    // multiple times.
    if (!is_impl) {
      auto [entry, success] = api_map.insert({import_key, &unit_info});
      if (!success) {
        llvm::StringRef prev_filename =
            entry->second->unit->tokens->source().filename();
        if (packaging) {
          CARBON_DIAGNOSTIC(DuplicateLibraryApi, Error,
                            "Library's API previously provided by `{0}`.",
                            std::string);
          unit_info.emitter.Emit(packaging->names.node_id, DuplicateLibraryApi,
                                 prev_filename.str());
        } else {
          CARBON_DIAGNOSTIC(DuplicateMainApi, Error,
                            "Main//default previously provided by `{0}`.",
                            std::string);
          // Use the invalid node because there's no node to associate with.
          unit_info.emitter.Emit(Parse::NodeId::Invalid, DuplicateMainApi,
                                 prev_filename.str());
        }
      }
    }

    // Validate file extensions. Note imports rely the packaging directive, not
    // the extension. If the input is not a regular file, for example because it
    // is stdin, no filename checking is performed.
    if (unit_info.unit->tokens->source().is_regular_file()) {
      auto filename = unit_info.unit->tokens->source().filename();
      static constexpr llvm::StringLiteral ApiExt = ".carbon";
      static constexpr llvm::StringLiteral ImplExt = ".impl.carbon";
      bool is_api_with_impl_ext = !is_impl && filename.ends_with(ImplExt);
      auto want_ext = is_impl ? ImplExt : ApiExt;
      if (is_api_with_impl_ext || !filename.ends_with(want_ext)) {
        CARBON_DIAGNOSTIC(IncorrectExtension, Error,
                          "File extension of `{0}` required for `{1}`.",
                          llvm::StringLiteral, Lex::TokenKind);
        auto diag = unit_info.emitter.Build(
            packaging ? packaging->names.node_id : Parse::NodeId::Invalid,
            IncorrectExtension, want_ext,
            is_impl ? Lex::TokenKind::Impl : Lex::TokenKind::Api);
        if (is_api_with_impl_ext) {
          CARBON_DIAGNOSTIC(IncorrectExtensionImplNote, Note,
                            "File extension of `{0}` only allowed for `{1}`.",
                            llvm::StringLiteral, Lex::TokenKind);
          diag.Note(Parse::NodeId::Invalid, IncorrectExtensionImplNote, ImplExt,
                    Lex::TokenKind::Impl);
        }
        diag.Emit();
      }
    }
  }
  return api_map;
}

auto CheckParseTrees(const SemIR::File& builtin_ir,
                     llvm::MutableArrayRef<Unit> units,
                     llvm::raw_ostream* vlog_stream) -> void {
  // Prepare diagnostic emitters in case we run into issues during package
  // checking.
  //
  // UnitInfo is big due to its SmallVectors, so we default to 0 on the stack.
  llvm::SmallVector<UnitInfo, 0> unit_infos;
  unit_infos.reserve(units.size());
  for (auto& unit : units) {
    unit_infos.emplace_back(unit);
  }

  llvm::DenseMap<ImportKey, UnitInfo*> api_map =
      BuildApiMapAndDiagnosePackaging(unit_infos);

  // Mark down imports for all files.
  llvm::SmallVector<UnitInfo*> ready_to_check;
  ready_to_check.reserve(units.size());
  for (auto& unit_info : unit_infos) {
    if (const auto& packaging =
            unit_info.unit->parse_tree->packaging_directive()) {
      if (packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl) {
        // An `impl` has an implicit import of its `api`.
        auto implicit_names = packaging->names;
        implicit_names.package_id = IdentifierId::Invalid;
        TrackImport(api_map, nullptr, unit_info, implicit_names);
      }
    }

    llvm::DenseMap<ImportKey, Parse::NodeId> explicit_import_map;
    for (const auto& import : unit_info.unit->parse_tree->imports()) {
      TrackImport(api_map, &explicit_import_map, unit_info, import);
    }

    // If there were no imports, mark the file as ready to check for below.
    if (unit_info.imports_remaining == 0) {
      ready_to_check.push_back(&unit_info);
    }
  }

  llvm::DenseMap<const SemIR::File*, Parse::NodeLocationConverter*>
      node_converters;

  // Check everything with no dependencies. Earlier entries with dependencies
  // will be checked as soon as all their dependencies have been checked.
  for (int check_index = 0;
       check_index < static_cast<int>(ready_to_check.size()); ++check_index) {
    auto* unit_info = ready_to_check[check_index];
    CheckParseTree(&node_converters, builtin_ir, *unit_info, vlog_stream);
    for (auto* incoming_import : unit_info->incoming_imports) {
      --incoming_import->imports_remaining;
      if (incoming_import->imports_remaining == 0) {
        ready_to_check.push_back(incoming_import);
      }
    }
  }

  // If there are still units with remaining imports, it means there's a
  // dependency loop.
  if (ready_to_check.size() < unit_infos.size()) {
    // Go through units and mask out unevaluated imports. This breaks everything
    // associated with a loop equivalently, whether it's part of it or depending
    // on a part of it.
    // TODO: Better identify cycles, maybe try to untangle them.
    for (auto& unit_info : unit_infos) {
      if (unit_info.imports_remaining > 0) {
        for (auto& [package_id, package_imports] :
             unit_info.package_imports_map) {
          for (auto* import_it = package_imports.imports.begin();
               import_it != package_imports.imports.end();) {
            if (*import_it->unit_info->unit->sem_ir) {
              // The import is checked, so continue.
              ++import_it;
            } else {
              // The import hasn't been checked, indicating a cycle.
              CARBON_DIAGNOSTIC(ImportCycleDetected, Error,
                                "Import cannot be used due to a cycle. Cycle "
                                "must be fixed to import.");
              unit_info.emitter.Emit(import_it->names.node_id,
                                     ImportCycleDetected);
              // Make this look the same as an import which wasn't found.
              package_imports.has_load_error = true;
              import_it = package_imports.imports.erase(import_it);
            }
          }
        }
      }
    }

    // Check the remaining file contents, which are probably broken due to
    // incomplete imports.
    for (auto& unit_info : unit_infos) {
      if (unit_info.imports_remaining > 0) {
        CheckParseTree(&node_converters, builtin_ir, unit_info, vlog_stream);
      }
    }
  }
}

}  // namespace Carbon::Check
