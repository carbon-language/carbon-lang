// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter.h"

#include "common/ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

namespace {
// Assigns names to instructions, blocks, and scopes in the Semantics IR.
//
// TODOs / future work ideas:
// - Add a documentation file for the textual format and link to the
//   naming section here.
// - Consider representing literals as just `literal` in the IR and using the
//   type to distinguish.
class InstNamer {
 public:
  // int32_t matches the input value size.
  // NOLINTNEXTLINE(performance-enum-size)
  enum class ScopeId : int32_t {
    None = -1,
    File = 0,
    ImportRef = 1,
    Constants = 2,
    FirstFunction = 3,
  };
  static_assert(sizeof(ScopeId) == sizeof(FunctionId));

  struct NumberOfScopesTag {};

  InstNamer(const Lex::TokenizedBuffer& tokenized_buffer,
            const Parse::Tree& parse_tree, const File& sem_ir)
      : tokenized_buffer_(tokenized_buffer),
        parse_tree_(parse_tree),
        sem_ir_(sem_ir) {
    insts.resize(sem_ir.insts().size());
    labels.resize(sem_ir.inst_blocks().size());
    scopes.resize(static_cast<size_t>(GetScopeFor(NumberOfScopesTag())));

    // Build the constants scope.
    GetScopeInfo(ScopeId::Constants).name =
        globals.AddNameUnchecked("constants");
    CollectNamesInBlock(ScopeId::Constants, sem_ir.constants().GetAsVector());

    // Build the file scope.
    GetScopeInfo(ScopeId::File).name = globals.AddNameUnchecked("file");
    CollectNamesInBlock(ScopeId::File, sem_ir.top_inst_block_id());

    // Build the imports scope, used only by import-related instructions without
    // a block.
    // TODO: Consider other approaches for ImportRef constant formatting, as the
    // actual source of these remains unclear even though they're referenced in
    // constants.
    GetScopeInfo(ScopeId::ImportRef).name = globals.AddNameUnchecked("imports");

    // Build each function scope.
    for (auto [i, fn] : llvm::enumerate(sem_ir.functions().array_ref())) {
      auto fn_id = FunctionId(i);
      auto fn_scope = GetScopeFor(fn_id);
      // TODO: Provide a location for the function for use as a
      // disambiguator.
      auto fn_loc = Parse::NodeId::Invalid;
      GetScopeInfo(fn_scope).name = globals.AllocateName(
          *this, fn_loc, sem_ir.names().GetIRBaseName(fn.name_id).str());
      CollectNamesInBlock(fn_scope, fn.implicit_param_refs_id);
      CollectNamesInBlock(fn_scope, fn.param_refs_id);
      if (fn.return_slot_id.is_valid()) {
        insts[fn.return_slot_id.index] = {
            fn_scope,
            GetScopeInfo(fn_scope).insts.AllocateName(
                *this, sem_ir.insts().GetLocId(fn.return_slot_id), "return")};
      }
      if (!fn.body_block_ids.empty()) {
        AddBlockLabel(fn_scope, fn.body_block_ids.front(), "entry", fn_loc);
      }
      for (auto block_id : fn.body_block_ids) {
        CollectNamesInBlock(fn_scope, block_id);
      }
      for (auto block_id : fn.body_block_ids) {
        AddBlockLabel(fn_scope, block_id);
      }
    }

    // Build each class scope.
    for (auto [i, class_info] : llvm::enumerate(sem_ir.classes().array_ref())) {
      auto class_id = ClassId(i);
      auto class_scope = GetScopeFor(class_id);
      // TODO: Provide a location for the class for use as a disambiguator.
      auto class_loc = Parse::NodeId::Invalid;
      GetScopeInfo(class_scope).name = globals.AllocateName(
          *this, class_loc,
          sem_ir.names().GetIRBaseName(class_info.name_id).str());
      AddBlockLabel(class_scope, class_info.body_block_id, "class", class_loc);
      CollectNamesInBlock(class_scope, class_info.body_block_id);
    }

    // Build each interface scope.
    for (auto [i, interface_info] :
         llvm::enumerate(sem_ir.interfaces().array_ref())) {
      auto interface_id = InterfaceId(i);
      auto interface_scope = GetScopeFor(interface_id);
      // TODO: Provide a location for the interface for use as a disambiguator.
      auto interface_loc = Parse::NodeId::Invalid;
      GetScopeInfo(interface_scope).name = globals.AllocateName(
          *this, interface_loc,
          sem_ir.names().GetIRBaseName(interface_info.name_id).str());
      AddBlockLabel(interface_scope, interface_info.body_block_id, "interface",
                    interface_loc);
      CollectNamesInBlock(interface_scope, interface_info.body_block_id);
    }

    // Build each impl scope.
    for (auto [i, impl_info] : llvm::enumerate(sem_ir.impls().array_ref())) {
      auto impl_id = ImplId(i);
      auto impl_scope = GetScopeFor(impl_id);
      // TODO: Provide a location for the impl for use as a disambiguator.
      auto impl_loc = Parse::NodeId::Invalid;
      // TODO: Invent a name based on the self and constraint types.
      GetScopeInfo(impl_scope).name =
          globals.AllocateName(*this, impl_loc, "impl");
      AddBlockLabel(impl_scope, impl_info.body_block_id, "impl", impl_loc);
      CollectNamesInBlock(impl_scope, impl_info.body_block_id);
    }
  }

  // Returns the scope ID corresponding to an ID of a function, class, or
  // interface.
  template <typename IdT>
  auto GetScopeFor(IdT id) -> ScopeId {
    auto index = static_cast<int32_t>(ScopeId::FirstFunction);

    if constexpr (!std::same_as<FunctionId, IdT>) {
      index += sem_ir_.functions().size();
      if constexpr (!std::same_as<ClassId, IdT>) {
        index += sem_ir_.classes().size();
        if constexpr (!std::same_as<InterfaceId, IdT>) {
          index += sem_ir_.interfaces().size();
          if constexpr (!std::same_as<ImplId, IdT>) {
            index += sem_ir_.impls().size();
            static_assert(std::same_as<NumberOfScopesTag, IdT>,
                          "Unknown ID kind for scope");
          }
        }
      }
    }
    if constexpr (!std::same_as<NumberOfScopesTag, IdT>) {
      index += id.index;
    }
    return static_cast<ScopeId>(index);
  }

  // Returns the IR name to use for a function, class, or interface.
  template <typename IdT>
  auto GetNameFor(IdT id) -> llvm::StringRef {
    if (!id.is_valid()) {
      return "invalid";
    }
    return GetScopeInfo(GetScopeFor(id)).name.str();
  }

  // Returns the IR name to use for an instruction, when referenced from a given
  // scope.
  auto GetNameFor(ScopeId scope_id, InstId inst_id) -> std::string {
    if (!inst_id.is_valid()) {
      return "invalid";
    }

    // Check for a builtin.
    if (inst_id.is_builtin()) {
      return inst_id.builtin_kind().label().str();
    }

    if (inst_id == InstId::PackageNamespace) {
      return "package";
    }

    auto& [inst_scope, inst_name] = insts[inst_id.index];
    if (!inst_name) {
      // This should not happen in valid IR.
      std::string str;
      llvm::raw_string_ostream(str) << "<unexpected instref " << inst_id << ">";
      return str;
    }
    if (inst_scope == scope_id) {
      return inst_name.str().str();
    }
    return (GetScopeInfo(inst_scope).name.str() + "." + inst_name.str()).str();
  }

  // Returns the IR name to use for a label, when referenced from a given scope.
  auto GetLabelFor(ScopeId scope_id, InstBlockId block_id) -> std::string {
    if (!block_id.is_valid()) {
      return "!invalid";
    }

    auto& [label_scope, label_name] = labels[block_id.index];
    if (!label_name) {
      // This should not happen in valid IR.
      std::string str;
      llvm::raw_string_ostream(str)
          << "<unexpected instblockref " << block_id << ">";
      return str;
    }
    if (label_scope == scope_id) {
      return label_name.str().str();
    }
    return (GetScopeInfo(label_scope).name.str() + "." + label_name.str())
        .str();
  }

 private:
  // A space in which unique names can be allocated.
  struct Namespace {
    // A result of a name lookup.
    struct NameResult;

    // A name in a namespace, which might be redirected to refer to another name
    // for disambiguation purposes.
    class Name {
     public:
      Name() : value_(nullptr) {}
      explicit Name(llvm::StringMapIterator<NameResult> it) : value_(&*it) {}

      explicit operator bool() const { return value_; }

      auto str() const -> llvm::StringRef {
        llvm::StringMapEntry<NameResult>* value = value_;
        CARBON_CHECK(value) << "cannot print a null name";
        while (value->second.ambiguous && value->second.fallback) {
          value = value->second.fallback.value_;
        }
        return value->first();
      }

      auto SetFallback(Name name) -> void { value_->second.fallback = name; }

      auto SetAmbiguous() -> void { value_->second.ambiguous = true; }

     private:
      llvm::StringMapEntry<NameResult>* value_ = nullptr;
    };

    struct NameResult {
      bool ambiguous = false;
      Name fallback = Name();
    };

    llvm::StringRef prefix;
    llvm::StringMap<NameResult> allocated = {};
    int unnamed_count = 0;

    auto AddNameUnchecked(llvm::StringRef name) -> Name {
      return Name(allocated.insert({name, NameResult()}).first);
    }

    auto AllocateName(const InstNamer& namer, SemIR::LocId loc_id,
                      std::string name) -> Name {
      // The best (shortest) name for this instruction so far, and the current
      // name for it.
      Name best;
      Name current;

      // Add `name` as a name for this entity.
      auto add_name = [&](bool mark_ambiguous = true) {
        auto [it, added] = allocated.insert({name, NameResult()});
        Name new_name = Name(it);

        if (!added) {
          if (mark_ambiguous) {
            // This name was allocated for a different instruction. Mark it as
            // ambiguous and keep looking for a name for this instruction.
            new_name.SetAmbiguous();
          }
        } else {
          if (!best) {
            best = new_name;
          } else {
            CARBON_CHECK(current);
            current.SetFallback(new_name);
          }
          current = new_name;
        }
        return added;
      };

      // All names start with the prefix.
      name.insert(0, prefix);

      // Use the given name if it's available and not just the prefix.
      if (name.size() > prefix.size()) {
        add_name();
      }

      // Append location information to try to disambiguate.
      // TODO: Consider handling inst_id cases.
      if (loc_id.is_node_id()) {
        auto token = namer.parse_tree_.node_token(loc_id.node_id());
        llvm::raw_string_ostream(name)
            << ".loc" << namer.tokenized_buffer_.GetLineNumber(token);
        add_name();

        llvm::raw_string_ostream(name)
            << "_" << namer.tokenized_buffer_.GetColumnNumber(token);
        add_name();
      }

      // Append numbers until we find an available name.
      name += ".";
      auto name_size_without_counter = name.size();
      for (int counter = 1;; ++counter) {
        name.resize(name_size_without_counter);
        llvm::raw_string_ostream(name) << counter;
        if (add_name(/*mark_ambiguous=*/false)) {
          return best;
        }
      }
    }
  };

  // A named scope that contains named entities.
  struct Scope {
    Namespace::Name name;
    Namespace insts = {.prefix = "%"};
    Namespace labels = {.prefix = "!"};
  };

  auto GetScopeInfo(ScopeId scope_id) -> Scope& {
    return scopes[static_cast<int>(scope_id)];
  }

  auto AddBlockLabel(ScopeId scope_id, InstBlockId block_id,
                     std::string name = "",
                     SemIR::LocId loc_id = SemIR::LocId::Invalid) -> void {
    if (!block_id.is_valid() || labels[block_id.index].second) {
      return;
    }

    if (!loc_id.is_valid()) {
      if (const auto& block = sem_ir_.inst_blocks().Get(block_id);
          !block.empty()) {
        loc_id = sem_ir_.insts().GetLocId(block.front());
      }
    }

    labels[block_id.index] = {
        scope_id, GetScopeInfo(scope_id).labels.AllocateName(*this, loc_id,
                                                             std::move(name))};
  }

  // Finds and adds a suitable block label for the given SemIR instruction that
  // represents some kind of branch.
  auto AddBlockLabel(ScopeId scope_id, SemIR::LocId loc_id, AnyBranch branch)
      -> void {
    llvm::StringRef name;
    switch (parse_tree_.node_kind(loc_id.node_id())) {
      case Parse::NodeKind::IfExprIf:
        switch (branch.kind) {
          case BranchIf::Kind:
            name = "if.expr.then";
            break;
          case Branch::Kind:
            name = "if.expr.else";
            break;
          case BranchWithArg::Kind:
            name = "if.expr.result";
            break;
          default:
            break;
        }
        break;

      case Parse::NodeKind::IfCondition:
        switch (branch.kind) {
          case BranchIf::Kind:
            name = "if.then";
            break;
          case Branch::Kind:
            name = "if.else";
            break;
          default:
            break;
        }
        break;

      case Parse::NodeKind::IfStatement:
        name = "if.done";
        break;

      case Parse::NodeKind::ShortCircuitOperandAnd:
        name = branch.kind == BranchIf::Kind ? "and.rhs" : "and.result";
        break;
      case Parse::NodeKind::ShortCircuitOperandOr:
        name = branch.kind == BranchIf::Kind ? "or.rhs" : "or.result";
        break;

      case Parse::NodeKind::WhileConditionStart:
        name = "while.cond";
        break;

      case Parse::NodeKind::WhileCondition:
        switch (branch.kind) {
          case InstKind::BranchIf:
            name = "while.body";
            break;
          case InstKind::Branch:
            name = "while.done";
            break;
          default:
            break;
        }
        break;

      default:
        break;
    }

    AddBlockLabel(scope_id, branch.target_id, name.str(), loc_id);
  }

  auto CollectNamesInBlock(ScopeId scope_id, InstBlockId block_id) -> void {
    if (block_id.is_valid()) {
      CollectNamesInBlock(scope_id, sem_ir_.inst_blocks().Get(block_id));
    }
  }

  auto CollectNamesInBlock(ScopeId scope_id, llvm::ArrayRef<InstId> block)
      -> void {
    Scope& scope = GetScopeInfo(scope_id);

    // Use bound names where available. Otherwise, assign a backup name.
    for (auto inst_id : block) {
      if (!inst_id.is_valid()) {
        continue;
      }

      auto untyped_inst = sem_ir_.insts().Get(inst_id);
      auto add_inst_name = [&](std::string name) {
        insts[inst_id.index] = {
            scope_id, scope.insts.AllocateName(
                          *this, sem_ir_.insts().GetLocId(inst_id), name)};
      };
      auto add_inst_name_id = [&](NameId name_id, llvm::StringRef suffix = "") {
        add_inst_name(
            (sem_ir_.names().GetIRBaseName(name_id).str() + suffix).str());
      };

      if (auto branch = untyped_inst.TryAs<AnyBranch>()) {
        AddBlockLabel(scope_id, sem_ir_.insts().GetLocId(inst_id), *branch);
      }

      CARBON_KIND_SWITCH(untyped_inst) {
        case CARBON_KIND(AddrPattern inst): {
          // TODO: We need to assign names to parameters that appear in
          // function declarations, which may be nested within a pattern. For
          // now, just look through `addr`, but we should find a better way to
          // visit parameters.
          CollectNamesInBlock(scope_id, inst.inner_id);
          break;
        }
        case CARBON_KIND(AssociatedConstantDecl inst): {
          add_inst_name_id(inst.name_id);
          continue;
        }
        case BindAlias::Kind:
        case BindName::Kind:
        case BindSymbolicName::Kind: {
          auto inst = untyped_inst.As<AnyBindName>();
          add_inst_name_id(sem_ir_.bind_names().Get(inst.bind_name_id).name_id);
          continue;
        }
        case CARBON_KIND(ClassDecl inst): {
          add_inst_name_id(sem_ir_.classes().Get(inst.class_id).name_id,
                           ".decl");
          CollectNamesInBlock(scope_id, inst.decl_block_id);
          continue;
        }
        case CARBON_KIND(ClassType inst): {
          add_inst_name_id(sem_ir_.classes().Get(inst.class_id).name_id);
          continue;
        }
        case CARBON_KIND(FunctionDecl inst): {
          add_inst_name_id(sem_ir_.functions().Get(inst.function_id).name_id);
          CollectNamesInBlock(scope_id, inst.decl_block_id);
          continue;
        }
        case CARBON_KIND(ImplDecl inst): {
          CollectNamesInBlock(scope_id, inst.decl_block_id);
          break;
        }
        case ImportRefUnused::Kind:
        case ImportRefUsed::Kind: {
          add_inst_name("import_ref");
          // When building import refs, we frequently add instructions without
          // a block. Constants that refer to them need to be separately
          // named.
          auto const_id = sem_ir_.constant_values().Get(inst_id);
          if (const_id.is_valid() && const_id.is_template() &&
              !insts[const_id.inst_id().index].second) {
            CollectNamesInBlock(ScopeId::ImportRef, const_id.inst_id());
          }
          continue;
        }
        case CARBON_KIND(InterfaceDecl inst): {
          add_inst_name_id(sem_ir_.interfaces().Get(inst.interface_id).name_id,
                           ".decl");
          CollectNamesInBlock(scope_id, inst.decl_block_id);
          continue;
        }
        case CARBON_KIND(NameRef inst): {
          add_inst_name_id(inst.name_id, ".ref");
          continue;
        }
        // The namespace is specified here due to the name conflict.
        case CARBON_KIND(SemIR::Namespace inst): {
          add_inst_name_id(
              sem_ir_.name_scopes().Get(inst.name_scope_id).name_id);
          continue;
        }
        case CARBON_KIND(Param inst): {
          add_inst_name_id(inst.name_id);
          continue;
        }
        case CARBON_KIND(SpliceBlock inst): {
          CollectNamesInBlock(scope_id, inst.block_id);
          break;
        }
        case CARBON_KIND(VarStorage inst): {
          add_inst_name_id(inst.name_id, ".var");
          continue;
        }
        default: {
          break;
        }
      }

      // Sequentially number all remaining values.
      if (untyped_inst.kind().value_kind() != InstValueKind::None) {
        add_inst_name("");
      }
    }
  }

  const Lex::TokenizedBuffer& tokenized_buffer_;
  const Parse::Tree& parse_tree_;
  const File& sem_ir_;

  Namespace globals = {.prefix = "@"};
  std::vector<std::pair<ScopeId, Namespace::Name>> insts;
  std::vector<std::pair<ScopeId, Namespace::Name>> labels;
  std::vector<Scope> scopes;
};
}  // namespace

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  enum class AddSpace : bool { Before, After };

  explicit Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir,
                     llvm::raw_ostream& out)
      : sem_ir_(sem_ir),
        out_(out),
        inst_namer_(tokenized_buffer, parse_tree, sem_ir) {}

  // Prints the SemIR.
  //
  // Constants are printed first and may be referenced by later sections,
  // including file-scoped instructions. The file scope may contain entity
  // declarations which are defined later, such as classes.
  auto Format() -> void {
    out_ << "--- " << sem_ir_.filename() << "\n\n";

    FormatConstants();

    out_ << "file ";
    OpenBrace();

    // TODO: Handle the case where there are multiple top-level instruction
    // blocks. For example, there may be branching in the initializer of a
    // global or a type expression.
    if (auto block_id = sem_ir_.top_inst_block_id(); block_id.is_valid()) {
      llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::File);
      FormatCodeBlock(block_id);
    }

    CloseBrace();
    out_ << '\n';

    for (int i : llvm::seq(sem_ir_.interfaces().size())) {
      FormatInterface(InterfaceId(i));
    }

    for (int i : llvm::seq(sem_ir_.impls().size())) {
      FormatImpl(ImplId(i));
    }

    for (int i : llvm::seq(sem_ir_.classes().size())) {
      FormatClass(ClassId(i));
    }

    for (int i : llvm::seq(sem_ir_.functions().size())) {
      FormatFunction(FunctionId(i));
    }

    // End-of-file newline.
    out_ << "\n";
  }

  // Begins a braced block. Writes an open brace, and prepares to insert a
  // newline after it if the braced block is non-empty.
  auto OpenBrace() -> void {
    // Put the constant value of an instruction before any braced block, rather
    // than at the end.
    FormatPendingConstantValue(AddSpace::After);

    out_ << '{';
    indent_ += 2;
    after_open_brace_ = true;
  }

  // Ends a braced block by writing a close brace.
  auto CloseBrace() -> void {
    indent_ -= 2;
    if (!after_open_brace_) {
      Indent();
    }
    out_ << '}';
    after_open_brace_ = false;
  }

  // Adds beginning-of-line indentation. If we're at the start of a braced
  // block, first starts a new line.
  auto Indent(int offset = 0) -> void {
    if (after_open_brace_) {
      out_ << '\n';
      after_open_brace_ = false;
    }
    out_.indent(indent_ + offset);
  }

  // Adds beginning-of-label indentation. This is one level less than normal
  // indentation. Labels also get a preceding blank line unless they're at the
  // start of a block.
  auto IndentLabel() -> void {
    CARBON_CHECK(indent_ >= 2);
    if (!after_open_brace_) {
      out_ << '\n';
    }
    Indent(-2);
  }

  auto FormatConstants() -> void {
    if (!sem_ir_.constants().size()) {
      return;
    }

    llvm::SaveAndRestore constants_scope(scope_, InstNamer::ScopeId::Constants);
    out_ << "constants ";
    OpenBrace();
    FormatCodeBlock(sem_ir_.constants().GetAsVector());
    CloseBrace();
    out_ << "\n\n";
  }

  auto FormatClass(ClassId id) -> void {
    const Class& class_info = sem_ir_.classes().Get(id);

    out_ << "\nclass ";
    FormatClassName(id);

    llvm::SaveAndRestore class_scope(scope_, inst_namer_.GetScopeFor(id));

    if (class_info.scope_id.is_valid()) {
      out_ << ' ';
      OpenBrace();
      FormatCodeBlock(class_info.body_block_id);
      FormatNameScope(class_info.scope_id, "!members:\n");
      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }
  }

  auto FormatInterface(InterfaceId id) -> void {
    const Interface& interface_info = sem_ir_.interfaces().Get(id);

    out_ << "\ninterface ";
    FormatInterfaceName(id);

    llvm::SaveAndRestore interface_scope(scope_, inst_namer_.GetScopeFor(id));

    if (interface_info.scope_id.is_valid()) {
      out_ << ' ';
      OpenBrace();
      FormatCodeBlock(interface_info.body_block_id);

      // Always include the !members label because we always list the witness in
      // this section.
      IndentLabel();
      out_ << "!members:\n";
      FormatNameScope(interface_info.scope_id);

      Indent();
      out_ << "witness = ";
      FormatArg(interface_info.associated_entities_id);
      out_ << "\n";

      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }
  }

  auto FormatImpl(ImplId id) -> void {
    const Impl& impl_info = sem_ir_.impls().Get(id);

    out_ << "\nimpl ";
    FormatImplName(id);
    out_ << ": ";
    // TODO: Include the deduced parameter list if present.
    FormatType(impl_info.self_id);
    out_ << " as ";
    FormatType(impl_info.constraint_id);

    llvm::SaveAndRestore impl_scope(scope_, inst_namer_.GetScopeFor(id));

    if (impl_info.scope_id.is_valid()) {
      out_ << ' ';
      OpenBrace();
      FormatCodeBlock(impl_info.body_block_id);

      // Print the !members label even if the name scope is empty because we
      // always list the witness in this section.
      IndentLabel();
      out_ << "!members:\n";
      FormatNameScope(impl_info.scope_id);

      Indent();
      out_ << "witness = ";
      FormatArg(impl_info.witness_id);
      out_ << "\n";

      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }
  }

  auto FormatFunction(FunctionId id) -> void {
    const Function& fn = sem_ir_.functions().Get(id);

    out_ << "\n";

    if (fn.is_extern) {
      out_ << "extern ";
    }

    out_ << "fn ";
    FormatFunctionName(id);

    llvm::SaveAndRestore function_scope(scope_, inst_namer_.GetScopeFor(id));

    if (fn.implicit_param_refs_id != InstBlockId::Empty) {
      out_ << "[";
      FormatParamList(fn.implicit_param_refs_id);
      out_ << "]";
    }

    out_ << "(";
    FormatParamList(fn.param_refs_id);
    out_ << ")";

    if (fn.return_type_id.is_valid()) {
      out_ << " -> ";
      if (fn.return_slot_id.is_valid()) {
        FormatInstName(fn.return_slot_id);
        out_ << ": ";
      }
      FormatType(fn.return_type_id);
    }

    if (fn.builtin_kind != BuiltinFunctionKind::None) {
      out_ << " = \"";
      out_.write_escaped(fn.builtin_kind.name(),
                         /*UseHexEscapes=*/true);
      out_ << "\"";
    }

    if (!fn.body_block_ids.empty()) {
      out_ << ' ';
      OpenBrace();

      for (auto block_id : fn.body_block_ids) {
        IndentLabel();
        FormatLabel(block_id);
        out_ << ":\n";

        FormatCodeBlock(block_id);
      }

      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }
  }

  auto FormatParamList(InstBlockId param_refs_id) -> void {
    llvm::ListSeparator sep;
    for (InstId param_id : sem_ir_.inst_blocks().Get(param_refs_id)) {
      out_ << sep;
      if (!param_id.is_valid()) {
        out_ << "invalid";
        continue;
      }
      if (auto addr = sem_ir_.insts().TryGetAs<SemIR::AddrPattern>(param_id)) {
        out_ << "addr ";
        param_id = addr->inner_id;
      }
      FormatInstName(param_id);
      out_ << ": ";
      FormatType(sem_ir_.insts().Get(param_id).type_id());
    }
  }

  auto FormatCodeBlock(InstBlockId block_id) -> void {
    if (block_id.is_valid()) {
      FormatCodeBlock(sem_ir_.inst_blocks().Get(block_id));
    }
  }

  auto FormatCodeBlock(llvm::ArrayRef<InstId> block) -> void {
    for (const InstId inst_id : block) {
      FormatInstruction(inst_id);
    }
  }

  auto FormatTrailingBlock(InstBlockId block_id) -> void {
    out_ << ' ';
    OpenBrace();
    FormatCodeBlock(block_id);
    CloseBrace();
  }

  auto FormatNameScope(NameScopeId id, llvm::StringRef label = "") -> void {
    const auto& scope = sem_ir_.name_scopes().Get(id);

    if (scope.names.empty() && scope.extended_scopes.empty() &&
        !scope.has_error) {
      // Name scope is empty.
      return;
    }

    if (!label.empty()) {
      IndentLabel();
      out_ << label;
    }

    // Name scopes aren't kept in any particular order. Sort the entries before
    // we print them for stability and consistency.
    llvm::SmallVector<std::pair<InstId, NameId>> entries;
    for (auto [name_id, inst_id] : scope.names) {
      entries.push_back({inst_id, name_id});
    }
    llvm::sort(entries,
               [](auto a, auto b) { return a.first.index < b.first.index; });

    for (auto [inst_id, name_id] : entries) {
      Indent();
      out_ << ".";
      FormatName(name_id);
      out_ << " = ";
      FormatInstName(inst_id);
      out_ << "\n";
    }

    for (auto extended_scope_id : scope.extended_scopes) {
      // TODO: Print this scope in a better way.
      Indent();
      out_ << "extend " << extended_scope_id << "\n";
    }

    if (scope.has_error) {
      Indent();
      out_ << "has_error\n";
    }
  }

  auto FormatInstruction(InstId inst_id) -> void {
    if (!inst_id.is_valid()) {
      Indent();
      out_ << "invalid\n";
      return;
    }

    FormatInstruction(inst_id, sem_ir_.insts().Get(inst_id));
  }

  auto FormatInstruction(InstId inst_id, Inst inst) -> void {
    switch (inst.kind()) {
#define CARBON_SEM_IR_INST_KIND(InstT)            \
  case InstT::Kind:                               \
    FormatInstruction(inst_id, inst.As<InstT>()); \
    break;
#include "toolchain/sem_ir/inst_kind.def"
    }
  }

  template <typename InstT>
  auto FormatInstruction(InstId inst_id, InstT inst) -> void {
    Indent();
    FormatInstructionLHS(inst_id, inst);
    out_ << InstT::Kind.ir_name();
    pending_constant_value_ = sem_ir_.constant_values().Get(inst_id);
    pending_constant_value_is_self_ =
        pending_constant_value_.inst_id() == inst_id;
    FormatInstructionRHS(inst);
    FormatPendingConstantValue(AddSpace::Before);
    out_ << "\n";
  }

  // Don't print a constant for ImportRefUnused.
  auto FormatInstruction(InstId inst_id, ImportRefUnused inst) -> void {
    Indent();
    FormatInstructionLHS(inst_id, inst);
    out_ << ImportRefUnused::Kind.ir_name();
    FormatInstructionRHS(inst);
    out_ << "\n";
  }

  // If there is a pending constant value attached to the current instruction,
  // print it now and clear it out. The constant value gets printed before the
  // first braced block argument, or at the end of the instruction if there are
  // no such arguments.
  auto FormatPendingConstantValue(AddSpace space_where) -> void {
    if (pending_constant_value_ == ConstantId::NotConstant) {
      return;
    }

    if (space_where == AddSpace::Before) {
      out_ << ' ';
    }
    out_ << '[';
    if (pending_constant_value_.is_valid()) {
      out_ << (pending_constant_value_.is_symbolic() ? "symbolic" : "template");
      if (!pending_constant_value_is_self_) {
        out_ << " = ";
        FormatInstName(pending_constant_value_.inst_id());
      }
    } else {
      out_ << pending_constant_value_;
    }
    out_ << ']';
    if (space_where == AddSpace::After) {
      out_ << ' ';
    }
    pending_constant_value_ = ConstantId::NotConstant;
  }

  auto FormatInstructionLHS(InstId inst_id, Inst inst) -> void {
    switch (inst.kind().value_kind()) {
      case InstValueKind::Typed:
        FormatInstName(inst_id);
        out_ << ": ";
        switch (GetExprCategory(sem_ir_, inst_id)) {
          case ExprCategory::NotExpr:
          case ExprCategory::Error:
          case ExprCategory::Value:
          case ExprCategory::Mixed:
            break;
          case ExprCategory::DurableRef:
          case ExprCategory::EphemeralRef:
            out_ << "ref ";
            break;
          case ExprCategory::Initializing:
            out_ << "init ";
            break;
        }
        FormatType(inst.type_id());
        out_ << " = ";
        break;
      case InstValueKind::None:
        break;
    }
  }

  // Print ImportRefUnused with type-like semantics even though it lacks a
  // type_id.
  auto FormatInstructionLHS(InstId inst_id, ImportRefUnused /*inst*/) -> void {
    FormatInstName(inst_id);
    out_ << " = ";
  }

  template <typename InstT>
  auto FormatInstructionRHS(InstT inst) -> void {
    // By default, an instruction has a comma-separated argument list.
    using Info = Internal::InstLikeTypeInfo<InstT>;
    if constexpr (Info::NumArgs == 2) {
      FormatArgs(Info::template Get<0>(inst), Info::template Get<1>(inst));
    } else if constexpr (Info::NumArgs == 1) {
      FormatArgs(Info::template Get<0>(inst));
    } else {
      FormatArgs();
    }
  }

  auto FormatInstructionRHS(BindSymbolicName inst) -> void {
    // A BindSymbolicName with no value is a purely symbolic binding, such as
    // the `Self` in an interface. Don't print out `invalid` for the value.
    if (inst.value_id.is_valid()) {
      FormatArgs(inst.bind_name_id, inst.value_id);
    } else {
      FormatArgs(inst.bind_name_id);
    }
  }

  auto FormatInstructionRHS(BlockArg inst) -> void {
    out_ << " ";
    FormatLabel(inst.block_id);
  }

  auto FormatInstructionRHS(Namespace inst) -> void {
    if (inst.import_id.is_valid()) {
      FormatArgs(inst.import_id, inst.name_scope_id);
    } else {
      FormatArgs(inst.name_scope_id);
    }
  }

  auto FormatInstruction(InstId /*inst_id*/, BranchIf inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << "if ";
    FormatInstName(inst.cond_id);
    out_ << " " << Branch::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << " else ";
    in_terminator_sequence_ = true;
  }

  auto FormatInstruction(InstId /*inst_id*/, BranchWithArg inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << BranchWithArg::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << "(";
    FormatInstName(inst.arg_id);
    out_ << ")\n";
    in_terminator_sequence_ = false;
  }

  auto FormatInstruction(InstId /*inst_id*/, Branch inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << Branch::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << "\n";
    in_terminator_sequence_ = false;
  }

  auto FormatInstructionRHS(Call inst) -> void {
    out_ << " ";
    FormatArg(inst.callee_id);

    if (!inst.args_id.is_valid()) {
      out_ << "(<invalid>)";
      return;
    }

    llvm::ArrayRef<InstId> args = sem_ir_.inst_blocks().Get(inst.args_id);

    bool has_return_slot = GetInitRepr(sem_ir_, inst.type_id).has_return_slot();
    InstId return_slot_id = InstId::Invalid;
    if (has_return_slot) {
      return_slot_id = args.back();
      args = args.drop_back();
    }

    llvm::ListSeparator sep;
    out_ << '(';
    for (auto inst_id : args) {
      out_ << sep;
      FormatArg(inst_id);
    }
    out_ << ')';

    if (has_return_slot) {
      FormatReturnSlot(return_slot_id);
    }
  }

  auto FormatInstructionRHS(ArrayInit inst) -> void {
    FormatArgs(inst.inits_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstructionRHS(InitializeFrom inst) -> void {
    FormatArgs(inst.src_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstructionRHS(StructInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlot(init.dest_id);
  }

  auto FormatInstructionRHS(TupleInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlot(init.dest_id);
  }

  auto FormatInstructionRHS(FunctionDecl inst) -> void {
    FormatArgs(inst.function_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(ClassDecl inst) -> void {
    FormatArgs(inst.class_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(ImplDecl inst) -> void {
    FormatArgs(inst.impl_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(InterfaceDecl inst) -> void {
    FormatArgs(inst.interface_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(IntLiteral inst) -> void {
    out_ << " ";
    sem_ir_.ints()
        .Get(inst.int_id)
        .print(out_, sem_ir_.types().IsSignedInt(inst.type_id));
  }

  auto FormatInstructionRHS(ImportRefUnused inst) -> void {
    // Don't format the inst_id because it refers to a different IR.
    // TODO: Consider a better way to format the InstID from other IRs.
    out_ << " " << inst.ir_id << ", " << inst.inst_id << ", unused";
  }

  auto FormatInstructionRHS(ImportRefUsed inst) -> void {
    // Don't format the inst_id because it refers to a different IR.
    // TODO: Consider a better way to format the InstID from other IRs.
    out_ << " " << inst.ir_id << ", " << inst.inst_id << ", used";
  }

  auto FormatInstructionRHS(SpliceBlock inst) -> void {
    FormatArgs(inst.result_id);
    FormatTrailingBlock(inst.block_id);
  }

  // StructTypeFields are formatted as part of their StructType.
  auto FormatInstruction(InstId /*inst_id*/, StructTypeField /*inst*/) -> void {
  }

  auto FormatInstructionRHS(StructType inst) -> void {
    out_ << " {";
    llvm::ListSeparator sep;
    for (auto field_id : sem_ir_.inst_blocks().Get(inst.fields_id)) {
      out_ << sep << ".";
      auto field = sem_ir_.insts().GetAs<StructTypeField>(field_id);
      FormatName(field.name_id);
      out_ << ": ";
      FormatType(field.field_type_id);
    }
    out_ << "}";
  }

  auto FormatArgs() -> void {}

  template <typename... Args>
  auto FormatArgs(Args... args) -> void {
    out_ << ' ';
    llvm::ListSeparator sep;
    ((out_ << sep, FormatArg(args)), ...);
  }

  auto FormatArg(BoolValue v) -> void { out_ << v; }

  auto FormatArg(BuiltinKind kind) -> void { out_ << kind.label(); }

  auto FormatArg(BindNameId id) -> void {
    FormatName(sem_ir_.bind_names().Get(id).name_id);
  }

  auto FormatArg(FunctionId id) -> void { FormatFunctionName(id); }

  auto FormatArg(ClassId id) -> void { FormatClassName(id); }

  auto FormatArg(InterfaceId id) -> void { FormatInterfaceName(id); }

  auto FormatArg(ImplId id) -> void { FormatImplName(id); }

  auto FormatArg(ImportIRId id) -> void { out_ << id; }

  auto FormatArg(IntId id) -> void {
    // We don't know the signedness to use here. Default to unsigned.
    sem_ir_.ints().Get(id).print(out_, /*isSigned=*/false);
  }

  auto FormatArg(ElementIndex index) -> void { out_ << index; }

  auto FormatArg(NameScopeId id) -> void {
    OpenBrace();
    FormatNameScope(id);
    CloseBrace();
  }

  auto FormatArg(InstId id) -> void { FormatInstName(id); }

  auto FormatArg(InstBlockId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
      return;
    }

    out_ << '(';
    llvm::ListSeparator sep;
    for (auto inst_id : sem_ir_.inst_blocks().Get(id)) {
      out_ << sep;
      FormatArg(inst_id);
    }
    out_ << ')';
  }

  auto FormatArg(RealId id) -> void {
    // TODO: Format with a `.` when the exponent is near zero.
    const auto& real = sem_ir_.reals().Get(id);
    real.mantissa.print(out_, /*isSigned=*/false);
    out_ << (real.is_decimal ? 'e' : 'p') << real.exponent;
  }

  auto FormatArg(StringLiteralValueId id) -> void {
    out_ << '"';
    out_.write_escaped(sem_ir_.string_literal_values().Get(id),
                       /*UseHexEscapes=*/true);
    out_ << '"';
  }

  auto FormatArg(NameId id) -> void { FormatName(id); }

  auto FormatArg(TypeId id) -> void { FormatType(id); }

  auto FormatArg(TypeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto type_id : sem_ir_.type_blocks().Get(id)) {
      out_ << sep;
      FormatArg(type_id);
    }
    out_ << ')';
  }

  auto FormatReturnSlot(InstId dest_id) -> void {
    out_ << " to ";
    FormatArg(dest_id);
  }

  auto FormatName(NameId id) -> void {
    out_ << sem_ir_.names().GetFormatted(id);
  }

  auto FormatInstName(InstId id) -> void {
    out_ << inst_namer_.GetNameFor(scope_, id);
  }

  auto FormatLabel(InstBlockId id) -> void {
    out_ << inst_namer_.GetLabelFor(scope_, id);
  }

  auto FormatFunctionName(FunctionId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatClassName(ClassId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatInterfaceName(InterfaceId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatImplName(ImplId id) -> void { out_ << inst_namer_.GetNameFor(id); }

  auto FormatType(TypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << sem_ir_.StringifyType(id);
    }
  }

 private:
  const File& sem_ir_;
  llvm::raw_ostream& out_;
  InstNamer inst_namer_;

  // The current scope that we are formatting within. References to names in
  // this scope will not have a `@scope.` prefix added.
  InstNamer::ScopeId scope_ = InstNamer::ScopeId::None;

  // Whether we are formatting in a terminator sequence, that is, a sequence of
  // branches at the end of a block. The entirety of a terminator sequence is
  // formatted on a single line, despite being multiple instructions.
  bool in_terminator_sequence_ = false;

  // The indent depth to use for new instructions.
  int indent_ = 0;

  // Whether we are currently formatting immediately after an open brace. If so,
  // a newline will be inserted before the next line indent.
  bool after_open_brace_ = false;

  // The constant value of the current instruction, if it has one that has not
  // yet been printed. The value `NotConstant` is used as a sentinel to indicate
  // there is nothing to print.
  ConstantId pending_constant_value_ = ConstantId::NotConstant;

  // Whether `pending_constant_value_`'s instruction is the same as the
  // instruction currently being printed. If true, only the phase of the
  // constant is printed, and the value is omitted.
  bool pending_constant_value_is_self_ = false;
};

auto FormatFile(const Lex::TokenizedBuffer& tokenized_buffer,
                const Parse::Tree& parse_tree, const File& sem_ir,
                llvm::raw_ostream& out) -> void {
  Formatter(tokenized_buffer, parse_tree, sem_ir, out).Format();
}

}  // namespace Carbon::SemIR
