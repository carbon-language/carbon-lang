// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_

#include <concepts>

#include "common/concepts.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/formatter_chunks.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  // sem_ir and include_ir_in_dumps must be non-null.
  explicit Formatter(
      const File* sem_ir, int total_ir_count,
      Parse::GetTreeAndSubtreesFn get_tree_and_subtrees,
      const FixedSizeValueStore<CheckIRId, bool>* include_ir_in_dumps,
      bool use_dump_sem_ir_ranges);

  // Prints the SemIR into an internal buffer. Must only be called once.
  //
  // We first print top-level scopes (`constants`, `imports`, `generated`, and
  // `file`) then entities (types and functions). The ordering is based on
  // references:
  //
  // - `constants` is self-contained.
  // - `imports` can refer to `constants`.
  // - `generated` can refer to both of the above.
  // - `file` can refer to any of the above, and also entities.
  // - Entities are difficult to order (forward declarations may lead to
  //   circular references), and so are simply grouped by type.
  //
  // When formatting scopes other than `file`, we use `FormatterChunks` to only
  // print entities which are referenced. For example, `imports` speculatively
  // create constants which may never be referenced, or for which the
  // referencing instruction may be hidden and we normally hide those; those are
  // excluded from the `constants` scope output. See `FormatterChunks` for
  // additional information.
  //
  // Beyond the reference-based printing, `ShouldFormatEntity` and
  // `ShouldFormatInst` can also hide instructions. These interact because an
  // hidden instruction means its references are unused for `FormatterChunks`
  // visibility.
  auto Format() -> void;

  // Write buffered output to the given stream. `Format` must be called first.
  auto Write(llvm::raw_ostream& out) -> void;

 private:
  enum class AddSpace : bool { Before, After };

  // Fills `node_parents_` with parent information. Called at most once during
  // construction.
  auto ComputeNodeParents() -> void;

  // Returns true if the instruction should be included according to its
  // originating IR. Typically `ShouldFormatEntity` should be used instead.
  auto ShouldIncludeInstByIR(InstId inst_id) -> bool;

  // Determines whether the specified entity should be included in the formatted
  // output.
  auto ShouldFormatEntity(InstId decl_id) -> bool;

  auto ShouldFormatEntity(const EntityWithParamsBase& entity) -> bool;

  // Determines whether a single instruction should be included in the
  // formatted output.
  auto ShouldFormatInst(InstId inst_id) -> bool;

  // Begins a braced block. Writes an open brace, and prepares to insert a
  // newline after it if the braced block is non-empty.
  auto OpenBrace() -> void;

  // Ends a braced block by writing a close brace.
  auto CloseBrace() -> void;

  auto Semicolon() -> void;

  // Adds beginning-of-line indentation. If we're at the start of a braced
  // block, first starts a new line.
  auto Indent(int offset = 0) -> void;

  // Adds beginning-of-label indentation. This is one level less than normal
  // indentation. Labels also get a preceding blank line unless they're at the
  // start of a block.
  auto IndentLabel() -> void;

  // Formats a top-level scope, and any of the instructions in that scope that
  // are used.
  auto FormatTopLevelScope(InstNamer::ScopeId scope_id,
                           llvm::ArrayRef<InstId> block) -> void;

  // Formats a full class.
  auto FormatClass(ClassId id, const Class& class_info) -> void;

  // Formats a full vtable.
  auto FormatVtable(VtableId id, const Vtable& vtable_info) -> void;

  // Formats a full interface.
  auto FormatInterface(InterfaceId id, const Interface& interface_info) -> void;

  // Formats a full named constraint.
  auto FormatNamedConstraint(NamedConstraintId id,
                             const NamedConstraint& constraint_info) -> void;

  // Formats a full require declaration.
  auto FormatRequireImpls(RequireImplsId id, const RequireImpls& require)
      -> void;

  // Formats a full impl.
  auto FormatImpl(ImplId id, const Impl& impl) -> void;

  // Formats a full function.
  auto FormatFunction(FunctionId id, const Function& fn) -> void;

  // Helper for FormatSpecific to print regions.
  auto FormatSpecificRegion(const Generic& generic, const Specific& specific,
                            GenericInstIndex::Region region,
                            llvm::StringRef region_name) -> void;

  // Formats a full specific.
  auto FormatSpecific(SpecificId id, const Specific& specific) -> void;

  // Handles generic-specific setup for FormatEntityStart.
  auto FormatGenericStart(llvm::StringRef entity_kind, GenericId generic_id)
      -> void;

  // Before formatting a decl (typically an Entity), collect import information
  // (if there is any) needed to format it.
  auto PrepareToFormatDecl(InstId first_owning_decl_id) -> void;

  // Provides common formatting for entities, paired with FormatEntityEnd.
  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind, GenericId generic_id,
                         IdT entity_id) -> void;

  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind,
                         const EntityWithParamsBase& entity, IdT entity_id)
      -> void;

  // Provides common formatting for entities, paired with FormatEntityStart.
  auto FormatEntityEnd(GenericId generic_id) -> void;

  // Provides common formatting for generics, paired with FormatGenericStart.
  // Normally this is just called from FormatEntityEnd, as most generics are
  // entities.
  auto FormatGenericEnd() -> void;

  // Formats `params_id` as a parameter list in parentheses. Does nothing if
  // `params_id` is None.
  auto FormatParamList(InstBlockId params_id) -> void {
    FormatFunctionSignature(params_id, SemIR::CallParamIndex::None,
                            SemIR::InstId::None);
  }

  // Formats a function signature with `Call` parameter block `params_id` and
  // declared return form `return_form`. `return_begin` must be the starting
  // index of the return-parameter subrange of `params_id` (so if there are
  // no return parameters it should be equal to the number of parameters).
  auto FormatFunctionSignature(InstBlockId params_id,
                               SemIR::CallParamIndex return_begin,
                               SemIR::InstId return_form_id) -> void;

  // Prints instructions for a code block.
  auto FormatCodeBlock(InstBlockId block_id) -> void;

  // Prints a code block with braces, intended to be used trailing after other
  // content on the same line. If non-empty, instructions are on separate lines.
  auto FormatTrailingBlock(InstBlockId block_id) -> void;

  // Prints the contents of a name scope, with an optional label.
  auto FormatNameScope(NameScopeId id, llvm::StringRef label = "") -> void;

  // Prints the contents of a require impls as a block.
  auto FormatRequireImpls(RequireImplsId id) -> void;

  // Prints a single instruction. This typically formats as:
  //   `FormatInstLhs()` `<ir_name>` `FormatInstRhs()` `<constant>`
  //
  // Some instruction kinds are special-cased here. However, it's more common to
  // provide special-casing of `FormatInstRhs`, for custom argument
  // formatting.
  auto FormatInst(InstId inst_id) -> void;

  // If there is a pending library name that the current instruction was
  // imported from, print it now and clear it out.
  auto FormatPendingImportedFrom(AddSpace space_where) -> void;

  // If there is a pending constant value attached to the current instruction,
  // print it now and clear it out. The constant value gets printed before the
  // first braced block argument, or at the end of the instruction if there are
  // no such arguments.
  auto FormatPendingConstantValue(AddSpace space_where) -> void;

  // Formats `<name>[: <form>] = `. Skips unnamed instructions (according to
  // `inst_namer_`). Typed instructions must be named.
  auto FormatInstLhs(InstId inst_id, Inst inst) -> void;

  // Formats `<name>[: <form>]`. The inst must have a name.
  auto FormatNameAndForm(InstId inst_id, Inst inst) -> void;

  // Formats arguments to an instruction. This will typically look like "
  // <arg0>, <arg1>".
  auto FormatInstRhs(Inst inst) -> void;

  // Formats the default case for `FormatInstRhs`.
  auto FormatInstRhsDefault(Inst inst) -> void;

  // Formats arguments as " <callee>(<args>) -> <return>".
  auto FormatCallRhs(Call inst) -> void;

  // Standard formatting for a declaration instruction's arguments.
  template <typename IdT>
  auto FormatDeclRhs(IdT decl_id, InstBlockId pattern_block_id,
                     InstBlockId decl_block_id) {
    FormatArgs(decl_id);
    llvm::SaveAndRestore scope(scope_, inst_namer_.GetScopeFor(decl_id));
    FormatTrailingBlock(pattern_block_id);
    FormatTrailingBlock(decl_block_id);
  }

  // Format the metadata in File for `import Cpp`.
  auto FormatImportCppDeclRhs() -> void;

  // Formats an import ref. In an ideal case, this looks like " <ir>, <entity
  // name>, <loaded|unloaded>". However, if the entity name isn't present, this
  // may fall back to printing location information from the import source.
  auto FormatImportRefRhs(AnyImportRef inst) -> void;

  // Format a block of `require` declarations from their `RequireImplsDecl`
  // instructions. Starts with a `!requires:` label.
  auto FormatRequireImplsBlock(RequireImplsBlockId block_id) -> void;

  template <typename... Args>
  auto FormatArgs(Args... args) -> void {
    out() << ' ';
    llvm::ListSeparator sep;
    ((out() << sep, FormatArg(args)), ...);
  }

  // FormatArg variants handling printing instruction arguments. Several things
  // provide equivalent behavior with `FormatName`, so we provide that as the
  // default.
  template <typename IdT>
    requires(
        InstNamer::ScopeIdTypeEnum::Contains<IdT> ||
        SameAsOneOf<IdT, GenericId, NameId, SpecificId, SpecificInterfaceId> ||
        std::derived_from<IdT, InstId>)
  auto FormatArg(IdT id) -> void {
    FormatName(id);
  }

  auto FormatArg(BoolValue v) -> void { out() << v; }
  auto FormatArg(CharId c) -> void { out() << c; }
  auto FormatArg(EntityNameId id) -> void;
  auto FormatArg(FacetTypeId id) -> void;
  auto FormatArg(IntKind k) -> void { k.Print(out()); }
  auto FormatArg(FloatKind k) -> void { k.Print(out()); }
  auto FormatArg(ImportIRId id) -> void;
  auto FormatArg(IntId id) -> void;
  auto FormatArg(ElementIndex index) -> void { out() << index; }
  auto FormatArg(CallParamIndex index) -> void { out() << index; }
  auto FormatArg(NameScopeId id) -> void;
  auto FormatArg(InstBlockId id) -> void;
  auto FormatArg(AbsoluteInstBlockId id) -> void;
  auto FormatArg(ExprRegionId id) -> void;
  auto FormatArg(RealId id) -> void;
  auto FormatArg(StringLiteralValueId id) -> void;
  auto FormatArg(ConstantId id) -> void { FormatConstant(id); }

  // Calls `FormatArg` from an `IdAndKind`.
  auto FormatInstArgAndKind(IdAndKind arg_and_kind) -> void;

  auto FormatReturnSlotArg(InstId dest_id) -> void;

  // `FormatName` is used when we need the name from an id. Most id types use
  // equivalent name formatting from InstNamer, although there are a few special
  // formats below.
  template <typename IdT>
    requires(InstNamer::ScopeIdTypeEnum::Contains<IdT> ||
             std::same_as<IdT, GenericId>)
  auto FormatName(IdT id) -> void {
    out() << inst_namer_.GetNameFor(id);
  }

  auto FormatName(NameId id) -> void;
  auto FormatName(InstId id) -> void;
  auto FormatName(SpecificId id) -> void;
  auto FormatName(SpecificInterfaceId id) -> void;

  auto FormatLabel(InstBlockId id) -> void;

  auto FormatConstant(ConstantId id) -> void;

  auto FormatInstAsType(InstId id) -> void;

  auto FormatTypeOfInst(InstId id) -> void;

  // Returns the label for the indicated IR.
  auto GetImportIRLabel(ImportIRId id) -> std::string;

  // The output stream.
  auto out() -> llvm::raw_ostream& { return chunks_.out(); }

  // The output chunks.
  FormatterChunks chunks_;

  const File* sem_ir_;
  InstNamer inst_namer_;
  Parse::GetTreeAndSubtreesFn get_tree_and_subtrees_;

  // For each CheckIRId, whether entities from it should be formatted.
  const FixedSizeValueStore<CheckIRId, bool>* include_ir_in_dumps_;

  // Whether to use ranges when dumping, or to dump the full SemIR.
  bool use_dump_sem_ir_ranges_;

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

  // The name of the IR file from which the current entity was imported, if it
  // was imported and no file has been printed yet. This is printed before the
  // first open brace or the semicolon in the entity declaration.
  llvm::StringRef pending_imported_from_;

  // Chunks for each scope's labels, including `File` scope. These are parents
  // of chunks containing a scope's `<label> {` and `}` output, and children of
  // the scope's instructions in `tentative_inst_chunks_`.
  std::array<FormatterChunks::ChunkId,
             static_cast<size_t>(InstNamer::ScopeId::FirstEntityScope)>
      scope_label_chunks_;

  // Chunks for each instruction in a tentative top-level scope. These don't
  // directly contain content, and are instead parents of the instruction's
  // output to help indirect inclusion.
  FixedSizeValueStore<InstId, FormatterChunks::ChunkId, Tag<CheckIRId>>
      tentative_inst_chunks_;

  // Maps nodes to their parents. Only set when dump ranges are in use, because
  // the parents aren't used otherwise.
  using NodeParentStore = FixedSizeValueStore<Parse::NodeId, Parse::NodeId>;
  std::optional<NodeParentStore> node_parents_;
};

template <typename IdT>
auto Formatter::FormatEntityStart(llvm::StringRef entity_kind,
                                  GenericId generic_id, IdT entity_id) -> void {
  if (generic_id.has_value()) {
    FormatGenericStart(entity_kind, generic_id);
  }

  out() << "\n";
  after_open_brace_ = false;
  Indent();
  out() << entity_kind;

  // If there's a generic, it will have attached the name. Otherwise, add the
  // name here.
  if (!generic_id.has_value()) {
    out() << " ";
    FormatName(entity_id);
  }
}

template <typename IdT>
auto Formatter::FormatEntityStart(llvm::StringRef entity_kind,
                                  const EntityWithParamsBase& entity,
                                  IdT entity_id) -> void {
  FormatEntityStart(entity_kind, entity.generic_id, entity_id);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
