// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_

#include <concepts>

#include "llvm/Support/raw_ostream.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  // A callback that indicates whether a specific entity, identified by its
  // declaration, should be included in the output.
  using ShouldFormatEntityFn =
      llvm::function_ref<auto(InstId decl_inst_id)->bool>;

  explicit Formatter(const File* sem_ir,
                     ShouldFormatEntityFn should_format_entity,
                     Parse::GetTreeAndSubtreesFn get_tree_and_subtrees);

  // Prints the SemIR into an internal buffer.
  //
  // Constants are printed first and may be referenced by later sections,
  // including file-scoped instructions. The file scope may contain entity
  // declarations which are defined later, such as classes.
  auto Format() -> void;

  // Write buffered output to the given stream.
  auto Write(llvm::raw_ostream& out) -> void;

 private:
  enum class AddSpace : bool { Before, After };

  // A chunk of the buffered output. Chunks of the output, such as constant
  // values, are buffered until we reach the end of formatting so that we can
  // decide whether to include them based on whether they are referenced.
  struct OutputChunk {
    // Whether this chunk is known to be included in the output.
    bool include_in_output;
    // The textual contents of this chunk.
    std::string chunk = std::string();
    // Chunks that should be included in the output if this one is.
    llvm::SmallVector<size_t> dependencies = {};
  };

  // A scope in which output should be buffered because we don't yet know
  // whether to include it in the final formatted SemIR.
  struct TentativeOutputScope {
    explicit TentativeOutputScope(Formatter& f, size_t parent_chunk_index)
        : formatter(f) {
      // If our parent is not known to be included, create a new chunk and
      // include it only if the parent is later found to be used.
      if (!f.output_chunks_[parent_chunk_index].include_in_output) {
        index = formatter.AddChunk(false);
        f.output_chunks_[parent_chunk_index].dependencies.push_back(index);
      }
    }
    ~TentativeOutputScope() {
      auto next_index = formatter.AddChunk(true);
      CARBON_CHECK(next_index == index + 1, "Nested TentativeOutputScope");
    }
    Formatter& formatter;
    size_t index;
  };

  // Flushes the buffered output to the current chunk.
  auto FlushChunk() -> void;

  // Adds a new chunk to the output. Does not flush existing output, so should
  // only be called if there is no buffered output.
  auto AddChunkNoFlush(bool include_in_output) -> size_t;

  // Flushes the current chunk and add a new chunk to the output.
  auto AddChunk(bool include_in_output) -> size_t;

  // Marks the given chunk as being included in the output if the current chunk
  // is.
  auto IncludeChunkInOutput(size_t chunk) -> void;

  // Returns true if the node subtree for the instruction or body overlaps with
  // a dump range, or if there are no ranges.
  auto OverlapsWithDumpSemIRRange(InstId inst_id,
                                  llvm::ArrayRef<InstBlockId> body_block_ids)
      -> bool;

  // Determines whether the specified entity should be included in the formatted
  // output.
  auto ShouldFormatEntity(InstId decl_id,
                          llvm::ArrayRef<InstBlockId> body_block_ids) -> bool;

  auto ShouldFormatEntity(const EntityWithParamsBase& entity,
                          llvm::ArrayRef<InstBlockId> body_block_ids) -> bool;

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
  auto FormatScopeIfUsed(InstNamer::ScopeId scope_id,
                         llvm::ArrayRef<InstId> block) -> void;

  // Formats a full class.
  auto FormatClass(ClassId id) -> void;

  // Formats a full interface.
  auto FormatInterface(InterfaceId id) -> void;

  // Formats an associated constant entity.
  auto FormatAssociatedConstant(AssociatedConstantId id) -> void;

  // Formats a full impl.
  auto FormatImpl(ImplId id) -> void;

  // Formats a full function.
  auto FormatFunction(FunctionId id) -> void;

  // Helper for FormatSpecific to print regions.
  auto FormatSpecificRegion(const Generic& generic, const Specific& specific,
                            GenericInstIndex::Region region,
                            llvm::StringRef region_name) -> void;

  // Formats a full specific.
  auto FormatSpecific(SpecificId id) -> void;

  // Handles generic-specific setup for FormatEntityStart.
  auto FormatGenericStart(llvm::StringRef entity_kind, GenericId generic_id)
      -> void;

  // Provides common formatting for entities, paired with FormatEntityEnd.
  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind,
                         InstId first_owning_decl_id, GenericId generic_id,
                         IdT entity_id) -> void;

  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind,
                         const EntityWithParamsBase& entity, IdT entity_id)
      -> void;

  // Provides common formatting for entities, paired with FormatEntityStart.
  auto FormatEntityEnd(GenericId generic_id) -> void;

  // Formats parameters, eliding them completely if they're empty. Wraps input
  // parameters in parentheses. Formats output parameter as a return type.
  auto FormatParamList(InstBlockId params_id, bool has_return_slot = false)
      -> void;

  // Prints instructions for a code block.
  auto FormatCodeBlock(InstBlockId block_id) -> void;

  // Prints a code block with braces, intended to be used trailing after other
  // content on the same line. If non-empty, instructions are on separate lines.
  auto FormatTrailingBlock(InstBlockId block_id) -> void;

  // Prints the contents of a name scope, with an optional label.
  auto FormatNameScope(NameScopeId id, llvm::StringRef label = "") -> void;

  auto FormatInst(InstId inst_id, Inst inst) -> void;

  // Don't print a constant for ImportRefUnloaded.
  auto FormatInst(InstId inst_id, ImportRefUnloaded inst) -> void;

  // Prints a single instruction.
  auto FormatInst(InstId inst_id) -> void;

  template <typename InstT>
  auto FormatInst(InstId inst_id, InstT inst) -> void;

  // If there is a pending library name that the current instruction was
  // imported from, print it now and clear it out.
  auto FormatPendingImportedFrom(AddSpace space_where) -> void;

  // If there is a pending constant value attached to the current instruction,
  // print it now and clear it out. The constant value gets printed before the
  // first braced block argument, or at the end of the instruction if there are
  // no such arguments.
  auto FormatPendingConstantValue(AddSpace space_where) -> void;

  auto FormatInstLhs(InstId inst_id, Inst inst) -> void;

  // Format ImportCppDecl name.
  auto FormatInstLhs(InstId inst_id, ImportCppDecl inst) -> void;

  // Format ImportDecl with its name.
  auto FormatInstLhs(InstId inst_id, ImportDecl inst) -> void;

  // Print ImportRefUnloaded with type-like semantics even though it lacks a
  // type_id.
  auto FormatInstLhs(InstId inst_id, ImportRefUnloaded inst) -> void;

  // Format ImplWitnessTable with its name even though it lacks a type_id.
  auto FormatInstLhs(InstId inst_id, ImplWitnessTable inst) -> void;

  template <typename InstT>
  auto FormatInstRhs(InstT inst) -> void;

  auto FormatInstRhs(BindSymbolicName inst) -> void;

  auto FormatInstRhs(BlockArg inst) -> void;
  auto FormatInstRhs(Namespace inst) -> void;

  auto FormatInst(InstId inst_id, BranchIf inst) -> void;
  auto FormatInst(InstId inst_id, BranchWithArg inst) -> void;
  auto FormatInst(InstId inst_id, Branch inst) -> void;

  auto FormatInstRhs(Call inst) -> void;
  auto FormatInstRhs(ArrayInit inst) -> void;
  auto FormatInstRhs(InitializeFrom inst) -> void;
  auto FormatInstRhs(ValueParam inst) -> void;
  auto FormatInstRhs(RefParam inst) -> void;
  auto FormatInstRhs(OutParam inst) -> void;
  auto FormatInstRhs(ReturnExpr ret) -> void;
  auto FormatInstRhs(ReturnSlot inst) -> void;
  auto FormatInstRhs(ReturnSlotPattern inst) -> void;
  auto FormatInstRhs(StructInit init) -> void;
  auto FormatInstRhs(TupleInit init) -> void;
  auto FormatInstRhs(FunctionDecl inst) -> void;
  auto FormatInstRhs(ClassDecl inst) -> void;
  auto FormatInstRhs(ImplDecl inst) -> void;
  auto FormatInstRhs(InterfaceDecl inst) -> void;
  auto FormatInstRhs(AssociatedConstantDecl inst) -> void;
  auto FormatInstRhs(IntValue inst) -> void;
  auto FormatInstRhs(FloatLiteral inst) -> void;

  // Format the metadata in File for `import Cpp`.
  auto FormatInstRhs(ImportCppDecl inst) -> void;

  auto FormatImportRefRhs(ImportIRInstId import_ir_inst_id,
                          EntityNameId entity_name_id,
                          llvm::StringLiteral loaded_label) -> void;

  auto FormatInstRhs(ImportRefLoaded inst) -> void;
  auto FormatInstRhs(ImportRefUnloaded inst) -> void;
  auto FormatInstRhs(InstValue inst) -> void;
  auto FormatInstRhs(NameBindingDecl inst) -> void;
  auto FormatInstRhs(SpliceBlock inst) -> void;
  auto FormatInstRhs(WhereExpr inst) -> void;
  auto FormatInstRhs(StructType inst) -> void;

  auto FormatArgs() -> void {}

  template <typename... Args>
  auto FormatArgs(Args... args) -> void {
    out_ << ' ';
    llvm::ListSeparator sep;
    ((out_ << sep, FormatArg(args)), ...);
  }

  // FormatArg variants handling printing instruction arguments. Several things
  // provide equivalent behavior with `FormatName`, so we provide that as the
  // default.
  template <typename IdT>
  auto FormatArg(IdT id) -> void {
    FormatName(id);
  }

  auto FormatArg(BoolValue v) -> void { out_ << v; }
  auto FormatArg(EntityNameId id) -> void;
  auto FormatArg(FacetTypeId id) -> void;
  auto FormatArg(IntKind k) -> void { k.Print(out_); }
  auto FormatArg(FloatKind k) -> void { k.Print(out_); }
  auto FormatArg(ImportIRId id) -> void;
  auto FormatArg(IntId id) -> void;
  auto FormatArg(ElementIndex index) -> void { out_ << index; }
  auto FormatArg(CallParamIndex index) -> void { out_ << index; }
  auto FormatArg(NameScopeId id) -> void;
  auto FormatArg(InstBlockId id) -> void;
  auto FormatArg(AbsoluteInstBlockId id) -> void;
  auto FormatArg(RealId id) -> void;
  auto FormatArg(StringLiteralValueId id) -> void;

  auto FormatReturnSlotArg(InstId dest_id) -> void;

  // `FormatName` is used when we need the name from an id. Most id types use
  // equivalent name formatting from InstNamer, although there are a few special
  // formats below.
  template <typename IdT>
  // Force `InstId` children to use the `InstId` overload.
    requires(!std::derived_from<IdT, InstId>)
  auto FormatName(IdT id) -> void {
    out_ << inst_namer_.GetNameFor(id);
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

  const File* sem_ir_;
  InstNamer inst_namer_;
  ShouldFormatEntityFn should_format_entity_;
  Parse::GetTreeAndSubtreesFn get_tree_and_subtrees_;

  // The output stream buffer.
  std::string buffer_;

  // The output stream.
  llvm::raw_string_ostream out_ = llvm::raw_string_ostream(buffer_);

  // Chunks of output text that we have created so far.
  llvm::SmallVector<OutputChunk> output_chunks_;

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

  // Indexes of chunks of output that should be included when an instruction is
  // referenced, indexed by the instruction's index. This is resized in advance
  // to the correct size.
  llvm::SmallVector<size_t, 0> tentative_inst_chunks_;
};

template <typename IdT>
auto Formatter::FormatEntityStart(llvm::StringRef entity_kind,
                                  InstId first_owning_decl_id,
                                  GenericId generic_id, IdT entity_id) -> void {
  // If this entity was imported from a different IR, annotate the name of
  // that IR in the output before the `{` or `;`.
  if (first_owning_decl_id.has_value()) {
    auto loc_id = sem_ir_->insts().GetCanonicalLocId(first_owning_decl_id);
    if (loc_id.kind() == LocId::Kind::ImportIRInstId) {
      auto import_ir_id =
          sem_ir_->import_ir_insts().Get(loc_id.import_ir_inst_id()).ir_id();
      const auto* import_file = sem_ir_->import_irs().Get(import_ir_id).sem_ir;
      pending_imported_from_ = import_file->filename();
    }
  }

  if (generic_id.has_value()) {
    FormatGenericStart(entity_kind, generic_id);
  }

  out_ << "\n";
  after_open_brace_ = false;
  Indent();
  out_ << entity_kind;

  // If there's a generic, it will have attached the name. Otherwise, add the
  // name here.
  if (!generic_id.has_value()) {
    out_ << " ";
    FormatName(entity_id);
  }
}

template <typename IdT>
auto Formatter::FormatEntityStart(llvm::StringRef entity_kind,
                                  const EntityWithParamsBase& entity,
                                  IdT entity_id) -> void {
  FormatEntityStart(entity_kind, entity.first_owning_decl_id, entity.generic_id,
                    entity_id);
}

template <typename InstT>
auto Formatter::FormatInst(InstId inst_id, InstT inst) -> void {
  Indent();
  FormatInstLhs(inst_id, inst);
  out_ << InstT::Kind.ir_name();
  pending_constant_value_ = sem_ir_->constant_values().GetAttached(inst_id);
  pending_constant_value_is_self_ = sem_ir_->constant_values().GetInstIdIfValid(
                                        pending_constant_value_) == inst_id;
  FormatInstRhs(inst);
  FormatPendingConstantValue(AddSpace::Before);
  out_ << "\n";
}

template <typename InstT>
auto Formatter::FormatInstRhs(InstT inst) -> void {
  // By default, an instruction has a comma-separated argument list.
  using Info = Internal::InstLikeTypeInfo<InstT>;
  if constexpr (Info::NumArgs == 2) {
    // Several instructions have a second operand that's a specific ID. We
    // don't include it in the argument list if there is no corresponding
    // specific, that is, when we're not in a generic context.
    if constexpr (std::is_same_v<typename Info::template ArgType<1>,
                                 SpecificId>) {
      if (!Info::template Get<1>(inst).has_value()) {
        FormatArgs(Info::template Get<0>(inst));
        return;
      }
    }
    FormatArgs(Info::template Get<0>(inst), Info::template Get<1>(inst));
  } else if constexpr (Info::NumArgs == 1) {
    FormatArgs(Info::template Get<0>(inst));
  } else {
    FormatArgs();
  }
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
