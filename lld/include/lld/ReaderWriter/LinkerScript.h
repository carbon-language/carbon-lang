//===- ReaderWriter/LinkerScript.h ----------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Linker script parser.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_LINKER_SCRIPT_H
#define LLD_READER_WRITER_LINKER_SCRIPT_H

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/range.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace lld {
namespace script {
class Token {
public:
  enum Kind {
    unknown,
    eof,
    exclaim,
    exclaimequal,
    amp,
    ampequal,
    l_paren,
    r_paren,
    star,
    starequal,
    plus,
    plusequal,
    comma,
    minus,
    minusequal,
    slash,
    slashequal,
    number,
    colon,
    semicolon,
    less,
    lessequal,
    lessless,
    lesslessequal,
    equal,
    equalequal,
    greater,
    greaterequal,
    greatergreater,
    greatergreaterequal,
    question,
    identifier,
    libname,
    kw_align,
    kw_align_with_input,
    kw_as_needed,
    kw_at,
    kw_discard,
    kw_entry,
    kw_exclude_file,
    kw_extern,
    kw_filehdr,
    kw_fill,
    kw_flags,
    kw_group,
    kw_hidden,
    kw_input,
    kw_keep,
    kw_length,
    kw_memory,
    kw_origin,
    kw_phdrs,
    kw_provide,
    kw_provide_hidden,
    kw_only_if_ro,
    kw_only_if_rw,
    kw_output,
    kw_output_arch,
    kw_output_format,
    kw_overlay,
    kw_search_dir,
    kw_sections,
    kw_sort_by_alignment,
    kw_sort_by_init_priority,
    kw_sort_by_name,
    kw_sort_none,
    kw_subalign,
    l_brace,
    pipe,
    pipeequal,
    r_brace,
    tilde
  };

  Token() : _kind(unknown) {}
  Token(StringRef range, Kind kind) : _range(range), _kind(kind) {}

  void dump(raw_ostream &os) const;

  StringRef _range;
  Kind _kind;
};

class Lexer {
public:
  explicit Lexer(std::unique_ptr<MemoryBuffer> mb) : _buffer(mb->getBuffer()) {
    _sourceManager.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
  }

  void lex(Token &tok);

  const llvm::SourceMgr &getSourceMgr() const { return _sourceManager; }

private:
  bool canStartNumber(char c) const;
  bool canContinueNumber(char c) const;
  bool canStartName(char c) const;
  bool canContinueName(char c) const;
  void skipWhitespace();

  Token _current;
  /// \brief The current buffer state.
  StringRef _buffer;
  // Lexer owns the input files.
  llvm::SourceMgr _sourceManager;
};

/// All linker scripts commands derive from this class. High-level, sections and
/// output section commands are all subclasses of this class.
/// Examples:
///
/// OUTPUT_FORMAT("elf64-x86-64") /* A linker script command */
/// OUTPUT_ARCH(i386:x86-64)      /* Another command */
/// ENTRY(_start)                 /* Another command */
///
/// SECTIONS                      /* Another command */
/// {
///   .interp : {                 /* A sections-command */
///              *(.interp)       /* An output-section-command */
///              }
///  }
///
class Command {
public:
  enum class Kind {
    Entry,
    Extern,
    Fill,
    Group,
    Input,
    InputSectionsCmd,
    InputSectionName,
    Memory,
    Output,
    OutputArch,
    OutputFormat,
    OutputSectionDescription,
    Overlay,
    PHDRS,
    SearchDir,
    Sections,
    SortedGroup,
    SymbolAssignment,
  };

  Kind getKind() const { return _kind; }
  inline llvm::BumpPtrAllocator &getAllocator() const;

  virtual void dump(raw_ostream &os) const = 0;

  virtual ~Command() {}

protected:
  Command(class Parser &ctx, Kind k) : _ctx(ctx), _kind(k) {}

private:
  Parser &_ctx;
  Kind _kind;
};

template <class T>
ArrayRef<T> save_array(llvm::BumpPtrAllocator &alloc, ArrayRef<T> array) {
  size_t num = array.size();
  T *start = alloc.Allocate<T>(num);
  std::uninitialized_copy(std::begin(array), std::end(array), start);
  return llvm::makeArrayRef(start, num);
}

class Output : public Command {
public:
  Output(Parser &ctx, StringRef outputFileName)
      : Command(ctx, Kind::Output), _outputFileName(outputFileName) {}

  static bool classof(const Command *c) { return c->getKind() == Kind::Output; }

  void dump(raw_ostream &os) const override {
    os << "OUTPUT(" << _outputFileName << ")\n";
  }

  StringRef getOutputFileName() const { return _outputFileName; }

private:
  StringRef _outputFileName;
};

class OutputFormat : public Command {
public:
  OutputFormat(Parser &ctx, const SmallVectorImpl<StringRef> &formats)
      : Command(ctx, Kind::OutputFormat) {
    _formats = save_array<StringRef>(getAllocator(), formats);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::OutputFormat;
  }

  void dump(raw_ostream &os) const override {
    os << "OUTPUT_FORMAT(";
    bool first = true;
    for (StringRef format : _formats) {
      if (!first)
        os << ",";
      first = false;
      os << "\"" << format << "\"";
    }
    os << ")\n";
  }

  llvm::ArrayRef<StringRef> getFormats() { return _formats; }

private:
  llvm::ArrayRef<StringRef> _formats;
};

class OutputArch : public Command {
public:
  OutputArch(Parser &ctx, StringRef arch)
      : Command(ctx, Kind::OutputArch), _arch(arch) {}

  static bool classof(const Command *c) {
    return c->getKind() == Kind::OutputArch;
  }

  void dump(raw_ostream &os) const override {
    os << "OUTPUT_ARCH(" << getArch() << ")\n";
  }

  StringRef getArch() const { return _arch; }

private:
  StringRef _arch;
};

struct Path {
  StringRef _path;
  bool _asNeeded;
  bool _isDashlPrefix;

  Path() : _asNeeded(false), _isDashlPrefix(false) {}
  Path(StringRef path, bool asNeeded = false, bool isLib = false)
      : _path(path), _asNeeded(asNeeded), _isDashlPrefix(isLib) {}
};

template<Command::Kind K>
class PathList : public Command {
public:
  PathList(Parser &ctx, StringRef name, const SmallVectorImpl<Path> &paths)
      : Command(ctx, K), _name(name) {
    _paths = save_array<Path>(getAllocator(), paths);
  }

  static bool classof(const Command *c) { return c->getKind() == K; }

  void dump(raw_ostream &os) const override {
    os << _name << "(";
    bool first = true;
    for (const Path &path : getPaths()) {
      if (!first)
        os << " ";
      first = false;
      if (path._asNeeded)
        os << "AS_NEEDED(";
      if (path._isDashlPrefix)
        os << "-l";
      os << path._path;
      if (path._asNeeded)
        os << ")";
    }
    os << ")\n";
  }

  llvm::ArrayRef<Path> getPaths() const { return _paths; }

private:
  StringRef _name;
  llvm::ArrayRef<Path> _paths;
};

class Group : public PathList<Command::Kind::Group> {
public:
  template <class RangeT>
  Group(Parser &ctx, RangeT range)
      : PathList(ctx, "GROUP", std::move(range)) {}
};

class Input : public PathList<Command::Kind::Input> {
public:
  template <class RangeT>
  Input(Parser &ctx, RangeT range)
      : PathList(ctx, "INPUT", std::move(range)) {}
};

class Entry : public Command {
public:
  Entry(Parser &ctx, StringRef entryName)
      : Command(ctx, Kind::Entry), _entryName(entryName) {}

  static bool classof(const Command *c) { return c->getKind() == Kind::Entry; }

  void dump(raw_ostream &os) const override {
    os << "ENTRY(" << _entryName << ")\n";
  }

  StringRef getEntryName() const { return _entryName; }

private:
  StringRef _entryName;
};

class SearchDir : public Command {
public:
  SearchDir(Parser &ctx, StringRef searchPath)
      : Command(ctx, Kind::SearchDir), _searchPath(searchPath) {}

  static bool classof(const Command *c) {
    return c->getKind() == Kind::SearchDir;
  }

  void dump(raw_ostream &os) const override {
    os << "SEARCH_DIR(\"" << _searchPath << "\")\n";
  }

  StringRef getSearchPath() const { return _searchPath; }

private:
  StringRef _searchPath;
};

/// Superclass for expression nodes. Linker scripts accept C-like expressions in
/// many places, such as when defining the value of a symbol or the address of
/// an output section.
/// Example:
///
/// SECTIONS {
///   my_symbol = 1 + 1 * 2;
///               | |     ^~~~> Constant : Expression
///               | | ^~~~> Constant : Expression
///               | |   ^~~~> BinOp : Expression
///               ^~~~> Constant : Expression
///                 ^~~~> BinOp : Expression  (the top-level Expression node)
/// }
///
class Expression {
public:
  // The symbol table does not need to own its string keys and the use of StringMap
  // here is an overkill.
  typedef llvm::StringMap<int64_t, llvm::BumpPtrAllocator> SymbolTableTy;

  enum class Kind { Constant, Symbol, FunctionCall, Unary, BinOp,
                    TernaryConditional };
  Kind getKind() const { return _kind; }
  inline llvm::BumpPtrAllocator &getAllocator() const;
  virtual void dump(raw_ostream &os) const = 0;
  virtual ErrorOr<int64_t>
  evalExpr(const SymbolTableTy &symbolTable = SymbolTableTy()) const = 0;
  virtual ~Expression() {}

protected:
  Expression(class Parser &ctx, Kind k) : _ctx(ctx), _kind(k) {}

private:
  Parser &_ctx;
  Kind _kind;
};

/// A constant value is stored as unsigned because it represents absolute
/// values. We represent negative numbers by composing the unary '-' operator
/// with a constant.
class Constant : public Expression {
public:
  Constant(Parser &ctx, uint64_t num)
      : Expression(ctx, Kind::Constant), _num(num) {}
  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::Constant;
  }

  ErrorOr<int64_t> evalExpr(const SymbolTableTy &symbolTable) const override;

private:
  uint64_t _num;
};

class Symbol : public Expression {
public:
  Symbol(Parser &ctx, StringRef name)
      : Expression(ctx, Kind::Symbol), _name(name) {}
  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::Symbol;
  }

  ErrorOr<int64_t> evalExpr(const SymbolTableTy &symbolTable) const override;

private:
  StringRef _name;
};

class FunctionCall : public Expression {
public:
  FunctionCall(Parser &ctx, StringRef name,
               const SmallVectorImpl<const Expression *> &args)
      : Expression(ctx, Kind::FunctionCall), _name(name) {
    _args = save_array<const Expression *>(getAllocator(), args);
  }

  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::FunctionCall;
  }

  ErrorOr<int64_t> evalExpr(const SymbolTableTy &symbolTable) const override;

private:
  StringRef _name;
  llvm::ArrayRef<const Expression *> _args;
};

class Unary : public Expression {
public:
  enum Operation {
    Minus,
    Not
  };

  Unary(Parser &ctx, Operation op, const Expression *child)
      : Expression(ctx, Kind::Unary), _op(op), _child(child) {}
  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::Unary;
  }

  ErrorOr<int64_t> evalExpr(const SymbolTableTy &symbolTable) const override;

private:
  Operation _op;
  const Expression *_child;
};

class BinOp : public Expression {
public:
  enum Operation {
    And,
    CompareDifferent,
    CompareEqual,
    CompareGreater,
    CompareGreaterEqual,
    CompareLess,
    CompareLessEqual,
    Div,
    Mul,
    Or,
    Shl,
    Shr,
    Sub,
    Sum
  };

  BinOp(Parser &ctx, const Expression *lhs, Operation op, const Expression *rhs)
      : Expression(ctx, Kind::BinOp), _op(op), _lhs(lhs), _rhs(rhs) {}

  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::BinOp;
  }

  ErrorOr<int64_t> evalExpr(const SymbolTableTy &symbolTable) const override;

private:
  Operation _op;
  const Expression *_lhs;
  const Expression *_rhs;
};

/// Operands of the ternary operator can be any expression, similar to the other
/// operations, including another ternary operator. To disambiguate the parse
/// tree, note that ternary conditionals have precedence 13 and, different from
/// other operators, associates right-to-left. For example:
///
/// i = i > 3 ? i < 5 ? 1 : 2 : 0;
///
/// will have the following parse tree:
///
/// i = ((i > 3) ? ((i < 5) ? 1 : 2) : 0);
///
/// The '>' binds tigher because it has precedence 6. When faced with two "?"
/// ternary operators back-to-back, the parser prioritized the rightmost one.
///
class TernaryConditional : public Expression {
public:
  TernaryConditional(Parser &ctx, const Expression *conditional,
                     const Expression *trueExpr, const Expression *falseExpr)
      : Expression(ctx, Kind::TernaryConditional), _conditional(conditional),
        _trueExpr(trueExpr), _falseExpr(falseExpr) {}

  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::TernaryConditional;
  }

  ErrorOr<int64_t> evalExpr(const SymbolTableTy &symbolTable) const override;

private:
  const Expression *_conditional;
  const Expression *_trueExpr;
  const Expression *_falseExpr;
};

/// Symbol assignments of the form "symbolname = <expression>" may occur either
/// as sections-commands or as output-section-commands.
/// Example:
///
/// SECTIONS {
///   mysymbol = .         /* SymbolAssignment as a sections-command */
///   .data : {
///     othersymbol = .    /* SymbolAssignment as an output-section-command */
///   }
///}
///
class SymbolAssignment : public Command {
public:
  enum AssignmentKind { Simple, Sum, Sub, Mul, Div, Shl, Shr, And, Or };
  enum AssignmentVisibility { Default, Hidden, Provide, ProvideHidden };

  SymbolAssignment(Parser &ctx, StringRef name, const Expression *expr,
                   AssignmentKind kind, AssignmentVisibility visibility)
      : Command(ctx, Kind::SymbolAssignment), _expression(expr), _symbol(name),
        _assignmentKind(Simple), _assignmentVisibility(visibility) {}

  static bool classof(const Command *c) {
    return c->getKind() == Kind::SymbolAssignment;
  }

  void dump(raw_ostream &os) const override;
  const Expression *expr() const { return _expression; }
  StringRef symbol() const { return _symbol; }
  AssignmentKind assignmentKind() const { return _assignmentKind; }
  AssignmentVisibility assignmentVisibility() const {
    return _assignmentVisibility;
  }

private:
  const Expression *_expression;
  StringRef _symbol;
  AssignmentKind _assignmentKind;
  AssignmentVisibility _assignmentVisibility;
};

/// Encodes how to sort file names or section names that are expanded from
/// wildcard operators. This typically occurs in constructs such as
/// SECTIONS {  .data : SORT_BY_NAME(*)(*) }}, where the order of the expanded
/// names is important to determine which sections go first.
enum class WildcardSortMode {
  NA,
  ByAlignment,
  ByAlignmentAndName,
  ByInitPriority,
  ByName,
  ByNameAndAlignment,
  None
};

/// Represents either a single input section name or a group of sorted input
/// section names. They specify which sections to map to a given output section.
/// Example:
///
/// SECTIONS {
///   .x: { *(.text) }
///   /*      ^~~~^         InputSectionName : InputSection  */
///   .y: { *(SORT(.text*)) }
///   /*      ^~~~~~~~~~~^  InputSectionSortedGroup : InputSection  */
/// }
class InputSection : public Command {
public:
  static bool classof(const Command *c) {
    return c->getKind() == Kind::InputSectionName ||
           c->getKind() == Kind::SortedGroup;
  }

protected:
  InputSection(Parser &ctx, Kind k) : Command(ctx, k) {}
};

class InputSectionName : public InputSection {
public:
  InputSectionName(Parser &ctx, StringRef name, bool excludeFile)
      : InputSection(ctx, Kind::InputSectionName), _name(name),
        _excludeFile(excludeFile) {}

  void dump(raw_ostream &os) const override;

  static bool classof(const Command *c) {
    return c->getKind() == Kind::InputSectionName;
  }
  bool hasExcludeFile() const { return _excludeFile; }
  StringRef name() const { return _name; }

private:
  StringRef _name;
  bool _excludeFile;
};

class InputSectionSortedGroup : public InputSection {
public:
  typedef llvm::ArrayRef<const InputSection *>::const_iterator const_iterator;

  InputSectionSortedGroup(Parser &ctx, WildcardSortMode sort,
                          const SmallVectorImpl<const InputSection *> &sections)
      : InputSection(ctx, Kind::SortedGroup), _sortMode(sort) {
    _sections = save_array<const InputSection *>(getAllocator(), sections);
  }

  void dump(raw_ostream &os) const override;
  WildcardSortMode sortMode() const { return _sortMode; }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::SortedGroup;
  }

  const_iterator begin() const { return _sections.begin(); }
  const_iterator end() const { return _sections.end(); }

private:
  WildcardSortMode _sortMode;
  llvm::ArrayRef<const InputSection *> _sections;
};

/// An output-section-command that maps a series of sections inside a given
/// file-archive pair to an output section.
/// Example:
///
/// SECTIONS {
///   .x: { *(.text) }
///   /*    ^~~~~~~^ InputSectionsCmd   */
///   .y: { w:z(SORT(.text*)) }
///   /*    ^~~~~~~~~~~~~~~~^  InputSectionsCmd  */
/// }
class InputSectionsCmd : public Command {
public:
  typedef llvm::ArrayRef<const InputSection *>::const_iterator const_iterator;
  typedef std::vector<const InputSection *> VectorTy;

  InputSectionsCmd(Parser &ctx, StringRef memberName, StringRef archiveName,
                   bool keep, WildcardSortMode fileSortMode,
                   WildcardSortMode archiveSortMode,
                   const SmallVectorImpl<const InputSection *> &sections)
      : Command(ctx, Kind::InputSectionsCmd), _memberName(memberName),
        _archiveName(archiveName), _keep(keep), _fileSortMode(fileSortMode),
        _archiveSortMode(archiveSortMode) {
    _sections = save_array<const InputSection *>(getAllocator(), sections);
  }

  void dump(raw_ostream &os) const override;

  static bool classof(const Command *c) {
    return c->getKind() == Kind::InputSectionsCmd;
  }

  StringRef memberName() const { return _memberName; }
  StringRef archiveName() const { return _archiveName; }
  const_iterator begin() const { return _sections.begin(); }
  const_iterator end() const { return _sections.end(); }
  WildcardSortMode archiveSortMode() const { return _archiveSortMode; }
  WildcardSortMode fileSortMode() const { return _fileSortMode; }

private:
  StringRef _memberName;
  StringRef _archiveName;
  bool _keep;
  WildcardSortMode _fileSortMode;
  WildcardSortMode _archiveSortMode;
  llvm::ArrayRef<const InputSection *> _sections;
};

class FillCmd : public Command {
public:
  FillCmd(Parser &ctx, ArrayRef<uint8_t> bytes) : Command(ctx, Kind::Fill) {
    _bytes = save_array<uint8_t>(getAllocator(), bytes);
  }

  void dump(raw_ostream &os) const override;

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Fill;
  }

  ArrayRef<uint8_t> bytes() { return _bytes; }

private:
  ArrayRef<uint8_t> _bytes;
};

/// A sections-command to specify which input sections and symbols compose a
/// given output section.
/// Example:
///
/// SECTIONS {
///   .x: { *(.text) ; symbol = .; }
/// /*^~~~~~~~~~~~~~~~~~~~~~~~~~~~~^   OutputSectionDescription */
///   .y: { w:z(SORT(.text*)) }
/// /*^~~~~~~~~~~~~~~~~~~~~~~~^  OutputSectionDescription  */
///   .a 0x10000 : ONLY_IF_RW { *(.data*) ; *:libc.a(SORT(*)); }
/// /*^~~~~~~~~~~~~  OutputSectionDescription ~~~~~~~~~~~~~~~~~^ */
/// }
class OutputSectionDescription : public Command {
public:
  enum Constraint { C_None, C_OnlyIfRO, C_OnlyIfRW };

  typedef llvm::ArrayRef<const Command *>::const_iterator const_iterator;

  OutputSectionDescription(
      Parser &ctx, StringRef sectionName, const Expression *address,
      const Expression *align, const Expression *subAlign, const Expression *at,
      const Expression *fillExpr, StringRef fillStream, bool alignWithInput,
      bool discard, Constraint constraint,
      const SmallVectorImpl<const Command *> &outputSectionCommands,
      ArrayRef<StringRef> phdrs)
      : Command(ctx, Kind::OutputSectionDescription), _sectionName(sectionName),
        _address(address), _align(align), _subAlign(subAlign), _at(at),
        _fillExpr(fillExpr), _fillStream(fillStream),
        _alignWithInput(alignWithInput), _discard(discard),
        _constraint(constraint) {
    _outputSectionCommands =
        save_array<const Command *>(getAllocator(), outputSectionCommands);
    _phdrs = save_array<StringRef>(getAllocator(), phdrs);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::OutputSectionDescription;
  }

  void dump(raw_ostream &os) const override;

  const_iterator begin() const { return _outputSectionCommands.begin(); }
  const_iterator end() const { return _outputSectionCommands.end(); }
  StringRef name() const { return _sectionName; }
  bool isDiscarded() const { return _discard; }
  ArrayRef<StringRef> PHDRs() const { return _phdrs; }

private:
  StringRef _sectionName;
  const Expression *_address;
  const Expression *_align;
  const Expression *_subAlign;
  const Expression *_at;
  const Expression *_fillExpr;
  StringRef _fillStream;
  bool _alignWithInput;
  bool _discard;
  Constraint _constraint;
  llvm::ArrayRef<const Command *> _outputSectionCommands;
  ArrayRef<StringRef> _phdrs;
};

/// Represents an Overlay structure as documented in
/// https://sourceware.org/binutils/docs/ld/Overlay-Description.html#Overlay-Description
class Overlay : public Command {
public:
  Overlay(Parser &ctx) : Command(ctx, Kind::Overlay) {}

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Overlay;
  }

  void dump(raw_ostream &os) const override { os << "Overlay description\n"; }
};

class PHDR {
public:
  PHDR(StringRef name, uint64_t type, bool includeFileHdr, bool includePHDRs,
       const Expression *at, uint64_t flags)
      : _name(name), _type(type), _includeFileHdr(includeFileHdr),
        _includePHDRs(includePHDRs), _at(at), _flags(flags) {}

  StringRef name() const { return _name; }
  uint64_t type() const { return _type; }
  bool hasFileHdr() const { return _includeFileHdr; }
  bool hasPHDRs() const { return _includePHDRs; }
  uint64_t flags() const { return _flags; }
  bool isNone() const;

  void dump(raw_ostream &os) const;

private:
  StringRef _name;
  uint64_t _type;
  bool _includeFileHdr;
  bool _includePHDRs;
  const Expression *_at;
  uint64_t _flags;
};

class PHDRS : public Command {
public:
  typedef ArrayRef<const PHDR *>::const_iterator const_iterator;

  PHDRS(Parser &ctx, const SmallVectorImpl<const PHDR *> &phdrs)
      : Command(ctx, Kind::PHDRS) {
    _phdrs = save_array<const PHDR *>(getAllocator(), phdrs);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::PHDRS;
  }

  void dump(raw_ostream &os) const override;
  const_iterator begin() const { return _phdrs.begin(); }
  const_iterator end() const { return _phdrs.end(); }

private:
  ArrayRef<const PHDR *> _phdrs;
};

/// Represents all the contents of the SECTIONS {} construct.
class Sections : public Command {
public:
  typedef llvm::ArrayRef<const Command *>::const_iterator const_iterator;

  Sections(Parser &ctx,
           const SmallVectorImpl<const Command *> &sectionsCommands)
      : Command(ctx, Kind::Sections) {
    _sectionsCommands =
        save_array<const Command *>(getAllocator(), sectionsCommands);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Sections;
  }

  void dump(raw_ostream &os) const override;
  const_iterator begin() const { return _sectionsCommands.begin(); }
  const_iterator end() const { return _sectionsCommands.end(); }

private:
  llvm::ArrayRef<const Command *> _sectionsCommands;
};

/// Represents a single memory block definition in a MEMORY {} command.
class MemoryBlock {
public:
  MemoryBlock(StringRef name, StringRef attr,
              const Expression *origin, const Expression *length)
      : _name(name), _attr(attr), _origin(origin), _length(length) {}

  void dump(raw_ostream &os) const;

private:
  StringRef _name;
  StringRef _attr;
  const Expression *_origin;
  const Expression *_length;
};

/// Represents all the contents of the MEMORY {} command.
class Memory : public Command {
public:
  Memory(Parser &ctx,
         const SmallVectorImpl<const MemoryBlock *> &blocks)
      : Command(ctx, Kind::Memory) {
    _blocks = save_array<const MemoryBlock *>(getAllocator(), blocks);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Memory;
  }

  void dump(raw_ostream &os) const override;

private:
  llvm::ArrayRef<const MemoryBlock *> _blocks;
};

/// Represents an extern command.
class Extern : public Command {
public:
  typedef llvm::ArrayRef<StringRef>::const_iterator const_iterator;

  Extern(Parser &ctx,
         const SmallVectorImpl<StringRef> &symbols)
      : Command(ctx, Kind::Extern) {
    _symbols = save_array<StringRef>(getAllocator(), symbols);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Extern;
  }

  void dump(raw_ostream &os) const override;
  const_iterator begin() const { return _symbols.begin(); }
  const_iterator end() const { return _symbols.end(); }

private:
  llvm::ArrayRef<StringRef> _symbols;
};

/// Stores the parse tree of a linker script.
class LinkerScript {
public:
  void dump(raw_ostream &os) const {
    for (const Command *c : _commands) {
      c->dump(os);
      if (isa<SymbolAssignment>(c))
        os << "\n";
    }
  }

  std::vector<const Command *> _commands;
};

/// Recognizes syntactic constructs of a linker script using a predictive
/// parser/recursive descent implementation.
///
/// Based on the linker script documentation available at
/// https://sourceware.org/binutils/docs/ld/Scripts.html
class Parser {
public:
  explicit Parser(std::unique_ptr<MemoryBuffer> mb)
      : _lex(std::move(mb)), _peekAvailable(false) {}

  /// Let's not allow copying of Parser class because it would be expensive
  /// to update all the AST pointers to a new buffer.
  Parser(const Parser &instance) = delete;

  /// Lex and parse the current memory buffer to create a linker script AST.
  std::error_code parse();

  /// Returns a reference to the top level node of the linker script AST.
  LinkerScript *get() { return &_script; }

  /// Returns a reference to the underlying allocator.
  llvm::BumpPtrAllocator &getAllocator() { return _alloc; }

private:
  /// Advances to the next token, either asking the Lexer to lex the next token
  /// or obtaining it from the look ahead buffer.
  void consumeToken() {
    // First check if the look ahead buffer cached the next token
    if (_peekAvailable) {
      _tok = _bufferedToken;
      _peekAvailable = false;
      return;
    }
    _lex.lex(_tok);
  }

  /// Returns the token that succeeds the current one without consuming the
  /// current token. This operation will lex an additional token and store it in
  /// a private buffer.
  const Token &peek() {
    if (_peekAvailable)
      return _bufferedToken;

    _lex.lex(_bufferedToken);
    _peekAvailable = true;
    return _bufferedToken;
  }

  void error(const Token &tok, Twine msg) {
    _lex.getSourceMgr().PrintMessage(
        llvm::SMLoc::getFromPointer(tok._range.data()),
        llvm::SourceMgr::DK_Error, msg);
  }

  bool expectAndConsume(Token::Kind kind, Twine msg) {
    if (_tok._kind != kind) {
      error(_tok, msg);
      return false;
    }
    consumeToken();
    return true;
  }

  bool isNextToken(Token::Kind kind) { return (_tok._kind == kind); }

  // Recursive descent parsing member functions
  // All of these functions consumes tokens and return an AST object,
  // represented by the Command superclass. However, note that not all AST
  // objects derive from Command. For nodes of C-like expressions, used in
  // linker scripts, the superclass is Expression. For nodes that represent
  // input sections that map to an output section, the superclass is
  // InputSection.
  //
  // Example mapping common constructs to AST nodes:
  //
  // SECTIONS {             /* Parsed to Sections class */
  //   my_symbol = 1 + 1;   /* Parsed to SymbolAssignment class */
  //   /*          ^~~> Parsed to Expression class         */
  //   .data : { *(.data) } /* Parsed to OutputSectionDescription class */
  //   /*          ^~~> Parsed to InputSectionName class   */
  //   /*        ^~~~~> Parsed to InputSectionsCmd class   */
  // }

  // ==== Expression parsing member functions ====

  /// Parse "identifier(param [, param]...)"
  ///
  /// Example:
  ///
  /// SECTIONS {
  ///   my_symbol = 0x1000 | ALIGN(other_symbol);
  ///   /*                   ^~~~> parseFunctionCall()
  /// }
  const Expression *parseFunctionCall();

  /// Ensures that the current token is an expression operand. If it is not,
  /// issues an error to the user and returns false.
  bool expectExprOperand();

  /// Parse operands of an expression, such as function calls, identifiers,
  /// literal numbers or unary operators.
  ///
  /// Example:
  ///
  /// SECTIONS {
  ///   my_symbol = 0x1000 | ALIGN(other_symbol);
  ///               ^~~~> parseExprTerminal()
  /// }
  const Expression *parseExprOperand();

  // As a reference to the precedence of C operators, consult
  // http://en.cppreference.com/w/c/language/operator_precedence

  /// Parse either a single expression operand and returns or parse an entire
  /// expression if its top-level node has a lower or equal precedence than the
  /// indicated.
  const Expression *parseExpression(unsigned precedence = 13);

  /// Parse an operator and its rhs operand, assuming that the lhs was already
  /// consumed. Keep parsing subsequent operator-operand pairs that do not
  /// exceed highestPrecedence.
  /// * lhs points to the left-hand-side operand of this operator
  /// * maxPrecedence has the maximum operator precedence level that this parse
  /// function is allowed to consume.
  const Expression *parseOperatorOperandLoop(const Expression *lhs,
                                             unsigned maxPrecedence);

  /// Parse ternary conditionals such as "(condition)? true: false;". This
  /// operator has precedence level 13 and associates right-to-left.
  const Expression *parseTernaryCondOp(const Expression *lhs);

  // ==== High-level commands parsing ====

  /// Parse the OUTPUT linker script command.
  /// Example:
  /// OUTPUT(/path/to/file)
  /// ^~~~> parseOutput()
  ///
  Output *parseOutput();

  /// Parse the OUTPUT_FORMAT linker script command.
  /// Example:
  ///
  /// OUTPUT_FORMAT(elf64-x86-64,elf64-x86-64,elf64-x86-64)
  /// ^~~~> parseOutputFormat()
  ///
  OutputFormat *parseOutputFormat();

  /// Parse the OUTPUT_ARCH linker script command.
  /// Example:
  ///
  /// OUTPUT_ARCH(i386:x86-64)
  /// ^~~~> parseOutputArch()
  ///
  OutputArch *parseOutputArch();

  /// Parse the INPUT or GROUP linker script command.
  /// Example:
  ///
  /// GROUP ( /lib/x86_64-linux-gnu/libc.so.6
  ///         /usr/lib/x86_64-linux-gnu/libc_nonshared.a
  ///         AS_NEEDED ( /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 )
  ///         -lm -l:libgcc.a )
  ///
  template<class T> T *parsePathList();
  bool parseAsNeeded(SmallVectorImpl<Path> &paths);

  /// Parse the ENTRY linker script command.
  /// Example:
  ///
  /// ENTRY(init)
  /// ^~~~> parseEntry()
  ///
  Entry *parseEntry();

  /// Parse the SEARCH_DIR linker script command.
  /// Example:
  ///
  /// SEARCH_DIR("/usr/x86_64-linux-gnu/lib64");
  /// ^~~~> parseSearchDir()
  ///
  SearchDir *parseSearchDir();

  /// Parse "symbol = expression" commands that live inside the
  /// SECTIONS directive.
  /// Example:
  ///
  /// SECTIONS {
  ///   my_symbol = 1 + 1;
  ///               ^~~~> parseExpression()
  ///   ^~~~ parseSymbolAssignment()
  /// }
  ///
  const SymbolAssignment *parseSymbolAssignment();

  /// Parse "EXCLUDE_FILE" used inside the listing of input section names.
  /// Example:
  ///
  /// SECTIONS {
  ///   .data :  { *(EXCLUDE_FILE (*crtend.o *otherfile.o) .ctors) }
  ///                ^~~~> parseExcludeFile()
  /// }
  ///
  ErrorOr<InputSectionsCmd::VectorTy> parseExcludeFile();

  /// Helper to parse SORT_BY_NAME(, SORT_BY_ALIGNMENT( and SORT_NONE(,
  /// possibly nested. Returns the number of Token::r_paren tokens that need
  /// to be consumed, while sortMode is updated with the parsed sort
  /// criteria.
  /// Example:
  ///
  /// SORT_BY_NAME(SORT_BY_ALIGNMENT(*))
  /// ^~~~ parseSortDirectives()  ~~^
  /// Returns 2, finishes with sortMode = WildcardSortMode::ByNameAndAlignment
  ///
  int parseSortDirectives(WildcardSortMode &sortMode);

  /// Parse a group of input section names that are sorted via SORT* directives.
  /// Example:
  ///   SORT_BY_NAME(SORT_BY_ALIGNMENT(*data *bss))
  const InputSection *parseSortedInputSections();

  /// Parse input section description statements.
  /// Example:
  ///
  /// SECTIONS {
  ///   .mysection : crt.o(.data* .bss SORT_BY_NAME(name*))
  ///                ^~~~ parseInputSectionsCmd()
  /// }
  const InputSectionsCmd *parseInputSectionsCmd();

  const FillCmd *parseFillCmd();

  /// Parse output section description statements.
  /// Example:
  ///
  /// SECTIONS {
  ///   .data : { crt.o(.data* .bss SORT_BY_NAME(name*)) }
  ///   ^~~~ parseOutputSectionDescription()
  /// }
  const OutputSectionDescription *parseOutputSectionDescription();

  /// Stub for parsing overlay commands. Currently unimplemented.
  const Overlay *parseOverlay();

  const PHDR *parsePHDR();

  PHDRS *parsePHDRS();

  /// Parse the SECTIONS linker script command.
  /// Example:
  ///
  ///   SECTIONS {
  ///   ^~~~ parseSections()
  ///     . = 0x100000;
  ///     .data : { *(.data) }
  ///   }
  ///
  Sections *parseSections();

  /// Parse the MEMORY linker script command.
  /// Example:
  ///
  ///   MEMORY {
  ///   ^~~~ parseMemory()
  ///     ram (rwx) : ORIGIN = 0x20000000, LENGTH = 96K
  ///     rom (rx)  : ORIGIN = 0x0,        LENGTH = 256K
  ///   }
  ///
  Memory *parseMemory();

  /// Parse the EXTERN linker script command.
  /// Example:
  ///
  /// EXTERN(symbol symbol ...)
  /// ^~~~> parseExtern()
  ///
  Extern *parseExtern();

private:
  // Owns the entire linker script AST nodes
  llvm::BumpPtrAllocator _alloc;

  // The top-level/entry-point linker script AST node
  LinkerScript _script;

  Lexer _lex;

  // Current token being analyzed
  Token _tok;

  // Annotate whether we buffered the next token to allow peeking
  bool _peekAvailable;
  Token _bufferedToken;
};

/// script::Sema traverses all parsed linker script structures and populate
/// internal data structures to be able to answer the following questions:
///
///   * According to the linker script, which input section goes first in the
///     output file layout, input section A or input section B?
///
///   * What is the name of the output section that input section A should be
///     mapped to?
///
///   * Which linker script expressions should be calculated before emitting
///     a given section?
///
///   * How to evaluate a given linker script expression?
///
class Sema {
public:
  /// From the linker script point of view, this class represents the minimum
  /// set of information to uniquely identify an input section.
  struct SectionKey {
    StringRef archivePath;
    StringRef memberPath;
    StringRef sectionName;
  };

  Sema();

  /// We can parse several linker scripts via command line whose ASTs are stored
  /// here via addLinkerScript().
  void addLinkerScript(std::unique_ptr<Parser> script) {
    _scripts.push_back(std::move(script));
  }

  const std::vector<std::unique_ptr<Parser>> &getLinkerScripts() {
    return _scripts;
  }

  /// Prepare our data structures according to the linker scripts currently in
  /// our control (control given via addLinkerScript()). Called once all linker
  /// scripts have been parsed.
  std::error_code perform();

  /// Answer if we have layout commands (section mapping rules). If we don't,
  /// the output file writer can assume there is no linker script special rule
  /// to handle.
  bool hasLayoutCommands() const { return _layoutCommands.size() > 0; }

  /// Return true if this section has a mapping rule in the linker script
  bool hasMapping(const SectionKey &key) const {
    return getLayoutOrder(key, true) >= 0;
  }

  /// Order function - used to sort input sections in the output file according
  /// to linker script custom mappings. Return true if lhs should appear before
  /// rhs.
  bool less(const SectionKey &lhs, const SectionKey &rhs) const;

  /// Retrieve the name of the output section that this input section is mapped
  /// to, according to custom linker script mappings.
  StringRef getOutputSection(const SectionKey &key) const;

  /// Retrieve all the linker script expressions that need to be evaluated
  /// before the given section is emitted. This is *not* const because the
  /// first section to retrieve a given set of expression is the only one to
  /// receive it. This set is marked as "delivered" and no other sections can
  /// retrieve this set again. If we don't do this, multiple sections may map
  /// to the same set of expressions because of wildcards rules.
  std::vector<const SymbolAssignment *> getExprs(const SectionKey &key);

  /// Evaluate a single linker script expression according to our current
  /// context (symbol table). This function is *not* constant because it can
  /// update our symbol table with new symbols calculated in this expression.
  std::error_code evalExpr(const SymbolAssignment *assgn, uint64_t &curPos);

  /// Retrieve the set of symbols defined in linker script expressions.
  const llvm::StringSet<> &getScriptDefinedSymbols() const;

  /// Queries the linker script symbol table for the value of a given symbol.
  /// This function must be called after linker script expressions evaluation
  /// has been performed (by calling evalExpr() for all expressions).
  uint64_t getLinkerScriptExprValue(StringRef name) const;

  /// Check if there are custom headers available.
  bool hasPHDRs() const;

  /// Retrieve all the headers the given output section is assigned to.
  std::vector<const PHDR *> getPHDRsForOutputSection(StringRef name) const;

  /// Retrieve program header if available.
  const PHDR *getProgramPHDR() const;

  void dump() const;

private:
  /// A custom hash operator to teach the STL how to handle our custom keys.
  /// This will be used in our hash table mapping Sections to a Layout Order
  /// number (caching results).
  struct SectionKeyHash {
    int64_t operator()(const SectionKey &k) const {
      return llvm::hash_combine(k.archivePath, k.memberPath, k.sectionName);
    }
  };

  /// Teach the STL when two section keys are the same. This will be used in
  /// our hash table mapping Sections to a Layout Order number (caching results)
  struct SectionKeyEq {
    bool operator()(const SectionKey &lhs, const SectionKey &rhs) const {
      return ((lhs.archivePath == rhs.archivePath) &&
              (lhs.memberPath == rhs.memberPath) &&
              (lhs.sectionName == rhs.sectionName));
    }
  };

  /// Given an order id, check if it matches the tuple
  /// <archivePath, memberPath, sectionName> and returns the
  /// internal id that matched, or -1 if no matches.
  int matchSectionName(int id, const SectionKey &key) const;

  /// Returns a number that will determine the order of this input section
  /// in the final layout. If coarse is true, we simply return the layour order
  /// of the higher-level node InputSectionsCmd, used to order input sections.
  /// If coarse is false, we return the layout index down to the internal
  /// InputSectionsCmd arrangement, used to get the set of preceding linker
  ///expressions.
  int getLayoutOrder(const SectionKey &key, bool coarse) const;

  /// Compare two sections that have the same mapping rule (i.e., are matched
  /// by the same InputSectionsCmd).
  /// Determine if lhs < rhs by analyzing the InputSectionsCmd structure.
  bool localCompare(int order, const SectionKey &lhs,
                    const SectionKey &rhs) const;

  /// Convert the PHDRS command into map of names to headers.
  /// Determine program header during processing.
  std::error_code collectPHDRs(const PHDRS *ph,
                               llvm::StringMap<const PHDR *> &phdrs);

  /// Build map that matches output section names to segments they should be
  /// put into.
  std::error_code buildSectionToPHDR(llvm::StringMap<const PHDR *> &phdrs);

  /// Our goal with all linearizeAST overloaded functions is to
  /// traverse the linker script AST while putting nodes in a vector and
  /// thus enforcing order among nodes (which comes first).
  ///
  /// The order among nodes is determined by their indexes in this vector
  /// (_layoutCommands). This index allows us to solve the problem of
  /// establishing the order among two different input sections: we match each
  /// input sections with their respective layout command and use the indexes
  /// of these commands to order these sections.
  ///
  /// Example:
  ///
  ///     Given the linker script:
  ///       SECTIONS {
  ///         .text : { *(.text) }
  ///         .data : { *(.data) }
  ///       }
  ///
  ///     The _layoutCommands vector should contain:
  ///         id 0 : <OutputSectionDescription> (_sectionName = ".text")
  ///         id 1 : <InputSectionsCmd> (_memberName = "*")
  ///         id 2 : <InputSectionName> (_name = ".text)
  ///         id 3 : <OutputSectionDescription> (_sectionName = ".data")
  ///         id 4 : <InputSectionsCmd> (_memberName = "*")
  ///         id 5 : <InputSectionName> (_name = ".data")
  ///
  ///     If we need to sort the following input sections:
  ///
  ///     input section A:  .text from libc.a (member errno.o)
  ///     input section B:  .data from libc.a (member write.o)
  ///
  ///     Then we match input section A with the InputSectionsCmd of id 1, and
  ///     input section B with the InputSectionsCmd of id 4. Since 1 < 4, we
  ///     put A before B.
  ///
  /// The second problem handled by the linearization of the AST is the task
  /// of finding all preceding expressions that need to be calculated before
  /// emitting a given section. This task is easier to deal with when all nodes
  /// are in a vector because otherwise we would need to traverse multiple
  /// levels of the AST to find the set of expressions that preceed a layout
  /// command.
  ///
  /// The linker script commands that are linearized ("layout commands") are:
  ///
  ///   * OutputSectionDescription, containing an output section name
  ///   * InputSectionsCmd, containing an input file name
  ///   * InputSectionName, containing a single input section name
  ///   * InputSectionSortedName, a group of input section names
  ///   * SymbolAssignment, containing an expression that may
  ///     change the address where the linker is outputting data
  ///
  void linearizeAST(const Sections *sections);
  void linearizeAST(const InputSectionsCmd *inputSections);
  void linearizeAST(const InputSection *inputSection);

  std::vector<std::unique_ptr<Parser>> _scripts;
  std::vector<const Command *> _layoutCommands;
  std::unordered_multimap<std::string, int> _memberToLayoutOrder;
  std::vector<std::pair<StringRef, int>> _memberNameWildcards;
  mutable std::unordered_map<SectionKey, int, SectionKeyHash, SectionKeyEq>
      _cacheSectionOrder, _cacheExpressionOrder;
  llvm::DenseSet<int> _deliveredExprs;
  mutable llvm::StringSet<> _definedSymbols;

  llvm::StringMap<llvm::SmallVector<const PHDR *, 2>> _sectionToPHDR;
  const PHDR *_programPHDR;

  Expression::SymbolTableTy _symbolTable;
};

llvm::BumpPtrAllocator &Command::getAllocator() const {
  return _ctx.getAllocator();
}
llvm::BumpPtrAllocator &Expression::getAllocator() const {
  return _ctx.getAllocator();
}
} // end namespace script
} // end namespace lld

#endif
