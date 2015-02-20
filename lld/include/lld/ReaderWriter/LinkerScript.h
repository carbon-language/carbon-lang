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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>
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
    kw_group,
    kw_hidden,
    kw_input,
    kw_keep,
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
    Group,
    Input,
    InputSectionsCmd,
    Output,
    OutputArch,
    OutputFormat,
    OutputSectionDescription,
    Overlay,
    SearchDir,
    Sections,
    SymbolAssignment,
  };

  Kind getKind() const { return _kind; }
  inline llvm::BumpPtrAllocator &getAllocator() const;

  virtual void dump(raw_ostream &os) const = 0;

  virtual ~Command() {}

protected:
  explicit Command(class Parser &ctx, Kind k) : _ctx(ctx), _kind(k) {}

private:
  Parser &_ctx;
  Kind _kind;
};

class Output : public Command {
public:
  explicit Output(Parser &ctx, StringRef outputFileName)
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
  explicit OutputFormat(Parser &ctx, const SmallVectorImpl<StringRef> &formats)
      : Command(ctx, Kind::OutputFormat) {
    size_t numFormats = formats.size();
    StringRef *formatsStart = getAllocator().Allocate<StringRef>(numFormats);
    std::copy(std::begin(formats), std::end(formats), formatsStart);
    _formats = llvm::makeArrayRef(formatsStart, numFormats);
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
  explicit OutputArch(Parser &ctx, StringRef arch)
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
  explicit Path(StringRef path, bool asNeeded = false, bool isLib = false)
      : _path(path), _asNeeded(asNeeded), _isDashlPrefix(isLib) {}
};

template<Command::Kind K>
class PathList : public Command {
public:
  PathList(Parser &ctx, StringRef name, const SmallVectorImpl<Path> &paths)
      : Command(ctx, K), _name(name) {
    size_t numPaths = paths.size();
    Path *pathsStart = getAllocator().template Allocate<Path>(numPaths);
    std::copy(std::begin(paths), std::end(paths), pathsStart);
    _paths = llvm::makeArrayRef(pathsStart, numPaths);
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
  explicit Entry(Parser &ctx, StringRef entryName)
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
  explicit SearchDir(Parser &ctx, StringRef searchPath)
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
  enum class Kind { Constant, Symbol, FunctionCall, Unary, BinOp,
                    TernaryConditional };
  Kind getKind() const { return _kind; }
  inline llvm::BumpPtrAllocator &getAllocator() const;
  virtual void dump(raw_ostream &os) const = 0;
  virtual ~Expression() {}

protected:
  explicit Expression(class Parser &ctx, Kind k) : _ctx(ctx), _kind(k) {}

private:
  Parser &_ctx;
  Kind _kind;
};

/// A constant value is stored as unsigned because it represents absolute
/// values. We represent negative numbers by composing the unary '-' operator
/// with a constant.
class Constant : public Expression {
public:
  explicit Constant(Parser &ctx, uint64_t num)
      : Expression(ctx, Kind::Constant), _num(num) {}
  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::Constant;
  }

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

private:
  StringRef _name;
};

class FunctionCall : public Expression {
public:
  FunctionCall(Parser &ctx, StringRef name,
               const SmallVectorImpl<const Expression *> &args)
      : Expression(ctx, Kind::FunctionCall), _name(name) {
    size_t numArgs = args.size();
    const Expression **argsStart =
        getAllocator().Allocate<const Expression *>(numArgs);
    std::copy(std::begin(args), std::end(args), argsStart);
    _args = llvm::makeArrayRef(argsStart, numArgs);
  }

  void dump(raw_ostream &os) const override;

  static bool classof(const Expression *c) {
    return c->getKind() == Kind::FunctionCall;
  }

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
  enum AssignmentVisibility { Normal, Hidden, Provide, ProvideHidden };

  SymbolAssignment(Parser &ctx, StringRef name, const Expression *expr,
                   AssignmentKind kind, AssignmentVisibility visibility)
      : Command(ctx, Kind::SymbolAssignment), _expression(expr), _symbol(name),
        _assignmentKind(Simple), _assignmentVisibility(visibility) {}

  static bool classof(const Command *c) {
    return c->getKind() == Kind::SymbolAssignment;
  }

  void dump(raw_ostream &os) const override;

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
class InputSection {
public:
  enum class Kind { InputSectionName, SortedGroup };

  Kind getKind() const { return _kind; }
  inline llvm::BumpPtrAllocator &getAllocator() const;

  virtual void dump(raw_ostream &os) const = 0;

  virtual ~InputSection() {}

protected:
  explicit InputSection(Parser &ctx, Kind k) : _ctx(ctx), _kind(k) {}

private:
  Parser &_ctx;
  Kind _kind;
};

class InputSectionName : public InputSection {
public:
  InputSectionName(Parser &ctx, StringRef name, bool excludeFile)
      : InputSection(ctx, Kind::InputSectionName), _name(name),
        _excludeFile(excludeFile) {}

  void dump(raw_ostream &os) const override;

  static bool classof(const InputSection *c) {
    return c->getKind() == Kind::InputSectionName;
  }
  bool hasExcludeFile() const { return _excludeFile; }

private:
  StringRef _name;
  bool _excludeFile;
};

class InputSectionSortedGroup : public InputSection {
public:
  InputSectionSortedGroup(Parser &ctx, WildcardSortMode sort,
                          const SmallVectorImpl<const InputSection *> &sections)
      : InputSection(ctx, Kind::SortedGroup), _sortMode(sort) {
    size_t numSections = sections.size();
    const InputSection **sectionsStart =
        getAllocator().Allocate<const InputSection *>(numSections);
    std::copy(std::begin(sections), std::end(sections), sectionsStart);
    _sections = llvm::makeArrayRef(sectionsStart, numSections);
  }

  void dump(raw_ostream &os) const override;
  WildcardSortMode getSortMode() const { return _sortMode; }

  static bool classof(const InputSection *c) {
    return c->getKind() == Kind::SortedGroup;
  }

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
  typedef std::vector<const InputSection *> VectorTy;

  InputSectionsCmd(Parser &ctx, StringRef fileName, StringRef archiveName,
                   bool keep, WildcardSortMode fileSortMode,
                   WildcardSortMode archiveSortMode,
                   const SmallVectorImpl<const InputSection *> &sections)
      : Command(ctx, Kind::InputSectionsCmd), _fileName(fileName),
        _archiveName(archiveName), _keep(keep), _fileSortMode(fileSortMode),
        _archiveSortMode(archiveSortMode) {
    size_t numSections = sections.size();
    const InputSection **sectionsStart =
        getAllocator().Allocate<const InputSection *>(numSections);
    std::copy(std::begin(sections), std::end(sections), sectionsStart);
    _sections = llvm::makeArrayRef(sectionsStart, numSections);
  }

  void dump(raw_ostream &os) const override;

  static bool classof(const Command *c) {
    return c->getKind() == Kind::InputSectionsCmd;
  }

private:
  StringRef _fileName;
  StringRef _archiveName;
  bool _keep;
  WildcardSortMode _fileSortMode;
  WildcardSortMode _archiveSortMode;
  llvm::ArrayRef<const InputSection *> _sections;
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

  OutputSectionDescription(
      Parser &ctx, StringRef sectionName, const Expression *address,
      const Expression *align, const Expression *subAlign, const Expression *at,
      const Expression *fillExpr, StringRef fillStream, bool alignWithInput,
      bool discard, Constraint constraint,
      const SmallVectorImpl<const Command *> &outputSectionCommands)
      : Command(ctx, Kind::OutputSectionDescription), _sectionName(sectionName),
        _address(address), _align(align), _subAlign(subAlign), _at(at),
        _fillExpr(fillExpr), _fillStream(fillStream),
        _alignWithInput(alignWithInput), _discard(discard),
        _constraint(constraint) {
    size_t numCommands = outputSectionCommands.size();
    const Command **commandsStart =
        getAllocator().Allocate<const Command *>(numCommands);
    std::copy(std::begin(outputSectionCommands),
              std::end(outputSectionCommands), commandsStart);
    _outputSectionCommands = llvm::makeArrayRef(commandsStart, numCommands);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::OutputSectionDescription;
  }

  void dump(raw_ostream &os) const override;

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

/// Represents all the contents of the SECTIONS {} construct.
class Sections : public Command {
public:
  Sections(Parser &ctx,
           const SmallVectorImpl<const Command *> &sectionsCommands)
      : Command(ctx, Kind::Sections) {
    size_t numCommands = sectionsCommands.size();
    const Command **commandsStart =
        getAllocator().Allocate<const Command *>(numCommands);
    std::copy(std::begin(sectionsCommands), std::end(sectionsCommands),
              commandsStart);
    _sectionsCommands = llvm::makeArrayRef(commandsStart, numCommands);
  }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Sections;
  }

  void dump(raw_ostream &os) const override;

private:
  llvm::ArrayRef<const Command *> _sectionsCommands;
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

llvm::BumpPtrAllocator &Command::getAllocator() const {
  return _ctx.getAllocator();
}
llvm::BumpPtrAllocator &Expression::getAllocator() const {
  return _ctx.getAllocator();
}
llvm::BumpPtrAllocator &InputSection::getAllocator() const {
  return _ctx.getAllocator();
}
} // end namespace script
} // end namespace lld

#endif
