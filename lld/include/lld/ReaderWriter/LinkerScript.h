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
    identifier,
    libname,
    comma,
    l_paren,
    r_paren,
    kw_entry,
    kw_group,
    kw_output_format,
    kw_output_arch,
    kw_as_needed
  };

  Token() : _kind(unknown) {}
  Token(StringRef range, Kind kind) : _range(range), _kind(kind) {}

  void dump(raw_ostream &os) const;

  StringRef _range;
  Kind _kind;
};

class Lexer {
public:
  explicit Lexer(std::unique_ptr<MemoryBuffer> mb)
      : _buffer(mb->getBuffer()) {
    _sourceManager.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
  }

  void lex(Token &tok);

  const llvm::SourceMgr &getSourceMgr() const { return _sourceManager; }

private:
  bool canStartName(char c) const;
  bool canContinueName(char c) const;
  void skipWhitespace();

  Token _current;
  /// \brief The current buffer state.
  StringRef _buffer;
  // Lexer owns the input files.
  llvm::SourceMgr _sourceManager;
};

class Command {
public:
  enum class Kind { Entry, OutputFormat, OutputArch, Group, };

  Kind getKind() const { return _kind; }

  virtual void dump(raw_ostream &os) const = 0;

  virtual ~Command() {}

protected:
  explicit Command(Kind k) : _kind(k) {}

private:
  Kind _kind;
};

class OutputFormat : public Command {
public:
  explicit OutputFormat(StringRef format) : Command(Kind::OutputFormat) {
    _formats.push_back(format);
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
      os << format;
    }
    os << ")\n";
  }

  virtual void addOutputFormat(StringRef format) { _formats.push_back(format); }

  range<StringRef *> getFormats() { return _formats; }

private:
  std::vector<StringRef> _formats;
};

class OutputArch : public Command {
public:
  explicit OutputArch(StringRef arch)
      : Command(Kind::OutputArch), _arch(arch) {}

  static bool classof(const Command *c) {
    return c->getKind() == Kind::OutputArch;
  }

  void dump(raw_ostream &os) const override {
    os << "OUTPUT_arch(" << getArch() << ")\n";
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

class Group : public Command {
public:
  template <class RangeT>
  explicit Group(RangeT range) : Command(Kind::Group) {
    std::copy(std::begin(range), std::end(range), std::back_inserter(_paths));
  }

  static bool classof(const Command *c) { return c->getKind() == Kind::Group; }

  void dump(raw_ostream &os) const override {
    os << "GROUP(";
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

  const std::vector<Path> &getPaths() const { return _paths; }

private:
  std::vector<Path> _paths;
};

class Entry : public Command {
public:
  explicit Entry(StringRef entryName) :
      Command(Kind::Entry), _entryName(entryName) { }

  static bool classof(const Command *c) {
    return c->getKind() == Kind::Entry;
  }

  void dump(raw_ostream &os) const override {
    os << "ENTRY(" << _entryName << ")\n";
  }

  const StringRef getEntryName() const {
    return _entryName;
  }

private:
  StringRef _entryName;
};

class LinkerScript {
public:
  void dump(raw_ostream &os) const {
    for (const Command *c : _commands)
      c->dump(os);
  }

  std::vector<Command *> _commands;
};

class Parser {
public:
  explicit Parser(Lexer &lex) : _lex(lex) {}

  LinkerScript *parse();

private:
  void consumeToken() { _lex.lex(_tok); }

  void error(const Token &tok, Twine msg) {
    _lex.getSourceMgr()
        .PrintMessage(llvm::SMLoc::getFromPointer(tok._range.data()),
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

  OutputFormat *parseOutputFormat();
  OutputArch *parseOutputArch();
  Group *parseGroup();
  bool parseAsNeeded(std::vector<Path> &paths);
  Entry *parseEntry();

private:
  llvm::BumpPtrAllocator _alloc;
  LinkerScript _script;
  Lexer &_lex;
  Token _tok;
};
} // end namespace script
} // end namespace lld

#endif
