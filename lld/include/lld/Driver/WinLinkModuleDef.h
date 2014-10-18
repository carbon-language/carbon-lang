//===- lld/Driver/WinLinkModuleDef.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Windows module definition file parser.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_WIN_LINK_MODULE_DEF_H
#define LLD_DRIVER_WIN_LINK_MODULE_DEF_H

#include "lld/Core/LLVM.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace lld {
namespace moduledef {

enum class Kind {
  unknown,
  eof,
  identifier,
  comma,
  equal,
  kw_base,
  kw_data,
  kw_exports,
  kw_heapsize,
  kw_library,
  kw_name,
  kw_noname,
  kw_private,
  kw_stacksize,
  kw_version,
};

class Token {
public:
  Token() : _kind(Kind::unknown) {}
  Token(Kind kind, StringRef range) : _kind(kind), _range(range) {}

  Kind _kind;
  StringRef _range;
};

class Lexer {
public:
  explicit Lexer(std::unique_ptr<MemoryBuffer> mb) : _buffer(mb->getBuffer()) {
    _sourceManager.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
  }

  Token lex();
  const llvm::SourceMgr &getSourceMgr() const { return _sourceManager; }

private:
  StringRef _buffer;
  llvm::SourceMgr _sourceManager;
};

class Directive {
public:
  enum class Kind { exports, heapsize, library, name, stacksize, version };

  Kind getKind() const { return _kind; }
  virtual ~Directive() {}

protected:
  explicit Directive(Kind k) : _kind(k) {}

private:
  Kind _kind;
};

class Exports : public Directive {
public:
  explicit Exports(const std::vector<PECOFFLinkingContext::ExportDesc> &exports)
      : Directive(Kind::exports), _exports(exports) {}

  static bool classof(const Directive *dir) {
    return dir->getKind() == Kind::exports;
  }

  const std::vector<PECOFFLinkingContext::ExportDesc> &getExports() const {
    return _exports;
  }

private:
  const std::vector<PECOFFLinkingContext::ExportDesc> _exports;
};

template <Directive::Kind kind>
class MemorySize : public Directive {
public:
  MemorySize(uint64_t reserve, uint64_t commit)
      : Directive(kind), _reserve(reserve), _commit(commit) {}

  static bool classof(const Directive *dir) {
    return dir->getKind() == kind;
  }

  uint64_t getReserve() const { return _reserve; }
  uint64_t getCommit() const { return _commit; }

private:
  const uint64_t _reserve;
  const uint64_t _commit;
};

typedef MemorySize<Directive::Kind::heapsize> Heapsize;
typedef MemorySize<Directive::Kind::stacksize> Stacksize;

class Name : public Directive {
public:
  Name(StringRef outputPath, uint64_t baseaddr)
      : Directive(Kind::name), _outputPath(outputPath), _baseaddr(baseaddr) {}

  static bool classof(const Directive *dir) {
    return dir->getKind() == Kind::name;
  }

  StringRef getOutputPath() const { return _outputPath; }
  uint64_t getBaseAddress() const { return _baseaddr; }

private:
  const std::string _outputPath;
  const uint64_t _baseaddr;
};

class Library : public Directive {
public:
  Library(StringRef name, uint64_t baseaddr)
      : Directive(Kind::library), _name(name), _baseaddr(baseaddr) {}

  static bool classof(const Directive *dir) {
    return dir->getKind() == Kind::library;
  }

  StringRef getName() const { return _name; }
  uint64_t getBaseAddress() const { return _baseaddr; }

private:
  const std::string _name;
  const uint64_t _baseaddr;
};

class Version : public Directive {
public:
  Version(int major, int minor)
      : Directive(Kind::version), _major(major), _minor(minor) {}

  static bool classof(const Directive *dir) {
    return dir->getKind() == Kind::version;
  }

  int getMajorVersion() const { return _major; }
  int getMinorVersion() const { return _minor; }

private:
  const int _major;
  const int _minor;
};

class Parser {
public:
  Parser(Lexer &lex, llvm::BumpPtrAllocator &alloc)
      : _lex(lex), _alloc(alloc) {}

  bool parse(std::vector<Directive *> &ret);

private:
  void consumeToken();
  bool consumeTokenAsInt(uint64_t &result);
  bool expectAndConsume(Kind kind, Twine msg);

  void ungetToken();
  void error(const Token &tok, Twine msg);

  bool parseOne(Directive *&dir);
  bool parseExport(PECOFFLinkingContext::ExportDesc &result);
  bool parseMemorySize(uint64_t &reserve, uint64_t &commit);
  bool parseName(std::string &outfile, uint64_t &baseaddr);
  bool parseVersion(int &major, int &minor);

  Lexer &_lex;
  llvm::BumpPtrAllocator &_alloc;
  Token _tok;
  std::vector<Token> _tokBuf;
};
}
}

#endif
