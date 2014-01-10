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

namespace lld {
namespace moduledef {

enum class Kind {
  unknown,
  eof,
  identifier,
  equal,
  kw_data,
  kw_exports,
  kw_noname,
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
    _sourceManager.AddNewSourceBuffer(mb.release(), llvm::SMLoc());
  }

  Token lex();
  const llvm::SourceMgr &getSourceMgr() const { return _sourceManager; }

private:
  StringRef _buffer;
  llvm::SourceMgr _sourceManager;
};

class Parser {
public:
  explicit Parser(Lexer &lex) : _lex(lex) {}
  bool parse(std::vector<PECOFFLinkingContext::ExportDesc> &result);

private:
  void consumeToken();
  void ungetToken();
  void error(const Token &tok, Twine msg);

  bool parseExport(PECOFFLinkingContext::ExportDesc &result);

  Lexer &_lex;
  Token _tok;
  std::vector<Token> _tokBuf;
};
}
}

#endif
