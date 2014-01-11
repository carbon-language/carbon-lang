//===- lib/Driver/WinLinkModuleDef.cpp ------------------------------------===//
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

#include "lld/Driver/WinLinkModuleDef.h"
#include "llvm/ADT/StringSwitch.h"

namespace lld {
namespace moduledef {

Token Lexer::lex() {
  _buffer = _buffer.trim();
  if (_buffer.empty() || _buffer[0] == '\0')
    return Token(Kind::eof, _buffer);

  switch (_buffer[0]) {
  case '=':
    _buffer = _buffer.drop_front();
    return Token(Kind::equal, "=");
  case ',':
    _buffer = _buffer.drop_front();
    return Token(Kind::comma, ",");
  case '"': {
    size_t end = _buffer.find('"', 1);
    Token ret;
    if (end == _buffer.npos) {
      ret = Token(Kind::identifier, _buffer.substr(1, end));
      _buffer = "";
    } else {
      ret = Token(Kind::identifier, _buffer.substr(1, end - 1));
      _buffer = _buffer.drop_front(end);
    }
    return ret;
  }
  default: {
    size_t end = _buffer.find_first_not_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_.*~+!@#$%^&*()/");
    StringRef word = _buffer.substr(0, end);
    Kind kind = llvm::StringSwitch<Kind>(word)
                    .Case("DATA", Kind::kw_data)
                    .Case("EXPORTS", Kind::kw_exports)
                    .Case("HEAPSIZE", Kind::kw_heapsize)
                    .Case("NONAME", Kind::kw_noname)
                    .Default(Kind::identifier);
    _buffer = (end == _buffer.npos) ? "" : _buffer.drop_front(end);
    return Token(kind, word);
  }
  }
}

void Parser::consumeToken() {
  if (_tokBuf.empty()) {
    _tok = _lex.lex();
    return;
  }
  _tok = _tokBuf.back();
  _tokBuf.pop_back();
}

bool Parser::consumeTokenAsInt(uint64_t &result) {
  consumeToken();
  if (_tok._kind != Kind::identifier) {
    ungetToken();
    error(_tok, "Integer expected");
    return false;
  }
  if (_tok._range.getAsInteger(10, result)) {
    error(_tok, "Integer expected");
    return false;
  }
  return true;
}

void Parser::ungetToken() { _tokBuf.push_back(_tok); }

void Parser::error(const Token &tok, Twine msg) {
  _lex.getSourceMgr().PrintMessage(
      llvm::SMLoc::getFromPointer(tok._range.data()), llvm::SourceMgr::DK_Error,
      msg);
}

llvm::Optional<Directive *> Parser::parse() {
  consumeToken();
  if (_tok._kind == Kind::kw_exports) {
    std::vector<PECOFFLinkingContext::ExportDesc> exports;
    for (;;) {
      PECOFFLinkingContext::ExportDesc desc;
      if (!parseExport(desc))
        break;
      exports.push_back(desc);
    }
    return new (_alloc) Exports(exports);
  }
  if (_tok._kind == Kind::kw_heapsize) {
    uint64_t reserve, commit;
    if (!parseHeapsize(reserve, commit))
      return llvm::None;
    return new (_alloc) Heapsize(reserve, commit);
  }
  error(_tok, Twine("Unknown directive: ") + _tok._range);
  return llvm::None;
}

bool Parser::parseExport(PECOFFLinkingContext::ExportDesc &result) {
  consumeToken();
  if (_tok._kind != Kind::identifier) {
    ungetToken();
    return false;
  }
  result.name = _tok._range;

  for (;;) {
    consumeToken();
    if (_tok._kind == Kind::identifier && _tok._range[0] == '@') {
      _tok._range.drop_front().getAsInteger(10, result.ordinal);
      consumeToken();
      if (_tok._kind == Kind::kw_noname) {
        result.noname = true;
      } else {
        ungetToken();
      }
      continue;
    }
    if (_tok._kind == Kind::kw_data) {
      result.isData = true;
      continue;
    }
    ungetToken();
    return true;
  }
}

bool Parser::parseHeapsize(uint64_t &reserve, uint64_t &commit) {
  if (!consumeTokenAsInt(reserve))
    return false;

  consumeToken();
  if (_tok._kind != Kind::comma) {
    ungetToken();
    commit = 0;
    return true;
  }

  if (!consumeTokenAsInt(commit))
    return false;
  return true;
}

} // moddef
} // namespace lld
