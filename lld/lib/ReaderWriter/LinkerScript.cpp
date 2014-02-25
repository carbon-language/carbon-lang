//===- ReaderWriter/LinkerScript.cpp --------------------------------------===//
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

#include "lld/ReaderWriter/LinkerScript.h"

namespace lld {
namespace script {
void Token::dump(raw_ostream &os) const {
  switch (_kind) {
#define CASE(name)                              \
  case Token::name:                             \
    os << #name ": ";                           \
    break;
  CASE(eof)
  CASE(identifier)
  CASE(kw_as_needed)
  CASE(kw_entry)
  CASE(kw_group)
  CASE(kw_output_format)
  CASE(kw_output_arch)
  CASE(quotedString)
  CASE(comma)
  CASE(l_paren)
  CASE(r_paren)
  CASE(unknown)
#undef CASE
  }
  os << _range << "\n";
}

bool Lexer::canStartName(char c) const {
  switch (c) {
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
  case '_': case '.': case '$': case '/': case '\\':
    return true;
  default:
    return false;
  }
}

bool Lexer::canContinueName(char c) const {
  switch (c) {
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
  case '0': case '1': case '2': case '3': case '4': case '5': case '6':
  case '7': case '8': case '9':
  case '_': case '.': case '$': case '/': case '\\': case '~': case '=':
  case '+':
  case '[':
  case ']':
  case '*':
  case '?':
  case '-':
  case ':':
    return true;
  default:
    return false;
  }
}

void Lexer::lex(Token &tok) {
  skipWhitespace();
  if (_buffer.empty()) {
    tok = Token(_buffer, Token::eof);
    return;
  }
  switch (_buffer[0]) {
  case 0:
    tok = Token(_buffer.substr(0, 1), Token::eof);
    _buffer = _buffer.drop_front();
    return;
  case '(':
    tok = Token(_buffer.substr(0, 1), Token::l_paren);
    _buffer = _buffer.drop_front();
    return;
  case ')':
    tok = Token(_buffer.substr(0, 1), Token::r_paren);
    _buffer = _buffer.drop_front();
    return;
  case ',':
    tok = Token(_buffer.substr(0, 1), Token::comma);
    _buffer = _buffer.drop_front();
    return;
  default:
    // Quoted strings ?
    if ((_buffer[0] == '\"') || (_buffer[0] == '\'')) {
      char c = _buffer[0];
      _buffer = _buffer.drop_front();
      auto quotedStringEnd = _buffer.find(c);
      if (quotedStringEnd == StringRef::npos || quotedStringEnd == 0)
        break;
      StringRef word = _buffer.substr(0, quotedStringEnd);
      tok = Token(word, Token::quotedString);
      _buffer = _buffer.drop_front(quotedStringEnd + 1);
      return;
    }
    /// keyword or identifer.
    if (!canStartName(_buffer[0]))
      break;
    auto endIter =
        std::find_if(_buffer.begin() + 1, _buffer.end(), [=](char c) {
      return !canContinueName(c);
    });
    StringRef::size_type end =
        endIter == _buffer.end() ? StringRef::npos
                                 : std::distance(_buffer.begin(), endIter);
    if (end == StringRef::npos || end == 0)
      break;
    StringRef word = _buffer.substr(0, end);
    Token::Kind kind = llvm::StringSwitch<Token::Kind>(word)
                           .Case("OUTPUT_FORMAT", Token::kw_output_format)
                           .Case("OUTPUT_ARCH", Token::kw_output_arch)
                           .Case("GROUP", Token::kw_group)
                           .Case("AS_NEEDED", Token::kw_as_needed)
                           .Case("ENTRY", Token::kw_entry)
                           .Default(Token::identifier);
    tok = Token(word, kind);
    _buffer = _buffer.drop_front(end);
    return;
  }
  tok = Token(_buffer.substr(0, 1), Token::unknown);
  _buffer = _buffer.drop_front();
}

void Lexer::skipWhitespace() {
  while (true) {
    if (_buffer.empty())
      return;
    switch (_buffer[0]) {
    case ' ':
    case '\r':
    case '\n':
    case '\t':
      _buffer = _buffer.drop_front();
      break;
    // Potential comment.
    case '/':
      if (_buffer.size() >= 2 && _buffer[1] == '*') {
        // Skip starting /*
        _buffer = _buffer.drop_front(2);
        // If the next char is also a /, it's not the end.
        if (!_buffer.empty() && _buffer[0] == '/')
          _buffer = _buffer.drop_front();

        // Scan for /'s. We're done if it is preceded by a *.
        while (true) {
          if (_buffer.empty())
            break;
          _buffer = _buffer.drop_front();
          if (_buffer.data()[-1] == '/' && _buffer.data()[-2] == '*')
            break;
        }
      } else
        return;
      break;
    default:
      return;
    }
  }
}

LinkerScript *Parser::parse() {
  // Get the first token.
  _lex.lex(_tok);
  // Parse top level commands.
  while (true) {
    switch (_tok._kind) {
    case Token::eof:
      return &_script;
    case Token::kw_output_format: {
      auto outputFormat = parseOutputFormat();
      if (!outputFormat)
        return nullptr;
      _script._commands.push_back(outputFormat);
      break;
    }
    case Token::kw_output_arch: {
      auto outputArch = parseOutputArch();
      if (!outputArch)
        return nullptr;
      _script._commands.push_back(outputArch);
      break;
    }
    case Token::kw_group: {
      auto group = parseGroup();
      if (!group)
        return nullptr;
      _script._commands.push_back(group);
      break;
    }
    case Token::kw_as_needed:
      // Not allowed at top level.
      return nullptr;
    case Token::kw_entry: {
      Entry *entry = parseEntry();
      if (!entry)
        return nullptr;
      _script._commands.push_back(entry);
      break;
    }
    default:
      // Unexpected.
      return nullptr;
    }
  }

  return nullptr;
}

// Parse OUTPUT_FORMAT(ident)
OutputFormat *Parser::parseOutputFormat() {
  assert(_tok._kind == Token::kw_output_format && "Expected OUTPUT_FORMAT!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  if (_tok._kind != Token::quotedString && _tok._kind != Token::identifier) {
    error(_tok, "Expected identifier/string in OUTPUT_FORMAT.");
    return nullptr;
  }

  auto ret = new (_alloc) OutputFormat(_tok._range);
  consumeToken();

  do {
    if (isNextToken(Token::comma))
      consumeToken();
    else
      break;
    if (_tok._kind != Token::quotedString && _tok._kind != Token::identifier) {
      error(_tok, "Expected identifier/string in OUTPUT_FORMAT.");
      return nullptr;
    }
    ret->addOutputFormat(_tok._range);
    consumeToken();
  } while (isNextToken(Token::comma));

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;

  return ret;
}

// Parse OUTPUT_ARCH(ident)
OutputArch *Parser::parseOutputArch() {
  assert(_tok._kind == Token::kw_output_arch && "Expected OUTPUT_ARCH!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  if (_tok._kind != Token::identifier) {
    error(_tok, "Expected identifier in OUTPUT_ARCH.");
    return nullptr;
  }

  auto ret = new (_alloc) OutputArch(_tok._range);
  consumeToken();

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;

  return ret;
}

// Parse GROUP(file ...)
Group *Parser::parseGroup() {
  assert(_tok._kind == Token::kw_group && "Expected GROUP!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  std::vector<Path> paths;

  while (_tok._kind == Token::identifier || _tok._kind == Token::kw_as_needed) {
    switch (_tok._kind) {
    case Token::identifier:
      paths.push_back(Path(_tok._range));
      consumeToken();
      break;
    case Token::kw_as_needed:
      if (!parseAsNeeded(paths))
        return nullptr;
      break;
    default:
      llvm_unreachable("Invalid token.");
    }
  }

  auto ret = new (_alloc) Group(paths);

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;

  return ret;
}

// Parse AS_NEEDED(file ...)
bool Parser::parseAsNeeded(std::vector<Path> &paths) {
  assert(_tok._kind == Token::kw_as_needed && "Expected AS_NEEDED!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return false;

  while (_tok._kind == Token::identifier) {
    paths.push_back(Path(_tok._range, true));
    consumeToken();
  }

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return false;
  return true;
}

// Parse ENTRY(ident)
Entry *Parser::parseEntry() {
  assert(_tok._kind == Token::kw_entry && "Expected ENTRY!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;
  if (_tok._kind != Token::identifier) {
    error(_tok, "expected identifier in ENTRY");
    return nullptr;
  }
  StringRef entryName(_tok._range);
  consumeToken();
  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;
  return new (_alloc) Entry(entryName);
}

} // end namespace script
} // end namespace lld
