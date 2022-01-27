//===- Lexer.cpp - MLIR Lexer Implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexer for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;

using llvm::SMLoc;

// Returns true if 'c' is an allowable punctuation character: [$._-]
// Returns false otherwise.
static bool isPunct(char c) {
  return c == '$' || c == '.' || c == '_' || c == '-';
}

Lexer::Lexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

/// Encode the specified source location information into an attribute for
/// attachment to the IR.
Location Lexer::getEncodedSourceLocation(llvm::SMLoc loc) {
  auto &sourceMgr = getSourceMgr();
  unsigned mainFileID = sourceMgr.getMainFileID();

  // TODO: Fix performance issues in SourceMgr::getLineAndColumn so that we can
  //       use it here.
  auto &bufferInfo = sourceMgr.getBufferInfo(mainFileID);
  unsigned lineNo = bufferInfo.getLineNumber(loc.getPointer());
  unsigned column =
      (loc.getPointer() - bufferInfo.getPointerForLineNumber(lineNo)) + 1;
  auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);

  return FileLineColLoc::get(context, buffer->getBufferIdentifier(), lineNo,
                             column);
}

/// emitError - Emit an error message and return an Token::error token.
Token Lexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(getEncodedSourceLocation(SMLoc::getFromPointer(loc)),
                  message);
  return formToken(Token::error, loc);
}

Token Lexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    default:
      // Handle bare identifiers.
      if (isalpha(curPtr[-1]))
        return lexBareIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    case '_':
      // Handle bare identifiers.
      return lexBareIdentifierOrKeyword(tokStart);

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(Token::eof, tokStart);
      continue;

    case ':':
      return formToken(Token::colon, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '.':
      return lexEllipsis(tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '[':
      return formToken(Token::l_square, tokStart);
    case ']':
      return formToken(Token::r_square, tokStart);
    case '<':
      return formToken(Token::less, tokStart);
    case '>':
      return formToken(Token::greater, tokStart);
    case '=':
      return formToken(Token::equal, tokStart);

    case '+':
      return formToken(Token::plus, tokStart);
    case '*':
      return formToken(Token::star, tokStart);
    case '-':
      if (*curPtr == '>') {
        ++curPtr;
        return formToken(Token::arrow, tokStart);
      }
      return formToken(Token::minus, tokStart);

    case '?':
      return formToken(Token::question, tokStart);

    case '/':
      if (*curPtr == '/') {
        skipComment();
        continue;
      }
      return emitError(tokStart, "unexpected character");

    case '@':
      return lexAtIdentifier(tokStart);

    case '!':
      LLVM_FALLTHROUGH;
    case '^':
      LLVM_FALLTHROUGH;
    case '#':
      LLVM_FALLTHROUGH;
    case '%':
      return lexPrefixedIdentifier(tokStart);
    case '"':
      return lexString(tokStart);

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex an '@foo' identifier.
///
///   symbol-ref-id ::= `@` (bare-id | string-literal)
///
Token Lexer::lexAtIdentifier(const char *tokStart) {
  char cur = *curPtr++;

  // Try to parse a string literal, if present.
  if (cur == '"') {
    Token stringIdentifier = lexString(curPtr);
    if (stringIdentifier.is(Token::error))
      return stringIdentifier;
    return formToken(Token::at_identifier, tokStart);
  }

  // Otherwise, these always start with a letter or underscore.
  if (!isalpha(cur) && cur != '_')
    return emitError(curPtr - 1,
                     "@ identifier expected to start with letter or '_'");

  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '.')
    ++curPtr;
  return formToken(Token::at_identifier, tokStart);
}

/// Lex a bare identifier or keyword that starts with a letter.
///
///   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
///   integer-type ::= `[su]?i[1-9][0-9]*`
///
Token Lexer::lexBareIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_.$]*
  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '.')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr - tokStart);

  auto isAllDigit = [](StringRef str) {
    return llvm::all_of(str, [](char c) { return llvm::isDigit(c); });
  };

  // Check for i123, si456, ui789.
  if ((spelling.size() > 1 && tokStart[0] == 'i' &&
       isAllDigit(spelling.drop_front())) ||
      ((spelling.size() > 2 && tokStart[1] == 'i' &&
        (tokStart[0] == 's' || tokStart[0] == 'u')) &&
       isAllDigit(spelling.drop_front(2))))
    return Token(Token::inttype, spelling);

  Token::Kind kind = StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
                         .Default(Token::bare_identifier);

  return Token(kind, spelling);
}

/// Skip a comment line, starting with a '//'.
///
///   TODO: add a regex for comments here and to the spec.
///
void Lexer::skipComment() {
  // Advance over the second '/' in a '//' comment.
  assert(*curPtr == '/');
  ++curPtr;

  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      LLVM_FALLTHROUGH;
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Lex an ellipsis.
///
///   ellipsis ::= '...'
///
Token Lexer::lexEllipsis(const char *tokStart) {
  assert(curPtr[-1] == '.');

  if (curPtr == curBuffer.end() || *curPtr != '.' || *(curPtr + 1) != '.')
    return emitError(curPtr, "expected three consecutive dots for an ellipsis");

  curPtr += 2;
  return formToken(Token::ellipsis, tokStart);
}

/// Lex a number literal.
///
///   integer-literal ::= digit+ | `0x` hex_digit+
///   float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
///
Token Lexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]));

  // Handle the hexadecimal case.
  if (curPtr[-1] == '0' && *curPtr == 'x') {
    // If we see stuff like 0xi32, this is a literal `0` followed by an
    // identifier `xi32`, stop after `0`.
    if (!isxdigit(curPtr[1]))
      return formToken(Token::integer, tokStart);

    curPtr += 2;
    while (isxdigit(*curPtr))
      ++curPtr;

    return formToken(Token::integer, tokStart);
  }

  // Handle the normal decimal case.
  while (isdigit(*curPtr))
    ++curPtr;

  if (*curPtr != '.')
    return formToken(Token::integer, tokStart);
  ++curPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(*curPtr))
    ++curPtr;

  if (*curPtr == 'e' || *curPtr == 'E') {
    if (isdigit(static_cast<unsigned char>(curPtr[1])) ||
        ((curPtr[1] == '-' || curPtr[1] == '+') &&
         isdigit(static_cast<unsigned char>(curPtr[2])))) {
      curPtr += 2;
      while (isdigit(*curPtr))
        ++curPtr;
    }
  }
  return formToken(Token::floatliteral, tokStart);
}

/// Lex an identifier that starts with a prefix followed by suffix-id.
///
///   attribute-id  ::= `#` suffix-id
///   ssa-id        ::= '%' suffix-id
///   block-id      ::= '^' suffix-id
///   type-id       ::= '!' suffix-id
///   suffix-id     ::= digit+ | (letter|id-punct) (letter|id-punct|digit)*
///   id-punct      ::= `$` | `.` | `_` | `-`
///
Token Lexer::lexPrefixedIdentifier(const char *tokStart) {
  Token::Kind kind;
  StringRef errorKind;
  switch (*tokStart) {
  case '#':
    kind = Token::hash_identifier;
    errorKind = "invalid attribute name";
    break;
  case '%':
    kind = Token::percent_identifier;
    errorKind = "invalid SSA name";
    break;
  case '^':
    kind = Token::caret_identifier;
    errorKind = "invalid block name";
    break;
  case '!':
    kind = Token::exclamation_identifier;
    errorKind = "invalid type identifier";
    break;
  default:
    llvm_unreachable("invalid caller");
  }

  // Parse suffix-id.
  if (isdigit(*curPtr)) {
    // If suffix-id starts with a digit, the rest must be digits.
    while (isdigit(*curPtr)) {
      ++curPtr;
    }
  } else if (isalpha(*curPtr) || isPunct(*curPtr)) {
    do {
      ++curPtr;
    } while (isalpha(*curPtr) || isdigit(*curPtr) || isPunct(*curPtr));
  } else {
    return emitError(curPtr - 1, errorKind);
  }

  return formToken(kind, tokStart);
}

/// Lex a string literal.
///
///   string-literal ::= '"' [^"\n\f\v\r]* '"'
///
/// TODO: define escaping rules.
Token Lexer::lexString(const char *tokStart) {
  assert(curPtr[-1] == '"');

  while (true) {
    switch (*curPtr++) {
    case '"':
      return formToken(Token::string, tokStart);
    case 0:
      // If this is a random nul character in the middle of a string, just
      // include it.  If it is the end of file, then it is an error.
      if (curPtr - 1 != curBuffer.end())
        continue;
      LLVM_FALLTHROUGH;
    case '\n':
    case '\v':
    case '\f':
      return emitError(curPtr - 1, "expected '\"' in string literal");
    case '\\':
      // Handle explicitly a few escapes.
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' || *curPtr == 't')
        ++curPtr;
      else if (llvm::isHexDigit(*curPtr) && llvm::isHexDigit(curPtr[1]))
        // Support \xx for two hex digits.
        curPtr += 2;
      else
        return emitError(curPtr - 1, "unknown escape in string literal");
      continue;

    default:
      continue;
    }
  }
}
