//===- ScriptParser.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the base parser class for linker script and dynamic
// list.
//
//===----------------------------------------------------------------------===//

#include "ScriptParser.h"
#include "Error.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;
using namespace lld;
using namespace lld::elf;

// Returns a line containing a token.
static StringRef getLine(StringRef S, StringRef Tok) {
  size_t Pos = S.rfind('\n', Tok.data() - S.data());
  if (Pos != StringRef::npos)
    S = S.substr(Pos + 1);
  return S.substr(0, S.find_first_of("\r\n"));
}

// Returns 1-based line number of a given token.
static size_t getLineNumber(StringRef S, StringRef Tok) {
  return S.substr(0, Tok.data() - S.data()).count('\n') + 1;
}

// Returns 0-based column number of a given token.
static size_t getColumnNumber(StringRef S, StringRef Tok) {
  return Tok.data() - getLine(S, Tok).data();
}

ScriptParserBase::ScriptParserBase(MemoryBufferRef MB) { tokenize(MB); }

// We don't want to record cascading errors. Keep only the first one.
void ScriptParserBase::setError(const Twine &Msg) {
  if (Error)
    return;
  Error = true;

  MemoryBufferRef MB = currentBuffer();
  std::string Filename = MB.getBufferIdentifier();

  if (!Pos) {
    error(Filename + ": " + Msg);
    return;
  }

  StringRef Buf = MB.getBuffer();
  StringRef Tok = Tokens[Pos - 1];
  std::string S = (Filename + ":" + Twine(getLineNumber(Buf, Tok))).str();

  error(S + ": " + Msg);
  error(S + ": " + getLine(Buf, Tok));
  error(S + ": " + std::string(getColumnNumber(Buf, Tok), ' ') + "^");
}

// Split S into linker script tokens.
void ScriptParserBase::tokenize(MemoryBufferRef MB) {
  std::vector<StringRef> Ret;
  MBs.push_back(MB);
  StringRef S = MB.getBuffer();
  StringRef Begin = S;
  for (;;) {
    S = skipSpace(S);
    if (S.empty())
      break;

    // Quoted token. Note that double-quote characters are parts of a token
    // because, in a glob match context, only unquoted tokens are interpreted
    // as glob patterns. Double-quoted tokens are literal patterns in that
    // context.
    if (S.startswith("\"")) {
      size_t E = S.find("\"", 1);
      if (E == StringRef::npos) {
        error(MB.getBufferIdentifier() + ":" + Twine(getLineNumber(Begin, S)) +
              ": unclosed quote");
        return;
      }
      Ret.push_back(S.take_front(E + 1));
      S = S.substr(E + 1);
      continue;
    }

    // Unquoted token. This is more relaxed than tokens in C-like language,
    // so that you can write "file-name.cpp" as one bare token, for example.
    size_t Pos = S.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789_.$/\\~=+[]*?-:!<>^");

    // A character that cannot start a word (which is usually a
    // punctuation) forms a single character token.
    if (Pos == 0)
      Pos = 1;
    Ret.push_back(S.substr(0, Pos));
    S = S.substr(Pos);
  }
  Tokens.insert(Tokens.begin() + Pos, Ret.begin(), Ret.end());
}

// Skip leading whitespace characters or comments.
StringRef ScriptParserBase::skipSpace(StringRef S) {
  for (;;) {
    if (S.startswith("/*")) {
      size_t E = S.find("*/", 2);
      if (E == StringRef::npos) {
        error("unclosed comment in a linker script");
        return "";
      }
      S = S.substr(E + 2);
      continue;
    }
    if (S.startswith("#")) {
      size_t E = S.find('\n', 1);
      if (E == StringRef::npos)
        E = S.size() - 1;
      S = S.substr(E + 1);
      continue;
    }
    size_t Size = S.size();
    S = S.ltrim();
    if (S.size() == Size)
      return S;
  }
}

// An erroneous token is handled as if it were the last token before EOF.
bool ScriptParserBase::atEOF() { return Error || Tokens.size() == Pos; }

StringRef ScriptParserBase::next() {
  if (Error)
    return "";
  if (atEOF()) {
    setError("unexpected EOF");
    return "";
  }
  return Tokens[Pos++];
}

StringRef ScriptParserBase::peek() {
  StringRef Tok = next();
  if (Error)
    return "";
  --Pos;
  return Tok;
}

bool ScriptParserBase::consume(StringRef Tok) {
  if (peek() == Tok) {
    skip();
    return true;
  }
  return false;
}

void ScriptParserBase::skip() { (void)next(); }

void ScriptParserBase::expect(StringRef Expect) {
  if (Error)
    return;
  StringRef Tok = next();
  if (Tok != Expect)
    setError(Expect + " expected, but got " + Tok);
}

std::string ScriptParserBase::currentLocation() {
  MemoryBufferRef MB = currentBuffer();
  return (MB.getBufferIdentifier() + ":" +
          Twine(getLineNumber(MB.getBuffer(), Tokens[Pos - 1])))
      .str();
}

// Returns true if string 'Bigger' contains string 'Shorter'.
static bool containsString(StringRef Bigger, StringRef Shorter) {
  const char *BiggerEnd = Bigger.data() + Bigger.size();
  const char *ShorterEnd = Shorter.data() + Shorter.size();

  return Bigger.data() <= Shorter.data() && BiggerEnd >= ShorterEnd;
}

MemoryBufferRef ScriptParserBase::currentBuffer() {
  // Find input buffer containing the current token.
  assert(!MBs.empty());
  if (Pos)
    for (MemoryBufferRef MB : MBs)
      if (containsString(MB.getBuffer(), Tokens[Pos - 1]))
        return MB;

  return MBs.front();
}
