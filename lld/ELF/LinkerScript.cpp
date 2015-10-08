//===- LinkerScript.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the parser/evaluator of the linker script.
// It does not construct an AST but consume linker script directives directly.
// Results are written to Symtab or Config object.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "SymbolTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace lld;
using namespace lld::elf2;

namespace {
class LinkerScript {
public:
  LinkerScript(StringRef S) : Tokens(tokenize(S)) {}
  void run();

private:
  static std::vector<StringRef> tokenize(StringRef S);
  static StringRef skipSpace(StringRef S);
  StringRef next();
  bool atEOF() { return Tokens.size() == Pos; }
  void expect(StringRef Expect);

  void readAsNeeded();
  void readEntry();
  void readGroup();
  void readOutput();
  void readOutputFormat();

  std::vector<StringRef> Tokens;
  size_t Pos = 0;
};
}

void LinkerScript::run() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Tok == "ENTRY") {
      readEntry();
    } else if (Tok == "GROUP") {
      readGroup();
    } else if (Tok == "OUTPUT") {
      readOutput();
    } else if (Tok == "OUTPUT_FORMAT") {
      readOutputFormat();
    } else {
      error("unknown directive: " + Tok);
    }
  }
}

// Split S into linker script tokens.
std::vector<StringRef> LinkerScript::tokenize(StringRef S) {
  std::vector<StringRef> Ret;
  for (;;) {
    S = skipSpace(S);
    if (S.empty())
      return Ret;

    // Quoted token
    if (S.startswith("\"")) {
      size_t E = S.find("\"", 1);
      if (E == StringRef::npos)
        error("unclosed quote");
      Ret.push_back(S.substr(1, E));
      S = S.substr(E + 1);
      continue;
    }

    // Unquoted token
    size_t Pos = S.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789_.$/\\~=+[]*?-:");
    // A character that cannot start a word (which is usually a
    // punctuation) forms a single character token.
    if (Pos == 0)
      Pos = 1;
    Ret.push_back(S.substr(0, Pos));
    S = S.substr(Pos);
  }
}

// Skip leading whitespace characters or /**/-style comments.
StringRef LinkerScript::skipSpace(StringRef S) {
  for (;;) {
    if (S.startswith("/*")) {
      size_t E = S.find("*/", 2);
      if (E == StringRef::npos)
        error("unclosed comment in a linker script");
      S = S.substr(E + 2);
      continue;
    }
    size_t Size = S.size();
    S = S.ltrim();
    if (S.size() == Size)
      return S;
  }
}

StringRef LinkerScript::next() {
  if (Pos == Tokens.size())
    error("unexpected EOF");
  return Tokens[Pos++];
}

void LinkerScript::expect(StringRef Expect) {
  StringRef Tok = next();
  if (Tok != Expect)
    error(Expect + " expected, but got " + Tok);
}

void LinkerScript::readAsNeeded() {
  expect("(");
  for (;;) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    Driver->addFile(Tok);
  }
}

void LinkerScript::readEntry() {
  // -e <symbol> takes predecence over ENTRY(<symbol>).
  expect("(");
  StringRef Tok = next();
  if (Config->Entry.empty())
    Config->Entry = Tok;
  expect(")");
}

void LinkerScript::readGroup() {
  expect("(");
  for (;;) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    if (Tok == "AS_NEEDED") {
      readAsNeeded();
      continue;
    }
    Driver->addFile(Tok);
  }
}

void LinkerScript::readOutput() {
  // -o <file> takes predecence over OUTPUT(<file>).
  expect("(");
  StringRef Tok = next();
  if (Config->OutputFile.empty())
    Config->OutputFile = Tok;
  expect(")");
}

void LinkerScript::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  next();
  expect(")");
}

// Entry point. The other functions or classes are private to this file.
void lld::elf2::readLinkerScript(MemoryBufferRef MB) {
  LinkerScript(MB.getBuffer()).run();
}
