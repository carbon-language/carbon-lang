//===- DriverUtils.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the driver. Because there
// are so many small functions, we created this separate file to make
// Driver.cpp less cluttered.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)            \
  {                                                                            \
    X1, X2, X9, X10, OPT_##ID, opt::Option::KIND##Class, X8, X7, OPT_##GROUP,  \
        OPT_##ALIAS, X6                                                        \
  }                                                                            \
  ,
#include "Options.inc"
#undef OPTION
};

class ELFOptTable : public opt::OptTable {
public:
  ELFOptTable() : OptTable(infoTable, array_lengthof(infoTable)) {}
};

// Parses a given list of options.
opt::InputArgList ArgParser::parse(ArrayRef<const char *> Argv) {
  // Make InputArgList from string vectors.
  ELFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;

  // Expand response files. '@<filename>' is replaced by the file's contents.
  SmallVector<const char *, 256> Vec(Argv.data(), Argv.data() + Argv.size());
  StringSaver Saver(Alloc);
  llvm::cl::ExpandResponseFiles(Saver, llvm::cl::TokenizeGNUCommandLine, Vec);

  // Parse options and then do error checking.
  opt::InputArgList Args = Table.ParseArgs(Vec, MissingIndex, MissingCount);
  if (MissingCount)
    error(Twine("missing arg value for \"") + Args.getArgString(MissingIndex) +
          "\", expected " + Twine(MissingCount) +
          (MissingCount == 1 ? " argument.\n" : " arguments"));

  iterator_range<opt::arg_iterator> Unknowns = Args.filtered(OPT_UNKNOWN);
  for (auto *Arg : Unknowns)
    warning("warning: unknown argument: " + Arg->getSpelling());
  if (Unknowns.begin() != Unknowns.end())
    error("unknown argument(s) found");

  return Args;
}

// Parser and evaluator of the linker script.
// Results are directly written to the Config object.
namespace {
class LinkerScript {
public:
  LinkerScript(SymbolTable *T, StringRef S) : Symtab(T), Tokens(tokenize(S)) {}
  void run();

private:
  static std::vector<StringRef> tokenize(StringRef S);
  static StringRef skipSpace(StringRef S);
  StringRef next();
  bool atEOF() { return Tokens.size() == Pos; }
  void expect(StringRef Expect);

  void readAsNeeded();
  void readGroup();
  void readOutputFormat();

  SymbolTable *Symtab;
  std::vector<StringRef> Tokens;
  size_t Pos = 0;
};
}

void LinkerScript::run() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Tok == "GROUP") {
      readGroup();
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
    Symtab->addFile(createFile(openFile(Tok)));
  }
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
    Symtab->addFile(createFile(openFile(Tok)));
  }
}

void LinkerScript::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  next();
  expect(")");
}

void lld::elf2::readLinkerScript(SymbolTable *Symtab, MemoryBufferRef MB) {
  LinkerScript(Symtab, MB.getBuffer()).run();
}
