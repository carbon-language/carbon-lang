//===- COFF/ModuleDef.cpp -------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Windows-specific.
// A parser for the module-definition file (.def file).
// Parsed results are directly written to Config global variable.
//
// The format of module-definition files are described in this document:
// https://msdn.microsoft.com/en-us/library/28d6s79h.aspx
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Error.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

using namespace llvm;

namespace lld {
namespace coff {
namespace {

enum Kind {
  Unknown,
  Eof,
  Identifier,
  Comma,
  Equal,
  KwBase,
  KwData,
  KwExports,
  KwHeapsize,
  KwLibrary,
  KwName,
  KwNoname,
  KwPrivate,
  KwStacksize,
  KwVersion,
};

struct Token {
  explicit Token(Kind T = Unknown, StringRef S = "") : K(T), Value(S) {}
  Kind K;
  StringRef Value;
};

class Lexer {
public:
  explicit Lexer(StringRef S) : Buf(S) {}

  Token lex() {
    Buf = Buf.trim();
    if (Buf.empty())
      return Token(Eof);

    switch (Buf[0]) {
    case '\0':
      return Token(Eof);
    case ';': {
      size_t End = Buf.find('\n');
      Buf = (End == Buf.npos) ? "" : Buf.drop_front(End);
      return lex();
    }
    case '=':
      Buf = Buf.drop_front();
      return Token(Equal, "=");
    case ',':
      Buf = Buf.drop_front();
      return Token(Comma, ",");
    case '"': {
      StringRef S;
      std::tie(S, Buf) = Buf.substr(1).split('"');
      return Token(Identifier, S);
    }
    default: {
      size_t End = Buf.find_first_of("=,\r\n \t\v");
      StringRef Word = Buf.substr(0, End);
      Kind K = llvm::StringSwitch<Kind>(Word)
                   .Case("BASE", KwBase)
                   .Case("DATA", KwData)
                   .Case("EXPORTS", KwExports)
                   .Case("HEAPSIZE", KwHeapsize)
                   .Case("LIBRARY", KwLibrary)
                   .Case("NAME", KwName)
                   .Case("NONAME", KwNoname)
                   .Case("PRIVATE", KwPrivate)
                   .Case("STACKSIZE", KwStacksize)
                   .Case("VERSION", KwVersion)
                   .Default(Identifier);
      Buf = (End == Buf.npos) ? "" : Buf.drop_front(End);
      return Token(K, Word);
    }
    }
  }

private:
  StringRef Buf;
};

class Parser {
public:
  explicit Parser(StringRef S) : Lex(S) {}

  std::error_code parse() {
    do {
      if (auto EC = parseOne())
        return EC;
    } while (Tok.K != Eof);
    return std::error_code();
  }

private:
  void read() {
    if (Stack.empty()) {
      Tok = Lex.lex();
      return;
    }
    Tok = Stack.back();
    Stack.pop_back();
  }

  std::error_code readAsInt(uint64_t *I) {
    read();
    if (Tok.K != Identifier || Tok.Value.getAsInteger(10, *I)) {
      llvm::errs() << "integer expected\n";
      return make_error_code(LLDError::InvalidOption);
    }
    return std::error_code();
  }

  std::error_code expect(Kind Expected, StringRef Msg) {
    read();
    if (Tok.K != Expected) {
      llvm::errs() << Msg << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    return std::error_code();
  }

  void unget() { Stack.push_back(Tok); }

  std::error_code parseOne() {
    read();
    switch (Tok.K) {
    case Eof:
      return std::error_code();
    case KwExports:
      for (;;) {
        read();
        if (Tok.K != Identifier) {
          unget();
          return std::error_code();
        }
        if (auto EC = parseExport())
          return EC;
      }
    case KwHeapsize:
      if (auto EC = parseNumbers(&Config->HeapReserve, &Config->HeapCommit))
        return EC;
      return std::error_code();
    case KwLibrary:
      if (auto EC = parseName(&Config->OutputFile, &Config->ImageBase))
        return EC;
      if (!StringRef(Config->OutputFile).endswith_lower(".dll"))
        Config->OutputFile += ".dll";
      return std::error_code();
    case KwStacksize:
      if (auto EC = parseNumbers(&Config->StackReserve, &Config->StackCommit))
        return EC;
      return std::error_code();
    case KwName:
      if (auto EC = parseName(&Config->OutputFile, &Config->ImageBase))
        return EC;
      return std::error_code();
    case KwVersion:
      if (auto EC = parseVersion(&Config->MajorImageVersion,
                                 &Config->MinorImageVersion))
        return EC;
      return std::error_code();
    default:
      llvm::errs() << "unknown directive: " << Tok.Value << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
  }

  std::error_code parseExport() {
    Export E;
    E.ExtName = Tok.Value;
    read();
    if (Tok.K == Equal) {
      read();
      if (Tok.K != Identifier) {
        llvm::errs() << "identifier expected, but got " << Tok.Value << "\n";
        return make_error_code(LLDError::InvalidOption);
      }
      E.Name = Tok.Value;
    } else {
      unget();
      E.Name = E.ExtName;
    }

    for (;;) {
      read();
      if (Tok.K == Identifier && Tok.Value[0] == '@') {
        Tok.Value.drop_front().getAsInteger(10, E.Ordinal);
        read();
        if (Tok.K == KwNoname) {
          E.Noname = true;
        } else {
          unget();
        }
        continue;
      }
      if (Tok.K == KwData) {
        E.Data = true;
        continue;
      }
      if (Tok.K == KwPrivate) {
        E.Private = true;
        continue;
      }
      unget();
      Config->Exports.push_back(E);
      return std::error_code();
    }
  }

  // HEAPSIZE/STACKSIZE reserve[,commit]
  std::error_code parseNumbers(uint64_t *Reserve, uint64_t *Commit) {
    if (auto EC = readAsInt(Reserve))
      return EC;
    read();
    if (Tok.K != Comma) {
      unget();
      Commit = 0;
      return std::error_code();
    }
    if (auto EC = readAsInt(Commit))
      return EC;
    return std::error_code();
  }

  // NAME outputPath [BASE=address]
  std::error_code parseName(std::string *Out, uint64_t *Baseaddr) {
    read();
    if (Tok.K == Identifier) {
      *Out = Tok.Value;
    } else {
      *Out = "";
      unget();
      return std::error_code();
    }
    read();
    if (Tok.K == KwBase) {
      if (auto EC = expect(Equal, "'=' expected"))
        return EC;
      if (auto EC = readAsInt(Baseaddr))
        return EC;
    } else {
      unget();
      *Baseaddr = 0;
    }
    return std::error_code();
  }

  // VERSION major[.minor]
  std::error_code parseVersion(uint32_t *Major, uint32_t *Minor) {
    read();
    if (Tok.K != Identifier) {
      llvm::errs() << "identifier expected, but got " << Tok.Value << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    StringRef V1, V2;
    std::tie(V1, V2) = Tok.Value.split('.');
    if (V1.getAsInteger(10, *Major)) {
      llvm::errs() << "integer expected, but got " << Tok.Value << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    if (V2.empty()) {
      *Minor = 0;
    } else if (V2.getAsInteger(10, *Minor)) {
      llvm::errs() << "integer expected, but got " << Tok.Value << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    return std::error_code();
  }

  Lexer Lex;
  Token Tok;
  std::vector<Token> Stack;
};

} // anonymous namespace

std::error_code parseModuleDefs(MemoryBufferRef MB) {
  return Parser(MB.getBuffer()).parse();
}

} // namespace coff
} // namespace lld
