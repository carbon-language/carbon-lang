//===---- QueryParser.cpp - clang-query command parser --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "clang/ASTMatchers/Dynamic/Parser.h"
#include "clang/Basic/CharInfo.h"

using namespace llvm;
using namespace clang::ast_matchers::dynamic;

namespace clang {
namespace query {

// Lex any amount of whitespace followed by a "word" (any sequence of
// non-whitespace characters) from the start of region [Begin,End).  If no word
// is found before End, return StringRef().  Begin is adjusted to exclude the
// lexed region.
static StringRef LexWord(const char *&Begin, const char *End) {
  while (true) {
    if (Begin == End)
      return StringRef();

    if (!isWhitespace(*Begin))
      break;

    ++Begin;
  }

  const char *WordBegin = Begin;

  while (true) {
    ++Begin;

    if (Begin == End || isWhitespace(*Begin))
      return StringRef(WordBegin, Begin - WordBegin);
  }
}

static QueryRef ParseSetBool(bool QuerySession::*Var, StringRef ValStr) {
  unsigned Value = StringSwitch<unsigned>(ValStr)
                      .Case("false", 0)
                      .Case("true", 1)
                      .Default(~0u);
  if (Value == ~0u) {
    return new InvalidQuery("expected 'true' or 'false', got '" + ValStr + "'");
  }
  return new SetQuery<bool>(Var, Value);
}

static QueryRef ParseSetOutputKind(StringRef ValStr) {
  unsigned OutKind = StringSwitch<unsigned>(ValStr)
                         .Case("diag", OK_Diag)
                         .Case("print", OK_Print)
                         .Case("dump", OK_Dump)
                         .Default(~0u);
  if (OutKind == ~0u) {
    return new InvalidQuery("expected 'diag', 'print' or 'dump', got '" +
                            ValStr + "'");
  }
  return new SetQuery<OutputKind>(&QuerySession::OutKind, OutputKind(OutKind));
}

static QueryRef EndQuery(const char *Begin, const char *End, QueryRef Q) {
  const char *Extra = Begin;
  if (!LexWord(Begin, End).empty())
    return new InvalidQuery("unexpected extra input: '" +
                            StringRef(Extra, End - Extra) + "'");
  return Q;
}

enum ParsedQueryKind {
  PQK_Invalid,
  PQK_NoOp,
  PQK_Help,
  PQK_Match,
  PQK_Set
};

enum ParsedQueryVariable {
  PQV_Invalid,
  PQV_Output,
  PQV_BindRoot
};

QueryRef ParseQuery(StringRef Line) {
  const char *Begin = Line.data();
  const char *End = Line.data() + Line.size();

  StringRef CommandStr = LexWord(Begin, End);
  ParsedQueryKind QKind = StringSwitch<ParsedQueryKind>(CommandStr)
                              .Case("", PQK_NoOp)
                              .Case("help", PQK_Help)
                              .Case("m", PQK_Match)
                              .Case("match", PQK_Match)
                              .Case("set", PQK_Set)
                              .Default(PQK_Invalid);

  switch (QKind) {
  case PQK_NoOp:
    return new NoOpQuery;

  case PQK_Help:
    return EndQuery(Begin, End, new HelpQuery);

  case PQK_Match: {
    Diagnostics Diag;
    Optional<DynTypedMatcher> Matcher =
        Parser::parseMatcherExpression(StringRef(Begin, End - Begin), &Diag);
    if (!Matcher) {
      std::string ErrStr;
      llvm::raw_string_ostream OS(ErrStr);
      Diag.printToStreamFull(OS);
      return new InvalidQuery(OS.str());
    }
    return new MatchQuery(*Matcher);
  }

  case PQK_Set: {
    StringRef VarStr = LexWord(Begin, End);
    if (VarStr.empty())
      return new InvalidQuery("expected variable name");

    ParsedQueryVariable Var = StringSwitch<ParsedQueryVariable>(VarStr)
                .Case("output", PQV_Output)
                .Case("bind-root", PQV_BindRoot)
                .Default(PQV_Invalid);
    if (Var == PQV_Invalid)
      return new InvalidQuery("unknown variable: '" + VarStr + "'");

    StringRef ValStr = LexWord(Begin, End);
    if (ValStr.empty())
      return new InvalidQuery("expected variable value");

    QueryRef Q;
    switch (Var) {
    case PQV_Output:
      Q = ParseSetOutputKind(ValStr);
      break;
    case PQV_BindRoot:
      Q = ParseSetBool(&QuerySession::BindRoot, ValStr);
      break;
    case PQV_Invalid:
      llvm_unreachable("Invalid query kind");
    }

    return EndQuery(Begin, End, Q);
  }

  case PQK_Invalid:
    return new InvalidQuery("unknown command: " + CommandStr);
  }
}

} // namespace query
} // namespace clang
