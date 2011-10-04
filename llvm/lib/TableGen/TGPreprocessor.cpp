//===- TGPreprocessor.cpp - Preprocessor for TableGen ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Preprocessor for TableGen.
//
//===----------------------------------------------------------------------===//

#include "TGPreprocessor.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TableGen/Error.h"
#include <map>
#include <string>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace llvm {
typedef std::map<std::string, std::string> TGPPEnvironment;

enum TGPPTokenKind {
  tgpptoken_symbol,
  tgpptoken_literal,
  tgpptoken_newline,
  tgpptoken_error,
  tgpptoken_end
};

enum TGPPRecordKind {
  tgpprecord_for,
  tgpprecord_variable,
  tgpprecord_literal
};

enum TGPPRangeKind {
  tgpprange_list,
  tgpprange_sequence
};

bool MatchSymbol(TGPPTokenKind Kind,
                 const char *BeginOfToken, const char *EndOfToken,
                 char Symbol);

bool MatchSymbol(TGPPTokenKind Kind,
                 const char *BeginOfToken, const char *EndOfToken,
                 const char *Symbol);

bool MatchIdNum(TGPPTokenKind Kind,
                const char *BeginOfToken, const char *EndOfToken);

bool MatchIdentifier(TGPPTokenKind Kind,
                     const char *BeginOfToken, const char *EndOfToken);

bool MatchNumber(TGPPTokenKind Kind,
                 const char *BeginOfToken, const char *EndOfToken,
                 long int *Val);

class TGPPLexer {
  const MemoryBuffer *CurBuf;
  const char *CurPtr;
  bool IsInsideMacroStatement, WasEndOfLine;

  bool IsEndOfBuffer(const char *Ptr) const {
    return (!*Ptr && Ptr == CurBuf->getBufferEnd());
  }

  bool IsNewLine() {
    if (*CurPtr == '\r' || *CurPtr == '\n') {
      if ((CurPtr[1] == '\r' || CurPtr[1] == '\n') && CurPtr[0] != CurPtr[1])
        ++CurPtr;
      return true;
    }
    return false;
  }

  bool MatchPrefix(const char *Prefix, const char *Ptr) const {
    while (*Ptr == ' ' || *Ptr == '\t')
      ++Ptr;
    return !strncmp(Prefix, Ptr, strlen(Prefix));
  }
public:
  TGPPLexer(const SourceMgr &SM)
    : CurBuf(SM.getMemoryBuffer(0)),
      CurPtr(CurBuf->getBufferStart()),
      IsInsideMacroStatement(false),
      WasEndOfLine(true) {
  }

  TGPPTokenKind NextToken(const char **BeginOfToken, const char **EndOfToken);
};

// preprocessor records
class TGPPRecord {
  TGPPRecordKind Kind;

  // tgpprecord_for
  std::vector<std::string> IndexVars;
  std::vector<TGPPRange> IndexRanges;
  TGPPRecords LoopBody;

  // tgpprecord_variable, tgpprecord_literal
  std::string Str;

  bool EvaluateFor(const TGPPEnvironment &Env, raw_fd_ostream &OS) const;

  bool EvaluateVariable(const TGPPEnvironment &Env, raw_fd_ostream &OS) const {
    TGPPEnvironment::const_iterator it_val = Env.find(Str);
    if (it_val == Env.end()) {
      PrintError("Var is not bound to any value: " + Str);
      return true;
    }
    OS << it_val->second;
    return false;
  }

  bool EvaluateLiteral(const TGPPEnvironment &Env, raw_fd_ostream &OS) const {
    OS << Str;
    return false;
  }

public:
  TGPPRecord(TGPPRecordKind K) : Kind(K) {}
  TGPPRecord(TGPPRecordKind K, const std::string &S) : Kind(K), Str(S) {}

  TGPPRecords *GetLoopBody() { return &LoopBody; }

  void AppendIndex(const std::string &V, const TGPPRange &R) {
    IndexVars.push_back(V);
    IndexRanges.push_back(R);
  }

  bool Evaluate(const TGPPEnvironment &Env, raw_fd_ostream &OS) const;
};

class TGPPRange {
  TGPPRangeKind Kind;

  // tgpprange_list
  std::vector<std::string> Vals;

  // tgpprange_sequence
  long int From, To;

public:
  TGPPRange() : Kind(tgpprange_list) {}
  TGPPRange(long int F, long int T)
    : Kind(tgpprange_sequence), From(F), To(T) {}

  size_t size() const {
    if (Kind == tgpprange_list)
      return Vals.size();
    else
      return To - From + 1;
  }

  std::string at(size_t i) const {
    if (Kind == tgpprange_list)
      return Vals.at(i);
    else {
      char buf[32];
      snprintf(buf, sizeof(buf), "%ld", From + (long int)i);
      return std::string(buf);
    }
  }

  void push_back(const std::string &S) {
    if (Kind == tgpprange_list)
      Vals.push_back(S);
  }
};
} // namespace llvm

using namespace llvm;

bool llvm::MatchSymbol(TGPPTokenKind Kind,
                       const char *BeginOfToken, const char *EndOfToken,
                       char Symbol) {
  return Kind == tgpptoken_symbol &&
         BeginOfToken + 1 == EndOfToken &&
         Symbol == *BeginOfToken;
}

bool llvm::MatchSymbol(TGPPTokenKind Kind,
                       const char *BeginOfToken, const char *EndOfToken,
                       const char *Symbol) {
  return Kind == tgpptoken_symbol &&
         BeginOfToken + strlen(Symbol) == EndOfToken &&
         !strncmp(Symbol, BeginOfToken, EndOfToken - BeginOfToken);
}

bool llvm::MatchIdNum(TGPPTokenKind Kind,
                      const char *BeginOfToken, const char *EndOfToken) {
  if (Kind != tgpptoken_symbol)
    return false;
  for (const char *i = BeginOfToken; i != EndOfToken; ++i)
    if (*i != '_' && !isalnum(*i))
      return false;
  return true;
}

bool llvm::MatchIdentifier(TGPPTokenKind Kind,
                           const char *BeginOfToken, const char *EndOfToken) {
  if (Kind != tgpptoken_symbol)
    return false;

  const char *i = BeginOfToken;
  if (*i != '_' && !isalpha(*i))
    return false;
  for (++i; i != EndOfToken; ++i)
    if (*i != '_' && !isalnum(*i))
      return false;

  return true;
}

bool llvm::MatchNumber(TGPPTokenKind Kind,
                       const char *BeginOfToken, const char *EndOfToken,
                       long int *Val) {
  if (Kind != tgpptoken_symbol)
    return false;
  char *e;
  *Val = strtol(BeginOfToken, &e, 10);
  return e == EndOfToken;
}

TGPPTokenKind TGPPLexer::
NextToken(const char **BeginOfToken, const char **EndOfToken) {
  bool IsBeginOfLine = WasEndOfLine;
  WasEndOfLine = false;

  if (IsEndOfBuffer(CurPtr))
    return tgpptoken_end;

  else if (IsInsideMacroStatement) {
    while (*CurPtr == ' ' || *CurPtr == '\t') // trim space, if any
      ++CurPtr;

    const char *BeginOfSymbol = CurPtr;

    if (IsNewLine()) {
      ++CurPtr;
      IsInsideMacroStatement = false;
      WasEndOfLine = true;
      return tgpptoken_newline;
    }

    else if (*CurPtr == '[' || *CurPtr == ']' ||
             *CurPtr == '(' || *CurPtr == ')' ||
             *CurPtr == ',' || *CurPtr == '=') {
      *BeginOfToken = BeginOfSymbol;
      *EndOfToken = ++CurPtr;
      return tgpptoken_symbol;
    }

    else if (*CurPtr == '_' || isalpha(*CurPtr)) {
      ++CurPtr;
      while (*CurPtr == '_' || isalnum(*CurPtr))
        ++CurPtr;
      *BeginOfToken = BeginOfSymbol;
      *EndOfToken = CurPtr;
      return tgpptoken_symbol;
    }

    else if (*CurPtr == '+' || *CurPtr == '-' || isdigit(*CurPtr)) {
      ++CurPtr;
      while (isdigit(*CurPtr))
        ++CurPtr;
      *BeginOfToken = BeginOfSymbol;
      *EndOfToken = CurPtr;
      return tgpptoken_symbol;
    }

    else {
      PrintError(BeginOfSymbol, "Unrecognizable token");
      return tgpptoken_error;
    }
  }

  else if (*CurPtr == '#') {
    if (IsBeginOfLine &&
        (MatchPrefix("for", CurPtr + 1) ||
         MatchPrefix("end", CurPtr + 1))) {
      ++CurPtr;
      IsInsideMacroStatement = true;
      return NextToken(BeginOfToken, EndOfToken);
    }

    // special token #"# is translate to literal "
    else if (CurPtr[1] == '"' && CurPtr[2] == '#') {
      *BeginOfToken = ++CurPtr;
      *EndOfToken = ++CurPtr;
      ++CurPtr;
      return tgpptoken_literal;
    }

    else {
      const char *BeginOfVar = ++CurPtr; // trim '#'
      if (*CurPtr != '_' && !isalpha(*CurPtr)) {
        PrintError(BeginOfVar, "Variable must start with [_A-Za-z]: ");
        return tgpptoken_error;
      }
      while (*CurPtr == '_' || isalnum(*CurPtr))
        ++CurPtr;
      if (*CurPtr != '#') {
        PrintError(BeginOfVar, "Variable must end with #");
        return tgpptoken_error;
      }
      *BeginOfToken = BeginOfVar;
      *EndOfToken = CurPtr++; // trim '#'
      return tgpptoken_symbol;
    }
  }

  const char *BeginOfLiteral = CurPtr;
  int CCommentLevel = 0;
  bool BCPLComment = false;
  bool StringLiteral = false;
  for (; !IsEndOfBuffer(CurPtr); ++CurPtr) {
    if (CCommentLevel > 0) {
      if (CurPtr[0] == '/' && CurPtr[1] == '*') {
        ++CurPtr;
        ++CCommentLevel;
      } else if (CurPtr[0] == '*' && CurPtr[1] == '/') {
        ++CurPtr;
        --CCommentLevel;
      } else if (IsNewLine())
        WasEndOfLine = true;
    }

    else if (BCPLComment) {
      if (IsNewLine()) {
        WasEndOfLine = true;
        BCPLComment = false;
      }
    }

    else if (StringLiteral) {
      // no string escape sequence in TableGen?
      if (*CurPtr == '"')
        StringLiteral = false;
    }

    else if (CurPtr[0] == '/' && CurPtr[1] == '*') {
      ++CurPtr;
      ++CCommentLevel;
    }

    else if (CurPtr[0] == '/' && CurPtr[1] == '/') {
      ++CurPtr;
      BCPLComment = true;
    }

    else if (*CurPtr == '"')
      StringLiteral = true;

    else if (IsNewLine()) {
      ++CurPtr;
      WasEndOfLine = true;
      break;
    }

    else if (*CurPtr == '#')
      break;
  }

  *BeginOfToken = BeginOfLiteral;
  *EndOfToken = CurPtr;
  return tgpptoken_literal;
}

bool TGPPRecord::
EvaluateFor(const TGPPEnvironment &Env, raw_fd_ostream &OS) const {
  std::vector<TGPPRange>::const_iterator ri, re;

  // calculate the min size
  ri = IndexRanges.begin();
  re = IndexRanges.begin();
  size_t n = ri->size();
  for (; ri != re; ++ri) {
    size_t m = ri->size();
    if (m < n)
      n = m;
  }

  for (size_t which_val = 0; which_val < n; ++which_val) {
    // construct nested environment
    TGPPEnvironment NestedEnv(Env);
    std::vector<std::string>::const_iterator vi = IndexVars.begin();
    for (ri = IndexRanges.begin(), re = IndexRanges.end();
        ri != re; ++vi, ++ri) {
      NestedEnv.insert(std::make_pair(*vi, ri->at(which_val)));
    }
    // evalute loop body
    for (TGPPRecords::const_iterator i = LoopBody.begin(), e = LoopBody.end();
        i != e; ++i)
      if (i->Evaluate(NestedEnv, OS))
        return true;
  }

  return false;
}

bool TGPPRecord::
Evaluate(const TGPPEnvironment &Env, raw_fd_ostream &OS) const {
  switch (Kind) {
  case tgpprecord_for:
    return EvaluateFor(Env, OS);
  case tgpprecord_variable:
    return EvaluateVariable(Env, OS);
  case tgpprecord_literal:
    return EvaluateLiteral(Env, OS);
  default:
    PrintError("Unknown kind of record: " + Kind);
    return true;
  }
  return false;
}

bool TGPreprocessor::ParseBlock(bool TopLevel) {
  TGPPTokenKind Kind;
  const char *BeginOfToken, *EndOfToken;
  while ((Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken)) !=
         tgpptoken_end) {
    std::string Symbol(BeginOfToken, EndOfToken);
    switch (Kind) {
    case tgpptoken_symbol:
      if (Symbol == "for") {
        if (ParseForLoop())
          return true;
      } else if (Symbol == "end") {
        if (TopLevel) {
          PrintError(BeginOfToken, "No block to end here");
          return true;
        }
        if ((Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken)) !=
            tgpptoken_newline) {
          PrintError(BeginOfToken, "Tokens after #end");
          return true;
        }
        return false;
      } else if (Symbol == "NAME") {
        // treat '#NAME#' as a literal
        CurRecords->push_back(
            TGPPRecord(tgpprecord_literal,
                       std::string("#NAME#")));
      } else {
        CurRecords->push_back(
            TGPPRecord(tgpprecord_variable,
                       std::string(BeginOfToken, EndOfToken)));
      }
      break;
    case tgpptoken_literal:
      CurRecords->push_back(
          TGPPRecord(tgpprecord_literal,
                     std::string(BeginOfToken, EndOfToken)));
      break;
    default:
      return true;
    }
  }
  return false;
}

bool TGPreprocessor::ParseForLoop() {
  TGPPRecord ForLoopRecord(tgpprecord_for);

  for (;;) {
    TGPPTokenKind Kind;
    const char *BeginOfToken, *EndOfToken;

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchIdentifier(Kind, BeginOfToken, EndOfToken)) {
      PrintError(BeginOfToken, "Not an identifier");
      return true;
    }
    std::string IndexVar(BeginOfToken, EndOfToken);

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchSymbol(Kind, BeginOfToken, EndOfToken, '=')) {
      PrintError(BeginOfToken, "Need a '=' here");
      return true;
    }

    TGPPRange Range;
    if (ParseRange(&Range))
      return true;
    ForLoopRecord.AppendIndex(IndexVar, Range);

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (Kind == tgpptoken_newline)
      break;
    if (!MatchSymbol(Kind, BeginOfToken, EndOfToken, ',')) {
      PrintError(BeginOfToken, "Need a ',' here");
      return true;
    }
  }

  // open a new level
  TGPPRecords *LastCurRecords = CurRecords;
  CurRecords = ForLoopRecord.GetLoopBody();

  if (ParseBlock(false))
    return true;

  CurRecords = LastCurRecords;
  CurRecords->push_back(ForLoopRecord);
  return false;
}

bool TGPreprocessor::ParseRange(TGPPRange *Range) {
  TGPPTokenKind Kind;
  const char *BeginOfToken, *EndOfToken;

  Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);

  if (MatchSymbol(Kind, BeginOfToken, EndOfToken, '[')) {
    for (;;) {
      Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
      if (!MatchIdNum(Kind, BeginOfToken, EndOfToken)) {
        PrintError(BeginOfToken, "Need a identifier or a number here");
        return true;
      }
      Range->push_back(std::string(BeginOfToken, EndOfToken));

      Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
      if (MatchSymbol(Kind, BeginOfToken, EndOfToken, ']'))
        break;
      if (!MatchSymbol(Kind, BeginOfToken, EndOfToken, ',')) {
        PrintError(BeginOfToken, "Need a comma here");
        return true;
      }
    }
    return false;
  }

  else if (MatchSymbol(Kind, BeginOfToken, EndOfToken, "sequence")) {
    long int from, to;

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchSymbol(Kind, BeginOfToken, EndOfToken, '(')) {
      PrintError(BeginOfToken, "Need a left parentheses here");
      return true;
    }

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchNumber(Kind, BeginOfToken, EndOfToken, &from)) {
      PrintError(BeginOfToken, "Not a number");
      return true;
    }

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchSymbol(Kind, BeginOfToken, EndOfToken, ',')) {
      PrintError(BeginOfToken, "Need a comma here");
      return true;
    }

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchNumber(Kind, BeginOfToken, EndOfToken, &to)) {
      PrintError(BeginOfToken, "Not a number");
      return true;
    }

    Kind = Lexer->NextToken(&BeginOfToken, &EndOfToken);
    if (!MatchSymbol(Kind, BeginOfToken, EndOfToken, ')')) {
      PrintError(BeginOfToken, "Need a right parentheses here");
      return true;
    }

    *Range = TGPPRange(from, to);
    return false;
  }

  PrintError(BeginOfToken, "illegal range of loop index");
  return true;
}

bool TGPreprocessor::PreprocessFile() {
  TGPPLexer TheLexer(SrcMgr);
  TGPPRecords TopLevelRecords;

  Lexer = &TheLexer;
  CurRecords = &TopLevelRecords;
  if (ParseBlock(true))
    return true;

  TGPPEnvironment Env;
  for (TGPPRecords::const_iterator i = TopLevelRecords.begin(),
                                   e = TopLevelRecords.end();
      i != e; ++i)
    if (i->Evaluate(Env, Out.os()))
      return true;

  return false;
}
