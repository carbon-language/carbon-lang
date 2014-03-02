//===--- QueryParser.h - clang-query ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PARSER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PARSER_H

#include "Query.h"

#include <stddef.h>
#include "llvm/LineEditor/LineEditor.h"

namespace clang {
namespace query {

class QuerySession;

class QueryParser {
public:
  /// Parse \a Line as a query.
  ///
  /// \return A QueryRef representing the query, which may be an InvalidQuery.
  static QueryRef parse(StringRef Line);

  /// Compute a list of completions for \a Line assuming a cursor at
  /// \param Pos characters past the start of \a Line, ordered from most
  /// likely to least likely.
  ///
  /// \return A vector of completions for \a Line.
  static std::vector<llvm::LineEditor::Completion> complete(StringRef Line,
                                                            size_t Pos);

private:
  QueryParser(StringRef Line)
      : Begin(Line.data()), End(Line.data() + Line.size()), CompletionPos(0) {}

  StringRef lexWord();

  template <typename T> struct LexOrCompleteWord;
  template <typename T> LexOrCompleteWord<T> lexOrCompleteWord(StringRef &Str);

  QueryRef parseSetBool(bool QuerySession::*Var);
  QueryRef parseSetOutputKind();

  QueryRef endQuery(QueryRef Q);

  /// \brief Parse [\p Begin,\p End).
  ///
  /// \return A reference to the parsed query object, which may be an
  /// \c InvalidQuery if a parse error occurs.
  QueryRef doParse();

  const char *Begin;
  const char *End;

  const char *CompletionPos;
  std::vector<llvm::LineEditor::Completion> Completions;
};

} // namespace query
} // namespace clang

#endif
