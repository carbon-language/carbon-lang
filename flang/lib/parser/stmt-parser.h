// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_PARSER_STMT_PARSER_H_
#define FORTRAN_PARSER_STMT_PARSER_H_

// Basic parsing of statements.

#include "basic-parsers.h"
#include "token-parsers.h"

namespace Fortran::parser {

// statement(p) parses Statement<P> for some statement type P that is the
// result type of the argument parser p, while also handling labels and
// end-of-statement markers.

// R611 label -> digit [digit]...
constexpr auto label = space >> digitString / spaceCheck;

template<typename PA> inline constexpr auto unterminatedStatement(const PA &p) {
  return skipEmptyLines >>
      sourced(construct<Statement<typename PA::resultType>>(
          maybe(label), space >> p));
}

constexpr auto endOfLine = "\n"_ch / skipEmptyLines ||
    consumedAllInput >> pure("\n") ||
    fail<const char *>("expected end of line"_err_en_US);

constexpr auto endOfStmt = space >>
    (";"_ch / skipMany(";"_tok) / maybe(endOfLine) || endOfLine);

template<typename PA> inline constexpr auto statement(const PA &p) {
  return unterminatedStatement(p) / endOfStmt;
}

constexpr auto ignoredStatementPrefix = skipEmptyLines >> maybe(label) >>
    maybe(name / ":") >> space;

// Error recovery within statements: skip to the end of the line,
// but not over an END or CONTAINS statement.
constexpr auto errorRecovery = construct<ErrorRecovery>();
constexpr auto skipToEndOfLine = SkipTo<'\n'>{} >> errorRecovery;
constexpr auto stmtErrorRecovery =
    !"END"_tok >> !"CONTAINS"_tok >> skipToEndOfLine;

// Error recovery across statements: skip the line, unless it looks
// like it might end the containing construct.
constexpr auto errorRecoveryStart = ignoredStatementPrefix;
constexpr auto skipBadLine = SkipPast<'\n'>{} >> errorRecovery;
constexpr auto executionPartErrorRecovery = errorRecoveryStart >> !"END"_tok >>
    !"CONTAINS"_tok >> !"ELSE"_tok >> !"CASE"_tok >> !"TYPE IS"_tok >>
    !"CLASS"_tok >> !"RANK"_tok >> skipBadLine;

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_STMT_PARSER_H_
