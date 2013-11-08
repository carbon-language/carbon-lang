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

namespace clang {
namespace query {

/// \brief Parse \p Line.
///
/// \return A reference to the parsed query object, which may be an
/// \c InvalidQuery if a parse error occurs.
QueryRef ParseQuery(StringRef Line);

} // namespace query
} // namespace clang

#endif
