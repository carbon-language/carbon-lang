//===--- CXX.h - Public interfaces for the C++ grammar -----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines public interfaces for the C++ grammar
//  (pseudo/lib/cxx.bnf). It provides a fast way to access core building pieces
//  of the LR parser, e.g. Grammar, LRTable, rather than parsing the grammar
//  file at the runtime.
//
//  We do a compilation of the C++ BNF grammar at build time, and generate
//  critical data sources. The implementation of the interfaces are based on the
//  generated data sources.
//
//  FIXME: not everything is fully compiled yet. The implementation of the
//  interfaces are still parsing the grammar file at the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_CXX_CXX_H
#define CLANG_PSEUDO_CXX_CXX_H

#include "clang-pseudo/grammar/Grammar.h"

namespace clang {
namespace pseudo {
class LRTable;

namespace cxx {
// Symbol represents nonterminal symbols in the C++ grammar.
// It provides a simple uniform way to access a particular nonterminal.
enum class Symbol : SymbolID {
#define NONTERMINAL(X, Y) X = Y,
#include "CXXSymbols.inc"
#undef NONTERMINAL
};

// Returns the C++ grammar.
const Grammar &getGrammar();
// Returns the corresponding LRTable for the C++ grammar.
const LRTable &getLRTable();

} // namespace cxx

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_CXX_CXX_H
