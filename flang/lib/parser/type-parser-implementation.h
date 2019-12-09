// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,ccc
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Macros for implementing per-type parsers

#ifndef FORTRAN_PARSER_TYPE_PARSER_IMPLEMENTATION_H_
#define FORTRAN_PARSER_TYPE_PARSER_IMPLEMENTATION_H_

#include "type-parsers.h"

#undef TYPE_PARSER
#undef TYPE_CONTEXT_PARSER

// The result type of a parser combinator expression is determined
// here via "decltype(attempt(pexpr))" to work around a g++ bug that
// causes it to crash on "decltype(pexpr)" when pexpr's top-level
// operator is an overridden || of parsing alternatives.
#define TYPE_PARSER(pexpr) \
  template<> \
  auto Parser<typename decltype(attempt(pexpr))::resultType>::Parse( \
      ParseState &state) \
      ->std::optional<resultType> { \
    static constexpr auto parser{(pexpr)}; \
    return parser.Parse(state); \
  }

#define TYPE_CONTEXT_PARSER(contextText, pexpr) \
  TYPE_PARSER(CONTEXT_PARSER((contextText), (pexpr)))

#endif

