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

#ifndef FORTRAN_PARSER_DEBUG_PARSER_H_
#define FORTRAN_PARSER_DEBUG_PARSER_H_

// Implements the parser with syntax "(YOUR MESSAGE HERE)"_debug for use
// in temporary modifications to the grammar intended for tracing the
// flow of the parsers.  Not to be used in production.

#include "basic-parsers.h"
#include "parse-state.h"
#include <cstddef>
#include <optional>

namespace Fortran::parser {

class DebugParser {
public:
  using resultType = Success;
  constexpr DebugParser(const DebugParser &) = default;
  constexpr DebugParser(const char *str, std::size_t n)
    : str_{str}, length_{n} {}
  std::optional<Success> Parse(ParseState &) const;

private:
  const char *const str_;
  const std::size_t length_;
};

constexpr DebugParser operator""_debug(const char str[], std::size_t n) {
  return DebugParser{str, n};
}
}
#endif  // FORTRAN_PARSER_DEBUG_PARSER_H_
