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

#ifndef FORTRAN_SEMANTICS_UNPARSE_WITH_SYMBOLS_H_
#define FORTRAN_SEMANTICS_UNPARSE_WITH_SYMBOLS_H_

#include "../parser/characters.h"
#include <iosfwd>

namespace Fortran::parser {
struct Program;
}

namespace Fortran::semantics {
void UnparseWithSymbols(std::ostream &, const parser::Program &,
    parser::Encoding encoding = parser::Encoding::UTF8);
}

#endif  // FORTRAN_SEMANTICS_UNPARSE_WITH_SYMBOLS_H_
