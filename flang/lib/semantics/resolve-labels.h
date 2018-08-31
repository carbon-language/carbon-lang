/* -*- mode: c++; c-basic-offset: 2 -*- */
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

#ifndef FORTRAN_SEMANTICS_RESOLVE_LABELS_H_
#define FORTRAN_SEMANTICS_RESOLVE_LABELS_H_

namespace Fortran::parser {
struct Program;
class CookedSource;
}  // namespace Fortran::parser

namespace Fortran::semantics {

/// \brief Validate the labels in the program
/// \param ParseTree  the parse tree
/// \param Source     the cooked source
/// \return true, iff the program's labels pass semantics checks
bool ValidateLabels(
    const parser::Program &ParseTree, const parser::CookedSource &Source);
}  // namespace Fortran::semantics

#endif  // FORTRAN_SEMANTICS_RESOLVE_LABELS_H_
