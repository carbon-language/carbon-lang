// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Maps a Carbon source location into an equivalent Clang source location.
auto GetCppLocation(Context& context, SemIR::LocId loc_id)
    -> clang::SourceLocation;

// Maps a Carbon source location into the Clang range covering the source a
// Carbon diagnostic marked with it would underline, which is everything the
// node spans rather than the one token it names.
//
// In Carbon source this is a character range, since Clang finds the end of a
// token range by lexing the token there as C++. Only the file being checked has
// its subtrees available, so a location in an imported file gives the range of
// the one token `GetCppLocation` names. A location that is already in C++ gives
// a token range at that token, which Clang can measure.
auto GetCppRange(Context& context, SemIR::LocId loc_id)
    -> clang::CharSourceRange;

// Adds an `ImportIRInst` referring to the given source range and returns a
// corresponding `ImportIRInstId` that can be used to construct a `LocId`. The
// range is what a diagnostic marking the location underlines.
auto AddImportIRInst(SemIR::File& file, clang::CharSourceRange clang_range)
    -> SemIR::ImportIRInstId;

// The same for a Clang location, which is a point and marks the one column it
// names.
auto AddImportIRInst(SemIR::File& file, clang::SourceLocation clang_source_loc)
    -> SemIR::ImportIRInstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_
