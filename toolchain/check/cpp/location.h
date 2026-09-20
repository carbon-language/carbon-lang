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
// This is a token range, as Clang's are: it reaches to the start of its last
// token, and is widened where it is read. A location whose subtree is not
// reachable -- one in an imported file, or one that is already C++ -- gives the
// token `GetCppLocation` names, which is what it marked before.
auto GetCppRange(Context& context, SemIR::LocId loc_id) -> clang::SourceRange;

// Adds an `ImportIRInst` referring to the given source range and returns a
// corresponding `ImportIRInstId` that can be used to construct a `LocId`. The
// range is what a diagnostic marking the location underlines.
auto AddImportIRInst(SemIR::File& file, clang::CharSourceRange clang_range)
    -> SemIR::ImportIRInstId;

// The same for a Clang location, which marks the column it names.
//
// A Clang location is the start of a token, and marking the whole token would
// say more, but measuring one needs the `SourceManager` and `LangOptions` that
// lexed it and neither is reachable from here. `CppDiagnosticListener` has both
// and does widen the locations it reports.
// TODO: Take what is needed to widen this too, so that a location from the
// importer and one from a Clang diagnostic mark the same thing.
auto AddImportIRInst(SemIR::File& file, clang::SourceLocation clang_source_loc)
    -> SemIR::ImportIRInstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_
