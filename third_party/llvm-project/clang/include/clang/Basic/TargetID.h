//===--- TargetID.h - Utilities for target ID -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETID_H
#define LLVM_CLANG_BASIC_TARGETID_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include <set>

namespace clang {

/// Get all feature strings that can be used in target ID for \p Processor.
/// Target ID is a processor name with optional feature strings
/// postfixed by a plus or minus sign delimited by colons, e.g.
/// gfx908:xnack+:sramecc-. Each processor have a limited
/// number of predefined features when showing up in a target ID.
llvm::SmallVector<llvm::StringRef, 4>
getAllPossibleTargetIDFeatures(const llvm::Triple &T,
                               llvm::StringRef Processor);

/// Get processor name from target ID.
/// Returns canonical processor name or empty if the processor name is invalid.
llvm::StringRef getProcessorFromTargetID(const llvm::Triple &T,
                                         llvm::StringRef OffloadArch);

/// Parse a target ID to get processor and feature map.
/// Returns canonicalized processor name or None if the target ID is invalid.
/// Returns target ID features in \p FeatureMap if it is not null pointer.
/// This function assumes \p OffloadArch is a valid target ID.
/// If the target ID contains feature+, map it to true.
/// If the target ID contains feature-, map it to false.
/// If the target ID does not contain a feature (default), do not map it.
llvm::Optional<llvm::StringRef>
parseTargetID(const llvm::Triple &T, llvm::StringRef OffloadArch,
              llvm::StringMap<bool> *FeatureMap);

/// Returns canonical target ID, assuming \p Processor is canonical and all
/// entries in \p Features are valid.
std::string getCanonicalTargetID(llvm::StringRef Processor,
                                 const llvm::StringMap<bool> &Features);

/// Get the conflicted pair of target IDs for a compilation or a bundled code
/// object, assuming \p TargetIDs are canonicalized. If there is no conflicts,
/// returns None.
llvm::Optional<std::pair<llvm::StringRef, llvm::StringRef>>
getConflictTargetIDCombination(const std::set<llvm::StringRef> &TargetIDs);
} // namespace clang

#endif
