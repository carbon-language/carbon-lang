//===--- Tooling.h - Framework for standalone Clang tools -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements functions to run clang tools standalone instead
//  of running them as a plugin.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TOOLING_H
#define LLVM_CLANG_TOOLING_TOOLING_H

#include "llvm/ADT/StringRef.h"

namespace clang {

class FrontendAction;

namespace tooling {

/// \brief Runs (and deletes) the tool on 'Code' with the -fsynatx-only flag.
///
/// \param ToolAction The action to run over the code.
//  \param Code C++ code.
///
/// \return - True if 'ToolAction' was successfully executed.
bool RunSyntaxOnlyToolOnCode(
    clang::FrontendAction *ToolAction, llvm::StringRef Code);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_TOOLING_H
