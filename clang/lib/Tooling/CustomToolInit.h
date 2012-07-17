//===--- CustomToolInit.h -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains a hook to supply a custom tool initialization routine.
//
//  The mechanism can be used by IDEs or non-public code bases to integrate with
//  their build system. Currently we support statically linking in an
//  implementation of \c customToolInit and enabling it with
//  -DUSE_CUSTOM_TOOL_INIT when compiling the Tooling library.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLING_CUSTOM_TOOL_INIT_H
#define LLVM_CLANG_TOOLING_CUSTOM_TOOL_INIT_H

namespace clang {
namespace tooling {

/// \brief Performs a custom initialization of a tool.
///
/// This function provides a hook for custom initialization of a clang tool. It
/// receives command-line arguments and can change them if needed.
/// If the initialization fails (say, custom command-line arguments are invalid)
/// this function should terminate the process.
void customToolInit(int argc, const char **argv);

} // namespace tooling
} // namespace clang

#endif  // LLVM_CLANG_TOOLING_CUSTOM_TOOL_INIT_H
