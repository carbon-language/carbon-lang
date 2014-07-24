//===--- RuntimeDebugBuilder.h --- Helper to insert prints into LLVM-IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_DEBUG_BUILDER_H
#define RUNTIME_DEBUG_BUILDER_H

#include "polly/CodeGen/IRBuilder.h"

#include <string>

namespace llvm {
class Value;
class Function;
}

namespace polly {

/// @brief Insert function calls that print certain LLVM values at run time.
///
/// This class inserts libc function calls to print certain LLVM values at
/// run time.
struct RuntimeDebugBuilder {

  /// @brief Print a string to stdout.
  ///
  /// @param String The string to print.
  static void createStrPrinter(PollyIRBuilder &Builder,
                               const std::string &String);

  /// @brief Print a value to stdout.
  ///
  /// @param V The value to print.
  ///
  /// @note Only integer, floating point and pointer values up to 64bit are
  ///       supported.
  static void createValuePrinter(PollyIRBuilder &Builder, llvm::Value *V);

  /// @brief Add a call to the fflush function with no file pointer given.
  ///
  /// This call will flush all opened file pointers including stdout and stderr.
  static void createFlush(PollyIRBuilder &Builder);

  /// @brief Get a reference to the 'printf' function.
  ///
  /// If the current module does not yet contain a reference to printf, we
  /// insert a reference to it. Otherwise the existing reference is returned.
  static llvm::Function *getPrintF(PollyIRBuilder &Builder);
};
}

#endif
