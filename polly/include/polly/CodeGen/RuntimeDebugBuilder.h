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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

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

  /// @brief Print a set of LLVM-IR Values or StringRefs on an NVIDIA GPU.
  ///
  ///  This function emits a call to vprintf that will print the given
  ///  arguments from within a kernel thread. It is useful for debugging
  ///  CUDA program kernels. All arguments given in this list will be
  ///  automatically concatenated and the resulting string will be printed
  ///  atomically. We also support ArrayRef arguments, which can be used to
  ///  provide for example a list of thread-id values.
  ///
  ///  @param Builder The builder used to emit the printer calls.
  ///  @param Args    The list of values to print.
  template <typename... Args>
  static void createGPUPrinter(PollyIRBuilder &Builder, Args... args) {
    std::vector<llvm::Value *> Vector;
    createGPUVAPrinter(Builder, Vector, args...);
  }

private:
  /// @brief GPU printing - Print a list of LLVM Values.
  ///
  static void createGPUVAPrinter(PollyIRBuilder &Builder,
                                 llvm::ArrayRef<llvm::Value *> Values);

  /// @brief GPU printing - Handle Values.
  template <typename... Args>
  static void createGPUVAPrinter(PollyIRBuilder &Builder,
                                 std::vector<llvm::Value *> &Values,
                                 llvm::Value *Value, Args... args) {
    Values.push_back(Value);
    createGPUVAPrinter(Builder, Values, args...);
  }

  /// @brief GPU printing - Handle StringRefs.
  template <typename... Args>
  static void createGPUVAPrinter(PollyIRBuilder &Builder,
                                 std::vector<llvm::Value *> &Values,
                                 llvm::StringRef String, Args... args) {
    Values.push_back(Builder.CreateGlobalStringPtr(String, "", 4));
    createGPUVAPrinter(Builder, Values, args...);
  }

  /// @brief GPU printing - Handle ArrayRefs.
  template <typename... Args>
  static void createGPUVAPrinter(PollyIRBuilder &Builder,
                                 std::vector<llvm::Value *> &Values,
                                 llvm::ArrayRef<llvm::Value *> Array,
                                 Args... args) {
    if (Array.size() >= 2)
      createGPUVAPrinter(
          Builder, Values, Array[0], " ",
          llvm::ArrayRef<llvm::Value *>(&Array[1], Array.size() - 1), args...);
    else if (Array.size() == 1)
      createGPUVAPrinter(Builder, Values, Array[0], args...);
    else
      createGPUVAPrinter(Builder, Values, args...);
  }

  /// @brief Get (and possibly insert) a vprintf declaration into the module.
  static llvm::Function *getVPrintF(PollyIRBuilder &Builder);

  /// @brief Get (and possibly insert) a NVIDIA address space cast call.
  static llvm::Function *getAddressSpaceCast(PollyIRBuilder &Builder,
                                             unsigned Src, unsigned Dst,
                                             unsigned SrcBits = 8,
                                             unsigned DstBits = 8);
};
}

#endif
