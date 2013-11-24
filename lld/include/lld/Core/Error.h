//===- Error.h - system_error extensions for lld ----------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the lld library.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ERROR_H
#define LLD_CORE_ERROR_H

#include "llvm/Support/system_error.h"

namespace lld {

const llvm::error_category &native_reader_category();

enum class NativeReaderError {
  success = 0,
  unknown_file_format,
  file_too_short,
  file_malformed,
  unknown_chunk_type,
  memory_error,
};

inline llvm::error_code make_error_code(NativeReaderError e) {
  return llvm::error_code(static_cast<int>(e), native_reader_category());
}

const llvm::error_category &YamlReaderCategory();

enum class YamlReaderError {
  success = 0,
  unknown_keyword,
  illegal_value
};

inline llvm::error_code make_error_code(YamlReaderError e) {
  return llvm::error_code(static_cast<int>(e), YamlReaderCategory());
}

const llvm::error_category &LinkerScriptReaderCategory();

enum class LinkerScriptReaderError {
  success = 0,
  parse_error
};

inline llvm::error_code make_error_code(LinkerScriptReaderError e) {
  return llvm::error_code(static_cast<int>(e), LinkerScriptReaderCategory());
}

/// \brief Errors returned by InputGraph functionality
const llvm::error_category &InputGraphErrorCategory();

enum class InputGraphError {
  success = 0,
  failure = 1,
  no_more_elements,
  no_more_files
};

inline llvm::error_code make_error_code(InputGraphError e) {
  return llvm::error_code(static_cast<int>(e), InputGraphErrorCategory());
}

/// \brief Errors returned by Reader.
const llvm::error_category &ReaderErrorCategory();

enum class ReaderError {
  success = 0,
  unknown_file_format = 1
};

inline llvm::error_code make_error_code(ReaderError e) {
  return llvm::error_code(static_cast<int>(e), ReaderErrorCategory());
}

} // end namespace lld

namespace llvm {

template <> struct is_error_code_enum<lld::NativeReaderError> : true_type {};
template <> struct is_error_code_enum<lld::YamlReaderError> : true_type {};
template <>
struct is_error_code_enum<lld::LinkerScriptReaderError> : true_type {};
template <> struct is_error_code_enum<lld::InputGraphError> : true_type {};
template <> struct is_error_code_enum<lld::ReaderError> : true_type {};
} // end namespace llvm

#endif
