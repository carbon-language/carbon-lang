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

struct native_reader_error {
  enum _ {
    success = 0,
    unknown_file_format,
    file_too_short,
    file_malformed,
    unknown_chunk_type,
    memory_error,
  };
  _ v_;

  native_reader_error(_ v) : v_(v) {}
  explicit native_reader_error(int v) : v_(_(v)) {}
  operator int() const {return v_;}
};

inline llvm::error_code make_error_code(native_reader_error e) {
  return llvm::error_code(static_cast<int>(e), native_reader_category());
}

const llvm::error_category &yaml_reader_category();

struct yaml_reader_error {
  enum _ {
    success = 0,
    unknown_keyword,
    illegal_value
  };
  _ v_;

  yaml_reader_error(_ v) : v_(v) {}
  explicit yaml_reader_error(int v) : v_(_(v)) {}
  operator int() const {return v_;}
};

inline llvm::error_code make_error_code(yaml_reader_error e) {
  return llvm::error_code(static_cast<int>(e), yaml_reader_category());
}

const llvm::error_category &linker_script_reader_category();

enum class linker_script_reader_error {
  success = 0,
  parse_error
};

inline llvm::error_code make_error_code(linker_script_reader_error e) {
  return llvm::error_code(static_cast<int>(e), linker_script_reader_category());
}

/// \brief Errors returned by InputGraph functionality
const llvm::error_category &input_graph_error_category();

struct input_graph_error {
  enum _ {
    success = 0,
    failure = 1,
    no_more_elements,
    no_more_files,
  };
  _ v_;

  input_graph_error(_ v) : v_(v) {}
  explicit input_graph_error(int v) : v_(_(v)) {}
  operator int() const { return v_; }
};

inline llvm::error_code make_error_code(input_graph_error e) {
  return llvm::error_code(static_cast<int>(e), input_graph_error_category());
}

} // end namespace lld

namespace llvm {

template <> struct is_error_code_enum<lld::native_reader_error> : true_type { };
template <>
struct is_error_code_enum<lld::native_reader_error::_> : true_type { };

template <> struct is_error_code_enum<lld::yaml_reader_error> : true_type { };
template <>
struct is_error_code_enum<lld::yaml_reader_error::_> : true_type { };

template <>
struct is_error_code_enum<lld::linker_script_reader_error> : true_type { };

template <> struct is_error_code_enum<lld::input_graph_error> : true_type {};
template <> struct is_error_code_enum<lld::input_graph_error::_> : true_type {};
} // end namespace llvm

#endif
