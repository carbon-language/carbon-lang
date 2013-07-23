//===- Error.cpp - system_error extensions for lld --------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Error.h"

#include "llvm/Support/ErrorHandling.h"

using namespace lld;

class _native_reader_error_category : public llvm::_do_message {
public:
  virtual const char* name() const {
    return "lld.native.reader";
  }

  virtual std::string message(int ev) const {
    switch (ev) {
    case native_reader_error::success:
      return "Success";
    case native_reader_error::unknown_file_format:
      return "Unknown file format";
    case native_reader_error::file_too_short:
      return "file truncated";
    case native_reader_error::file_malformed:
      return "file malformed";
    case native_reader_error::memory_error:
      return "out of memory";
    case native_reader_error::unknown_chunk_type:
      return "unknown chunk type";
    default:
      llvm_unreachable("An enumerator of native_reader_error does not have a "
                       "message defined.");
    }
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (ev == native_reader_error::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::native_reader_category() {
  static _native_reader_error_category o;
  return o;
}

class _yaml_reader_error_category : public llvm::_do_message {
public:
  virtual const char* name() const {
    return "lld.yaml.reader";
  }

  virtual std::string message(int ev) const {
    switch (ev) {
    case yaml_reader_error::success:
      return "Success";
    case yaml_reader_error::unknown_keyword:
      return "Unknown keyword found in yaml file";
    case yaml_reader_error::illegal_value:
      return "Bad value found in yaml file";
    default:
      llvm_unreachable("An enumerator of yaml_reader_error does not have a "
                       "message defined.");
    }
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (ev == yaml_reader_error::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::yaml_reader_category() {
  static _yaml_reader_error_category o;
  return o;
}

class _linker_script_reader_error_category : public llvm::_do_message {
public:
  virtual const char *name() const { return "lld.linker-script.reader"; }

  virtual std::string message(int ev) const {
    switch (ev) {
    case static_cast<int>(linker_script_reader_error::success):
      return "Success";
    case static_cast<int>(linker_script_reader_error::parse_error):
      return "Error parsing linker script";
    default:
      llvm_unreachable(
          "An enumerator of linker_script_reader_error does not have a "
          "message defined.");
    }
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (ev == static_cast<int>(linker_script_reader_error::success))
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::linker_script_reader_category() {
  static _linker_script_reader_error_category o;
  return o;
}

