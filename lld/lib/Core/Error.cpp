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
    if (native_reader_error(ev) == native_reader_error::success)
      return "Success";
    if (native_reader_error(ev) == native_reader_error::unknown_file_format)
      return "Unknown file format";
    if (native_reader_error(ev) == native_reader_error::file_too_short)
      return "file truncated";
    if (native_reader_error(ev) == native_reader_error::file_malformed)
      return "file malformed";
    if (native_reader_error(ev) == native_reader_error::memory_error)
      return "out of memory";
    if (native_reader_error(ev) == native_reader_error::unknown_chunk_type)
      return "unknown chunk type";
    llvm_unreachable("An enumerator of native_reader_error does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (native_reader_error(ev) == native_reader_error::success)
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
    if (yaml_reader_error(ev) == yaml_reader_error::success)
      return "Success";
    if (yaml_reader_error(ev) == yaml_reader_error::unknown_keyword)
      return "Unknown keyword found in yaml file";
    if (yaml_reader_error(ev) == yaml_reader_error::illegal_value)
      return "Bad value found in yaml file";
    llvm_unreachable("An enumerator of yaml_reader_error does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (yaml_reader_error(ev) == yaml_reader_error::success)
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
    linker_script_reader_error e = linker_script_reader_error(ev);
    if (e == linker_script_reader_error::success)
      return "Success";
    if (e == linker_script_reader_error::parse_error)
      return "Error parsing linker script";
    llvm_unreachable(
        "An enumerator of linker_script_reader_error does not have a "
        "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    linker_script_reader_error e = linker_script_reader_error(ev);
    if (e == linker_script_reader_error::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::linker_script_reader_category() {
  static _linker_script_reader_error_category o;
  return o;
}

class _input_graph_error_category : public llvm::_do_message {
public:
  virtual const char *name() const { return "lld.inputGraph.parse"; }

  virtual std::string message(int ev) const {
    if (input_graph_error(ev) == input_graph_error::success)
      return "Success";
    llvm_unreachable("An enumerator of input_graph_error does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (input_graph_error(ev) == input_graph_error::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::input_graph_error_category() {
  static _input_graph_error_category i;
  return i;
}
