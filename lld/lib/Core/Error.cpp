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

class _NativeReaderErrorCategory : public llvm::_do_message {
public:
  virtual const char* name() const {
    return "lld.native.reader";
  }

  virtual std::string message(int ev) const {
    if (NativeReaderError(ev) == NativeReaderError::success)
      return "Success";
    if (NativeReaderError(ev) == NativeReaderError::unknown_file_format)
      return "Unknown file format";
    if (NativeReaderError(ev) == NativeReaderError::file_too_short)
      return "file truncated";
    if (NativeReaderError(ev) == NativeReaderError::file_malformed)
      return "file malformed";
    if (NativeReaderError(ev) == NativeReaderError::memory_error)
      return "out of memory";
    if (NativeReaderError(ev) == NativeReaderError::unknown_chunk_type)
      return "unknown chunk type";
    llvm_unreachable("An enumerator of NativeReaderError does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (NativeReaderError(ev) == NativeReaderError::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::native_reader_category() {
  static _NativeReaderErrorCategory o;
  return o;
}

class _YamlReaderErrorCategory : public llvm::_do_message {
public:
  virtual const char* name() const {
    return "lld.yaml.reader";
  }

  virtual std::string message(int ev) const {
    if (YamlReaderError(ev) == YamlReaderError::success)
      return "Success";
    if (YamlReaderError(ev) == YamlReaderError::unknown_keyword)
      return "Unknown keyword found in yaml file";
    if (YamlReaderError(ev) == YamlReaderError::illegal_value)
      return "Bad value found in yaml file";
    llvm_unreachable("An enumerator of YamlReaderError does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (YamlReaderError(ev) == YamlReaderError::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::YamlReaderCategory() {
  static _YamlReaderErrorCategory o;
  return o;
}

class _LinkerScriptReaderErrorCategory : public llvm::_do_message {
public:
  virtual const char *name() const { return "lld.linker-script.reader"; }

  virtual std::string message(int ev) const {
    LinkerScriptReaderError e = LinkerScriptReaderError(ev);
    if (e == LinkerScriptReaderError::success)
      return "Success";
    if (e == LinkerScriptReaderError::parse_error)
      return "Error parsing linker script";
    llvm_unreachable(
        "An enumerator of LinkerScriptReaderError does not have a "
        "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    LinkerScriptReaderError e = LinkerScriptReaderError(ev);
    if (e == LinkerScriptReaderError::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::LinkerScriptReaderCategory() {
  static _LinkerScriptReaderErrorCategory o;
  return o;
}

class _InputGraphErrorCategory : public llvm::_do_message {
public:
  virtual const char *name() const { return "lld.inputGraph.parse"; }

  virtual std::string message(int ev) const {
    if (InputGraphError(ev) == InputGraphError::success)
      return "Success";
    llvm_unreachable("An enumerator of InputGraphError does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (InputGraphError(ev) == InputGraphError::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::InputGraphErrorCategory() {
  static _InputGraphErrorCategory i;
  return i;
}

class _ReaderErrorCategory : public llvm::_do_message {
public:
  virtual const char *name() const { return "lld.inputGraph.parse"; }

  virtual std::string message(int ev) const {
    if (ReaderError(ev) == ReaderError::success)
      return "Success";
    else if (ReaderError(ev) == ReaderError::unknown_file_format)
      return "File format for the input file is not recognized by this flavor";

    llvm_unreachable("An enumerator of ReaderError does not have a "
                     "message defined.");
  }

  virtual llvm::error_condition default_error_condition(int ev) const {
    if (ReaderError(ev) == ReaderError::success)
      return llvm::errc::success;
    return llvm::errc::invalid_argument;
  }
};

const llvm::error_category &lld::ReaderErrorCategory() {
  static _ReaderErrorCategory i;
  return i;
}
