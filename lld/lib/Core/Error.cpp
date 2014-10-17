//===- Error.cpp - system_error extensions for lld --------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Error.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mutex.h"

#include <string>
#include <vector>

using namespace lld;

class _NativeReaderErrorCategory : public std::error_category {
public:
  const char* name() const LLVM_NOEXCEPT override {
    return "lld.native.reader";
  }

  std::string message(int ev) const override {
    switch (static_cast<NativeReaderError>(ev)) {
    case NativeReaderError::success:
      return "Success";
    case NativeReaderError::unknown_file_format:
      return "Unknown file format";
    case NativeReaderError::file_too_short:
      return "file truncated";
    case NativeReaderError::file_malformed:
      return "file malformed";
    case NativeReaderError::memory_error:
      return "out of memory";
    case NativeReaderError::unknown_chunk_type:
      return "unknown chunk type";
    case NativeReaderError::conflicting_target_machine:
      return "conflicting target machine";
    }
    llvm_unreachable("An enumerator of NativeReaderError does not have a "
                     "message defined.");
  }
};

const std::error_category &lld::native_reader_category() {
  static _NativeReaderErrorCategory o;
  return o;
}

class _YamlReaderErrorCategory : public std::error_category {
public:
  const char* name() const LLVM_NOEXCEPT override {
    return "lld.yaml.reader";
  }

  std::string message(int ev) const override {
    switch (static_cast<YamlReaderError>(ev)) {
    case YamlReaderError::success:
      return "Success";
    case YamlReaderError::unknown_keyword:
      return "Unknown keyword found in yaml file";
    case YamlReaderError::illegal_value:
      return "Bad value found in yaml file";
    }
    llvm_unreachable("An enumerator of YamlReaderError does not have a "
                     "message defined.");
  }
};

const std::error_category &lld::YamlReaderCategory() {
  static _YamlReaderErrorCategory o;
  return o;
}

class _LinkerScriptReaderErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override {
    return "lld.linker-script.reader";
  }

  std::string message(int ev) const override {
    switch (static_cast<LinkerScriptReaderError>(ev)) {
    case LinkerScriptReaderError::success:
      return "Success";
    case LinkerScriptReaderError::parse_error:
      return "Error parsing linker script";
    }
    llvm_unreachable("An enumerator of LinkerScriptReaderError does not have a "
                     "message defined.");
  }
};

const std::error_category &lld::LinkerScriptReaderCategory() {
  static _LinkerScriptReaderErrorCategory o;
  return o;
}

class _InputGraphErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override {
    return "lld.inputGraph.parse";
  }

  std::string message(int ev) const override {
    switch (static_cast<InputGraphError>(ev)) {
    case InputGraphError::success:
      return "Success";
    case InputGraphError::failure:
      return "failure";
    case InputGraphError::no_more_elements:
      return "no more elements";
    case InputGraphError::no_more_files:
      return "no more files";
    }
    llvm_unreachable("An enumerator of InputGraphError does not have a "
                     "message defined.");
  }
};

const std::error_category &lld::InputGraphErrorCategory() {
  static _InputGraphErrorCategory i;
  return i;
}

class _ReaderErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override {
    return "lld.inputGraph.parse";
  }

  std::string message(int ev) const override {
    switch (static_cast<ReaderError>(ev)) {
    case ReaderError::success:
      return "Success";
    case ReaderError::unknown_file_format:
      return "File format for the input file is not recognized by this flavor";
    }
    llvm_unreachable("An enumerator of ReaderError does not have a "
                     "message defined.");
  }
};

const std::error_category &lld::ReaderErrorCategory() {
  static _ReaderErrorCategory i;
  return i;
}




namespace lld {


/// Temporary class to enable make_dynamic_error_code() until
/// llvm::ErrorOr<> is updated to work with error encapsulations 
/// other than error_code.
class dynamic_error_category : public std::error_category {
public:
  ~dynamic_error_category() LLVM_NOEXCEPT {}

  const char *name() const LLVM_NOEXCEPT override {
    return "lld.dynamic_error";
  }

  std::string message(int ev) const override {
    assert(ev >= 0);
    assert(ev < (int)_messages.size());
    // The value is an index into the string vector.
    return _messages[ev];
  }
  
  int add(std::string msg) {
    llvm::sys::SmartScopedLock<true> lock(_mutex);
    // Value zero is always the successs value.
    if (_messages.empty())
      _messages.push_back("Success");
    _messages.push_back(msg);
    // Return the index of the string just appended.
    return _messages.size() - 1;
  }
  
private:
  std::vector<std::string> _messages;
  llvm::sys::SmartMutex<true> _mutex;
};

static dynamic_error_category categorySingleton;

std::error_code make_dynamic_error_code(StringRef msg) {
  return std::error_code(categorySingleton.add(msg), categorySingleton);
}

std::error_code make_dynamic_error_code(const Twine &msg) {
  return std::error_code(categorySingleton.add(msg.str()), categorySingleton);
}

}

