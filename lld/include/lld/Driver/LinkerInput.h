//===- lld/Core/LinkerInput.h - Files to be linked ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// All linker options needed by core linking.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_LINKER_INPUT_H
#define LLD_CORE_LINKER_INPUT_H

#include "lld/Core/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <vector>

namespace lld {

/// \brief An input to the linker.
///
/// This class represents an input to the linker. It create the MemoryBuffer
/// lazily when needed based on the file path. It can also take a MemoryBuffer
/// directly.
///
/// The intent is that we only open each file once. And have strong ownership
/// semantics.
class LinkerInput {
  LinkerInput(const LinkerInput &) LLVM_DELETED_FUNCTION;

public:
  LinkerInput(StringRef file) : _file(file) {}

  LinkerInput(std::unique_ptr<llvm::MemoryBuffer> buffer)
      : _buffer(std::move(buffer)), _file(_buffer->getBufferIdentifier()) {
  }

  LinkerInput(LinkerInput &&other)
      : _buffer(std::move(other._buffer)), _file(std::move(other._file)) {
  }

  LinkerInput &operator=(LinkerInput &&rhs) {
    _buffer = std::move(rhs._buffer);
    _file = std::move(rhs._file);
    return *this;
  }

  ErrorOr<llvm::MemoryBuffer&> getBuffer() const {
    if (!_buffer) {
      llvm::OwningPtr<llvm::MemoryBuffer> buf;
      if (error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(_file, buf))
        return ec;
      _buffer.reset(buf.take());
    }

    return *_buffer;
  }

  StringRef getPath() const {
    return _file;
  }

  std::unique_ptr<llvm::MemoryBuffer> takeBuffer() {
    getBuffer();
    return std::move(_buffer);
  }

private:
  mutable std::unique_ptr<llvm::MemoryBuffer> _buffer;
  std::string _file;
};



} // namespace lld

#endif
