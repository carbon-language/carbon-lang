//===- lld/Driver/LinkerOptions.h - Linker Options ------------------------===//
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
/// All linker options to be provided to a LinkerInvocation.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_LINKER_OPTIONS_H
#define LLD_DRIVER_LINKER_OPTIONS_H

#include "lld/Core/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <vector>

namespace lld {
enum class InputKind {
  Unknown,
  YAML,
  Native,
  Object,
  LLVM,
  Script
};

class LinkerInput {
  LinkerInput(const LinkerInput &) LLVM_DELETED_FUNCTION;

public:
  LinkerInput(StringRef file, InputKind kind = InputKind::Unknown)
    : _file(file)
    , _kind(kind) {}

  LinkerInput(llvm::MemoryBuffer *buffer, InputKind kind = InputKind::Unknown)
    : _buffer(buffer)
    , _file(_buffer->getBufferIdentifier())
    , _kind(kind) {}

  LinkerInput(LinkerInput &&other)
    : _buffer(std::move(other._buffer))
    , _file(std::move(other._file))
    , _kind(other._kind) {}

  LinkerInput &operator=(LinkerInput &&rhs) {
    _buffer = std::move(rhs._buffer);
    _file = std::move(rhs._file);
    _kind = rhs._kind;
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

  ErrorOr<InputKind> getKind() const {
    if (_kind == InputKind::Unknown) {
      _kind = llvm::StringSwitch<InputKind>(getPath())
        .EndsWith(".objtxt", InputKind::YAML)
        .EndsWith(".yaml", InputKind::YAML)
        .Default(InputKind::Unknown);

      if (_kind != InputKind::Unknown)
        return _kind;

      auto buf = getBuffer();
      if (!buf)
        return error_code(buf);

      llvm::sys::fs::file_magic magic =
        llvm::sys::fs::identify_magic(buf->getBuffer());

      switch (magic) {
      case llvm::sys::fs::file_magic::elf_relocatable:
        _kind = InputKind::Object;
        break;
      }
    }

    return _kind;
  }

  StringRef getPath() const {
    return _file;
  }

private:
  mutable std::unique_ptr<llvm::MemoryBuffer> _buffer;
  std::string _file;
  mutable InputKind _kind;
};

struct LinkerOptions {
  LinkerOptions() {}

  // This exists because MSVC doesn't support = default :(
  LinkerOptions(LinkerOptions &&other)
    : _input(std::move(other._input))
    , _llvmArgs(std::move(other._llvmArgs))
    , _target(std::move(other._target))
    , _outputPath(std::move(other._outputPath))
    , _entrySymbol(std::move(other._entrySymbol))
    , _relocatable(other._relocatable)
    , _outputCommands(other._outputCommands)
    , _outputYAML(other._outputYAML)
    , _noInhibitExec(other._noInhibitExec) {}

  std::vector<LinkerInput> _input;
  std::vector<std::string> _llvmArgs;
  std::string _target;
  std::string _outputPath;
  std::string _entrySymbol;
  unsigned _relocatable : 1;
  /// \brief -###
  unsigned _outputCommands : 1;
  unsigned _outputYAML : 1;
  unsigned _noInhibitExec : 1;

private:
  LinkerOptions(const LinkerOptions&) LLVM_DELETED_FUNCTION;
};
}

#endif
