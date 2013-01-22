//===- lld/Core/LinkerOptions.h - Linker Options --------------------------===//
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

#ifndef LLD_CORE_LINKER_OPTIONS_H
#define LLD_CORE_LINKER_OPTIONS_H

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

  LinkerInput(std::unique_ptr<llvm::MemoryBuffer> buffer,
              InputKind kind = InputKind::Unknown)
    : _buffer(std::move(buffer))
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

  std::unique_ptr<llvm::MemoryBuffer> takeBuffer() {
    getBuffer();
    return std::move(_buffer);
  }

private:
  mutable std::unique_ptr<llvm::MemoryBuffer> _buffer;
  std::string _file;
  mutable InputKind _kind;
};

enum class OutputKind {
  Executable,
  Relocatable,
  Shared,
};

struct LinkerOptions {
  LinkerOptions()
    : _baseAddress(0)
    , _outputKind(OutputKind::Executable)
    , _outputCommands(false)
    , _outputYAML(false)
    , _noInhibitExec(true)
    , _deadStrip(false)
    , _globalsAreDeadStripRoots(false)
    , _searchArchivesToOverrideTentativeDefinitions(false)
    , _searchSharedLibrariesToOverrideTentativeDefinitions(false)
    , _warnIfCoalesableAtomsHaveDifferentCanBeNull(false)
    , _warnIfCoalesableAtomsHaveDifferentLoadName(false)
    , _forceLoadArchives(false)
    , _textRelocations(false)
    , _relocatable(false) {}

  // This exists because MSVC doesn't support = default :(
  LinkerOptions(LinkerOptions &&other)
    : _input(std::move(other._input))
    , _llvmArgs(std::move(other._llvmArgs))
    , _deadStripRoots(std::move(other._deadStripRoots))
    , _target(std::move(other._target))
    , _outputPath(std::move(other._outputPath))
    , _entrySymbol(std::move(other._entrySymbol))
    , _baseAddress(other._baseAddress)
    , _outputKind(other._outputKind)
    , _outputCommands(other._outputCommands)
    , _outputYAML(other._outputYAML)
    , _noInhibitExec(other._noInhibitExec)
    , _deadStrip(other._deadStrip)
    , _globalsAreDeadStripRoots(other._globalsAreDeadStripRoots)
    , _searchArchivesToOverrideTentativeDefinitions(
          other._searchArchivesToOverrideTentativeDefinitions)
    , _searchSharedLibrariesToOverrideTentativeDefinitions(
          other._searchSharedLibrariesToOverrideTentativeDefinitions)
    , _warnIfCoalesableAtomsHaveDifferentCanBeNull(
          other._warnIfCoalesableAtomsHaveDifferentCanBeNull)
    , _warnIfCoalesableAtomsHaveDifferentLoadName(
          other._warnIfCoalesableAtomsHaveDifferentLoadName)
    , _forceLoadArchives(other._forceLoadArchives)
    , _textRelocations(other._textRelocations)
    , _relocatable(other._relocatable) {}

  std::vector<LinkerInput> _input;
  std::vector<std::string> _llvmArgs;
  std::vector<std::string> _deadStripRoots;
  std::string _target;
  std::string _outputPath;
  std::string _entrySymbol;
  uint64_t _baseAddress;
  OutputKind _outputKind;
  /// \brief -###
  unsigned _outputCommands : 1;
  unsigned _outputYAML : 1;
  unsigned _noInhibitExec : 1;
  unsigned _deadStrip : 1;
  unsigned _globalsAreDeadStripRoots : 1;
  unsigned _searchArchivesToOverrideTentativeDefinitions : 1;
  unsigned _searchSharedLibrariesToOverrideTentativeDefinitions : 1;
  unsigned _warnIfCoalesableAtomsHaveDifferentCanBeNull : 1;
  unsigned _warnIfCoalesableAtomsHaveDifferentLoadName : 1;
  unsigned _forceLoadArchives : 1;
  unsigned _textRelocations : 1;
  unsigned _relocatable : 1;

private:
  LinkerOptions(const LinkerOptions&) LLVM_DELETED_FUNCTION;
};
}

#endif
