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

  LinkerInput(LinkerInput && other)
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

enum OutputKind {
  Invalid,
  StaticExecutable,
  DynamicExecutable,
  Relocatable,
  Shared,
  SharedStubs,
  Core,
  DebugSymbols,
  Bundle,
  Preload,
};

struct LinkerOptions {
  LinkerOptions()
    : _baseAddress(0)
    , _outputKind(OutputKind::Invalid)
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
  LinkerOptions(LinkerOptions && other)
      : _input(std::move(other._input)),
        _inputSearchPaths(std::move(other._inputSearchPaths)),
        _llvmArgs(std::move(other._llvmArgs)),
        _deadStripRoots(std::move(other._deadStripRoots)),
        _target(std::move(other._target)),
        _outputPath(std::move(other._outputPath)),
        _entrySymbol(std::move(other._entrySymbol)),
        _baseAddress(other._baseAddress), _outputKind(other._outputKind),
        _outputCommands(other._outputCommands), _outputYAML(other._outputYAML),
        _noInhibitExec(other._noInhibitExec), _deadStrip(other._deadStrip),
        _globalsAreDeadStripRoots(other._globalsAreDeadStripRoots),
        _searchArchivesToOverrideTentativeDefinitions(
            other._searchArchivesToOverrideTentativeDefinitions),
        _searchSharedLibrariesToOverrideTentativeDefinitions(
            other._searchSharedLibrariesToOverrideTentativeDefinitions),
        _warnIfCoalesableAtomsHaveDifferentCanBeNull(
            other._warnIfCoalesableAtomsHaveDifferentCanBeNull),
        _warnIfCoalesableAtomsHaveDifferentLoadName(
            other._warnIfCoalesableAtomsHaveDifferentLoadName),
        _forceLoadArchives(other._forceLoadArchives),
        _textRelocations(other._textRelocations),
        _relocatable(other._relocatable) {}

  std::vector<LinkerInput> _input;
  std::vector<std::string> _inputSearchPaths;
  std::vector<std::string> _llvmArgs;
  std::vector<std::string> _deadStripRoots;
  std::string _target;
  std::string _outputPath;
  std::string _entrySymbol;
  uint64_t _baseAddress;
  OutputKind _outputKind:4;
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
  unsigned _mergeCommonStrings: 1;

private:
  LinkerOptions(const LinkerOptions&) LLVM_DELETED_FUNCTION;
};
}

#endif
