//===- lib/ReaderWriter/PECOFF/PECOFFTargetInfo.cpp -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "GroupedSectionsPass.h"
#include "IdataPass.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Path.h"
#include "lld/Core/InputFiles.h"
#include "lld/Core/PassManager.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/ReaderWriter/PECOFFTargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Simple.h"
#include "lld/ReaderWriter/Writer.h"

namespace lld {

namespace {
bool containDirectoryName(StringRef path) {
  SmallString<128> smallStr = StringRef(path);
  llvm::sys::path::remove_filename(smallStr);
  return !smallStr.str().empty();
}

/// An instance of UndefinedSymbolFile has a list of undefined symbols
/// specified by "/include" command line option. This will be added to the
/// input file list to force the core linker to try to resolve the undefined
/// symbols.
class UndefinedSymbolFile : public SimpleFile {
public:
  UndefinedSymbolFile(const TargetInfo &ti)
      : SimpleFile(ti, "Linker Internal File") {
    for (StringRef symbol : ti.initialUndefinedSymbols()) {
      UndefinedAtom *atom = new (_alloc) coff::COFFUndefinedAtom(*this, symbol);
      addAtom(*atom);
    }
  };

private:
  llvm::BumpPtrAllocator _alloc;
};
} // anonymous namespace

error_code PECOFFTargetInfo::parseFile(
    std::unique_ptr<MemoryBuffer> &mb,
    std::vector<std::unique_ptr<File>> &result) const {
  return _reader->parseFile(mb, result);
}

bool PECOFFTargetInfo::validateImpl(raw_ostream &diagnostics) {
  if (_inputFiles.empty()) {
    diagnostics << "No input files\n";
    return true;
  }

  if (_stackReserve < _stackCommit) {
    diagnostics << "Invalid stack size: reserve size must be equal to or "
                << "greater than commit size, but got "
                << _stackCommit << " and " << _stackReserve << ".\n";
    return true;
  }

  if (_heapReserve < _heapCommit) {
    diagnostics << "Invalid heap size: reserve size must be equal to or "
                << "greater than commit size, but got "
                << _heapCommit << " and " << _heapReserve << ".\n";
    return true;
  }

  _reader = createReaderPECOFF(*this);
  _writer = createWriterPECOFF(*this);
  return false;
}

void PECOFFTargetInfo::addImplicitFiles(InputFiles &files) const {
  // Add a pseudo file for "/include" linker option.
  auto *file = new (_alloc) UndefinedSymbolFile(*this);
  files.prependFile(*file);
}

/// Append the given file to the input file list. The file must be an object
/// file or an import library file.
void PECOFFTargetInfo::appendInputFileOrLibrary(std::string path) {
  std::string ext = llvm::sys::path::extension(path).lower();
  // This is an import library file. Look for the library file in the search
  // paths, unless the path contains a directory name.
  if (ext == ".lib") {
    if (containDirectoryName(path)) {
      appendInputFile(path);
      return;
    }
    appendLibraryFile(path);
    return;
  }
  // This is an object file otherwise. Add ".obj" extension if the given path
  // name has no file extension.
  if (ext.empty())
    path.append(".obj");
  appendInputFile(allocateString(path));
}

/// Try to find the input library file from the search paths and append it to
/// the input file list. Returns true if the library file is found.
void PECOFFTargetInfo::appendLibraryFile(StringRef filename) {
  // Current directory always takes precedence over the search paths.
  if (llvm::sys::fs::exists(filename)) {
    appendInputFile(filename);
    return;
  }
  // Iterate over the search paths.
  for (StringRef dir : _inputSearchPaths) {
    SmallString<128> path = dir;
    llvm::sys::path::append(path, filename);
    if (llvm::sys::fs::exists(path.str())) {
      appendInputFile(allocateString(path.str()));
      return;
    }
  }
  appendInputFile(filename);
}

Writer &PECOFFTargetInfo::writer() const {
  return *_writer;
}

ErrorOr<Reference::Kind>
PECOFFTargetInfo::relocKindFromString(StringRef str) const {
  return make_error_code(yaml_reader_error::illegal_value);
}

ErrorOr<std::string>
PECOFFTargetInfo::stringFromRelocKind(Reference::Kind kind) const {
  return make_error_code(yaml_reader_error::illegal_value);
}

void PECOFFTargetInfo::addPasses(PassManager &pm) const {
  pm.add(std::unique_ptr<Pass>(new pecoff::GroupedSectionsPass()));
  pm.add(std::unique_ptr<Pass>(new pecoff::IdataPass()));
  pm.add(std::unique_ptr<Pass>(new LayoutPass()));
}

} // end namespace lld
