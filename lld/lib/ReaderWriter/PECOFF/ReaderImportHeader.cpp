//===- lib/ReaderWriter/PECOFF/ReaderImportHeader.cpp ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file \brief This file provides a way to read an import library
/// member in a .lib file.
///
/// In Windows, archive files with .lib file extension serve two different
/// purposes.
///
///  - For static linking: An archive file in this use case contains multiple
///    normal .obj files and is used for static linking. This is the same
///    usage as .a file in Unix.
///
///  - For dynamic linking: An archive file in this case contains pseudo .obj
///    files to describe exported symbols of a DLL. Each .obj file in an archive
///    has a name of an exported symbol and a DLL filename from which the symbol
///    can be imported. When you link a DLL on Windows, you pass the name of the
///    .lib file for the DLL instead of the DLL filename itself. That is the
///    Windows way of linking a shared library.
///
/// This file contains a function to parse the pseudo object file.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ReaderImportHeader"

#include "lld/Core/File.h"
#include "lld/Core/Error.h"
#include "lld/Core/SharedLibraryAtom.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <map>
#include <vector>
#include <cstring>

using namespace lld;
using namespace llvm;

namespace lld {
namespace coff {

namespace {

class COFFDynamicAtom : public SharedLibraryAtom {
public:
  COFFDynamicAtom(File &file, StringRef symbolName, StringRef dllName)
      : _owningFile(file), _symbolName(symbolName), _dllName(dllName) {}

  virtual const File &file() const { return _owningFile; }
  virtual StringRef name() const { return _symbolName; }
  virtual StringRef loadName() const { return _dllName; }
  virtual bool canBeNullAtRuntime() const { return true; }

private:
  const File &_owningFile;
  StringRef _symbolName;
  StringRef _dllName;
};

class FileImportLibrary : public File {
public:
  FileImportLibrary(const TargetInfo &ti,
                    std::unique_ptr<llvm::MemoryBuffer> mb,
                    llvm::error_code &ec)
      : File(mb->getBufferIdentifier(), kindSharedLibrary), _targetInfo(ti) {
    const char *buf = mb->getBufferStart();
    const char *end = mb->getBufferEnd();

    // The size of the string that follows the header.
    uint32_t dataSize =
        *reinterpret_cast<const support::ulittle32_t *>(buf + 12);

    // Check if the total size is valid. The file header is 20 byte long.
    if (end - buf != 20 + dataSize) {
      ec = make_error_code(native_reader_error::unknown_file_format);
      return;
    }

    StringRef symbolName(buf + 20);
    StringRef dllName(buf + 20 + symbolName.size() + 1);

    auto *atom = new (allocator.Allocate<COFFDynamicAtom>())
        COFFDynamicAtom(*this, symbolName, dllName);
    _sharedLibraryAtoms._atoms.push_back(atom);
    ec = error_code::success();
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _noDefinedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _noUndefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _noAbsoluteAtoms;
  }

  virtual const TargetInfo &getTargetInfo() const { return _targetInfo; }

private:
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  const TargetInfo &_targetInfo;
  mutable llvm::BumpPtrAllocator allocator;
};

} // end anonymous namespace

error_code parseCOFFImportLibrary(const TargetInfo &targetInfo,
                                  std::unique_ptr<MemoryBuffer> &mb,
                                  std::vector<std::unique_ptr<File> > &result) {
  // Check the file magic.
  const char *buf = mb->getBufferStart();
  const char *end = mb->getBufferEnd();
  if (end - buf < 20 || memcmp(buf, "\0\0\xFF\xFF", 4))
    return make_error_code(native_reader_error::unknown_file_format);

  error_code ec;
  auto file = std::unique_ptr<File>(
      new FileImportLibrary(targetInfo, std::move(mb), ec));
  if (ec)
    return ec;
  result.push_back(std::move(file));
  return error_code::success();
}

} // end namespace coff
} // end namespace lld
