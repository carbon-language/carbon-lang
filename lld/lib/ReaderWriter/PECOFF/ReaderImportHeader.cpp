//===- lib/ReaderWriter/PECOFF/ReaderImportHeader.cpp ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file \brief This file provides a way to read an import library member in a
/// .lib file.
///
/// Archive Files in Windows
/// ========================
///
/// In Windows, archive files with .lib file extension serve two different
/// purposes.
///
///  - For static linking: An archive file in this use case contains multiple
///    normal .obj files and is used for static linking. This is the same
///    usage as .a file in Unix.
///
///  - For dynamic linking: An archive file in this use case contains pseudo
///    .obj files to describe exported symbols of a DLL. Each pseudo .obj file
///    in an archive has a name of an exported symbol and a DLL filename from
///    which the symbol can be imported. When you link a DLL on Windows, you
///    pass the name of the .lib file for the DLL instead of the DLL filename
///    itself. That is the Windows way of linking against a shared library.
///
/// This file contains a function to handle the pseudo object file.
///
/// Windows Loader and Import Address Table
/// =======================================
///
/// Windows supports a GOT-like mechanism for DLLs. The executable using DLLs
/// contains a list of DLL names and list of symbols that need to be resolved by
/// the loader. Windows loader maps the executable and all the DLLs to memory,
/// resolves the symbols referencing items in DLLs, and updates the import
/// address table in memory. The import address table is an array of pointers to
/// all of the data or functions in DLL referenced by the executable. You cannot
/// access items in DLLs directly. They have to be accessed through an extra
/// level of indirection.
///
/// So, if you want to access an item in DLL, you have to go through a
/// pointer. How do you actually do that? For each symbol in DLL, there is
/// another set of symbols with "_imp__" prefix. For example, if you have a
/// global variable "foo" in a DLL, a pointer to the variable is exported from
/// the DLL as "_imp__foo". You cannot directly use "foo" but need to go through
/// "_imp__foo", because symbol "foo" is not exported.
///
/// Is this OK? That's not that complicated. Because items in a DLL are not
/// directly accessible, you need to access through a pointer, and the pointer
/// is available as a symbol with "_imp__" prefix.
///
/// Trick 1: Although you can write code with "_imp__" prefix, today's compiler
/// and linker let you write code as if there's no extra level of
/// indirection. That's why you haven't seen lots of _imp__ in your code. A
/// variable or a function declared with "dllimport" attributes is treated as an
/// item in a DLL, and the compiler automatically mangles its name and inserts
/// the extra level of indirection when accessing the item. Here are some
/// examples:
///
///   __declspec(dllimport) int var_in_dll;
///   var_in_dll = 3; // is equivalent to *_imp__var_in_dll = 3;
///
///   __declspec(dllimport) int fn_in_dll(void);
///   fn_in_dll();     // is equivalent to (*_imp__fn_in_dll)();
///
/// It's just the compiler rewrites code for you so that you don't need to
/// handle the indirection youself.
///
/// Trick 2: __declspec(dllimport) is mandatory for data but optional for
/// function. For a function, the linker creates a jump table with the original
/// symbol name, so that the function is accessible without "_imp__" prefix. The
/// same function in a DLL can be called through two different symbols if it's
/// not dllimport'ed.
///
///   (*_imp__fn)()
///   fn()
///
/// The above functions do the same thing. fn's content is a JMP instruction to
/// branch to the address pointed by _imp__fn. The latter may be a little bit
/// slower than the former because it will execute the extra JMP instruction, but
/// that's not an important point here.
///
/// If a function is dllimport'ed, which is usually done in a header file,
/// mangled name will be used at compile time so the jump table will not be
/// used.
///
/// Because there's no way to hide the indirection for data access at link time,
/// data has to be accessed through dllimport'ed symbols or explicit "_imp__"
/// prefix.
///
/// Creating Atoms for the Import Address Table
/// ===========================================
///
/// This file is to read a pseudo object file and create at most two atoms. One
/// is a shared library atom for "_imp__" symbol. The another is a defined atom
/// for the JMP instruction if the symbol is for a function.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ReaderImportHeader"

#include "Atoms.h"

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

/// The defined atom for jump table.
class FuncAtom : public COFFBaseDefinedAtom {
public:
  FuncAtom(const File &file, StringRef symbolName)
      : COFFBaseDefinedAtom(file, symbolName, &rawContent) {}

  virtual uint64_t ordinal() const { return 0; }
  virtual Scope scope() const { return scopeGlobal; }
  virtual ContentType contentType() const { return typeCode; }
  virtual Alignment alignment() const { return Alignment(1); }
  virtual ContentPermissions permissions() const { return permR_X; }

private:
  static std::vector<uint8_t> rawContent;
};

// MSVC doesn't seem to like C++11 initializer list, so initialize the
// vector from an array.
namespace {
uint8_t FuncAtomContent[] = {
  0xff, 0x25, 0x00, 0x00, 0x00, 0x00,  // jmp *0x0
  0x90, 0x90                           // nop; nop
};
} // anonymous namespace

std::vector<uint8_t> FuncAtom::rawContent(
    FuncAtomContent, FuncAtomContent + sizeof(FuncAtomContent));

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

    const COFFSharedLibraryAtom *dataAtom = addSharedLibraryAtom(symbolName,
                                                                 dllName);
    int type = *reinterpret_cast<const support::ulittle16_t *>(buf + 18) >> 16;
    if (type == llvm::COFF::IMPORT_CODE)
      addDefinedAtom(symbolName, dllName, dataAtom);

    ec = error_code::success();
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
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
  const COFFSharedLibraryAtom *addSharedLibraryAtom(StringRef symbolName,
                                                    StringRef dllName) {
    auto *name = new (allocator.Allocate<std::string>()) std::string("__imp_");
    name->append(symbolName);
    auto *atom = new (allocator.Allocate<COFFSharedLibraryAtom>())
        COFFSharedLibraryAtom(*this, *name, symbolName, dllName);
    _sharedLibraryAtoms._atoms.push_back(atom);
    return atom;
  }

  void addDefinedAtom(StringRef symbolName, StringRef dllName,
                      const COFFSharedLibraryAtom *dataAtom) {
    auto *atom = new (allocator.Allocate<FuncAtom>())
        FuncAtom(*this, symbolName);

    // The first two byte of the atom is JMP instruction.
    atom->addReference(std::unique_ptr<COFFReference>(
        new COFFReference(dataAtom, 2, llvm::COFF::IMAGE_REL_I386_DIR32)));
    _definedAtoms._atoms.push_back(atom);
  }

  atom_collection_vector<DefinedAtom> _definedAtoms;
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
