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
///    regular .obj files and is used for static linking. This is the same
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
/// address table (IAT) in memory. The IAT is an array of pointers to all of the
/// data or functions in DLL referenced by the executable. You cannot access
/// items in DLLs directly. They have to be accessed through an extra level of
/// indirection.
///
/// So, if you want to access an item in DLL, you have to go through a
/// pointer. How do you actually do that? You need a symbol for a pointer in the
/// IAT. For each symbol defined in a DLL, a symbol with "__imp_" prefix is
/// exported from the DLL for an IAT entry. For example, if you have a global
/// variable "foo" in a DLL, a pointer to the variable is available as
/// "_imp__foo". The IAT is an array of _imp__ symbols.
///
/// Is this OK? That's not that complicated. Because items in a DLL are not
/// directly accessible, you need to access through a pointer, and the pointer
/// is available as a symbol with _imp__ prefix.
///
/// Note 1: Although you can write code with _imp__ prefix, today's compiler and
/// linker let you write code as if there's no extra level of indirection.
/// That's why you haven't seen lots of _imp__ in your code. A variable or a
/// function declared with "dllimport" attribute is treated as an item in a DLL,
/// and the compiler automatically mangles its name and inserts the extra level
/// of indirection when accessing the item. Here are some examples:
///
///   __declspec(dllimport) int var_in_dll;
///   var_in_dll = 3;  // is equivalent to *_imp__var_in_dll = 3;
///
///   __declspec(dllimport) int fn_in_dll(void);
///   fn_in_dll();     // is equivalent to (*_imp__fn_in_dll)();
///
/// It's just the compiler rewrites code for you so that you don't need to
/// handle the indirection yourself.
///
/// Note 2: __declspec(dllimport) is mandatory for data but optional for
/// function. For a function, the linker creates a jump table with the original
/// symbol name, so that the function is accessible without _imp__ prefix. The
/// same function in a DLL can be called through two different symbols if it's
/// not dllimport'ed.
///
///   (*_imp__fn)()
///   fn()
///
/// The above functions do the same thing. fn's content is a JMP instruction to
/// branch to the address pointed by _imp__fn. The latter may be a little bit
/// slower than the former because it will execute the extra JMP instruction,
/// but that's usually negligible.
///
/// If a function is dllimport'ed, which is usually done in a header file,
/// mangled name will be used at compile time so the jump table will not be
/// used.
///
/// Because there's no way to hide the indirection for data access at link time,
/// data has to be accessed through dllimport'ed symbols or explicit _imp__
/// prefix.
///
/// Idata Sections in the Pseudo Object File
/// ========================================
///
/// The object file created by cl.exe has several sections whose name starts
/// with ".idata$" followed by a number. The contents of the sections seem the
/// fragments of a complete ".idata" section. These sections has relocations for
/// the data referenced from the idata secton. Generally, the linker discards
/// "$" and all characters that follow from the section name and merges their
/// contents to one section. So, it looks like if everything would work fine,
/// the idata section would naturally be constructed without having any special
/// code for doing that.
///
/// However, the LLD linker cannot do that. An idata section constructed in that
/// way was never be in valid format. We don't know the reason yet. Our
/// assumption on the idata fragment could simply be wrong, or the LLD linker is
/// not powerful enough to do the job. Meanwhile, we construct the idata section
/// ourselves. All the "idata$" sections in the pseudo object file are currently
/// ignored.
///
/// Creating Atoms for the Import Address Table
/// ===========================================
///
/// The function in this file reads a pseudo object file and creates at most two
/// atoms. One is a shared library atom for _imp__ symbol. The another is a
/// defined atom for the JMP instruction if the symbol is for a function.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ReaderImportHeader"

#include "Atoms.h"

#include "lld/Core/File.h"
#include "lld/Core/Error.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"

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
using namespace lld::pecoff;
using namespace llvm;

namespace lld {

namespace {

uint8_t FuncAtomContent[] = { 0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *0x0
                              0xcc, 0xcc // int 3; int 3
};

/// The defined atom for jump table.
class FuncAtom : public COFFLinkerInternalAtom {
public:
  FuncAtom(const File &file, StringRef symbolName)
      : COFFLinkerInternalAtom(
            file, /*oridnal*/ 0,
            std::vector<uint8_t>(FuncAtomContent,
                                 FuncAtomContent + sizeof(FuncAtomContent)),
            symbolName) {}

  uint64_t ordinal() const override { return 0; }
  Scope scope() const override { return scopeGlobal; }
  ContentType contentType() const override { return typeCode; }
  Alignment alignment() const override { return Alignment(1); }
  ContentPermissions permissions() const override { return permR_X; }
};

class FileImportLibrary : public File {
public:
  FileImportLibrary(std::unique_ptr<MemoryBuffer> mb, error_code &ec)
      : File(mb->getBufferIdentifier(), kindSharedLibrary) {
    const char *buf = mb->getBufferStart();
    const char *end = mb->getBufferEnd();

    // The size of the string that follows the header.
    uint32_t dataSize = *reinterpret_cast<const support::ulittle32_t *>(
                             buf + offsetof(COFF::ImportHeader, SizeOfData));

    // Check if the total size is valid.
    if (end - buf != sizeof(COFF::ImportHeader) + dataSize) {
      ec = make_error_code(NativeReaderError::unknown_file_format);
      return;
    }

    uint16_t hint = *reinterpret_cast<const support::ulittle16_t *>(
                         buf + offsetof(COFF::ImportHeader, OrdinalHint));
    StringRef symbolName(buf + sizeof(COFF::ImportHeader));
    StringRef dllName(buf + sizeof(COFF::ImportHeader) + symbolName.size() + 1);

    // TypeInfo is a bitfield. The least significant 2 bits are import
    // type, followed by 3 bit import name type.
    uint16_t typeInfo = *reinterpret_cast<const support::ulittle16_t *>(
                             buf + offsetof(COFF::ImportHeader, TypeInfo));
    int type = typeInfo & 0x3;
    int nameType = (typeInfo >> 2) & 0x7;

    // Symbol name used by the linker may be different from the symbol name used
    // by the loader. The latter may lack symbol decorations, or may not even
    // have name if it's imported by ordinal.
    StringRef importName = symbolNameToImportName(symbolName, nameType);

    const COFFSharedLibraryAtom *dataAtom =
        addSharedLibraryAtom(hint, symbolName, importName, dllName);
    if (type == llvm::COFF::IMPORT_CODE)
      addDefinedAtom(symbolName, dllName, dataAtom);

    ec = error_code::success();
  }

  const atom_collection<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }

  const atom_collection<UndefinedAtom> &undefined() const override {
    return _noUndefinedAtoms;
  }

  const atom_collection<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const atom_collection<AbsoluteAtom> &absolute() const override {
    return _noAbsoluteAtoms;
  }

private:
  const COFFSharedLibraryAtom *addSharedLibraryAtom(uint16_t hint,
                                                    StringRef symbolName,
                                                    StringRef importName,
                                                    StringRef dllName) {
    auto *atom = new (_alloc)
        COFFSharedLibraryAtom(*this, hint, symbolName, importName, dllName);
    _sharedLibraryAtoms._atoms.push_back(atom);
    return atom;
  }

  void addDefinedAtom(StringRef symbolName, StringRef dllName,
                      const COFFSharedLibraryAtom *dataAtom) {
    auto *atom = new (_alloc) FuncAtom(*this, symbolName);

    // The first two byte of the atom is JMP instruction.
    atom->addReference(std::unique_ptr<COFFReference>(
        new COFFReference(dataAtom, 2, llvm::COFF::IMAGE_REL_I386_DIR32)));
    _definedAtoms._atoms.push_back(atom);
  }

  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  mutable llvm::BumpPtrAllocator _alloc;

  // Does the same thing as StringRef::ltrim() but removes at most one
  // character.
  StringRef ltrim1(StringRef str, const char *chars) const {
    if (!str.empty() && strchr(chars, str[0]))
      return str.substr(1);
    return str;
  }

  // Convert the given symbol name to the import symbol name exported by the
  // DLL.
  StringRef symbolNameToImportName(StringRef symbolName, int nameType) const {
    StringRef ret;
    switch (nameType) {
    case llvm::COFF::IMPORT_ORDINAL:
      // The import is by ordinal. No symbol name will be used to identify the
      // item in the DLL. Only its ordinal will be used.
      return "";
    case llvm::COFF::IMPORT_NAME:
      // The import name in this case is identical to the symbol name.
      return symbolName;
    case llvm::COFF::IMPORT_NAME_NOPREFIX:
      // The import name is the symbol name without leading ?, @ or _.
      ret = ltrim1(symbolName, "?@_");
      break;
    case llvm::COFF::IMPORT_NAME_UNDECORATE:
      // Similar to NOPREFIX, but we also need to truncate at the first @.
      ret = ltrim1(symbolName, "?@_");
      ret = ret.substr(0, ret.find('@'));
      break;
    }
    std::string *str = new (_alloc) std::string(ret);
    return *str;
  }
};

class COFFImportLibraryReader : public Reader {
public:
  bool canParse(file_magic magic, StringRef,
                const MemoryBuffer &mb) const override {
    if (mb.getBufferSize() < sizeof(COFF::ImportHeader))
      return false;
    return (magic == llvm::sys::fs::file_magic::coff_import_library);
  }

  error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File> > &result) const override {
    error_code ec;
    auto file = std::unique_ptr<File>(new FileImportLibrary(std::move(mb), ec));
    if (ec)
      return ec;
    result.push_back(std::move(file));
    return error_code::success();
  }
};

} // end anonymous namespace

namespace pecoff {

error_code parseCOFFImportLibrary(const LinkingContext &targetInfo,
                                  std::unique_ptr<MemoryBuffer> &mb,
                                  std::vector<std::unique_ptr<File> > &result) {
  // Check the file magic.
  const char *buf = mb->getBufferStart();
  const char *end = mb->getBufferEnd();
  // Error if the file is too small or does not start with the magic.
  if (end - buf < static_cast<ptrdiff_t>(sizeof(COFF::ImportHeader)) ||
      memcmp(buf, "\0\0\xFF\xFF", 4))
    return make_error_code(NativeReaderError::unknown_file_format);

  error_code ec;
  auto file = std::unique_ptr<File>(new FileImportLibrary(std::move(mb), ec));
  if (ec)
    return ec;
  result.push_back(std::move(file));
  return error_code::success();
}

} // end namespace pecoff

void Registry::addSupportCOFFImportLibraries() {
  add(std::unique_ptr<Reader>(new COFFImportLibraryReader()));
}

} // end namespace lld
