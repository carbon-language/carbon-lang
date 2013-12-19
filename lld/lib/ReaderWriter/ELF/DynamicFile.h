//===- lib/ReaderWriter/ELF/DynamicFile.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_DYNAMIC_FILE_H
#define LLD_READER_WRITER_ELF_DYNAMIC_FILE_H

#include "Atoms.h"

#include "lld/Core/SharedLibraryFile.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/Path.h"

#include <unordered_map>

namespace lld {
namespace elf {
template <class ELFT> class DynamicFile LLVM_FINAL : public SharedLibraryFile {
public:
  static ErrorOr<std::unique_ptr<DynamicFile> >
  create(std::unique_ptr<llvm::MemoryBuffer> mb, bool useShlibUndefines) {
    std::unique_ptr<DynamicFile> file(
        new DynamicFile(mb->getBufferIdentifier()));

    error_code ec;
    file->_objFile.reset(new llvm::object::ELFFile<ELFT>(mb.release(), ec));

    if (ec)
      return ec;

    llvm::object::ELFFile<ELFT> &obj = *file->_objFile;

    file->_soname = obj.getLoadName();
    if (file->_soname.empty())
      file->_soname = llvm::sys::path::filename(file->path());

    // Create a map from names to dynamic symbol table entries.
    // TODO: This should use the object file's build in hash table instead if
    // it exists.
    for (auto i = obj.begin_dynamic_symbols(),
              e = obj.end_dynamic_symbols();
         i != e; ++i) {
      auto name = obj.getSymbolName(i);
      if (!name)
        return error_code(name);

      // TODO: Add absolute symbols
      if (i->st_shndx == llvm::ELF::SHN_ABS)
        continue;

      if (i->st_shndx == llvm::ELF::SHN_UNDEF) {
        if (!useShlibUndefines)
          continue;
        // Create an undefined atom.
        if (!name->empty()) {
          auto *newAtom =
              new (file->_alloc) ELFUndefinedAtom<ELFT>(*file.get(), *name, &*i);
          file->_undefinedAtoms._atoms.push_back(newAtom);
        }
        continue;
      }
      file->_nameToSym[*name]._symbol = &*i;
    }

    return std::move(file);
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _undefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }

  virtual const SharedLibraryAtom *exports(StringRef name,
                                           bool dataSymbolOnly) const {
    assert(!dataSymbolOnly && "Invalid option for ELF exports!");
    // See if we have the symbol.
    auto sym = _nameToSym.find(name);
    if (sym == _nameToSym.end())
      return nullptr;
    // Have we already created a SharedLibraryAtom for it?
    if (sym->second._atom)
      return sym->second._atom;
    // Create a SharedLibraryAtom for this symbol.
    return sym->second._atom = new (_alloc) ELFDynamicAtom<ELFT>(
        *this, name, _soname, sym->second._symbol);
  }

private:
  DynamicFile(StringRef name)
      : SharedLibraryFile(name) {}

  mutable llvm::BumpPtrAllocator _alloc;
  std::unique_ptr<llvm::object::ELFFile<ELFT>> _objFile;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
  /// \brief DT_SONAME
  StringRef _soname;

  struct SymAtomPair {
    SymAtomPair() : _symbol(nullptr), _atom(nullptr) {}
    const typename llvm::object::ELFFile<ELFT>::Elf_Sym *_symbol;
    const SharedLibraryAtom *_atom;
  };

  mutable std::unordered_map<StringRef, SymAtomPair> _nameToSym;
};
} // end namespace elf
} // end namespace lld

#endif
