//===- lib/ReaderWriter/ELF/ExecutableAtoms.h -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_EXECUTABLE_ATOM_H
#define LLD_READER_WRITER_ELF_EXECUTABLE_ATOM_H

#include "Atoms.h"
#include "File.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/ReaderWriter/Writer.h"

namespace lld {
namespace elf {
/// \brief All atoms are owned by a File. To add linker specific atoms
/// the atoms need to be inserted to a file called (CRuntimeFile) which
/// are basically additional symbols required by libc and other runtime
/// libraries part of executing a program. This class provides support
/// for adding absolute symbols and undefined symbols
template <class ELFT> class CRuntimeFile : public ELFFile<ELFT> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  CRuntimeFile(const ELFTargetInfo &ti, StringRef name = "C runtime")
      : ELFFile<ELFT>(ti, name) {}

  /// \brief add a global absolute atom
  virtual void addAbsoluteAtom(StringRef symbolName) {
    Elf_Sym *symbol = new (_allocator.Allocate<Elf_Sym>()) Elf_Sym;
    symbol->st_name = 0;
    symbol->st_value = 0;
    symbol->st_shndx = llvm::ELF::SHN_ABS;
    symbol->setBindingAndType(llvm::ELF::STB_GLOBAL,
                              llvm::ELF::STT_OBJECT);
    symbol->st_other = llvm::ELF::STV_DEFAULT;
    symbol->st_size = 0;
    auto *newAtom =
        new (_allocator) ELFAbsoluteAtom<ELFT>(*this, symbolName, symbol, -1);
    _absoluteAtoms._atoms.push_back(newAtom);
  }

  /// \brief add an undefined atom
  virtual void addUndefinedAtom(StringRef symbolName) {
    Elf_Sym *symbol = new (_allocator) Elf_Sym;
    symbol->st_name = 0;
    symbol->st_value = 0;
    symbol->st_shndx = llvm::ELF::SHN_UNDEF;
    symbol->st_other = llvm::ELF::STV_DEFAULT;
    symbol->st_size = 0;
    auto *newAtom =
        new (_allocator) ELFUndefinedAtom<ELFT>(*this, symbolName, symbol);
    _undefinedAtoms._atoms.push_back(newAtom);
  }

  virtual const File::atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
  }

  virtual const File::atom_collection<UndefinedAtom> &undefined() const {
    return _undefinedAtoms;
  }

  virtual const File::atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }

  virtual const File::atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }

  // cannot add atoms to C Runtime file
  virtual void addAtom(const Atom &) {
    llvm_unreachable("cannot add atoms to Runtime files");
  }

protected:
  llvm::BumpPtrAllocator _allocator;
  File::atom_collection_vector<DefinedAtom> _definedAtoms;
  File::atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  File::atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  File::atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
};
} // end namespace elf
} // end namespace lld

#endif
