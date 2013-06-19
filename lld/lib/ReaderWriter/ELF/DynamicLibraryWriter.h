//===- lib/ReaderWriter/ELF/DynamicLibraryWriter.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H
#define LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H

#include "OutputELFWriter.h"

namespace lld {
namespace elf {
using namespace llvm;
using namespace llvm::object;

template<class ELFT>
class DynamicLibraryWriter;

//===----------------------------------------------------------------------===//
//  DynamicLibraryWriter Class
//===----------------------------------------------------------------------===//
template<class ELFT>
class DynamicLibraryWriter : public OutputELFWriter<ELFT> {
public:
  DynamicLibraryWriter(const ELFTargetInfo &ti):OutputELFWriter<ELFT>(ti)
  {}

private:
  void buildDynamicSymbolTable(const File &file);
  void addDefaultAtoms();
  void finalizeDefaultAtomValues();

  llvm::BumpPtrAllocator _alloc;
};

//===----------------------------------------------------------------------===//
//  DynamicLibraryWriter
//===----------------------------------------------------------------------===//
template <class ELFT>
void DynamicLibraryWriter<ELFT>::buildDynamicSymbolTable(const File &file) {
  // Add all the defined symbols to the dynamic symbol table
  // we need hooks into the Atom to find out which atoms need
  // to be exported
  for (auto sec : this->_layout->sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms()) {
        const DefinedAtom *da = dyn_cast<const DefinedAtom>(atom->_atom);
        if (da && (da->scope() != DefinedAtom::scopeTranslationUnit))
          this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                               atom->_virtualAddr, atom);
      }

  for (const UndefinedAtom *a : file.undefined())
    this->_dynamicSymbolTable->addSymbol(a, ELF::SHN_UNDEF);

  OutputELFWriter<ELFT>::buildDynamicSymbolTable(file);
}

template<class ELFT>
void DynamicLibraryWriter<ELFT>::addDefaultAtoms() { }

template<class ELFT>
void DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  this->_targetHandler.finalizeSymbolValues();
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H
