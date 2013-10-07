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
  DynamicLibraryWriter(const ELFLinkingContext &context)
      : OutputELFWriter<ELFT>(context),
        _runtimeFile(new CRuntimeFile<ELFT>(context)) {}

private:
  void buildDynamicSymbolTable(const File &file);
  void addDefaultAtoms();
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File> > &);
  void finalizeDefaultAtomValues();

  llvm::BumpPtrAllocator _alloc;
  std::unique_ptr<CRuntimeFile<ELFT> > _runtimeFile;
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

template <class ELFT> void DynamicLibraryWriter<ELFT>::addDefaultAtoms() {
  _runtimeFile->addAbsoluteAtom("_end");
}

/// \brief Hook in lld to add CRuntime file
template <class ELFT>
bool DynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File> > &result) {
  // Add the default atoms as defined by executables
  addDefaultAtoms();
  OutputELFWriter<ELFT>::createImplicitFiles(result);
  result.push_back(std::move(_runtimeFile));
  return true;
}

template <class ELFT>
void DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  auto underScoreEndAtomIter = this->_layout->findAbsoluteAtom("_end");

  if (auto bssSection = this->_layout->findOutputSection(".bss")) {
    (*underScoreEndAtomIter)->_virtualAddr =
        bssSection->virtualAddr() + bssSection->memSize();
  } else if (auto dataSection = this->_layout->findOutputSection(".data")) {
    (*underScoreEndAtomIter)->_virtualAddr =
        dataSection->virtualAddr() + dataSection->memSize();
  }

  this->_targetHandler.finalizeSymbolValues();
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H
