//===- lib/ReaderWriter/ELF/FileELF.h ------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \brief Read a binary, find out based on the symbol table contents what kind
/// of symbol it is and create corresponding atoms for it

#ifndef LLD_READER_WRITER_FILE_ELF_H
#define LLD_READER_WRITER_FILE_ELF_H

#include "lld/Core/Reference.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"
#include "lld/ReaderWriter/ReaderArchive.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "AtomsELF.h"

namespace lld {

template <class ELFT> class FileELF : public File {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

public:
  FileELF(const ELFTargetInfo &ti, const StringRef name)
      : File(name), _elfTargetInfo(ti) {
  }

  FileELF(const ELFTargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> MB,
          llvm::error_code &EC)
      : File(MB->getBufferIdentifier()), _elfTargetInfo(ti) {
    llvm::OwningPtr<llvm::object::Binary> binaryFile;
    EC = createBinary(MB.release(), binaryFile);
    if (EC)
      return;

    // Point Obj to correct class and bitwidth ELF object
    _objFile.reset(
        llvm::dyn_cast<llvm::object::ELFObjectFile<ELFT> >(binaryFile.get()));

    if (!_objFile) {
      EC = make_error_code(llvm::object::object_error::invalid_file_type);
      return;
    }

    binaryFile.take();

    std::map<const Elf_Shdr *, std::vector<const Elf_Sym *> > sectionSymbols;

    // Handle: SHT_REL and SHT_RELA sections:
    // Increment over the sections, when REL/RELA section types are found add
    // the contents to the RelocationReferences map.
    llvm::object::section_iterator sit(_objFile->begin_sections());
    llvm::object::section_iterator sie(_objFile->end_sections());
    for (; sit != sie; sit.increment(EC)) {
      if (EC)
        return;

      const Elf_Shdr *section = _objFile->getElfSection(sit);

      if (section->sh_type == llvm::ELF::SHT_RELA) {
        llvm::StringRef sectionName;
        if ((EC = _objFile->getSectionName(section, sectionName)))
          return;
        // Get rid of the leading .rela so Atoms can use their own section
        // name to find the relocs.
        sectionName = sectionName.drop_front(5);

        auto rai(_objFile->beginELFRela(section));
        auto rae(_objFile->endELFRela(section));

        auto &Ref = _relocationAddendRefences[sectionName];
        for (; rai != rae; ++rai) {
          Ref.push_back(&*rai);
        }
      }

      if (section->sh_type == llvm::ELF::SHT_REL) {
        llvm::StringRef sectionName;
        if ((EC = _objFile->getSectionName(section, sectionName)))
          return;
        // Get rid of the leading .rel so Atoms can use their own section
        // name to find the relocs.
        sectionName = sectionName.drop_front(4);

        auto ri(_objFile->beginELFRel(section));
        auto re(_objFile->endELFRel(section));

        auto &Ref = _relocationReferences[sectionName];
        for (; ri != re; ++ri) {
          Ref.push_back(&*ri);
        }
      }
    }

    // Increment over all the symbols collecting atoms and symbol names for
    // later use.
    llvm::object::symbol_iterator it(_objFile->begin_symbols());
    llvm::object::symbol_iterator ie(_objFile->end_symbols());

    for (; it != ie; it.increment(EC)) {
      if (EC)
        return;

      if ((EC = it->getSection(sit)))
        return;

      const Elf_Shdr *section = _objFile->getElfSection(sit);
      const Elf_Sym *symbol = _objFile->getElfSymbol(it);

      llvm::StringRef symbolName;
      if ((EC = _objFile->getSymbolName(section, symbol, symbolName)))
        return;

      if (symbol->st_shndx == llvm::ELF::SHN_ABS) {
        // Create an absolute atom.
        auto *newAtom = new (_readerStorage)
            ELFAbsoluteAtom<ELFT>(*this, symbolName, symbol, symbol->st_value);

        _absoluteAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair(symbol, newAtom));
      } else if (symbol->st_shndx == llvm::ELF::SHN_UNDEF) {
        // Create an undefined atom.
        auto *newAtom = new (_readerStorage)
            ELFUndefinedAtom<ELFT>(*this, symbolName, symbol);

        _undefinedAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair(symbol, newAtom));
      } else {
        // This is actually a defined symbol. Add it to its section's list of
        // symbols.
        if (symbol->getType() == llvm::ELF::STT_NOTYPE || symbol->getType() ==
            llvm::ELF::STT_OBJECT || symbol->getType() == llvm::ELF::STT_FUNC ||
            symbol->getType() == llvm::ELF::STT_GNU_IFUNC ||
            symbol->getType() == llvm::ELF::STT_SECTION || symbol->getType() ==
            llvm::ELF::STT_FILE || symbol->getType() == llvm::ELF::STT_TLS ||
            symbol->getType() == llvm::ELF::STT_COMMON ||
            symbol->st_shndx == llvm::ELF::SHN_COMMON) {
          sectionSymbols[section].push_back(symbol);
        } else {
          llvm::errs() << "Unable to create atom for: " << symbolName << "\n";
          EC = llvm::object::object_error::parse_failed;
          return;
        }
      }
    }

    for (auto &i : sectionSymbols) {
      auto &symbols = i.second;
      // Sort symbols by position.
      std::stable_sort(symbols.begin(), symbols.end(),
                       [](const Elf_Sym * A, const Elf_Sym * B) {
        return A->st_value < B->st_value;
      });

      StringRef sectionContents;
      if ((EC = _objFile->getSectionContents(i.first, sectionContents)))
        return;

      llvm::StringRef sectionName;
      if ((EC = _objFile->getSectionName(i.first, sectionName)))
        return;

      // i.first is the section the symbol lives in
      for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
        llvm::StringRef symbolName;
        if ((EC = _objFile->getSymbolName(i.first, *si, symbolName)))
          return;

        bool isCommon = (*si)->getType() == llvm::ELF::STT_COMMON ||
                        (*si)->st_shndx == llvm::ELF::SHN_COMMON;

        DefinedAtom::ContentType c;

        if (((*si)->st_shndx >= llvm::ELF::SHN_LOPROC) &&
            ((*si)->st_shndx <= llvm::ELF::SHN_HIPROC)) {
          elf::ELFTargetHandler<ELFT> &elfTargetHandler =
              _elfTargetInfo.getTargetHandler<ELFT>();
          elf::ELFTargetAtomHandler<ELFT> &elfAtomHandler =
              elfTargetHandler.targetAtomHandler();
          c = elfAtomHandler.contentType(*si);

          if (c == DefinedAtom::typeZeroFill)
            isCommon = true;
        }

        // Get the symbol's content:
        uint64_t contentSize;
        if (si + 1 == se) {
          // if this is the last symbol, take up the remaining data.
          contentSize = isCommon ? 0 : i.first->sh_size - (*si)->st_value;
        } else {
          contentSize = isCommon ? 0 : (*(si + 1))->st_value - (*si)->st_value;
        }

        // Don't allocate content to a weak symbol, as they may be merged away.
        // Create an anonymous atom to hold the data.
        ELFDefinedAtom<ELFT> *anonAtom = nullptr;
        if ((*si)->getBinding() == llvm::ELF::STB_WEAK && contentSize != 0) {
          // Create a new non-weak ELF symbol.
          auto sym = new (_readerStorage) Elf_Sym;
          *sym = **si;
          sym->setBinding(llvm::ELF::STB_GLOBAL);
          anonAtom = createDefinedAtomAndAssignRelocations(
              "", sectionName, sym, i.first,
              ArrayRef<uint8_t>((uint8_t *)sectionContents.data() +
                                (*si)->st_value, contentSize));
          contentSize = 0;
        }

        ArrayRef<uint8_t> symbolData = ArrayRef<uint8_t>(
            (uint8_t *)sectionContents.data() + (*si)->st_value, contentSize);

        auto newAtom = createDefinedAtomAndAssignRelocations(
            symbolName, sectionName, *si, i.first, symbolData);

        _definedAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair((*si), newAtom));
        if (anonAtom)
          _definedAtoms._atoms.push_back(anonAtom);
      }
    }

    // All the Atoms and References are created.  Now update each Reference's
    // target with the Atom pointer it refers to.
    for (auto &ri : _references) {
      const Elf_Sym *Symbol = _objFile->getElfSymbol(ri->targetSymbolIndex());
      ri->setTarget(findAtom(Symbol));
    }
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

  virtual const ELFTargetInfo &getTargetInfo() const { return _elfTargetInfo; }

  Atom *findAtom(const Elf_Sym *symbol) {
    return _symbolToAtomMapping.lookup(symbol);
  }

private:
  ELFDefinedAtom<ELFT> *createDefinedAtomAndAssignRelocations(
      StringRef symbolName, StringRef sectionName, const Elf_Sym *symbol,
      const Elf_Shdr *section, ArrayRef<uint8_t> content) {
    unsigned int referenceStart = _references.size();

    // Only relocations that are inside the domain of the atom are added.

    // Add Rela (those with r_addend) references:
    for (auto &rai : _relocationAddendRefences[sectionName]) {
      if (!((rai->r_offset >= symbol->st_value) &&
            (rai->r_offset < symbol->st_value + content.size())))
        continue;
      auto *ERef = new (_readerStorage)
          ELFReference<ELFT>(rai, rai->r_offset - symbol->st_value, nullptr);
      _references.push_back(ERef);
    }

    // Add Rel references.
    for (auto &ri : _relocationReferences[sectionName]) {
      if ((ri->r_offset >= symbol->st_value) &&
          (ri->r_offset < symbol->st_value + content.size())) {
        auto *ERef = new (_readerStorage)
            ELFReference<ELFT>(ri, ri->r_offset - symbol->st_value, nullptr);
        _references.push_back(ERef);
      }
    }

    // Create the DefinedAtom and add it to the list of DefinedAtoms.
    return new (_readerStorage) ELFDefinedAtom<
        ELFT>(*this, symbolName, sectionName, symbol, section, content,
              referenceStart, _references.size(), _references);
  }

  std::unique_ptr<llvm::object::ELFObjectFile<ELFT> > _objFile;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;

  /// \brief _relocationAddendRefences and _relocationReferences contain the
  /// list of relocations references.  In ELF, if a section named, ".text" has
  /// relocations will also have a section named ".rel.text" or ".rela.text"
  /// which will hold the entries. -- .rel or .rela is prepended to create
  /// the SHT_REL(A) section name.
  std::map<llvm::StringRef,
           std::vector<const Elf_Rela *> > _relocationAddendRefences;
  std::map<llvm::StringRef,
           std::vector<const Elf_Rel *> > _relocationReferences;
  std::vector<ELFReference<ELFT> *> _references;
  llvm::DenseMap<const Elf_Sym *, Atom *> _symbolToAtomMapping;
  llvm::BumpPtrAllocator _readerStorage;
  const ELFTargetInfo &_elfTargetInfo;
};
} // lld

#endif // LLD_READER_WRITER_FILE_ELF_H
