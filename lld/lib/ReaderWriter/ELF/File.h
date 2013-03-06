//===- lib/ReaderWriter/ELF/File.h ----------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_FILE_H
#define LLD_READER_WRITER_ELF_FILE_H

#include "Atoms.h"

#include "lld/Core/File.h"
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
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <map>
#include <unordered_map>

namespace lld {
namespace elf {
/// \brief Read a binary, find out based on the symbol table contents what kind
/// of symbol it is and create corresponding atoms for it
template <class ELFT> class ELFFile : public File {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

  // A Map is used to hold the atoms that have been divided up
  // after reading the section that contains Merge String attributes
  struct MergeSectionKey {
    MergeSectionKey(const Elf_Shdr *shdr, int32_t offset)
        : _shdr(shdr), _offset(offset) {
    }
    // Data members
    const Elf_Shdr *_shdr;
    int32_t _offset;
  };
  struct MergeSectionEq {
    int64_t operator()(const MergeSectionKey &k) const {
      return llvm::hash_combine((int64_t)(k._shdr->sh_name),
                                (int64_t) k._offset);
    }
    bool operator()(const MergeSectionKey &lhs,
                    const MergeSectionKey &rhs) const {
      return ((lhs._shdr->sh_name == rhs._shdr->sh_name) &&
              (lhs._offset == rhs._offset));
    }
  };

  struct MergeString {
    MergeString(int32_t offset, StringRef str, const Elf_Shdr *shdr,
                StringRef sectionName)
        : _offset(offset), _string(str), _shdr(shdr),
          _sectionName(sectionName) {
    }
    // the offset of this atom
    int32_t _offset;
    // The content 
    StringRef _string;
    // Section header
    const Elf_Shdr *_shdr;
    // Section name
    StringRef _sectionName;
  };

  // This is used to find the MergeAtom given a relocation 
  // offset
  typedef std::vector<ELFMergeAtom<ELFT> *> MergeAtomsT;
  typedef typename MergeAtomsT::iterator MergeAtomsIter;

  /// \brief find a mergeAtom given a start offset
  struct FindByOffset {
    const Elf_Shdr *_shdr;
    uint64_t _offset;
    FindByOffset(const Elf_Shdr *shdr, uint64_t offset)
        : _shdr(shdr), _offset(offset) {
    }
    bool operator()(const ELFMergeAtom<ELFT> *a) {
      uint64_t off = a->offset();
      return (_shdr->sh_name == a->section()) &&
             ((_offset >= off) && (_offset <= off + a->size()));
    }
  };

  /// \brief find a merge atom given a offset
  MergeAtomsIter findMergeAtom(const Elf_Shdr *shdr, uint64_t offset) {
    return std::find_if(_mergeAtoms.begin(), _mergeAtoms.end(),
                        FindByOffset(shdr, offset));
  }

  typedef std::unordered_map<MergeSectionKey, DefinedAtom *, MergeSectionEq,
                             MergeSectionEq> MergedSectionMapT;
  typedef typename MergedSectionMapT::iterator MergedSectionMapIterT;

public:
  ELFFile(const ELFTargetInfo &ti, StringRef name)
      : File(name), _elfTargetInfo(ti) {
  }

  ELFFile(const ELFTargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> MB,
          llvm::error_code &EC)
      : File(MB->getBufferIdentifier()), _elfTargetInfo(ti) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;

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

    // Sections that have merge string property
    std::vector<const Elf_Shdr *> mergeStringSections;

    bool doStringsMerge = _elfTargetInfo.getLinkerOptions()._mergeCommonStrings;

    // Handle: SHT_REL and SHT_RELA sections:
    // Increment over the sections, when REL/RELA section types are found add
    // the contents to the RelocationReferences map.
    llvm::object::section_iterator sit(_objFile->begin_sections());
    llvm::object::section_iterator sie(_objFile->end_sections());
    for (; sit != sie; sit.increment(EC)) {
      if (EC)
        return;

      const Elf_Shdr *section = _objFile->getElfSection(sit);
      switch (section->sh_type) {
      case llvm::ELF::SHT_NOTE:
      case llvm::ELF::SHT_STRTAB:
      case llvm::ELF::SHT_SYMTAB:
      case llvm::ELF::SHT_SYMTAB_SHNDX:
        continue;
      }
      if (section->sh_size == 0)
        continue;

      if (doStringsMerge) {
        int64_t sectionFlags = section->sh_flags;
        sectionFlags &= ~llvm::ELF::SHF_ALLOC;

        // If the section have mergeable strings, the linker would 
        // need to split the section into multiple atoms and mark them
        // mergeByContent
        if ((section->sh_entsize < 2) &&
           (sectionFlags == (llvm::ELF::SHF_MERGE | llvm::ELF::SHF_STRINGS))) {
          mergeStringSections.push_back(section);
          continue;
        }
      }

      // Create a sectionSymbols entry for every progbits section.
      if (section->sh_type == llvm::ELF::SHT_PROGBITS)
        sectionSymbols[section];

      if (section->sh_type == llvm::ELF::SHT_RELA) {
        StringRef sectionName;
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
        StringRef sectionName;
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

    // Divide the section that contains mergeable strings into tokens
    // TODO
    // a) add resolver support to recognize multibyte chars
    // b) Create a seperate section chunk to write mergeable atoms
    std::vector<MergeString *> tokens;
    for (auto msi : mergeStringSections) {
      StringRef sectionContents;
      StringRef sectionName;
      if ((EC = _objFile->getSectionName(msi, sectionName)))
        return;

      if ((EC = _objFile->getSectionContents(msi, sectionContents)))
        return;

      unsigned int prev = 0;
      for (std::size_t i = 0, e = sectionContents.size(); i != e; ++i) {
        if (sectionContents[i] == '\0') {
          tokens.push_back(new (_readerStorage) MergeString(
              prev, sectionContents.slice(prev, i + 1), msi, sectionName));
          prev = i + 1;
        }
      }
    }

    // Create Mergeable atoms
    for (auto tai : tokens) {
      ArrayRef<uint8_t> content((const uint8_t *)tai->_string.data(),
                                tai->_string.size());
      ELFMergeAtom<ELFT> *mergeAtom = new (_readerStorage) ELFMergeAtom<ELFT>(
          *this, tai->_sectionName, tai->_shdr, content, tai->_offset);
      const MergeSectionKey mergedSectionKey(tai->_shdr, tai->_offset);
      if (_mergedSectionMap.find(mergedSectionKey) == _mergedSectionMap.end())
        _mergedSectionMap.insert(std::make_pair(mergedSectionKey, mergeAtom));
      _definedAtoms._atoms.push_back(mergeAtom);
      _mergeAtoms.push_back(mergeAtom);
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

      StringRef symbolName;
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

      StringRef sectionName;
      if ((EC = _objFile->getSectionName(i.first, sectionName)))
        return;

      // If the section has no symbols, create a custom atom for it.
      if (i.first->sh_type == llvm::ELF::SHT_PROGBITS && symbols.empty() &&
          !sectionContents.empty()) {
        Elf_Sym *sym = new (_readerStorage) Elf_Sym;
        sym->st_name = 0;
        sym->setBindingAndType(llvm::ELF::STB_LOCAL, llvm::ELF::STT_SECTION);
        sym->st_other = 0;
        sym->st_shndx = 0;
        sym->st_value = 0;
        sym->st_size = 0;
        ArrayRef<uint8_t> content((const uint8_t *)sectionContents.data(),
                                  sectionContents.size());
        _definedAtoms._atoms.push_back(
            new (_readerStorage)
            ELFDefinedAtom<ELFT>(*this, sectionName, sectionName, sym, i.first,
                                 content, 0, 0, _references));
      }

      ELFDefinedAtom<ELFT> *previous_atom = nullptr;
      // Don't allocate content to a weak symbol, as they may be merged away.
      // Create an anonymous atom to hold the data.
      ELFDefinedAtom<ELFT> *anonAtom = nullptr;
      ELFReference<ELFT> *anonPrecededBy = nullptr;
      ELFReference<ELFT> *anonFollowedBy = nullptr;

      // i.first is the section the symbol lives in
      for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
        StringRef symbolName;
        if ((EC = _objFile->getSymbolName(i.first, *si, symbolName)))
          return;

        const Elf_Shdr *section = _objFile->getSection(*si);

        bool isCommon = (*si)->getType() == llvm::ELF::STT_COMMON ||
                        (*si)->st_shndx == llvm::ELF::SHN_COMMON;

        if ((section && section->sh_flags & llvm::ELF::SHF_MASKPROC) ||
            (((*si)->st_shndx >= llvm::ELF::SHN_LOPROC) &&
             ((*si)->st_shndx <= llvm::ELF::SHN_HIPROC))) {
          TargetHandler<ELFT> &TargetHandler =
              _elfTargetInfo.template getTargetHandler<ELFT>();
          TargetAtomHandler<ELFT> &elfAtomHandler =
              TargetHandler.targetAtomHandler();
          int64_t targetSymType = elfAtomHandler.getType(*si);

          if (targetSymType == llvm::ELF::STT_COMMON)
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

        // Check to see if we need to add the FollowOn Reference
        // We dont want to do for symbols that are
        // a) common symbols
        ELFReference<ELFT> *followOn = nullptr;
        if (!isCommon && previous_atom) {
          // Replace the followon atom with the anonymous
          // atom that we created, so that the next symbol
          // that we create is a followon from the anonymous
          // atom
          if (!anonFollowedBy) {
            followOn = new (_readerStorage)
                ELFReference<ELFT>(lld::Reference::kindLayoutAfter);
            previous_atom->addReference(followOn);
          }
          else 
            followOn = anonFollowedBy;
        }

        // Don't allocate content to a weak symbol, as they may be merged away.
        // Create an anonymous atom to hold the data.
        anonAtom = nullptr;
        anonPrecededBy = nullptr;
        anonFollowedBy = nullptr;
        if ((*si)->getBinding() == llvm::ELF::STB_WEAK && contentSize != 0) {
          // Create a new non-weak ELF symbol.
          auto sym = new (_readerStorage) Elf_Sym;
          *sym = **si;
          sym->setBinding(llvm::ELF::STB_GLOBAL);
          anonAtom = createDefinedAtomAndAssignRelocations(
              "", sectionName, sym, i.first,
              ArrayRef<uint8_t>((uint8_t *)sectionContents.data() +
                                (*si)->st_value, contentSize));

          // If this is the last atom, lets not create a followon 
          // reference
          if ((si + 1) != se) 
            anonFollowedBy = new (_readerStorage)
               ELFReference<ELFT>(lld::Reference::kindLayoutAfter);
          anonPrecededBy = new (_readerStorage)
              ELFReference<ELFT>(lld::Reference::kindLayoutBefore);
          // Add the references to the anonymous atom that we created
          if (anonFollowedBy)
            anonAtom->addReference(anonFollowedBy);
          anonAtom->addReference(anonPrecededBy);
          if (previous_atom) 
            anonPrecededBy->setTarget(previous_atom);
          contentSize = 0;
        }

        ArrayRef<uint8_t> symbolData = ArrayRef<uint8_t>(
            (uint8_t *)sectionContents.data() + (*si)->st_value, contentSize);

        // If the linker finds that a section has global atoms that are in a 
        // mergeable section, treat them as defined atoms as they shouldnt be
        // merged away as well as these symbols have to be part of symbol
        // resolution
        int64_t sectionFlags = 0;
        if (section)
          sectionFlags = section->sh_flags;
        sectionFlags &= ~llvm::ELF::SHF_ALLOC;
        if (doStringsMerge && section && (section->sh_entsize < 2) &&
            (sectionFlags == (llvm::ELF::SHF_MERGE | llvm::ELF::SHF_STRINGS))) {
          if ((*si)->getBinding() == llvm::ELF::STB_GLOBAL) {
            auto definedMergeAtom = new (_readerStorage) ELFDefinedAtom<ELFT>(
                *this, symbolName, sectionName, (*si), section, symbolData,
                _references.size(), _references.size(), _references);
            _definedAtoms._atoms.push_back(definedMergeAtom);
          }
          continue;
        }

        auto newAtom = createDefinedAtomAndAssignRelocations(
            symbolName, sectionName, *si, i.first, symbolData);

        // If the atom was a weak symbol, lets create a followon 
        // reference to the anonymous atom that we created
        if ((*si)->getBinding() == llvm::ELF::STB_WEAK && anonAtom) {
          ELFReference<ELFT> *wFollowedBy = new (_readerStorage)
              ELFReference<ELFT>(lld::Reference::kindLayoutAfter);
          wFollowedBy->setTarget(anonAtom);
          newAtom->addReference(wFollowedBy);
        }

        if (followOn) {
          ELFReference<ELFT> *precededby = nullptr;
          // Set the followon atom to the weak atom 
          // that we have created, so that they would
          // alias when the file gets written
          if (anonAtom) 
            followOn->setTarget(anonAtom);
          else
            followOn->setTarget(newAtom);
          // Add a preceded by reference only if the current atom is not a 
          // weak atom
          if ((*si)->getBinding() != llvm::ELF::STB_WEAK) {
            precededby = new (_readerStorage)
                ELFReference<ELFT>(lld::Reference::kindLayoutBefore);
            precededby->setTarget(previous_atom);
            newAtom->addReference(precededby);
          }
        }

        // The previous atom is always the atom created before unless
        // the atom is a weak atom
        if (anonAtom)
          previous_atom = anonAtom;
        else
          previous_atom = newAtom;

        _definedAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair((*si), newAtom));
        if (anonAtom)
          _definedAtoms._atoms.push_back(anonAtom);
      }
    }

    // All the Atoms and References are created.  Now update each Reference's
    // target with the Atom pointer it refers to.
    for (auto &ri : _references) {
      if (ri->kind() >= lld::Reference::kindTargetLow) {
        const Elf_Sym *Symbol = _objFile->getElfSymbol(ri->targetSymbolIndex());
        const Elf_Shdr *shdr = _objFile->getSection(Symbol);
        int64_t sectionFlags = 0;
        if (shdr)
          sectionFlags = shdr->sh_flags;
        sectionFlags &= ~llvm::ELF::SHF_ALLOC;

        // If the section has mergeable strings, then make the relocation
        // refer to the MergeAtom to allow deduping
        if (doStringsMerge && shdr && (shdr->sh_entsize < 2) &&
            (sectionFlags == (llvm::ELF::SHF_MERGE | llvm::ELF::SHF_STRINGS))) {
          const TargetRelocationHandler<ELFT> &relHandler = _elfTargetInfo
              .template getTargetHandler<ELFT>().getRelocationHandler();
          int64_t relocAddend = relHandler.relocAddend(*ri);
          uint64_t addend = ri->addend() + relocAddend;
          const MergeSectionKey ms(shdr, addend);
          if (_mergedSectionMap.find(ms) == _mergedSectionMap.end()) {
            if (Symbol->getType() != llvm::ELF::STT_SECTION)
              addend = Symbol->st_value + addend;
            MergeAtomsIter mai = findMergeAtom(shdr, addend);
            if (mai != _mergeAtoms.end()) {
              ri->setOffset(addend - ((*mai)->offset()));
              ri->setAddend(0);
              ri->setTarget(*mai);
            } // check
                else
              llvm_unreachable("unable to find a merge atom");
          } // find
              else
            ri->setTarget(_mergedSectionMap[ms]);
        } else
          ri->setTarget(findAtom(Symbol));
      }
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
    auto rari = _relocationAddendRefences.find(sectionName);
    auto rri = _relocationReferences.find(sectionName);
    unsigned refs = 0;
    if (rari != _relocationAddendRefences.end())
      refs += rari->second.size();
    if (rri != _relocationReferences.end())
      refs += rri->second.size();
    _references.reserve(_references.size() + refs);
    if (rari != _relocationAddendRefences.end())
      for (auto &rai : rari->second) {
        if (!((rai->r_offset >= symbol->st_value) &&
              (rai->r_offset < symbol->st_value + content.size())))
          continue;
        auto *ERef = new (_readerStorage)
            ELFReference<ELFT>(rai, rai->r_offset - symbol->st_value, nullptr);
        _references.push_back(ERef);
      }

    // Add Rel references.
    if (rri != _relocationReferences.end())
      for (auto &ri : rri->second) {
        if ((ri->r_offset >= symbol->st_value) &&
            (ri->r_offset < symbol->st_value + content.size())) {
          auto *ERef = new (_readerStorage)
              ELFReference<ELFT>(ri, ri->r_offset - symbol->st_value, nullptr);
          // Read the addend from the section contents
          // TODO : We should move the way lld reads relocations totally from
          // ELFObjectFile
          int32_t addend = *(content.data() + ri->r_offset - symbol->st_value);
          ERef->setAddend(addend);
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
  std::unordered_map<StringRef,
                     std::vector<const Elf_Rela *> > _relocationAddendRefences;
  MergedSectionMapT _mergedSectionMap;
  std::unordered_map<StringRef,
                     std::vector<const Elf_Rel *> > _relocationReferences;
  std::vector<ELFReference<ELFT> *> _references;
  llvm::DenseMap<const Elf_Sym *, Atom *> _symbolToAtomMapping;
  llvm::BumpPtrAllocator _readerStorage;
  const ELFTargetInfo &_elfTargetInfo;
  MergeAtomsT _mergeAtoms;
};
} // end namespace elf
} // end namespace lld

#endif
