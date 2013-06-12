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
  ELFMergeAtom<ELFT> *findMergeAtom(const Elf_Shdr *shdr, uint64_t offset) {
    auto it = std::find_if(_mergeAtoms.begin(), _mergeAtoms.end(),
                           FindByOffset(shdr, offset));
    assert(it != _mergeAtoms.end());
    return *it;
  }

  typedef std::unordered_map<MergeSectionKey, DefinedAtom *, MergeSectionEq,
                             MergeSectionEq> MergedSectionMapT;
  typedef typename MergedSectionMapT::iterator MergedSectionMapIterT;

public:
  ELFFile(const ELFTargetInfo &ti, StringRef name)
      : File(name, kindObject), _elfTargetInfo(ti) {}

  ELFFile(const ELFTargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> MB,
          llvm::error_code &EC)
      : File(MB->getBufferIdentifier(), kindObject), _elfTargetInfo(ti),
        _ordinal(0), _doStringsMerge(false) {
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

    _doStringsMerge = _elfTargetInfo.mergeCommonStrings();

    // Read input sections from the input file that need to be converted to
    // atoms
    if (createAtomizableSections(EC))
      return;

    // For mergeable strings, we would need to split the section into various
    // atoms
    if (createMergeableAtoms(EC))
      return;

    // Create the necessary symbols that are part of the section that we
    // created in createAtomizableSections function
    if (createSymbolsFromAtomizableSections(EC))
      return;

    // Create the appropriate atoms from the file
    if (createAtoms(EC))
      return;
  }

  /// \brief Read input sections and populate necessary data structures
  /// to read them later and create atoms
  bool createAtomizableSections(llvm::error_code &EC) {
    // Handle: SHT_REL and SHT_RELA sections:
    // Increment over the sections, when REL/RELA section types are found add
    // the contents to the RelocationReferences map.
    llvm::object::section_iterator sit(_objFile->begin_sections());
    llvm::object::section_iterator sie(_objFile->end_sections());
    // Record the number of relocs to guess at preallocating the buffer.
    uint64_t totalRelocs = 0;
    for (; sit != sie; sit.increment(EC)) {
      if (EC)
        return true;

      const Elf_Shdr *section = _objFile->getElfSection(sit);

      if (isIgnoredSection(section))
        continue;

      if (isMergeableStringSection(section)) {
        _mergeStringSections.push_back(section);
        continue;
      }

      // Create a sectionSymbols entry for every progbits section.
      if (section->sh_type == llvm::ELF::SHT_PROGBITS)
        _sectionSymbols[section];

      if (section->sh_type == llvm::ELF::SHT_RELA) {
        StringRef sectionName;
        if ((EC = _objFile->getSectionName(section, sectionName)))
          return true;
        // Get rid of the leading .rela so Atoms can use their own section
        // name to find the relocs.
        sectionName = sectionName.drop_front(5);

        auto rai(_objFile->beginELFRela(section));
        auto rae(_objFile->endELFRela(section));

        _relocationAddendReferences[sectionName] = make_range(rai, rae);
        totalRelocs += std::distance(rai, rae);
      }

      if (section->sh_type == llvm::ELF::SHT_REL) {
        StringRef sectionName;
        if ((EC = _objFile->getSectionName(section, sectionName)))
          return true;
        // Get rid of the leading .rel so Atoms can use their own section
        // name to find the relocs.
        sectionName = sectionName.drop_front(4);

        auto ri(_objFile->beginELFRel(section));
        auto re(_objFile->endELFRel(section));

        _relocationReferences[sectionName] = make_range(ri, re);
        totalRelocs += std::distance(ri, re);
      }
    }
    _references.reserve(totalRelocs);
    return false;
  }

  /// \brief Create mergeable atoms from sections that have the merge attribute
  /// set
  bool createMergeableAtoms(llvm::error_code &EC) {
    // Divide the section that contains mergeable strings into tokens
    // TODO
    // a) add resolver support to recognize multibyte chars
    // b) Create a seperate section chunk to write mergeable atoms
    std::vector<MergeString *> tokens;
    for (const Elf_Shdr *msi : _mergeStringSections) {
      StringRef sectionContents;
      StringRef sectionName;
      if ((EC = _objFile->getSectionName(msi, sectionName)))
        return true;

      if ((EC = _objFile->getSectionContents(msi, sectionContents)))
        return true;

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
    for (const MergeString *tai : tokens) {
      ArrayRef<uint8_t> content((const uint8_t *)tai->_string.data(),
                                tai->_string.size());
      ELFMergeAtom<ELFT> *mergeAtom = new (_readerStorage) ELFMergeAtom<ELFT>(
          *this, tai->_sectionName, tai->_shdr, content, tai->_offset);
      const MergeSectionKey mergedSectionKey(tai->_shdr, tai->_offset);
      if (_mergedSectionMap.find(mergedSectionKey) == _mergedSectionMap.end())
        _mergedSectionMap.insert(std::make_pair(mergedSectionKey, mergeAtom));
      mergeAtom->setOrdinal(++_ordinal);
      _definedAtoms._atoms.push_back(mergeAtom);
      _mergeAtoms.push_back(mergeAtom);
    }
    return false;
  }

  /// \brief Add the symbols that the sections contain. The symbols will be
  /// converted to atoms for
  /// Undefined symbols, absolute symbols
  bool createSymbolsFromAtomizableSections(llvm::error_code &EC) {
    llvm::object::section_iterator sit(_objFile->begin_sections());

    // Increment over all the symbols collecting atoms and symbol names for
    // later use.
    llvm::object::symbol_iterator it(_objFile->begin_symbols());
    llvm::object::symbol_iterator ie(_objFile->end_symbols());

    // Skip ELF's first dummy symbol if we have one.
    if (it != ie)
      it.increment(EC);

    for (; it != ie; it.increment(EC)) {
      if (EC)
        return true;

      if ((EC = it->getSection(sit)))
        return true;

      const Elf_Shdr *section = _objFile->getElfSection(sit);
      const Elf_Sym *symbol = _objFile->getElfSymbol(it);

      StringRef symbolName;
      if ((EC = _objFile->getSymbolName(section, symbol, symbolName)))
        return true;

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
        if (symbol->getType() == llvm::ELF::STT_NOTYPE ||
            symbol->getType() == llvm::ELF::STT_OBJECT ||
            symbol->getType() == llvm::ELF::STT_FUNC ||
            symbol->getType() == llvm::ELF::STT_GNU_IFUNC ||
            symbol->getType() == llvm::ELF::STT_SECTION ||
            symbol->getType() == llvm::ELF::STT_FILE ||
            symbol->getType() == llvm::ELF::STT_TLS ||
            symbol->getType() == llvm::ELF::STT_COMMON ||
            symbol->st_shndx == llvm::ELF::SHN_COMMON) {
          _sectionSymbols[section].push_back(symbol);
        } else {
          llvm::errs() << "Unable to create atom for: " << symbolName << "\n";
          EC = llvm::object::object_error::parse_failed;
          return true;
        }
      }
    }
    return false;
  }

  /// \brief Create individual atoms
  bool createAtoms(llvm::error_code &EC) {
    for (auto &i : _sectionSymbols) {
      const Elf_Shdr *section = i.first;
      std::vector<const Elf_Sym *> &symbols = i.second;

      // Sort symbols by position.
      std::stable_sort(symbols.begin(), symbols.end(),
                       [](const Elf_Sym *A, const Elf_Sym *B) {
        return A->st_value < B->st_value;
      });

      StringRef sectionName;
      StringRef sectionContents;
      if ((EC = _objFile->getSectionName(section, sectionName)))
        return true;
      if ((EC = _objFile->getSectionContents(section, sectionContents)))
        return true;

      // If the section has no symbols, create a custom atom for it.
      if (section->sh_type == llvm::ELF::SHT_PROGBITS && symbols.empty() &&
          !sectionContents.empty()) {
        ELFDefinedAtom<ELFT> *newAtom = createSectionAtom(
            section, sectionName, sectionContents);
        _definedAtoms._atoms.push_back(newAtom);
        continue;
      }

      ELFDefinedAtom<ELFT> *previousAtom = nullptr;
      ELFReference<ELFT> *anonFollowedBy = nullptr;

      for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
        const Elf_Sym *symbol = *si;
        StringRef symbolName = "";
        if (symbol->getType() != llvm::ELF::STT_SECTION)
          if ((EC = _objFile->getSymbolName(section, symbol, symbolName)))
            return true;

        bool isCommon = isCommonSymbol(section, symbol);
        uint64_t contentSize = isCommon ? 0 : symbolContentSize(
            section, symbol, (si + 1 == se) ? nullptr : *(si + 1));

        // Check to see if we need to add the FollowOn Reference
        // We dont want to do for symbols that are
        // a) common symbols
        ELFReference<ELFT> *followOn = nullptr;
        if (!isCommon && previousAtom) {
          // Replace the followon atom with the anonymous atom that we created,
          // so that the next symbol that we create is a followon from the
          // anonymous atom.
          if (anonFollowedBy) {
            followOn = anonFollowedBy;
          } else {
            followOn = new (_readerStorage)
                ELFReference<ELFT>(lld::Reference::kindLayoutAfter);
            previousAtom->addReference(followOn);
          }
        }

        ArrayRef<uint8_t> symbolData(
            (uint8_t *)sectionContents.data() + symbol->st_value, contentSize);

        // If the linker finds that a section has global atoms that are in a
        // mergeable section, treat them as defined atoms as they shouldnt be
        // merged away as well as these symbols have to be part of symbol
        // resolution
        if (isMergeableStringSection(section)) {
          if (symbol->getBinding() == llvm::ELF::STB_GLOBAL) {
            auto definedMergeAtom = new (_readerStorage) ELFDefinedAtom<ELFT>(
                *this, symbolName, sectionName, (*si), section, symbolData,
                _references.size(), _references.size(), _references);
            _definedAtoms._atoms.push_back(definedMergeAtom);
          }
          continue;
        }

        // Don't allocate content to a weak symbol, as they may be merged away.
        // Create an anonymous atom to hold the data.
        ELFDefinedAtom<ELFT> *anonAtom = nullptr;
        anonFollowedBy = nullptr;
        if (symbol->getBinding() == llvm::ELF::STB_WEAK && contentSize != 0) {
          // Create anonymous new non-weak ELF symbol that holds the symbol
          // data.
          auto sym = new (_readerStorage) Elf_Sym(*symbol);
          sym->setBinding(llvm::ELF::STB_GLOBAL);
          anonAtom = createDefinedAtomAndAssignRelocations(
              "", sectionName, sym, section, symbolData);
          anonAtom->setOrdinal(++_ordinal);
          symbolData = ArrayRef<uint8_t>();

          if (previousAtom)
            createEdge(anonAtom, previousAtom,
                       lld::Reference::kindLayoutBefore);
        }

        ELFDefinedAtom<ELFT> *newAtom = createDefinedAtomAndAssignRelocations(
            symbolName, sectionName, symbol, section, symbolData);
        newAtom->setOrdinal(++_ordinal);

        // If this is the last atom, lets not create a followon reference.
        if (anonAtom && (si + 1) != se) {
          anonFollowedBy = new (_readerStorage)
              ELFReference<ELFT>(lld::Reference::kindLayoutAfter);
          anonAtom->addReference(anonFollowedBy);
        }

        // If the atom was a weak symbol, lets create a followon reference to
        // the anonymous atom that we created.
        if (anonAtom)
          createEdge(newAtom, anonAtom, lld::Reference::kindLayoutAfter);

        if (!isCommon && previousAtom) {
          // Set the followon atom to the weak atom that we have created, so
          // that they would alias when the file gets written.
          followOn->setTarget(anonAtom ? anonAtom : newAtom);

          // Add a preceded-by reference only if the current atom is not a weak
          // atom.
          if (symbol->getBinding() != llvm::ELF::STB_WEAK)
            createEdge(newAtom, previousAtom,
                       lld::Reference::kindLayoutBefore);
        }

        // The previous atom is always the atom created before unless the atom
        // is a weak atom.
        previousAtom = anonAtom ? anonAtom : newAtom;

        _definedAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair(symbol, newAtom));
        if (anonAtom)
          _definedAtoms._atoms.push_back(anonAtom);
      }
    }

    updateReferences();
    return false;
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
    auto rari = _relocationAddendReferences.find(sectionName);
    if (rari != _relocationAddendReferences.end()) {
      for (const Elf_Rela &rai : rari->second) {
        if (rai.r_offset < symbol->st_value ||
            symbol->st_value + content.size() <= rai.r_offset)
          continue;
        bool isMips64EL = _objFile->isMips64EL();
        Reference::Kind kind = (Reference::Kind) rai.getType(isMips64EL);
        uint32_t symbolIndex = rai.getSymbol(isMips64EL);
        auto *ERef = new (_readerStorage)
            ELFReference<ELFT>(&rai, rai.r_offset - symbol->st_value,
                               kind, symbolIndex);
        _references.push_back(ERef);
      }
    }

    // Add Rel references.
    auto rri = _relocationReferences.find(sectionName);
    if (rri != _relocationReferences.end()) {
      for (const Elf_Rel &ri : rri->second) {
        if (ri.r_offset < symbol->st_value ||
            symbol->st_value + content.size() <= ri.r_offset)
          continue;
        bool isMips64EL = _objFile->isMips64EL();
        Reference::Kind kind = (Reference::Kind) ri.getType(isMips64EL);
        uint32_t symbolIndex = ri.getSymbol(isMips64EL);
        auto *ERef = new (_readerStorage)
            ELFReference<ELFT>(&ri, ri.r_offset - symbol->st_value,
                               kind, symbolIndex);
        // Read the addend from the section contents
        // TODO : We should move the way lld reads relocations totally from
        // ELFObjectFile
        int32_t addend = *(content.data() + ri.r_offset - symbol->st_value);
        ERef->setAddend(addend);
        _references.push_back(ERef);
      }
    }

    // Create the DefinedAtom and add it to the list of DefinedAtoms.
    return new (_readerStorage) ELFDefinedAtom<ELFT>(
        *this, symbolName, sectionName, symbol, section, content,
        referenceStart, _references.size(), _references);
  }

  /// \brief After all the Atoms and References are created, update each
  /// Reference's target with the Atom pointer it refers to.
  void updateReferences() {
    /// cached value of target relocation handler
    const TargetRelocationHandler<ELFT> &_targetRelocationHandler =
        _elfTargetInfo.template getTargetHandler<ELFT>().getRelocationHandler();

    for (auto &ri : _references) {
      if (ri->kind() >= lld::Reference::kindTargetLow) {
        const Elf_Sym *symbol = _objFile->getElfSymbol(ri->targetSymbolIndex());
        const Elf_Shdr *shdr = _objFile->getSection(symbol);

        // If the atom is not in mergeable string section, the target atom is
        // simply that atom.
        if (!isMergeableStringSection(shdr)) {
          ri->setTarget(findAtom(symbol));
          continue;
        }

        // If the target atom is mergeable string atom, the atom might have been
        // merged with other atom having the same contents. Try to find the
        // merged one if that's the case.
        int64_t relocAddend = _targetRelocationHandler.relocAddend(*ri);
        uint64_t addend = ri->addend() + relocAddend;
        const MergeSectionKey ms(shdr, addend);
        auto msec = _mergedSectionMap.find(ms);
        if (msec != _mergedSectionMap.end()) {
          ri->setTarget(msec->second);
          continue;
        }

        // The target atom was not merged. Mergeable atoms are not in
        // _symbolToAtomMapping, so we cannot find it by calling findAtom(). We
        // instead call findMergeAtom().
        if (symbol->getType() != llvm::ELF::STT_SECTION)
          addend = symbol->st_value + addend;
        ELFMergeAtom<ELFT> *mergedAtom = findMergeAtom(shdr, addend);
        ri->setOffset(addend - mergedAtom->offset());
        ri->setAddend(0);
        ri->setTarget(mergedAtom);
      }
    }
  }

  /// \Brief Return true if the symbol is corresponding to an architecture
  /// specific section. We will let the TargetHandler to handle such atoms.
  inline bool isTargetSpecificAtom(const Elf_Shdr *shdr,
                                   const Elf_Sym *sym) {
    return ((shdr->sh_flags & llvm::ELF::SHF_MASKPROC) ||
            (sym->st_shndx >= llvm::ELF::SHN_LOPROC &&
             sym->st_shndx <= llvm::ELF::SHN_HIPROC));
  }

  /// \brief Do we want to ignore the section. Ignored sections are
  /// not processed to create atoms
  bool isIgnoredSection(const Elf_Shdr *section) {
    if (section->sh_size == 0)
      return true;
    switch (section->sh_type) {
    case llvm::ELF::SHT_NOTE:
    case llvm::ELF::SHT_STRTAB:
    case llvm::ELF::SHT_SYMTAB:
    case llvm::ELF::SHT_SYMTAB_SHNDX:
      return true;
    default:
      break;
    }
    return false;
  }

  /// \brief Is the current section be treated as a mergeable string section.
  /// The contents of a mergeable string section are null-terminated strings.
  /// If the section have mergeable strings, the linker would need to split
  /// the section into multiple atoms and mark them mergeByContent.
  bool isMergeableStringSection(const Elf_Shdr *section) {
    if (_doStringsMerge && section) {
      int64_t sectionFlags = section->sh_flags;
      sectionFlags &= ~llvm::ELF::SHF_ALLOC;
      // Mergeable string sections have both SHF_MERGE and SHF_STRINGS flags
      // set. sh_entsize is the size of each character which is normally 1.
      if ((section->sh_entsize < 2) &&
          (sectionFlags == (llvm::ELF::SHF_MERGE | llvm::ELF::SHF_STRINGS))) {
        return true;
      }
    }
    return false;
  }

  /// \brief Returns a new anonymous atom whose size is equal to the
  /// section size. That atom will be used to represent the entire
  /// section that have no symbols.
  ELFDefinedAtom<ELFT> *createSectionAtom(const Elf_Shdr *section,
                                          StringRef sectionName,
                                          StringRef sectionContents) {
    Elf_Sym *sym = new (_readerStorage) Elf_Sym;
    sym->st_name = 0;
    sym->setBindingAndType(llvm::ELF::STB_LOCAL, llvm::ELF::STT_SECTION);
    sym->st_other = 0;
    sym->st_shndx = 0;
    sym->st_value = 0;
    sym->st_size = 0;
    ArrayRef<uint8_t> content((const uint8_t *)sectionContents.data(),
                              sectionContents.size());
    auto *newAtom = new (_readerStorage) ELFDefinedAtom<ELFT>(
        *this, "", sectionName, sym, section, content, 0, 0, _references);
    newAtom->setOrdinal(++_ordinal);
    return newAtom;
  }

  /// Returns true if the symbol is common symbol. A common symbol represents a
  /// tentive definition in C. It has name, size and alignment constraint, but
  /// actual storage has not yet been allocated. (The linker will allocate
  /// storage for them in the later pass after coalescing tentative symbols by
  /// name.)
  bool isCommonSymbol(const Elf_Shdr *section, const Elf_Sym *symbol) {
    // This method handles only architecture independent stuffs, and don't know
    // whether an architecture dependent section is for common symbols or
    // not. Let the TargetHandler to make a decision if that's the case.
    if (isTargetSpecificAtom(section, symbol)) {
      TargetHandler<ELFT> &targetHandler =
          _elfTargetInfo.template getTargetHandler<ELFT>();
      TargetAtomHandler<ELFT> &targetAtomHandler =
          targetHandler.targetAtomHandler();
      return targetAtomHandler.getType(symbol) == llvm::ELF::STT_COMMON;
    }
    return symbol->getType() == llvm::ELF::STT_COMMON ||
        symbol->st_shndx == llvm::ELF::SHN_COMMON;
  }

  /// Returns the symbol's content size. The nextSymbol should be null if the
  /// symbol is the last one in the section.
  uint64_t symbolContentSize(const Elf_Shdr *section, const Elf_Sym *symbol,
                             const Elf_Sym *nextSymbol) {
    // if this is the last symbol, take up the remaining data.
    return nextSymbol
        ? nextSymbol->st_value - symbol->st_value
        : section->sh_size - symbol->st_value;
  }

  void createEdge(ELFDefinedAtom<ELFT> *from, ELFDefinedAtom<ELFT> *to,
                  lld::Reference::Kind kind) {
    auto reference = new (_readerStorage) ELFReference<ELFT>(kind);
    reference->setTarget(to);
    from->addReference(reference);
  }

  llvm::BumpPtrAllocator _readerStorage;
  std::unique_ptr<llvm::object::ELFObjectFile<ELFT> > _objFile;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;

  /// \brief _relocationAddendReferences and _relocationReferences contain the
  /// list of relocations references.  In ELF, if a section named, ".text" has
  /// relocations will also have a section named ".rel.text" or ".rela.text"
  /// which will hold the entries. -- .rel or .rela is prepended to create
  /// the SHT_REL(A) section name.
  std::unordered_map<
      StringRef,
      range<typename llvm::object::ELFObjectFile<ELFT>::Elf_Rela_Iter> >
  _relocationAddendReferences;
  MergedSectionMapT _mergedSectionMap;
  std::unordered_map<
      StringRef,
      range<typename llvm::object::ELFObjectFile<ELFT>::Elf_Rel_Iter> >
  _relocationReferences;
  std::vector<ELFReference<ELFT> *> _references;
  llvm::DenseMap<const Elf_Sym *, Atom *> _symbolToAtomMapping;
  const ELFTargetInfo &_elfTargetInfo;

  /// \brief Atoms that are created for a section that has the merge property
  /// set
  MergeAtomsT _mergeAtoms;

  /// \brief the section and the symbols that are contained within it to create
  /// used to create atoms
  std::map<const Elf_Shdr *, std::vector<const Elf_Sym *> > _sectionSymbols;

  /// \brief Sections that have merge string property
  std::vector<const Elf_Shdr *> _mergeStringSections;

  int64_t _ordinal;

  /// \brief the cached options relevant while reading the ELF File
  bool _doStringsMerge : 1;
};
} // end namespace elf
} // end namespace lld

#endif
