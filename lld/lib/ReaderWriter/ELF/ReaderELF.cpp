//===- lib/ReaderWriter/ELF/ReaderELF.cpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ELF Reader and all helper sub classes
// to consume an ELF file and produces atoms out of it.
//
//===----------------------------------------------------------------------===//
#include "lld/ReaderWriter/ReaderELF.h"
#include "lld/ReaderWriter/ReaderArchive.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/Path.h"


#include <map>
#include <vector>

using llvm::object::Elf_Sym_Impl;
using namespace lld;

namespace { // anonymous


/// \brief Relocation References: Defined Atoms may contain 
/// references that will need to be patched before
/// the executable is written.
template <llvm::support::endianness target_endianness, bool is64Bits>
class ELFReference final : public Reference {

  typedef llvm::object::Elf_Rel_Impl
                        <target_endianness, is64Bits, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl
                        <target_endianness, is64Bits, true> Elf_Rela;
public:

  ELFReference(const Elf_Rela *rela, uint64_t offset, const Atom *target)
    : _target(target)
    , _targetSymbolIndex(rela->getSymbol())
    , _offsetInAtom(offset)
    , _addend(rela->r_addend)
    , _kind(rela->getType()) {}

  ELFReference(const Elf_Rel *rel, uint64_t offset, const Atom *target)
    : _target(target)
    , _targetSymbolIndex(rel->getSymbol())
    , _offsetInAtom(offset)
    , _addend(0)
    , _kind(rel->getType()) {}


  virtual uint64_t offsetInAtom() const {
    return _offsetInAtom;
  }

  virtual Kind kind() const {
    return _kind;
  }

  virtual void setKind(Kind kind) {
    _kind = kind;
  }

  virtual const Atom *target() const {
    return _target;
  }

/// \brief targetSymbolIndex: This is the symbol table index that contains
/// the target reference.
  uint64_t targetSymbolIndex() const {
    return _targetSymbolIndex;
  }

  virtual Addend addend() const {
    return _addend;
  }

  virtual void setAddend(Addend A) {
    _addend = A;
  }

  virtual void setTarget(const Atom *newAtom) {
    _target = newAtom;
  }
private:
  const Atom  *_target;
  uint64_t     _targetSymbolIndex;
  uint64_t     _offsetInAtom;
  Addend       _addend;
  Kind         _kind;
};


/// \brief ELFAbsoluteAtom: These atoms store symbols that are fixed to a
/// particular address.  This atom has no content its address will be used by
/// the writer to fixup references that point to it.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFAbsoluteAtom final : public AbsoluteAtom {

  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

public:
  ELFAbsoluteAtom(const File &file,
                  llvm::StringRef name,
                  const Elf_Sym *symbol,
                  uint64_t value)
    : _owningFile(file)
    , _name(name)
    , _symbol(symbol)
    , _value(value)
  {}

  virtual const class File &file() const {
    return _owningFile;
  }

  virtual Scope scope() const {
    if (_symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    if (_symbol->getBinding() == llvm::ELF::STB_LOCAL)
      return scopeTranslationUnit;
    else
      return scopeGlobal;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

  virtual uint64_t value() const {
    return _value;
  }

private:
  const File &_owningFile;
  llvm::StringRef _name;
  const Elf_Sym *_symbol;
  uint64_t _value;
};


/// \brief ELFUndefinedAtom: These atoms store undefined symbols and are
/// place holders that will be replaced by defined atoms later in the
/// linking process.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFUndefinedAtom final: public UndefinedAtom {

  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

public:
  ELFUndefinedAtom(const File &file,
                   llvm::StringRef name,
                   const Elf_Sym *symbol)
    : _owningFile(file)
    , _name(name)
    , _symbol(symbol)
  {}

  virtual const class File &file() const {
    return _owningFile;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

  //   FIXME What distinguishes a symbol in ELF that can help
  //   decide if the symbol is undefined only during build and not
  //   runtime? This will make us choose canBeNullAtBuildtime and
  //   canBeNullAtRuntime
  //
  virtual CanBeNull canBeNull() const {

    if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
      return CanBeNull::canBeNullAtBuildtime;
    else
      return CanBeNull::canBeNullNever;
  }

private:
  const File &_owningFile;
  llvm::StringRef _name;
  const Elf_Sym *_symbol;
};


/// \brief ELFDefinedAtom: This atom stores defined symbols and will contain
/// either data or code.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFDefinedAtom final: public DefinedAtom {

  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;

public:
  ELFDefinedAtom(const File &file,
                 llvm::StringRef symbolName,
                 llvm::StringRef sectionName,
                 const Elf_Sym *symbol,
                 const Elf_Shdr *section,
                 llvm::ArrayRef<uint8_t> contentData,
                 unsigned int referenceStart,
                 unsigned int referenceEnd,
                 std::vector<ELFReference
                             <target_endianness, is64Bits> *> &referenceList)

    : _owningFile(file)
    , _symbolName(symbolName)
    , _sectionName(sectionName)
    , _symbol(symbol)
    , _section(section)
    , _contentData(contentData) 
    , _referenceStartIndex(referenceStart)
    , _referenceEndIndex(referenceEnd)
    , _referenceList(referenceList) {
    static uint64_t orderNumber = 0;
    _ordinal = ++orderNumber;
  }

  virtual const class File &file() const {
    return _owningFile;
  }

  virtual llvm::StringRef name() const {
    return _symbolName;
  }

  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  virtual uint64_t size() const {

    // Common symbols are not allocated in object files,
    // so use st_size to tell how many bytes are required.
    if ((_symbol->getType() == llvm::ELF::STT_COMMON)
        || _symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return (uint64_t)_symbol->st_size;

    return _contentData.size();

  }

  virtual Scope scope() const {
    if (_symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    else if (_symbol->getBinding() != llvm::ELF::STB_LOCAL)
      return scopeGlobal;
    else
      return scopeTranslationUnit;
  }

  //   FIXME   Need to revisit this in future.

  virtual Interposable interposable() const {
    return interposeNo;
  }

  //  FIXME What ways can we determine this in ELF?

  virtual Merge merge() const {

    if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
      return mergeAsWeak;

    if ((_symbol->getType() == llvm::ELF::STT_COMMON)
        || _symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return mergeAsTentative;

    return mergeNo;
  }

  virtual ContentType contentType() const {

    ContentType ret = typeUnknown;

    switch (_section->sh_type) {
    case llvm::ELF::SHT_PROGBITS:
    case llvm::ELF::SHT_DYNAMIC:
      switch (_section->sh_flags) {
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR
            | llvm::ELF::SHF_WRITE):
        ret = typeCode;
        break;
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR):
        ret = typeCode;
        break;
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE):
        ret = typeData;
        break;
      case llvm::ELF::SHF_ALLOC:
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_MERGE):
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_MERGE |
            llvm::ELF::SHF_STRINGS):
        ret = typeConstant;
        break;
      }
      break;
    case llvm::ELF::SHT_NOBITS:
      ret = typeZeroFill;
      break;
    case llvm::ELF::SHT_NULL:
      if ((_symbol->getType() == llvm::ELF::STT_COMMON)
          || _symbol->st_shndx == llvm::ELF::SHN_COMMON)
        ret = typeZeroFill;
      break;
    }

    return ret;
  }

  virtual Alignment alignment() const {

    // Unallocated common symbols specify their alignment
    // constraints in st_value.
    if ((_symbol->getType() == llvm::ELF::STT_COMMON)
        || _symbol->st_shndx == llvm::ELF::SHN_COMMON) {
      return Alignment(llvm::Log2_64(_symbol->st_value));
    }
    return Alignment(llvm::Log2_64(_section->sh_addralign));
  }

  // Do we have a choice for ELF?  All symbols
  // live in explicit sections.
  virtual SectionChoice sectionChoice() const {
    if (_symbol->st_shndx > llvm::ELF::SHN_LORESERVE)
      return sectionBasedOnContent;

    return sectionCustomRequired;
  }

  virtual llvm::StringRef customSectionName() const {
    return _sectionName;
  }

  // It isn't clear that __attribute__((used)) is transmitted to
  // the ELF object file.
  virtual DeadStripKind deadStrip() const {
    return deadStripNormal;
  }

  virtual ContentPermissions permissions() const {

    switch (_section->sh_type) {
    // permRW_L is for sections modified by the runtime
    // loader.
    case llvm::ELF::SHT_REL:
    case llvm::ELF::SHT_RELA:
      return permRW_L;

    case llvm::ELF::SHT_DYNAMIC:
    case llvm::ELF::SHT_PROGBITS:
      switch (_section->sh_flags) {

      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR):
        return permR_X;

      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE):
        return permRW_;

      case llvm::ELF::SHF_ALLOC:
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_MERGE):
      case (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_MERGE
            | llvm::ELF::SHF_STRINGS):
        return permR__;
      }
      default:
        return perm___;
    }
  }

  //   Many non ARM architectures use ELF file format
  //   This not really a place to put a architecture
  //   specific method in an atom. A better approach is
  //   needed.
  //
  virtual bool isThumb() const {
    return false;
  }

  //  FIXME Not Sure if ELF supports alias atoms. Find out more.
  virtual bool isAlias() const {
    return false;
  }

  virtual llvm::ArrayRef<uint8_t> rawContent() const {
    return _contentData;
  }

  DefinedAtom::reference_iterator begin() const {
    uintptr_t index = _referenceStartIndex;
    const void *it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  DefinedAtom::reference_iterator end() const {
    uintptr_t index = _referenceEndIndex;
    const void *it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  const Reference *derefIterator(const void *It) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(It);
    assert(index >= _referenceStartIndex);
    assert(index < _referenceEndIndex);
    return ((_referenceList)[index]);
  }

  void incrementIterator(const void*& It) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(It);
    ++index;
    It = reinterpret_cast<const void*>(index);
  }

private:

  const File &_owningFile;
  llvm::StringRef _symbolName;
  llvm::StringRef _sectionName;
  const Elf_Sym *_symbol;
  const Elf_Shdr *_section;
  // _contentData will hold the bits that make up the atom.
  llvm::ArrayRef<uint8_t> _contentData;

  uint64_t _ordinal;
  unsigned int _referenceStartIndex;
  unsigned int _referenceEndIndex;
  std::vector<ELFReference<target_endianness, is64Bits> *> &_referenceList;
};


//   FileELF will read a binary, find out based on the symbol table contents
//   what kind of symbol it is and create corresponding atoms for it

template<llvm::support::endianness target_endianness, bool is64Bits>
class FileELF: public File {

  typedef llvm::object::Elf_Sym_Impl
                        <target_endianness, is64Bits> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl
                        <target_endianness, is64Bits> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl
                        <target_endianness, is64Bits, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl
                        <target_endianness, is64Bits, true> Elf_Rela;

public:
  FileELF(std::unique_ptr<llvm::MemoryBuffer> MB, llvm::error_code &EC) :
          File(MB->getBufferIdentifier()) {

    llvm::OwningPtr<llvm::object::Binary> binaryFile;
    EC = llvm::object::createBinary(MB.release(), binaryFile);
    if (EC)
      return;

    // Point Obj to correct class and bitwidth ELF object
    _objFile.reset(llvm::dyn_cast<llvm::object::ELFObjectFile<target_endianness,
        is64Bits> >(binaryFile.get()));

    if (!_objFile) {
      EC = make_error_code(llvm::object::object_error::invalid_file_type);
      return;
    }

    binaryFile.take();

    std::map< const Elf_Shdr *, std::vector<const Elf_Sym *>> sectionSymbols;

//  Handle: SHT_REL and SHT_RELA sections:
//  Increment over the sections, when REL/RELA section types are
//  found add the contents to the RelocationReferences map.

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
        for (; rai != rae; rai++) {
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
        for (; ri != re; ri++) {
          Ref.push_back(&*ri);
        }
      }
    }


//  Increment over all the symbols collecting atoms and symbol
//  names for later use.

    llvm::object::symbol_iterator it(_objFile->begin_symbols());
    llvm::object::symbol_iterator ie(_objFile->end_symbols());

    for (; it != ie; it.increment(EC)) {
      if (EC)
        return;

      if ((EC = it->getSection(sit)))
        return;

      const Elf_Shdr *section = _objFile->getElfSection(sit);
      const Elf_Sym  *symbol  = _objFile->getElfSymbol(it);

      llvm::StringRef symbolName;
      if ((EC = _objFile->getSymbolName(section, symbol, symbolName)))
        return;

      if (symbol->st_shndx == llvm::ELF::SHN_ABS) {
        // Create an absolute atom.
        auto *newAtom = new (_readerStorage.Allocate
                       <ELFAbsoluteAtom<target_endianness, is64Bits> > ()) 
                        ELFAbsoluteAtom<target_endianness, is64Bits>
                          (*this, symbolName, symbol, symbol->st_value);

        _absoluteAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair(symbol, newAtom));

      } else if (symbol->st_shndx == llvm::ELF::SHN_UNDEF) {
        // Create an undefined atom.
        auto *newAtom = new (_readerStorage.Allocate
                       <ELFUndefinedAtom<target_endianness, is64Bits> > ()) 
                        ELFUndefinedAtom<target_endianness, is64Bits>
                          (*this, symbolName, symbol);

        _undefinedAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair(symbol, newAtom));

      } else {
        // This is actually a defined symbol. Add it to its section's list of
        // symbols.
        if (symbol->getType() == llvm::ELF::STT_NOTYPE
            || symbol->getType() == llvm::ELF::STT_OBJECT
            || symbol->getType() == llvm::ELF::STT_FUNC
            || symbol->getType() == llvm::ELF::STT_SECTION
            || symbol->getType() == llvm::ELF::STT_FILE
            || symbol->getType() == llvm::ELF::STT_TLS
            || symbol->getType() == llvm::ELF::STT_COMMON
            || symbol->st_shndx == llvm::ELF::SHN_COMMON) {
          sectionSymbols[section].push_back(symbol);
        }
        else {
          llvm::errs() << "Unable to create atom for: " << symbolName << "\n";
          EC = llvm::object::object_error::parse_failed;
          return;
        }
      }
    }

    for (auto &i : sectionSymbols) {
      auto &symbols = i.second;
      llvm::StringRef symbolName;
      llvm::StringRef sectionName;
      // Sort symbols by position.
      std::stable_sort(symbols.begin(), symbols.end(),
        // From ReaderCOFF.cpp:
        // For some reason MSVC fails to allow the lambda in this context with
        // a "illegal use of local type in type instantiation". MSVC is clearly
        // wrong here. Force a conversion to function pointer to work around.
        static_cast<bool(*)(const Elf_Sym*, const Elf_Sym*)>(
          [](const Elf_Sym *A, const Elf_Sym *B) -> bool {
        return A->st_value < B->st_value;
      }));

      // i.first is the section the symbol lives in
      for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {

        StringRef symbolContents;
        if ((EC = _objFile->getSectionContents(i.first, symbolContents)))
          return;

        if ((EC = _objFile->getSymbolName(i.first, *si, symbolName)))
          return;

        if ((EC = _objFile->getSectionName(i.first, sectionName)))
          return;

        bool isCommon = false;
        if (((*si)->getType() == llvm::ELF::STT_COMMON)
          || (*si)->st_shndx == llvm::ELF::SHN_COMMON)
          isCommon = true;

        // Get the symbol's content:
        llvm::ArrayRef<uint8_t> symbolData;
        uint64_t contentSize;
        if (si + 1 == se) {
          // if this is the last symbol, take up the remaining data.
          contentSize = (isCommon) ? 0 
                                   : ((i.first)->sh_size - (*si)->st_value);
        }
        else {
          contentSize = (isCommon) ? 0 
                                   : (*(si + 1))->st_value - (*si)->st_value;
        }

        symbolData = llvm::ArrayRef<uint8_t>((uint8_t *)symbolContents.data()
                                    + (*si)->st_value, contentSize);


        unsigned int referenceStart = _references.size();

        // Only relocations that are inside the domain of the atom are 
        // added.

        // Add Rela (those with r_addend) references:
        for (auto &rai : _relocationAddendRefences[sectionName]) {
          if ((rai->r_offset >= (*si)->st_value) &&
              (rai->r_offset < (*si)->st_value+contentSize)) {

            auto *ERef = new (_readerStorage.Allocate
                         <ELFReference<target_endianness, is64Bits> > ())
                          ELFReference<target_endianness, is64Bits> (
                          rai, rai->r_offset-(*si)->st_value, nullptr);

            _references.push_back(ERef);
          }
        }

        // Add Rel references:
        for (auto &ri : _relocationReferences[sectionName]) {
          if (((ri)->r_offset >= (*si)->st_value) &&
              ((ri)->r_offset < (*si)->st_value+contentSize)) {

            auto *ERef = new (_readerStorage.Allocate
                         <ELFReference<target_endianness, is64Bits> > ())
                          ELFReference<target_endianness, is64Bits> (
                         (ri), (ri)->r_offset-(*si)->st_value, nullptr);

            _references.push_back(ERef);
          }
        }

        // Create the DefinedAtom and add it to the list of DefinedAtoms.
        auto *newAtom = new (_readerStorage.Allocate
                       <ELFDefinedAtom<target_endianness, is64Bits> > ()) 
                        ELFDefinedAtom<target_endianness, is64Bits>
                           (*this, symbolName, sectionName,
                             *si, i.first, symbolData,
                             referenceStart, _references.size(), _references);

        _definedAtoms._atoms.push_back(newAtom);
        _symbolToAtomMapping.insert(std::make_pair((*si), newAtom));

      }
    }

// All the Atoms and References are created.  Now update each Reference's
// target with the Atom pointer it refers to.
    for (auto &ri : _references) {
      const Elf_Sym  *Symbol  = _objFile->getElfSymbol(ri->targetSymbolIndex());
      ri->setTarget(findAtom (Symbol));
    }
  }

  virtual void addAtom(const Atom&) {
    llvm_unreachable("cannot add atoms to native .o files");
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

  Atom *findAtom(const Elf_Sym  *symbol) {
    return (_symbolToAtomMapping.lookup(symbol));
  }


private:
  std::unique_ptr<llvm::object::ELFObjectFile<target_endianness, is64Bits> >
      _objFile;
  atom_collection_vector<DefinedAtom>       _definedAtoms;
  atom_collection_vector<UndefinedAtom>     _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      _absoluteAtoms;

/// \brief _relocationAddendRefences and _relocationReferences contain the list
/// of relocations references.  In ELF, if a section named, ".text" has
/// relocations will also have a section named ".rel.text" or ".rela.text"
/// which will hold the entries. -- .rel or .rela is prepended to create
/// the SHT_REL(A) section name.
///
  std::map<llvm::StringRef, std::vector<const Elf_Rela *> >
           _relocationAddendRefences;
  std::map<llvm::StringRef, std::vector<const Elf_Rel *> >
           _relocationReferences;

  std::vector<ELFReference<target_endianness, is64Bits> *> _references;
  llvm::DenseMap<const Elf_Sym *, Atom *> _symbolToAtomMapping;

  llvm::BumpPtrAllocator _readerStorage;
};

//  ReaderELF is reader object that will instantiate correct FileELF
//  by examining the memory buffer for ELF class and bitwidth

class ReaderELF: public Reader {
public:
  ReaderELF(const ReaderOptionsELF &,
            ReaderOptionsArchive &readerOptionsArchive)
    : _readerOptionsArchive(readerOptionsArchive)
    , _readerArchive(_readerOptionsArchive) { 
    _readerOptionsArchive.setReader(this);
  }

  error_code parseFile(std::unique_ptr<MemoryBuffer> mb, std::vector<
                       std::unique_ptr<File> > &result) {
    llvm::error_code ec; 
    std::unique_ptr<File> f;
    std::pair<unsigned char, unsigned char> Ident;

    llvm::sys::LLVMFileType fileType = 
          llvm::sys::IdentifyFileType(mb->getBufferStart(),
                                static_cast<unsigned>(mb->getBufferSize()));
    switch (fileType) {
    case llvm::sys::ELF_Relocatable_FileType:
      Ident = llvm::object::getElfArchType(&*mb);
      //    Instantiate the correct FileELF template instance
      //    based on the Ident pair. Once the File is created
      //     we push the file to the vector of files already
      //     created during parser's life.

      if (Ident.first == llvm::ELF::ELFCLASS32 && Ident.second
          == llvm::ELF::ELFDATA2LSB) {
        f.reset(new FileELF<llvm::support::little, false>(std::move(mb), ec));

      } else if (Ident.first == llvm::ELF::ELFCLASS32 && Ident.second
          == llvm::ELF::ELFDATA2MSB) {
        f.reset(new FileELF<llvm::support::big, false> (std::move(mb), ec));

      } else if (Ident.first == llvm::ELF::ELFCLASS64 && Ident.second
          == llvm::ELF::ELFDATA2MSB) {
        f.reset(new FileELF<llvm::support::big, true> (std::move(mb), ec));

      } else if (Ident.first == llvm::ELF::ELFCLASS64 && Ident.second
          == llvm::ELF::ELFDATA2LSB) {
        f.reset(new FileELF<llvm::support::little, true> (std::move(mb), ec));
      }
      if (!ec)
        result.push_back(std::move(f));
      break;

    case llvm::sys::Archive_FileType:
      ec = _readerArchive.parseFile(std::move(mb), result);
      break;

    default:
      llvm_unreachable("not supported format");
      break;
    }

    if (ec)
      return ec;

    return error_code::success();
  }

private:
  ReaderOptionsArchive &_readerOptionsArchive;
  ReaderArchive _readerArchive;
};

} // namespace anonymous

namespace lld {

ReaderOptionsELF::ReaderOptionsELF() {
}

ReaderOptionsELF::~ReaderOptionsELF() {
}

Reader *createReaderELF(const ReaderOptionsELF &options,
                        ReaderOptionsArchive &optionsArchive) {
  return new ReaderELF(options, optionsArchive);
}

} // namespace LLD
