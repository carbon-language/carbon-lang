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
#include "lld/Core/File.h"

#include "llvm/ADT/ArrayRef.h"
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


#include <map>
#include <vector>

using llvm::object::Elf_Sym_Impl;
using namespace lld;

namespace { // anonymous

// This atom class corresponds to absolute symbol
class ELFAbsoluteAtom: public AbsoluteAtom {

public:
  ELFAbsoluteAtom(const File &F,
                  llvm::StringRef N,
                  uint64_t V)
    : OwningFile(F)
    , Name(N)
    , Value(V)
  {}

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual llvm::StringRef name() const {
    return Name;
  }

  virtual uint64_t value() const {
    return Value;
  }

private:
  const File &OwningFile;
  llvm::StringRef Name;
  uint64_t Value;
};


//  This atom corresponds to undefined symbols.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFUndefinedAtom: public UndefinedAtom {

  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

public:
  ELFUndefinedAtom(const File &F,
                   llvm::StringRef N,
                   const Elf_Sym *E)
    : OwningFile(F)
    , Name(N)
    , Symbol(E)
  {}

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual llvm::StringRef name() const {
    return Name;
  }

  //   FIXME What distinguishes a symbol in ELF that can help
  //   decide if the symbol is undefined only during build and not
  //   runtime? This will make us choose canBeNullAtBuildtime and
  //   canBeNullAtRuntime
  //
  virtual CanBeNull canBeNull() const {

    if (Symbol->getBinding() == llvm::ELF::STB_WEAK)
      return CanBeNull::canBeNullAtBuildtime;
    else
      return CanBeNull::canBeNullNever;
  }

private:
  const File &OwningFile;
  llvm::StringRef Name;
  const Elf_Sym *Symbol;
};


//  This atom corresponds to defined symbols.
template<llvm::support::endianness target_endianness, bool is64Bits>
class ELFDefinedAtom: public DefinedAtom {

  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;

public:
  ELFDefinedAtom(const File &F,
                 llvm::StringRef N,
                 llvm::StringRef SN,
                 const Elf_Sym *E,
                 const Elf_Shdr *S,
                 llvm::ArrayRef<uint8_t> D)
    : OwningFile(F)
    , SymbolName(N)
    , SectionName(SN)
    , Symbol(E)
    , Section(S)
    , ContentData(D) {
    static uint64_t ordernumber = 0;
    _ordinal = ++ordernumber;
  }

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual llvm::StringRef name() const {
    return SymbolName;
  }

  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  virtual uint64_t size() const {

    // Common symbols are not allocated in object files so
    // their size is zero.
    if ((Symbol->getType() == llvm::ELF::STT_COMMON)
        || Symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return (uint64_t)0;

    return ContentData.size();

  }

  virtual Scope scope() const {
    if (Symbol->st_other == llvm::ELF::STV_HIDDEN)
      return scopeLinkageUnit;
    else if (Symbol->getBinding() != llvm::ELF::STB_LOCAL)
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

    if (Symbol->getBinding() == llvm::ELF::STB_WEAK)
      return mergeAsWeak;

    if ((Symbol->getType() == llvm::ELF::STT_COMMON)
        || Symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return mergeAsTentative;

    return mergeNo;
  }

  virtual ContentType contentType() const {

    if (Symbol->getType() == llvm::ELF::STT_FUNC)
      return typeCode;

    if ((Symbol->getType() == llvm::ELF::STT_COMMON)
        || Symbol->st_shndx == llvm::ELF::SHN_COMMON)
      return typeZeroFill;

    if (Symbol->getType() == llvm::ELF::STT_OBJECT)
      return typeData;

    return typeUnknown;
  }

  virtual Alignment alignment() const {

    // Unallocated common symbols specify their alignment
    // constraints in st_value.
    if ((Symbol->getType() == llvm::ELF::STT_COMMON)
        || Symbol->st_shndx == llvm::ELF::SHN_COMMON) {
      return (Alignment(Symbol->st_value));
    }

    return Alignment(1);
  }

  // Do we have a choice for ELF?  All symbols
  // live in explicit sections.
  virtual SectionChoice sectionChoice() const {
    if (Symbol->st_shndx > llvm::ELF::SHN_LORESERVE)
      return sectionBasedOnContent;

    return sectionCustomRequired;
  }

  virtual llvm::StringRef customSectionName() const {
    return SectionName;
  }

  // It isn't clear that __attribute__((used)) is transmitted to
  // the ELF object file.
  virtual DeadStripKind deadStrip() const {
    return deadStripNormal;
  }

  virtual ContentPermissions permissions() const {

    switch (Section->sh_type) {
    // permRW_L is for sections modified by the runtime
    // loader.
    case llvm::ELF::SHT_REL:
    case llvm::ELF::SHT_RELA:
      return permRW_L;

    case llvm::ELF::SHT_DYNAMIC:
    case llvm::ELF::SHT_PROGBITS:
      switch (Section->sh_flags) {

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
    return ContentData;
  }

  virtual reference_iterator begin() const {
    return reference_iterator(*this, nullptr);
  }

  virtual reference_iterator end() const {
    return reference_iterator(*this, nullptr);
  }

private:
  virtual const Reference *derefIterator(const void *iter) const {
    return nullptr;
  }
  virtual void incrementIterator(const void *&iter) const {
  }

  const File &OwningFile;
  llvm::StringRef SymbolName;
  llvm::StringRef SectionName;
  const Elf_Sym *Symbol;
  const Elf_Shdr *Section;

  // ContentData will hold the bits that make up the atom.
  llvm::ArrayRef<uint8_t> ContentData;

  uint64_t _ordinal;
};


//   FileELF will read a binary, find out based on the symbol table contents
//   what kind of symbol it is and create corresponding atoms for it

template<llvm::support::endianness target_endianness, bool is64Bits>
class FileELF: public File {

  typedef llvm::object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;

public:
  FileELF(std::unique_ptr<llvm::MemoryBuffer> MB, llvm::error_code &EC) :
          File(MB->getBufferIdentifier()) {

    llvm::OwningPtr<llvm::object::Binary> Bin;
    EC = llvm::object::createBinary(MB.release(), Bin);
    if (EC)
      return;

    // Point Obj to correct class and bitwidth ELF object
    Obj.reset(llvm::dyn_cast<llvm::object::ELFObjectFile<target_endianness,
        is64Bits> >(Bin.get()));

    if (!Obj) {
      EC = make_error_code(llvm::object::object_error::invalid_file_type);
      return;
    }

    Bin.take();

    std::map< const Elf_Shdr *, std::vector<const Elf_Sym *>> SectionSymbols;

    llvm::object::symbol_iterator it(Obj->begin_symbols());
    llvm::object::symbol_iterator ie(Obj->end_symbols());

    for (; it != ie; it.increment(EC)) {
      if (EC)
        return;
      llvm::object::SectionRef SR;
      llvm::object::section_iterator section(SR);

      if ((EC = it->getSection(section)))
        return;

      const Elf_Shdr *Section = Obj->getElfSection(section);
      const Elf_Sym  *Symbol  = Obj->getElfSymbol(it);

      llvm::StringRef SymbolName;
      if ((EC = Obj->getSymbolName(Section, Symbol, SymbolName)))
        return;

      if (Symbol->st_shndx == llvm::ELF::SHN_ABS) {
        // Create an absolute atom.
        AbsoluteAtoms._atoms.push_back(
                             new (AtomStorage.Allocate<ELFAbsoluteAtom> ())
                             ELFAbsoluteAtom(*this, SymbolName,
                                             Symbol->st_value));

      } else if (Symbol->st_shndx == llvm::ELF::SHN_UNDEF) {
        // Create an undefined atom.
        UndefinedAtoms._atoms.push_back(
            new (AtomStorage.Allocate<ELFUndefinedAtom<
                 target_endianness, is64Bits>>())
                 ELFUndefinedAtom<target_endianness, is64Bits> (
                                 *this, SymbolName, Symbol));
      } else {
        // This is actually a defined symbol. Add it to its section's list of
        // symbols.
        if (Symbol->getType() == llvm::ELF::STT_NOTYPE
            || Symbol->getType() == llvm::ELF::STT_OBJECT
            || Symbol->getType() == llvm::ELF::STT_FUNC
            || Symbol->getType() == llvm::ELF::STT_SECTION
            || Symbol->getType() == llvm::ELF::STT_FILE
            || Symbol->getType() == llvm::ELF::STT_TLS
            || Symbol->getType() == llvm::ELF::STT_COMMON
            || Symbol->st_shndx == llvm::ELF::SHN_COMMON) {
          SectionSymbols[Section].push_back(Symbol);
        }
        else {
          llvm::errs() << "Unable to create atom for: " << SymbolName << "\n";
          EC = llvm::object::object_error::parse_failed;
          return;
        }
      }
    }

    for (auto &i : SectionSymbols) {
      auto &Symbs = i.second;
      llvm::StringRef SymbolName;
      llvm::StringRef SectionName;
      // Sort symbols by position.
      std::stable_sort(Symbs.begin(), Symbs.end(),
        // From ReaderCOFF.cpp:
        // For some reason MSVC fails to allow the lambda in this context with
        // a "illegal use of local type in type instantiation". MSVC is clearly
        // wrong here. Force a conversion to function pointer to work around.
        static_cast<bool(*)(const Elf_Sym*, const Elf_Sym*)>(
          [](const Elf_Sym *A, const Elf_Sym *B) -> bool {
        return A->st_value < B->st_value;
      }));

      // i.first is the section the symbol lives in
      for (auto si = Symbs.begin(), se = Symbs.end(); si != se; ++si) {

        StringRef symbolContents;
        if ((EC = Obj->getSectionContents(i.first, symbolContents)))
          return;

        if ((EC = Obj->getSymbolName(i.first, *si, SymbolName)))
          return;

        if ((EC = Obj->getSectionName(i.first, SectionName)))
          return;

        bool IsCommon = false;
        if (((*si)->getType() == llvm::ELF::STT_COMMON)
             || (*si)->st_shndx == llvm::ELF::SHN_COMMON)
          IsCommon = true;

        // Get the symbol's content:
        llvm::ArrayRef<uint8_t> SymbolData;
        if (si + 1 == se) {
          // if this is the last symbol, take up the remaining data.
          SymbolData = llvm::ArrayRef<uint8_t>((uint8_t *)symbolContents.data()
                                    + (*si)->st_value,
                                    (IsCommon) ? 0 :
                                    ((i.first)->sh_size - (*si)->st_value));
        }
        else {
          SymbolData = llvm::ArrayRef<uint8_t>((uint8_t *)symbolContents.data()
                                    + (*si)->st_value,
                                    (IsCommon) ? 0 :
                                    (*(si + 1))->st_value - (*si)->st_value);
        }

        DefinedAtoms._atoms.push_back(
          new (AtomStorage.Allocate<ELFDefinedAtom<
               target_endianness, is64Bits> > ())
               ELFDefinedAtom<target_endianness, is64Bits> (*this,
                             SymbolName, SectionName,
                             *si, i.first, SymbolData));
      }
    }
  }

  virtual void addAtom(const Atom&) {
    llvm_unreachable("cannot add atoms to native .o files");
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return DefinedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return UndefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return SharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return AbsoluteAtoms;
  }

private:
  std::unique_ptr<llvm::object::ELFObjectFile<target_endianness, is64Bits> >
      Obj;
  atom_collection_vector<DefinedAtom>       DefinedAtoms;
  atom_collection_vector<UndefinedAtom>     UndefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> SharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      AbsoluteAtoms;
  llvm::BumpPtrAllocator AtomStorage;

};

//  ReaderELF is reader object that will instantiate correct FileELF
//  by examining the memory buffer for ELF class and bitwidth

class ReaderELF: public Reader {
public:
  ReaderELF(const ReaderOptionsELF &options) :
    _options(options) {
  }
  error_code parseFile(std::unique_ptr<MemoryBuffer> mb, std::vector<
      std::unique_ptr<File> > &result) {

    std::pair<unsigned char, unsigned char> Ident =
        llvm::object::getElfArchType(&*mb);
    llvm::error_code ec;
    //    Instantiate the correct FileELF template instance
    //    based on the Ident pair. Once the File is created
    //     we push the file to the vector of files already
    //     created during parser's life.

    std::unique_ptr<File> f;

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

    if (ec)
      return ec;

    result.push_back(std::move(f));
    return error_code::success();
  }
private:
  const ReaderOptionsELF &_options;
};

} // namespace anonymous

namespace lld {

ReaderOptionsELF::ReaderOptionsELF() {
}

ReaderOptionsELF::~ReaderOptionsELF() {
}


Reader *createReaderELF(const ReaderOptionsELF &options) {
  return new ReaderELF(options);
}

} // namespace LLD
