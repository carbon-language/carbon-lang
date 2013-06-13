//===- lib/ReaderWriter/PECOFF/ReaderCOFF.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ReaderCOFF"

#include "lld/ReaderWriter/Reader.h"
#include "lld/Core/File.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <map>
#include <vector>

using llvm::object::coff_symbol;
using namespace lld;

namespace { // anonymous

class COFFAbsoluteAtom : public AbsoluteAtom {
public:
  COFFAbsoluteAtom(const File &F, llvm::StringRef N, uint64_t V)
    : OwningFile(F)
    , Name(N)
    , Value(V)
  {}

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual Scope scope() const {
    return scopeGlobal;
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

class COFFUndefinedAtom : public UndefinedAtom {
public:
  COFFUndefinedAtom(const File &F, llvm::StringRef N)
    : OwningFile(F)
    , Name(N)
  {}

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual llvm::StringRef name() const {
    return Name;
  }

  virtual CanBeNull canBeNull() const {
    return CanBeNull::canBeNullNever;
  }

private:
  const File &OwningFile;
  llvm::StringRef Name;
};

class COFFDefinedAtom : public DefinedAtom {
public:
  COFFDefinedAtom( const File &F
                 , llvm::StringRef N
                 , const llvm::object::coff_symbol *Symb
                 , const llvm::object::coff_section *Sec
                 , llvm::ArrayRef<uint8_t> D)
    : OwningFile(F)
    , Name(N)
    , Symbol(Symb)
    , Section(Sec)
    , Data(D)
  {}

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual llvm::StringRef name() const {
    return Name;
  }

  virtual uint64_t ordinal() const {
    return reinterpret_cast<intptr_t>(Symbol);
  }

  virtual uint64_t size() const {
    return Data.size();
  }

  virtual Scope scope() const {
    if (!Symbol)
      return scopeTranslationUnit;
    switch (Symbol->StorageClass) {
    case llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL:
      return scopeGlobal;
    case llvm::COFF::IMAGE_SYM_CLASS_STATIC:
      return scopeTranslationUnit;
    }
    llvm_unreachable("Unknown scope!");
  }

  virtual Interposable interposable() const {
    return interposeNo;
  }

  virtual Merge merge() const {
    return mergeNo;
  }

  virtual ContentType contentType() const {
    if (Section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_CODE)
      return typeCode;
    if (Section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      return typeData;
    if (Section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      return typeZeroFill;
    return typeUnknown;
  }

  virtual Alignment alignment() const {
    return Alignment(1);
  }

  virtual SectionChoice sectionChoice() const {
    return sectionBasedOnContent;
  }

  virtual llvm::StringRef customSectionName() const {
    return "";
  }

  virtual SectionPosition sectionPosition() const {
    return sectionPositionAny;
  }

  virtual DeadStripKind deadStrip() const {
    return deadStripNormal;
  }

  virtual ContentPermissions permissions() const {
    if (   Section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_READ
        && Section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_WRITE)
      return permRW_;
    if (   Section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_READ
        && Section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_EXECUTE)
      return permR_X;
    if (Section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_READ)
      return permR__;
    return perm___;
  }

  virtual bool isAlias() const {
    return false;
  }

  virtual llvm::ArrayRef<uint8_t> rawContent() const {
    return Data;
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
  llvm::StringRef Name;
  const llvm::object::coff_symbol *Symbol;
  const llvm::object::coff_section *Section;
  llvm::ArrayRef<uint8_t> Data;
};

class FileCOFF : public File {
private:
  typedef std::vector<const llvm::object::coff_symbol*> SymbolVector;
  typedef std::map<const llvm::object::coff_section*,
                   std::vector<const llvm::object::coff_symbol*>>
      SectionToSymbolVectorMap;

public:
  FileCOFF(const TargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> MB,
           llvm::error_code &EC)
      : File(MB->getBufferIdentifier(), kindObject), _targetInfo(ti) {
    llvm::OwningPtr<llvm::object::Binary> Bin;
    EC = llvm::object::createBinary(MB.release(), Bin);
    if (EC)
      return;

    Obj.reset(llvm::dyn_cast<const llvm::object::COFFObjectFile>(Bin.get()));
    if (!Obj) {
      EC = make_error_code(llvm::object::object_error::invalid_file_type);
      return;
    }
    Bin.take();

    // Assign each symbol to the section it's in.
    SectionToSymbolVectorMap definedSymbols;
    if ((EC = readSymbolTable(AbsoluteAtoms._atoms, UndefinedAtoms._atoms,
                              definedSymbols)))
      return;

    // Atomize defined symbols. This is a separate pass from readSymbolTable()
    // because in order to create an atom for a symbol we need to the adjacent
    // symbols.
    for (auto &i : definedSymbols) {
      const llvm::object::coff_section *section = i.first;
      std::vector<const llvm::object::coff_symbol*> &symbols = i.second;
      if ((EC = AtomizeDefinedSymbols(section, symbols)))
        return;
    }
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

  virtual const TargetInfo &getTargetInfo() const { return _targetInfo; }

private:
  /// Iterate over symbol table to process all symbols. Absolute or undefined
  /// symbols are atomized in this method. Defined symbols are not atomized
  /// but added to DefinedSymbols as is for further processing. Note that this
  /// function is const, so it will not mutate objects other than arguments.
  error_code readSymbolTable(std::vector<const AbsoluteAtom*> &absoluteAtoms,
                             std::vector<const UndefinedAtom*> &undefinedAtoms,
                             SectionToSymbolVectorMap &definedSymbols) const {
    const llvm::object::coff_file_header *Header = nullptr;
    if (error_code ec = Obj->getHeader(Header))
      return ec;

    for (uint32_t i = 0, e = Header->NumberOfSymbols; i != e; ++i) {
      const llvm::object::coff_symbol *Symb;
      if (error_code ec = Obj->getSymbol(i, Symb))
        return ec;
      llvm::StringRef Name;
      if (error_code ec = Obj->getSymbolName(Symb, Name))
        return ec;
      int16_t SectionIndex = Symb->SectionNumber;
      assert(SectionIndex != llvm::COFF::IMAGE_SYM_DEBUG &&
             "Cannot atomize IMAGE_SYM_DEBUG!");
      // Skip aux symbols.
      i += Symb->NumberOfAuxSymbols;
      if (SectionIndex == llvm::COFF::IMAGE_SYM_ABSOLUTE) {
        // Create an absolute atom.
        absoluteAtoms.push_back(new (AtomStorage.Allocate<COFFAbsoluteAtom>())
            COFFAbsoluteAtom(*this, Name, Symb->Value));
        continue;
      }
      if (SectionIndex == llvm::COFF::IMAGE_SYM_UNDEFINED) {
        // Create an undefined atom.
        undefinedAtoms.push_back(new (AtomStorage.Allocate<COFFUndefinedAtom>())
            COFFUndefinedAtom(*this, Name));
        continue;
      }
      // A symbol with IMAGE_SYM_CLASS_STATIC and zero value represents a
      // section name. This is redundant and we can safely skip such a symbol
      // because the same section name is also in the section header.
      if (Symb->StorageClass != llvm::COFF::IMAGE_SYM_CLASS_STATIC
          || Symb->Value != 0) {
        // This is actually a defined symbol. Add it to its section's list of
        // symbols.
        uint8_t SC = Symb->StorageClass;
        if (SC != llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL
            && SC != llvm::COFF::IMAGE_SYM_CLASS_STATIC
            && SC != llvm::COFF::IMAGE_SYM_CLASS_FUNCTION) {
          llvm::errs() << "Unable to create atom for: " << Name << "\n";
          return llvm::object::object_error::parse_failed;
        }
        const llvm::object::coff_section *Sec;
        if (error_code ec = Obj->getSection(SectionIndex, Sec))
          return ec;
        assert(Sec && "SectionIndex > 0, Sec must be non-null!");
        definedSymbols[Sec].push_back(Symb);
      }
    }
    return error_code::success();
  }

  /// Atomize defined symbols.
  error_code AtomizeDefinedSymbols(
      const llvm::object::coff_section *section,
      std::vector<const llvm::object::coff_symbol*> &symbols) {
    // Sort symbols by position.
    std::stable_sort(symbols.begin(), symbols.end(),
      // For some reason MSVC fails to allow the lambda in this context with a
      // "illegal use of local type in type instantiation". MSVC is clearly
      // wrong here. Force a conversion to function pointer to work around.
      static_cast<bool(*)(const coff_symbol*, const coff_symbol*)>(
        [](const coff_symbol *A, const coff_symbol *B) -> bool {
      return A->Value < B->Value;
    }));

    if (symbols.empty()) {
      // Create an atom for the entire section.
      llvm::ArrayRef<uint8_t> Data;
      DefinedAtoms._atoms.push_back(
        new (AtomStorage.Allocate<COFFDefinedAtom>())
          COFFDefinedAtom(*this, "", nullptr, section, Data));
      return error_code::success();
    }

    llvm::ArrayRef<uint8_t> SecData;
    if (error_code ec = Obj->getSectionContents(section, SecData))
      return ec;

    // Create an unnamed atom if the first atom isn't at the start of the
    // section.
    if (symbols[0]->Value != 0) {
      uint64_t Size = symbols[0]->Value;
      llvm::ArrayRef<uint8_t> Data(SecData.data(), Size);
      DefinedAtoms._atoms.push_back(
        new (AtomStorage.Allocate<COFFDefinedAtom>())
          COFFDefinedAtom(*this, "", nullptr, section, Data));
    }

    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      const uint8_t *start = SecData.data() + (*si)->Value;
      // if this is the last symbol, take up the remaining data.
      const uint8_t *end = (si + 1 == se)
          ? start + SecData.size()
          : SecData.data() + (*(si + 1))->Value;
      llvm::ArrayRef<uint8_t> Data(start, end);
      llvm::StringRef Name;
      if (error_code ec = Obj->getSymbolName(*si, Name))
        return ec;
      DefinedAtoms._atoms.push_back(
        new (AtomStorage.Allocate<COFFDefinedAtom>())
          COFFDefinedAtom(*this, Name, *si, section, Data));
    }
    return error_code::success();
  }

  std::unique_ptr<const llvm::object::COFFObjectFile> Obj;
  atom_collection_vector<DefinedAtom> DefinedAtoms;
  atom_collection_vector<UndefinedAtom>     UndefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> SharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> AbsoluteAtoms;
  mutable llvm::BumpPtrAllocator AtomStorage;
  const TargetInfo &_targetInfo;
};

class ReaderCOFF : public Reader {
public:
  ReaderCOFF(const TargetInfo &ti) : Reader(ti) {}

  error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                       std::vector<std::unique_ptr<File> > &result) const {
    llvm::error_code ec;
    std::unique_ptr<File> file(new FileCOFF(_targetInfo, std::move(mb), ec));
    if (ec)
      return ec;

    DEBUG({
      llvm::dbgs() << "Defined atoms:\n";
      for (const auto &atom : file->defined())
        llvm::dbgs() << "  " << atom->name() << "\n";
    });

    result.push_back(std::move(file));
    return error_code::success();
  }
};

} // end namespace anonymous

namespace lld {
std::unique_ptr<Reader> createReaderPECOFF(const TargetInfo & ti) {
  return std::unique_ptr<Reader>(new ReaderCOFF(ti));
}

}
