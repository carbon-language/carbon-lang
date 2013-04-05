//===- lib/ReaderWriter/PECOFF/ReaderCOFF.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Reader.h"
#include "lld/Core/File.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Casting.h"
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

  virtual bool isThumb() const {
    return false;
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

    const llvm::object::coff_file_header *Header = nullptr;
    if ((EC = Obj->getHeader(Header)))
      return;

    // Assign each symbol to the section it's in. If it does not belong to a
    // section, create an atom for it now.
    std::map< const llvm::object::coff_section*
            , std::vector<const llvm::object::coff_symbol*>> SectionSymbols;

    for (uint32_t i = 0, e = Header->NumberOfSymbols; i != e; ++i) {
      const llvm::object::coff_symbol *Symb;
      if ((EC = Obj->getSymbol(i, Symb)))
        return;
      llvm::StringRef Name;
      if ((EC = Obj->getSymbolName(Symb, Name)))
        return;
      int16_t SectionIndex = Symb->SectionNumber;
      assert(SectionIndex != llvm::COFF::IMAGE_SYM_DEBUG &&
        "Cannot atomize IMAGE_SYM_DEBUG!");
      if (SectionIndex == llvm::COFF::IMAGE_SYM_ABSOLUTE) {
        // Create an absolute atom.
        AbsoluteAtoms._atoms.push_back(
          new (AtomStorage.Allocate<COFFAbsoluteAtom>())
            COFFAbsoluteAtom(*this, Name, Symb->Value));
      } else if (SectionIndex == llvm::COFF::IMAGE_SYM_UNDEFINED) {
        // Create an undefined atom.
        UndefinedAtoms._atoms.push_back(
          new (AtomStorage.Allocate<COFFUndefinedAtom>())
            COFFUndefinedAtom(*this, Name));
      } else {
        // This is actually a defined symbol. Add it to its section's list of
        // symbols.
        uint8_t SC = Symb->StorageClass;
        // If Symb->Value actually means section offset.
        if (   SC == llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL
            || SC == llvm::COFF::IMAGE_SYM_CLASS_STATIC
            || SC == llvm::COFF::IMAGE_SYM_CLASS_FUNCTION) {
          const llvm::object::coff_section *Sec;
          if ((EC = Obj->getSection(SectionIndex, Sec)))
            return;
          assert(Sec && "SectionIndex > 0, Sec must be non-null!");
          SectionSymbols[Sec].push_back(Symb);
        } else {
          llvm::errs() << "Unable to create atom for: " << Name << "\n";
          EC = llvm::object::object_error::parse_failed;
          return;
        }
      }
      // Skip aux symbols.
      i += Symb->NumberOfAuxSymbols;
    }

    // For each section, sort its symbols by address, then create a defined atom
    // for each range.
    for (auto &i : SectionSymbols) {
      auto &Symbs = i.second;
      // Sort symbols by position.
      std::stable_sort(Symbs.begin(), Symbs.end(),
        // For some reason MSVC fails to allow the lambda in this context with a
        // "illegal use of local type in type instantiation". MSVC is clearly
        // wrong here. Force a conversion to function pointer to work around.
        static_cast<bool(*)(const coff_symbol*, const coff_symbol*)>(
          [](const coff_symbol *A, const coff_symbol *B) -> bool {
        return A->Value < B->Value;
      }));

      if (Symbs.empty()) {
        // Create an atom for the entire section.
        llvm::ArrayRef<uint8_t> Data;
        DefinedAtoms._atoms.push_back(
          new (AtomStorage.Allocate<COFFDefinedAtom>())
            COFFDefinedAtom(*this, "", nullptr, i.first, Data));
        continue;
      }

      llvm::ArrayRef<uint8_t> SecData;
      if ((EC = Obj->getSectionContents(i.first, SecData)))
        return;

      // Create an unnamed atom if the first atom isn't at the start of the
      // section.
      if (Symbs[0]->Value != 0) {
        uint64_t Size = Symbs[0]->Value;
        llvm::ArrayRef<uint8_t> Data(SecData.data(), Size);
        DefinedAtoms._atoms.push_back(
          new (AtomStorage.Allocate<COFFDefinedAtom>())
            COFFDefinedAtom(*this, "", nullptr, i.first, Data));
      }

      for (auto si = Symbs.begin(), se = Symbs.end(); si != se; ++si) {
        // if this is the last symbol, take up the remaining data.
        llvm::ArrayRef<uint8_t> Data;
        if (si + 1 == se) {
          Data = llvm::ArrayRef<uint8_t>( SecData.data() + (*si)->Value
                                        , SecData.end());
        } else {
          Data = llvm::ArrayRef<uint8_t>( SecData.data() + (*si)->Value
                                        , (*(si + 1))->Value - (*si)->Value);
        }
        llvm::StringRef Name;
        if ((EC = Obj->getSymbolName(*si, Name)))
          return;
        DefinedAtoms._atoms.push_back(
          new (AtomStorage.Allocate<COFFDefinedAtom>())
            COFFDefinedAtom(*this, Name, *si, i.first, Data));
      }
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
  std::unique_ptr<const llvm::object::COFFObjectFile> Obj;
  atom_collection_vector<DefinedAtom> DefinedAtoms;
  atom_collection_vector<UndefinedAtom>     UndefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> SharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> AbsoluteAtoms;
  llvm::BumpPtrAllocator AtomStorage;
  const TargetInfo &_targetInfo;
};



class ReaderCOFF : public Reader {
public:
  ReaderCOFF(const TargetInfo &ti) : Reader(ti) {}

  error_code parseFile(std::unique_ptr<MemoryBuffer> mb,
                       std::vector<std::unique_ptr<File>> &result) const {
    llvm::error_code ec;
    std::unique_ptr<File> f(new FileCOFF(_targetInfo, std::move(mb), ec));
    if (ec) {
      return ec;
    }

    result.push_back(std::move(f));
    return error_code::success();
  }
};
} // end namespace anonymous

namespace lld {
std::unique_ptr<Reader> createReaderPECOFF(const TargetInfo & ti,
                                           std::function<ReaderFunc>) {
  return std::unique_ptr<Reader>(new ReaderCOFF(ti));
}
}
