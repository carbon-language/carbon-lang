//===- lib/ReaderWriter/PECOFF/ReaderCOFF.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ReaderCOFF"

#include "lld/Core/File.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/ReaderArchive.h"

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

using std::vector;
using llvm::object::coff_relocation;
using llvm::object::coff_section;
using llvm::object::coff_symbol;

using namespace lld;

namespace { // anonymous

/// A COFFReference represents relocation information for an atom. For
/// example, if atom X has a reference to atom Y with offsetInAtom=8, that
/// means that the address starting at 8th byte of the content of atom X needs
/// to be fixed up so that the address points to atom Y's address.
class COFFReference LLVM_FINAL : public Reference {
public:
  COFFReference(Kind kind) : _target(nullptr), _offsetInAtom(0) {
    _kind = kind;
  }

  COFFReference(const Atom *target, uint32_t offsetInAtom, uint16_t relocType)
      : _target(target), _offsetInAtom(offsetInAtom) {
    setKind(static_cast<Reference::Kind>(relocType));
  }

  virtual const Atom *target() const { return _target; }
  virtual void setTarget(const Atom *newAtom) { _target = newAtom; }

  // Addend is a value to be added to the relocation target. For example, if
  // target=AtomX and addend=4, the relocation address will become the address
  // of AtomX + 4. COFF does not support that sort of relocation, thus addend
  // is always zero.
  virtual Addend addend() const { return 0; }
  virtual void setAddend(Addend) {}

  virtual uint64_t offsetInAtom() const { return _offsetInAtom; }

private:
  const Atom *_target;
  uint32_t _offsetInAtom;
};

class COFFAbsoluteAtom : public AbsoluteAtom {
public:
  COFFAbsoluteAtom(const File &F, llvm::StringRef N, const coff_symbol *S)
    : OwningFile(F)
    , Name(N)
    , Symbol(S)
  {}

  virtual const class File &file() const {
    return OwningFile;
  }

  virtual Scope scope() const {
    if (Symbol->StorageClass == llvm::COFF::IMAGE_SYM_CLASS_STATIC)
      return scopeTranslationUnit;
    return scopeGlobal;
  }

  virtual llvm::StringRef name() const {
    return Name;
  }

  virtual uint64_t value() const {
    return Symbol->Value;
  }

private:
  const File &OwningFile;
  llvm::StringRef Name;
  const coff_symbol *Symbol;
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

  uint64_t originalOffset() const { return Symbol->Value; }

  void addReference(COFFReference *reference) {
    References.push_back(reference);
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
    return reference_iterator(*this, reinterpret_cast<const void *>(0));
  }

  virtual reference_iterator end() const {
    return reference_iterator(
        *this, reinterpret_cast<const void *>(References.size()));
  }

private:
  virtual const Reference *derefIterator(const void *iter) const {
    size_t index = reinterpret_cast<size_t>(iter);
    return References[index];
  }

  virtual void incrementIterator(const void *&iter) const {
    size_t index = reinterpret_cast<size_t>(iter);
    iter = reinterpret_cast<const void *>(index + 1);
  }

  const File &OwningFile;
  llvm::StringRef Name;
  const llvm::object::coff_symbol *Symbol;
  const llvm::object::coff_section *Section;
  std::vector<COFFReference *> References;
  llvm::ArrayRef<uint8_t> Data;
};

class FileCOFF : public File {
private:
  typedef vector<const coff_symbol *> SymbolVectorT;
  typedef std::map<const coff_section *, SymbolVectorT> SectionToSymbolsT;
  typedef std::map<const StringRef, Atom *> SymbolNameToAtomT;
  typedef std::map<const coff_section *, vector<COFFDefinedAtom *> >
      SectionToAtomsT;

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

    // Read the symbol table and atomize them if possible. Defined atoms
    // cannot be atomized in one pass, so they will be not be atomized but
    // added to symbolToAtom.
    SectionToSymbolsT definedSymbols;
    SymbolNameToAtomT symbolToAtom;
    if ((EC = readSymbolTable(AbsoluteAtoms._atoms, UndefinedAtoms._atoms,
                              definedSymbols, symbolToAtom)))
      return;

    // Atomize defined symbols. This is a separate pass from readSymbolTable()
    // because in order to create an atom for a symbol we need to the adjacent
    // symbols.
    SectionToAtomsT sectionToAtoms;
    if ((EC = AtomizeDefinedSymbols(definedSymbols, DefinedAtoms._atoms,
                                    symbolToAtom, sectionToAtoms)))
      return;

    EC = addRelocationReferenceToAtoms(symbolToAtom, sectionToAtoms);
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
  error_code readSymbolTable(vector<const AbsoluteAtom *> &absoluteAtoms,
                             vector<const UndefinedAtom *> &undefinedAtoms,
                             SectionToSymbolsT &definedSymbols,
                             SymbolNameToAtomT &symbolToAtom) const {
    const llvm::object::coff_file_header *Header = nullptr;
    if (error_code ec = Obj->getHeader(Header))
      return ec;

    for (uint32_t i = 0, e = Header->NumberOfSymbols; i != e; ++i) {
      const coff_symbol *Symb;
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
      // Create an absolute atom.
      if (SectionIndex == llvm::COFF::IMAGE_SYM_ABSOLUTE) {
        auto *atom = new (AtomStorage.Allocate<COFFAbsoluteAtom>())
            COFFAbsoluteAtom(*this, Name, Symb);
        if (!Name.empty())
          symbolToAtom[Name] = atom;
        absoluteAtoms.push_back(atom);
        continue;
      }
      // Create an undefined atom.
      if (SectionIndex == llvm::COFF::IMAGE_SYM_UNDEFINED) {
        auto *atom = new (AtomStorage.Allocate<COFFUndefinedAtom>())
            COFFUndefinedAtom(*this, Name);
        if (!Name.empty())
          symbolToAtom[Name] = atom;
        undefinedAtoms.push_back(atom);
        continue;
      }

      // This is actually a defined symbol. Add it to its section's list of
      // symbols.
      uint8_t SC = Symb->StorageClass;
      if (SC != llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL &&
          SC != llvm::COFF::IMAGE_SYM_CLASS_STATIC &&
          SC != llvm::COFF::IMAGE_SYM_CLASS_FUNCTION) {
        llvm::errs() << "Unable to create atom for: " << Name << "\n";
        return llvm::object::object_error::parse_failed;
      }
      const coff_section *Sec;
      if (error_code ec = Obj->getSection(SectionIndex, Sec))
        return ec;
      assert(Sec && "SectionIndex > 0, Sec must be non-null!");
      definedSymbols[Sec].push_back(Symb);
    }
    return error_code::success();
  }

  /// Atomize \p symbols and append the results to \p atoms. The symbols are
  /// assumed to have been defined in the \p section.
  error_code
  AtomizeDefinedSymbolsInSection(const coff_section *section,
                                 vector<const coff_symbol *> &symbols,
                                 vector<COFFDefinedAtom *> &atoms) const {
      // Sort symbols by position.
    std::stable_sort(symbols.begin(), symbols.end(),
      // For some reason MSVC fails to allow the lambda in this context with a
      // "illegal use of local type in type instantiation". MSVC is clearly
      // wrong here. Force a conversion to function pointer to work around.
      static_cast<bool(*)(const coff_symbol*, const coff_symbol*)>(
        [](const coff_symbol *A, const coff_symbol *B) -> bool {
      return A->Value < B->Value;
    }));

    llvm::ArrayRef<uint8_t> SecData;
    if (error_code ec = Obj->getSectionContents(section, SecData))
      return ec;

    // Create an atom for the entire section.
    if (symbols.empty()) {
      llvm::ArrayRef<uint8_t> Data(SecData.data(), SecData.size());
      atoms.push_back(new (AtomStorage.Allocate<COFFDefinedAtom>())
                      COFFDefinedAtom(*this, "", nullptr, section, Data));
      return error_code::success();
    }

    // Create an unnamed atom if the first atom isn't at the start of the
    // section.
    if (symbols[0]->Value != 0) {
      uint64_t Size = symbols[0]->Value;
      llvm::ArrayRef<uint8_t> Data(SecData.data(), Size);
      atoms.push_back(new (AtomStorage.Allocate<COFFDefinedAtom>())
                      COFFDefinedAtom(*this, "", nullptr, section, Data));
    }

    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      const uint8_t *start = SecData.data() + (*si)->Value;
      // if this is the last symbol, take up the remaining data.
      const uint8_t *end = (si + 1 == se)
          ? start + SecData.size()
          : SecData.data() + (*(si + 1))->Value;
      llvm::ArrayRef<uint8_t> Data(start, end);
      llvm::StringRef name;
      if (error_code ec = Obj->getSymbolName(*si, name))
        return ec;
      atoms.push_back(new (AtomStorage.Allocate<COFFDefinedAtom>())
                      COFFDefinedAtom(*this, name, *si, section, Data));
    }
    return error_code::success();
  }

  void addEdge(COFFDefinedAtom *a, COFFDefinedAtom *b,
               lld::Reference::Kind kind) const {
    auto ref = new (AtomStorage.Allocate<COFFReference>()) COFFReference(kind);
    ref->setTarget(b);
    a->addReference(ref);
  }

  void connectAtomsWithLayoutEdge(COFFDefinedAtom *a,
                                  COFFDefinedAtom *b) const {
    addEdge(a, b, lld::Reference::kindLayoutAfter);
    addEdge(b, a, lld::Reference::kindLayoutBefore);
  }

  /// Connect atoms appeared in the same section with layout-{before,after}
  /// edges. It has two purposes.
  ///
  ///   - To prevent atoms from being GC'ed (aka dead-stripped) if there is a
  ///     reference to one of the atoms. In that case we want to emit all the
  ///     atoms appeared in the same section, because the referenced "live"
  ///     atom may reference other atoms in the same section. If we don't add
  ///     edges between atoms, unreferenced atoms in the same section would be
  ///     GC'ed.
  ///   - To preserve the order of atmos. We want to emit the atoms in the
  ///     same order as they appeared in the input object file.
  void addLayoutEdges(vector<COFFDefinedAtom *> &definedAtoms) const {
    if (definedAtoms.size() <= 1)
      return;
    for (auto it = definedAtoms.begin(), e = definedAtoms.end(); it + 1 != e;
         ++it)
      connectAtomsWithLayoutEdge(*it, *(it + 1));
  }

  error_code AtomizeDefinedSymbols(SectionToSymbolsT &definedSymbols,
                                   vector<const DefinedAtom *> &definedAtoms,
                                   SymbolNameToAtomT &symbolToAtom,
                                   SectionToAtomsT &sectionToAtoms) const {
    // For each section, make atoms for all the symbols defined in the
    // section, and append the atoms to the result objects.
    for (auto &i : definedSymbols) {
      const coff_section *section = i.first;
      vector<const coff_symbol *> &symbols = i.second;
      vector<COFFDefinedAtom *> atoms;
      if (error_code ec =
              AtomizeDefinedSymbolsInSection(section, symbols, atoms))
        return ec;

      // Connect atoms with layout-before/layout-after edges.
      addLayoutEdges(atoms);

      for (COFFDefinedAtom *atom : atoms) {
        if (!atom->name().empty())
          symbolToAtom[atom->name()] = atom;
        sectionToAtoms[section].push_back(atom);
        definedAtoms.push_back(atom);
      }
    }
    return error_code::success();
  }

  /// Find the atom that is at \p targetOffset in \p section. It is assumed
  /// that \p atoms are sorted by position in the section.
  COFFDefinedAtom *findAtomAt(uint32_t targetOffset,
                              const coff_section *section,
                              const vector<COFFDefinedAtom *> &atoms) const {
    auto compareFn =
        [](const COFFDefinedAtom * a, const COFFDefinedAtom * b)->bool {
      return a->originalOffset() < b->originalOffset();
    }
    ;
    assert(std::is_sorted(atoms.begin(), atoms.end(), compareFn));

    for (COFFDefinedAtom *atom : atoms)
      if (targetOffset < atom->originalOffset() + atom->size())
        return atom;
    llvm_unreachable("Relocation target out of range");
  }

  /// Find the atom for the symbol that was at the \p index in the symbol
  /// table.
  error_code getAtomBySymbolIndex(uint32_t index,
                                  SymbolNameToAtomT symbolToAtom,
                                  Atom *&ret) const {
    const coff_symbol *symbol;
    if (error_code ec = Obj->getSymbol(index, symbol))
      return ec;
    StringRef symbolName;
    if (error_code ec = Obj->getSymbolName(symbol, symbolName))
      return ec;
    ret = symbolToAtom[symbolName];
    assert(ret);
    return error_code::success();
  }

  /// Add relocation information to an atom based on \p rel. \p rel is an
  /// relocation entry for the \p section, and \p atoms are all the atoms
  /// defined in the \p section.
  error_code
  addRelocationReference(const coff_relocation *rel,
                         const coff_section *section,
                         const vector<COFFDefinedAtom *> &atoms,
                         const SymbolNameToAtomT symbolToAtom) const {
    assert(atoms.size() > 0);
    // The address of the item which relocation is applied. Section's
    // VirtualAddress needs to be added for historical reasons, but the value
    // is usually just zero, so adding it is usually no-op.
    uint32_t itemAddress = rel->VirtualAddress + section->VirtualAddress;

    Atom *targetAtom = nullptr;
    if (error_code ec = getAtomBySymbolIndex(rel->SymbolTableIndex,
                                             symbolToAtom, targetAtom))
      return ec;

    COFFDefinedAtom *atom = findAtomAt(rel->VirtualAddress, section, atoms);
    uint32_t offsetInAtom = itemAddress - atom->originalOffset();
    assert(offsetInAtom < atom->size());
    COFFReference *ref = new (AtomStorage.Allocate<COFFReference>())
        COFFReference(targetAtom, offsetInAtom, rel->Type);
    atom->addReference(ref);
    return error_code::success();
  }

  /// Add relocation information to atoms.
  error_code addRelocationReferenceToAtoms(SymbolNameToAtomT symbolToAtom,
                                           SectionToAtomsT &sectionToAtoms) {
    // Relocation entries are defined for each section.
    error_code ec;
    for (auto si = Obj->begin_sections(), se = Obj->end_sections(); si != se;
         si.increment(ec)) {
      const coff_section *section = Obj->getCOFFSection(si);
      for (auto ri = si->begin_relocations(), re = si->end_relocations();
           ri != re; ri.increment(ec)) {
        const coff_relocation *rel = Obj->getCOFFRelocation(ri);
        if ((ec = addRelocationReference(rel, section, sectionToAtoms[section],
                                         symbolToAtom)))
          return ec;
      }
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
  ReaderCOFF(const TargetInfo &ti) : Reader(ti), _readerArchive(ti, *this) {}

  error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                       std::vector<std::unique_ptr<File> > &result) const {
    StringRef magic(mb->getBufferStart(), mb->getBufferSize());
    llvm::sys::fs::file_magic fileType = llvm::sys::fs::identify_magic(magic);
    if (fileType == llvm::sys::fs::file_magic::coff_object)
      return parseCOFFFile(mb, result);
    if (fileType == llvm::sys::fs::file_magic::archive)
      return _readerArchive.parseFile(mb, result);
    return make_error_code(llvm::object::object_error::invalid_file_type);
  }

private:
  error_code parseCOFFFile(std::unique_ptr<MemoryBuffer> &mb,
                           std::vector<std::unique_ptr<File> > &result) const {
    error_code ec;
    std::unique_ptr<File> file(new FileCOFF(_targetInfo, std::move(mb), ec));
    if (ec)
      return ec;

    DEBUG({
      llvm::dbgs() << "Defined atoms:\n";
      for (const auto &atom : file->defined()) {
        llvm::dbgs() << "  " << atom->name() << "\n";
        for (const Reference *ref : *atom)
          llvm::dbgs() << "    @" << ref->offsetInAtom() << " -> "
                       << ref->target()->name() << "\n";
      }
    });

    result.push_back(std::move(file));
    return error_code::success();
  }

  ReaderArchive _readerArchive;
};

} // end namespace anonymous

namespace lld {
std::unique_ptr<Reader> createReaderPECOFF(const TargetInfo & ti) {
  return std::unique_ptr<Reader>(new ReaderCOFF(ti));
}

}
