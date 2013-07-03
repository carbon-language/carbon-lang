//===- lib/ReaderWriter/PECOFF/ReaderCOFF.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ReaderCOFF"

#include "Atoms.h"
#include "ReaderImportHeader.h"

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
using lld::coff::COFFAbsoluteAtom;
using lld::coff::COFFDefinedAtom;
using lld::coff::COFFReference;
using lld::coff::COFFUndefinedAtom;
using llvm::object::coff_relocation;
using llvm::object::coff_section;
using llvm::object::coff_symbol;

using namespace lld;

namespace { // anonymous

class FileCOFF : public File {
private:
  typedef vector<const coff_symbol *> SymbolVectorT;
  typedef std::map<const coff_section *, SymbolVectorT> SectionToSymbolsT;
  typedef std::map<const StringRef, Atom *> SymbolNameToAtomT;
  typedef std::map<const coff_section *, vector<COFFDefinedAtom *> >
      SectionToAtomsT;

public:
  FileCOFF(const TargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> MB,
           error_code &EC)
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
      StringRef Name;
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

    ArrayRef<uint8_t> SecData;
    StringRef sectionName;
    if (error_code ec = Obj->getSectionContents(section, SecData))
      return ec;
    if (error_code ec = Obj->getSectionName(section, sectionName))
      return ec;
    uint64_t ordinal = 0;

    // We do not support debug information yet. We could keep data in ".debug$S"
    // section in the resultant binary by copying as opaque bytes, but it would
    // make the binary hard to debug because of extraneous data. So we'll skip
    // the debug info.
    if (sectionName == ".debug$S")
      return error_code::success();

    // Create an atom for the entire section.
    if (symbols.empty()) {
      ArrayRef<uint8_t> Data(SecData.data(), SecData.size());
      atoms.push_back(new (AtomStorage.Allocate<COFFDefinedAtom>())
                      COFFDefinedAtom(*this, "", nullptr, section, Data,
                                      sectionName, ordinal++));
      return error_code::success();
    }

    // Create an unnamed atom if the first atom isn't at the start of the
    // section.
    if (symbols[0]->Value != 0) {
      uint64_t Size = symbols[0]->Value;
      ArrayRef<uint8_t> Data(SecData.data(), Size);
      atoms.push_back(new (AtomStorage.Allocate<COFFDefinedAtom>())
                      COFFDefinedAtom(*this, "", nullptr, section, Data,
                                      sectionName, ordinal++));
    }

    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      const uint8_t *start = SecData.data() + (*si)->Value;
      // if this is the last symbol, take up the remaining data.
      const uint8_t *end = (si + 1 == se)
          ? start + SecData.size()
          : SecData.data() + (*(si + 1))->Value;
      ArrayRef<uint8_t> Data(start, end);
      StringRef name;
      if (error_code ec = Obj->getSymbolName(*si, name))
        return ec;
      atoms.push_back(new (AtomStorage.Allocate<COFFDefinedAtom>())
                      COFFDefinedAtom(*this, name, *si, section, Data,
                                      sectionName, ordinal++));
    }
    return error_code::success();
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
      connectAtomsWithLayoutEdge(atoms);

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
    assert(std::is_sorted(atoms.begin(), atoms.end(),
                          [](const COFFDefinedAtom * a,
                             const COFFDefinedAtom * b) -> bool {
                            return a->originalOffset() < b->originalOffset();
                          }));

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
    atom->addReference(std::unique_ptr<COFFReference>(
        new COFFReference(targetAtom, offsetInAtom, rel->Type)));
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
  explicit ReaderCOFF(const TargetInfo &ti)
      : Reader(ti), _readerArchive(ti, *this) {}

  error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                       std::vector<std::unique_ptr<File> > &result) const {
    StringRef magic(mb->getBufferStart(), mb->getBufferSize());
    llvm::sys::fs::file_magic fileType = llvm::sys::fs::identify_magic(magic);
    if (fileType == llvm::sys::fs::file_magic::coff_object)
      return parseCOFFFile(mb, result);
    if (fileType == llvm::sys::fs::file_magic::archive)
      return _readerArchive.parseFile(mb, result);
    return lld::coff::parseCOFFImportLibrary(_targetInfo, mb, result);
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
