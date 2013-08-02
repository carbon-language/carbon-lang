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
using lld::coff::COFFBSSAtom;
using lld::coff::COFFDefinedAtom;
using lld::coff::COFFDefinedFileAtom;
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
  typedef std::map<const coff_section *, vector<COFFDefinedFileAtom *> >
      SectionToAtomsT;

public:
  FileCOFF(const TargetInfo &ti, std::unique_ptr<llvm::MemoryBuffer> mb,
           error_code &ec)
      : File(mb->getBufferIdentifier(), kindObject), _targetInfo(ti) {
    llvm::OwningPtr<llvm::object::Binary> bin;
    ec = llvm::object::createBinary(mb.release(), bin);
    if (ec)
      return;

    _obj.reset(llvm::dyn_cast<const llvm::object::COFFObjectFile>(bin.get()));
    if (!_obj) {
      ec = make_error_code(llvm::object::object_error::invalid_file_type);
      return;
    }
    bin.take();

    // Read the symbol table and atomize them if possible. Defined atoms
    // cannot be atomized in one pass, so they will be not be atomized but
    // added to symbolToAtom.
    SymbolVectorT symbols;
    if ((ec = readSymbolTable(symbols)))
      return;

    createAbsoluteAtoms(symbols, _absoluteAtoms._atoms);
    createUndefinedAtoms(symbols, _undefinedAtoms._atoms);
    if ((ec = createDefinedSymbols(symbols, _definedAtoms._atoms)))
      return;

    ec = addRelocationReferenceToAtoms();
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

  virtual const TargetInfo &getTargetInfo() const { return _targetInfo; }

private:
  /// Iterate over the symbol table to retrieve all symbols.
  error_code readSymbolTable(vector<const coff_symbol *> &result) {
    const llvm::object::coff_file_header *header = nullptr;
    if (error_code ec = _obj->getHeader(header))
      return ec;

    for (uint32_t i = 0, e = header->NumberOfSymbols; i != e; ++i) {
      // Retrieve the symbol.
      const coff_symbol *sym;
      if (error_code ec = _obj->getSymbol(i, sym))
        return ec;
      assert(sym->SectionNumber != llvm::COFF::IMAGE_SYM_DEBUG &&
             "Cannot atomize IMAGE_SYM_DEBUG!");
      result.push_back(sym);

      // Cache the name.
      StringRef name;
      if (error_code ec = _obj->getSymbolName(sym, name))
        return ec;
      _symbolName[sym] = name;

      // Skip aux symbols.
      i += sym->NumberOfAuxSymbols;
    }
    return error_code::success();
  }

  /// Create atoms for the absolute symbols.
  void createAbsoluteAtoms(const SymbolVectorT &symbols,
                           vector<const AbsoluteAtom *> &result) {
    for (const coff_symbol *sym : symbols) {
      if (sym->SectionNumber != llvm::COFF::IMAGE_SYM_ABSOLUTE)
        continue;
      auto *atom = new (_alloc) COFFAbsoluteAtom(*this, _symbolName[sym], sym);
      result.push_back(atom);
      _symbolAtom[sym] = atom;
    }
  }

  /// Create atoms for the undefined symbols.
  void createUndefinedAtoms(const SymbolVectorT &symbols,
                            vector<const UndefinedAtom *> &result) {
    for (const coff_symbol *sym : symbols) {
      if (sym->SectionNumber != llvm::COFF::IMAGE_SYM_UNDEFINED)
        continue;
      auto *atom = new (_alloc) COFFUndefinedAtom(*this, _symbolName[sym]);
      result.push_back(atom);
      _symbolAtom[sym] = atom;
    }
  }

  /// Create atoms for the defined symbols. This pass is a bit complicated than
  /// the other two, because in order to create the atom for the defined symbol
  /// we need to know the adjacent symbols.
  error_code createDefinedSymbols(const SymbolVectorT &symbols,
                                  vector<const DefinedAtom *> &result) {
    // Filter non-defined atoms, and group defined atoms by its section.
    SectionToSymbolsT definedSymbols;
    for (const coff_symbol *sym : symbols) {
      // A symbol with section number 0 and non-zero value represents an
      // uninitialized data. I don't understand why there are two ways to define
      // an uninitialized data symbol (.bss and this way), but that's how COFF
      // works.
      if (sym->SectionNumber == llvm::COFF::IMAGE_SYM_UNDEFINED &&
          sym->Value > 0) {
        StringRef name = _symbolName[sym];
        uint32_t size = sym->Value;
        auto *atom = new (_alloc) COFFBSSAtom(*this, name, sym, size, 0);
        result.push_back(atom);
        continue;
      }

      // Skip if it's not for defined atom.
      if (sym->SectionNumber == llvm::COFF::IMAGE_SYM_ABSOLUTE ||
          sym->SectionNumber == llvm::COFF::IMAGE_SYM_UNDEFINED)
        continue;

      uint8_t sc = sym->StorageClass;
      if (sc != llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL &&
          sc != llvm::COFF::IMAGE_SYM_CLASS_STATIC &&
          sc != llvm::COFF::IMAGE_SYM_CLASS_FUNCTION &&
          sc != llvm::COFF::IMAGE_SYM_CLASS_LABEL) {
        llvm::errs() << "Unable to create atom for: " << _symbolName[sym]
                     << " (" << static_cast<int>(sc) << ")\n";
        return llvm::object::object_error::parse_failed;
      }

      const coff_section *sec;
      if (error_code ec = _obj->getSection(sym->SectionNumber, sec))
        return ec;
      assert(sec && "SectionIndex > 0, Sec must be non-null!");
      definedSymbols[sec].push_back(sym);
    }

    // Atomize the defined symbols.
    if (error_code ec = AtomizeDefinedSymbols(definedSymbols, result))
      return ec;

    return error_code::success();
  }

  /// Atomize \p symbols and append the results to \p atoms. The symbols are
  /// assumed to have been defined in the \p section.
  error_code
  AtomizeDefinedSymbolsInSection(const coff_section *section,
                                 vector<const coff_symbol *> &symbols,
                                 vector<COFFDefinedFileAtom *> &atoms) {
    // Sort symbols by position.
    std::stable_sort(symbols.begin(), symbols.end(),
      // For some reason MSVC fails to allow the lambda in this context with a
      // "illegal use of local type in type instantiation". MSVC is clearly
      // wrong here. Force a conversion to function pointer to work around.
      static_cast<bool(*)(const coff_symbol*, const coff_symbol*)>(
        [](const coff_symbol *a, const coff_symbol *b) -> bool {
      return a->Value < b->Value;
    }));

    StringRef sectionName;
    if (error_code ec = _obj->getSectionName(section, sectionName))
      return ec;
    uint64_t ordinal = -1;

    // BSS section does not have contents. If this is the BSS section, create
    // COFFBSSAtom instead of COFFDefinedAtom.
    if (section->Characteristics &
        llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) {
      for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
        const coff_symbol *sym = *si;
        uint32_t size = (si + 1 == se)
            ? section->SizeOfRawData - sym->Value
            : si[1]->Value - sym->Value;
        auto *atom = new (_alloc) COFFBSSAtom(
            *this, _symbolName[sym], sym, size, ++ordinal);
        atoms.push_back(atom);
        _symbolAtom[sym] = atom;
      }
      return error_code::success();
    }

    ArrayRef<uint8_t> secData;
    if (error_code ec = _obj->getSectionContents(section, secData))
      return ec;

    // We do not support debug information yet. We could keep data in ".debug$S"
    // section in the resultant binary by copying as opaque bytes, but it would
    // make the binary hard to debug because of extraneous data. So we'll skip
    // the debug info.
    if (sectionName == ".debug$S")
      return error_code::success();

    // A section with IMAGE_SCN_LNK_REMOVE attribute will never become
    // a part of the output image. That's what the COFF spec says.
    if (section->Characteristics & llvm::COFF::IMAGE_SCN_LNK_REMOVE)
      return error_code::success();

    // Create an atom for the entire section.
    if (symbols.empty()) {
      ArrayRef<uint8_t> Data(secData.data(), secData.size());
      atoms.push_back(new (_alloc) COFFDefinedAtom(
          *this, "", nullptr, section, Data, sectionName, 0));
      return error_code::success();
    }

    // Create an unnamed atom if the first atom isn't at the start of the
    // section.
    if (symbols[0]->Value != 0) {
      uint64_t size = symbols[0]->Value;
      ArrayRef<uint8_t> data(secData.data(), size);
      atoms.push_back(new (_alloc) COFFDefinedAtom(
          *this, "", nullptr, section, data, sectionName, ++ordinal));
    }

    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      const uint8_t *start = secData.data() + (*si)->Value;
      // if this is the last symbol, take up the remaining data.
      const uint8_t *end = (si + 1 == se)
          ? start + secData.size()
          : secData.data() + (*(si + 1))->Value;
      ArrayRef<uint8_t> data(start, end);
      auto *atom = new (_alloc) COFFDefinedAtom(
          *this, _symbolName[*si], *si, section, data, sectionName, ++ordinal);
      atoms.push_back(atom);
      _symbolAtom[*si] = atom;
    }
    return error_code::success();
  }

  error_code AtomizeDefinedSymbols(SectionToSymbolsT &definedSymbols,
                                   vector<const DefinedAtom *> &definedAtoms) {
    // For each section, make atoms for all the symbols defined in the
    // section, and append the atoms to the result objects.
    for (auto &i : definedSymbols) {
      const coff_section *section = i.first;
      vector<const coff_symbol *> &symbols = i.second;
      vector<COFFDefinedFileAtom *> atoms;
      if (error_code ec =
              AtomizeDefinedSymbolsInSection(section, symbols, atoms))
        return ec;

      // Connect atoms with layout-before/layout-after edges.
      connectAtomsWithLayoutEdge(atoms);

      for (COFFDefinedFileAtom *atom : atoms) {
        _sectionAtoms[section].push_back(atom);
        definedAtoms.push_back(atom);
      }
    }
    return error_code::success();
  }

  /// Find the atom that is at \p targetOffset in \p section. It is assumed
  /// that \p atoms are sorted by position in the section.
  COFFDefinedFileAtom *
  findAtomAt(uint32_t targetOffset,
             const vector<COFFDefinedFileAtom *> &atoms) const {
    for (COFFDefinedFileAtom *atom : atoms)
      if (targetOffset < atom->originalOffset() + atom->size())
        return atom;
    llvm_unreachable("Relocation target out of range");
  }

  /// Find the atom for the symbol that was at the \p index in the symbol
  /// table.
  error_code getAtomBySymbolIndex(uint32_t index, Atom *&ret) {
    const coff_symbol *symbol;
    if (error_code ec = _obj->getSymbol(index, symbol))
      return ec;
    ret = _symbolAtom[symbol];
    assert(ret);
    return error_code::success();
  }

  /// Add relocation information to an atom based on \p rel. \p rel is an
  /// relocation entry for the \p section, and \p atoms are all the atoms
  /// defined in the \p section.
  error_code
  addRelocationReference(const coff_relocation *rel,
                         const coff_section *section,
                         const vector<COFFDefinedFileAtom *> &atoms) {
    assert(atoms.size() > 0);
    // The address of the item which relocation is applied. Section's
    // VirtualAddress needs to be added for historical reasons, but the value
    // is usually just zero, so adding it is usually no-op.
    uint32_t itemAddress = rel->VirtualAddress + section->VirtualAddress;

    Atom *targetAtom = nullptr;
    if (error_code ec = getAtomBySymbolIndex(rel->SymbolTableIndex, targetAtom))
      return ec;

    COFFDefinedFileAtom *atom = findAtomAt(rel->VirtualAddress, atoms);
    uint32_t offsetInAtom = itemAddress - atom->originalOffset();
    assert(offsetInAtom < atom->size());
    atom->addReference(std::unique_ptr<COFFReference>(
        new COFFReference(targetAtom, offsetInAtom, rel->Type)));
    return error_code::success();
  }

  /// Add relocation information to atoms.
  error_code addRelocationReferenceToAtoms() {
    // Relocation entries are defined for each section.
    error_code ec;
    for (auto si = _obj->begin_sections(), se = _obj->end_sections(); si != se;
         si.increment(ec)) {
      const coff_section *section = _obj->getCOFFSection(si);

      // Skip there's no atom for the section. Currently we do not create any
      // atoms for some sections, such as "debug$S", and such sections need to
      // be skipped here too.
      if (_sectionAtoms.find(section) == _sectionAtoms.end())
        continue;

      for (auto ri = si->begin_relocations(), re = si->end_relocations();
           ri != re; ri.increment(ec)) {
        const coff_relocation *rel = _obj->getCOFFRelocation(ri);
        if ((ec = addRelocationReference(rel, section, _sectionAtoms[section])))
          return ec;
      }
    }
    return error_code::success();
  }

  std::unique_ptr<const llvm::object::COFFObjectFile> _obj;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;

  // A map from symbol to its name. All symbols should be in this map except
  // unnamed ones.
  std::map<const coff_symbol *, StringRef> _symbolName;

  // A map from symbol to its resultant atom.
  std::map<const coff_symbol *, Atom *> _symbolAtom;

  // A map from section to its atoms.
  std::map<const coff_section *, vector<COFFDefinedFileAtom *>> _sectionAtoms;

  mutable llvm::BumpPtrAllocator _alloc;
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
