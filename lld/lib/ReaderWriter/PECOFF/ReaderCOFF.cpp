//===- lib/ReaderWriter/PECOFF/ReaderCOFF.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "lld/Core/Alias.h"
#include "lld/Core/File.h"
#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <system_error>
#include <vector>

#define DEBUG_TYPE "ReaderCOFF"

using lld::pecoff::COFFAbsoluteAtom;
using lld::pecoff::COFFBSSAtom;
using lld::pecoff::COFFDefinedAtom;
using lld::pecoff::COFFDefinedFileAtom;
using lld::pecoff::COFFReference;
using lld::pecoff::COFFUndefinedAtom;
using llvm::object::coff_aux_section_definition;
using llvm::object::coff_aux_weak_external;
using llvm::object::coff_relocation;
using llvm::object::coff_section;
using llvm::object::coff_symbol;
using llvm::support::ulittle32_t;

using namespace lld;

namespace {

class FileCOFF : public File {
private:
  typedef std::vector<llvm::object::COFFSymbolRef> SymbolVectorT;
  typedef std::map<const coff_section *, SymbolVectorT> SectionToSymbolsT;
  typedef std::map<const StringRef, Atom *> SymbolNameToAtomT;
  typedef std::map<const coff_section *, std::vector<COFFDefinedFileAtom *>>
  SectionToAtomsT;

public:
  typedef const std::map<std::string, std::string> StringMap;

  FileCOFF(std::unique_ptr<MemoryBuffer> mb, std::error_code &ec);

  std::error_code parse();
  StringRef getLinkerDirectives() const { return _directives; }
  bool isCompatibleWithSEH() const { return _compatibleWithSEH; }

  const atom_collection<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }

  const atom_collection<UndefinedAtom> &undefined() const override {
    return _undefinedAtoms;
  }

  const atom_collection<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const atom_collection<AbsoluteAtom> &absolute() const override {
    return _absoluteAtoms;
  }

  void addDefinedAtom(AliasAtom *atom) {
    atom->setOrdinal(_ordinal++);
    _definedAtoms._atoms.push_back(atom);
  }

  void addUndefinedSymbol(StringRef sym) {
    _undefinedAtoms._atoms.push_back(new (_alloc) COFFUndefinedAtom(*this, sym));
  }

  mutable llvm::BumpPtrAllocator _alloc;

private:
  std::error_code readSymbolTable(SymbolVectorT &result);

  void createAbsoluteAtoms(const SymbolVectorT &symbols,
                           std::vector<const AbsoluteAtom *> &result);

  std::error_code
  createUndefinedAtoms(const SymbolVectorT &symbols,
                       std::vector<const UndefinedAtom *> &result);

  std::error_code
  createDefinedSymbols(const SymbolVectorT &symbols,
                       std::vector<const DefinedAtom *> &result);

  std::error_code cacheSectionAttributes();
  std::error_code maybeCreateSXDataAtoms();

  std::error_code
  AtomizeDefinedSymbolsInSection(const coff_section *section,
                                 SymbolVectorT &symbols,
                                 std::vector<COFFDefinedFileAtom *> &atoms);

  std::error_code
  AtomizeDefinedSymbols(SectionToSymbolsT &definedSymbols,
                        std::vector<const DefinedAtom *> &definedAtoms);

  std::error_code findAtomAt(const coff_section *section,
                             uint32_t targetAddress,
                             COFFDefinedFileAtom *&result,
                             uint32_t &offsetInAtom);

  std::error_code getAtomBySymbolIndex(uint32_t index, Atom *&ret);

  std::error_code
  addRelocationReference(const coff_relocation *rel,
                         const coff_section *section,
                         const std::vector<COFFDefinedFileAtom *> &atoms);

  std::error_code getSectionContents(StringRef sectionName,
                                     ArrayRef<uint8_t> &result);
  std::error_code getReferenceArch(Reference::KindArch &result);
  std::error_code addRelocationReferenceToAtoms();
  std::error_code findSection(StringRef name, const coff_section *&result);
  StringRef ArrayRefToString(ArrayRef<uint8_t> array);

  std::unique_ptr<const llvm::object::COFFObjectFile> _obj;
  std::unique_ptr<MemoryBuffer> _mb;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;

  // The target type of the object.
  Reference::KindArch _referenceArch;

  // The contents of .drectve section.
  StringRef _directives;

  // True if the object has "@feat.00" symbol.
  bool _compatibleWithSEH;

  // A map from symbol to its name. All symbols should be in this map except
  // unnamed ones.
  std::map<llvm::object::COFFSymbolRef, StringRef> _symbolName;

  // A map from symbol to its resultant atom.
  std::map<llvm::object::COFFSymbolRef, Atom *> _symbolAtom;

  // A map from symbol to its aux symbol.
  std::map<llvm::object::COFFSymbolRef, llvm::object::COFFSymbolRef> _auxSymbol;

  // A map from section to its atoms.
  std::map<const coff_section *, std::vector<COFFDefinedFileAtom *>>
  _sectionAtoms;

  // A set of COMDAT sections.
  std::set<const coff_section *> _comdatSections;

  // A map to get whether the section allows its contents to be merged or not.
  std::map<const coff_section *, DefinedAtom::Merge> _merge;

  // COMDAT associative sections
  std::map<const coff_section *, std::set<const coff_section *>> _association;

  // A sorted map to find an atom from a section and an offset within
  // the section.
  std::map<const coff_section *,
           std::map<uint32_t, std::vector<COFFDefinedAtom *>>>
  _definedAtomLocations;

  uint64_t _ordinal;
};

class BumpPtrStringSaver : public llvm::cl::StringSaver {
public:
  const char *SaveString(const char *str) override {
    size_t len = strlen(str);
    std::lock_guard<std::mutex> lock(_allocMutex);
    char *copy = _alloc.Allocate<char>(len + 1);
    memcpy(copy, str, len + 1);
    return copy;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  std::mutex _allocMutex;
};

// Converts the COFF symbol attribute to the LLD's atom attribute.
Atom::Scope getScope(llvm::object::COFFSymbolRef symbol) {
  switch (symbol.getStorageClass()) {
  case llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL:
    return Atom::scopeGlobal;
  case llvm::COFF::IMAGE_SYM_CLASS_STATIC:
  case llvm::COFF::IMAGE_SYM_CLASS_LABEL:
    return Atom::scopeTranslationUnit;
  }
  llvm_unreachable("Unknown scope");
}

DefinedAtom::ContentType getContentType(const coff_section *section) {
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_CODE)
    return DefinedAtom::typeCode;
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
    return DefinedAtom::typeData;
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
    return DefinedAtom::typeZeroFill;
  return DefinedAtom::typeUnknown;
}

DefinedAtom::ContentPermissions getPermissions(const coff_section *section) {
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_READ &&
      section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_WRITE)
    return DefinedAtom::permRW_;
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_READ &&
      section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_EXECUTE)
    return DefinedAtom::permR_X;
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_READ)
    return DefinedAtom::permR__;
  return DefinedAtom::perm___;
}

/// Returns the alignment of the section. The contents of the section must be
/// aligned by this value in the resulting executable/DLL.
DefinedAtom::Alignment getAlignment(const coff_section *section) {
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_TYPE_NO_PAD)
    return DefinedAtom::Alignment(0);

  // Bit [20:24] contains section alignment information. We need to decrease
  // the value stored by 1 in order to get the real exponent (e.g, ALIGN_1BYTE
  // is 0x00100000, but the exponent should be 0)
  uint32_t characteristics = (section->Characteristics >> 20) & 0xf;

  // If all bits are off, we treat it as if ALIGN_1BYTE was on. The PE/COFF spec
  // does not say anything about this case, but CVTRES.EXE does not set any bit
  // in characteristics[20:24], and its output is intended to be copied to .rsrc
  // section with no padding, so I think doing this is the right thing.
  if (characteristics == 0)
    return DefinedAtom::Alignment(0);

  uint32_t powerOf2 = characteristics - 1;
  return DefinedAtom::Alignment(powerOf2);
}

DefinedAtom::Merge getMerge(const coff_aux_section_definition *auxsym) {
  switch (auxsym->Selection) {
  case llvm::COFF::IMAGE_COMDAT_SELECT_NODUPLICATES:
    return DefinedAtom::mergeNo;
  case llvm::COFF::IMAGE_COMDAT_SELECT_ANY:
    return DefinedAtom::mergeAsWeakAndAddressUsed;
  case llvm::COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
    // TODO: This mapping is wrong. Fix it.
    return DefinedAtom::mergeByContent;
  case llvm::COFF::IMAGE_COMDAT_SELECT_SAME_SIZE:
    return DefinedAtom::mergeSameNameAndSize;
  case llvm::COFF::IMAGE_COMDAT_SELECT_LARGEST:
    return DefinedAtom::mergeByLargestSection;
  case llvm::COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE:
  case llvm::COFF::IMAGE_COMDAT_SELECT_NEWEST:
    // FIXME: These attributes has more complicated semantics than the regular
    // weak symbol. These are mapped to mergeAsWeakAndAddressUsed for now
    // because the core linker does not support them yet. We eventually have
    // to implement them for full COFF support.
    return DefinedAtom::mergeAsWeakAndAddressUsed;
  default:
    llvm_unreachable("Unknown merge type");
  }
}

FileCOFF::FileCOFF(std::unique_ptr<MemoryBuffer> mb, std::error_code &ec)
    : File(mb->getBufferIdentifier(), kindObject), _mb(std::move(mb)),
      _compatibleWithSEH(false), _ordinal(0) {
  auto binaryOrErr = llvm::object::createBinary(_mb->getMemBufferRef());
  if ((ec = binaryOrErr.getError()))
    return;
  std::unique_ptr<llvm::object::Binary> bin = std::move(binaryOrErr.get());

  _obj.reset(dyn_cast<const llvm::object::COFFObjectFile>(bin.get()));
  if (!_obj) {
    ec = make_error_code(llvm::object::object_error::invalid_file_type);
    return;
  }
  bin.release();

  // Read .drectve section if exists.
  ArrayRef<uint8_t> directives;
  if ((ec = getSectionContents(".drectve", directives)))
    return;
  if (!directives.empty())
    _directives = ArrayRefToString(directives);
}

std::error_code FileCOFF::parse() {
  if (std::error_code ec = getReferenceArch(_referenceArch))
    return ec;

  // Read the symbol table and atomize them if possible. Defined atoms
  // cannot be atomized in one pass, so they will be not be atomized but
  // added to symbolToAtom.
  SymbolVectorT symbols;
  if (std::error_code ec = readSymbolTable(symbols))
    return ec;

  createAbsoluteAtoms(symbols, _absoluteAtoms._atoms);
  if (std::error_code ec =
          createUndefinedAtoms(symbols, _undefinedAtoms._atoms))
    return ec;
  if (std::error_code ec = createDefinedSymbols(symbols, _definedAtoms._atoms))
    return ec;
  if (std::error_code ec = addRelocationReferenceToAtoms())
    return ec;
  if (std::error_code ec = maybeCreateSXDataAtoms())
    return ec;
  return std::error_code();
}

/// Iterate over the symbol table to retrieve all symbols.
std::error_code
FileCOFF::readSymbolTable(SymbolVectorT &result) {
  for (uint32_t i = 0, e = _obj->getNumberOfSymbols(); i != e; ++i) {
    // Retrieve the symbol.
    ErrorOr<llvm::object::COFFSymbolRef> sym = _obj->getSymbol(i);
    StringRef name;
    if (std::error_code ec = sym.getError())
      return ec;
    if (sym->getSectionNumber() == llvm::COFF::IMAGE_SYM_DEBUG)
      goto next;
    result.push_back(*sym);

    if (std::error_code ec = _obj->getSymbolName(*sym, name))
      return ec;

    // Existence of the symbol @feat.00 indicates that object file is compatible
    // with Safe Exception Handling.
    if (name == "@feat.00") {
      _compatibleWithSEH = true;
      goto next;
    }

    // Cache the name.
    _symbolName[*sym] = name;

    // Symbol may be followed by auxiliary symbol table records. The aux
    // record can be in any format, but the size is always the same as the
    // regular symbol. The aux record supplies additional information for the
    // standard symbol. We do not interpret the aux record here, but just
    // store it to _auxSymbol.
    if (sym->getNumberOfAuxSymbols() > 0) {
      ErrorOr<llvm::object::COFFSymbolRef> aux = _obj->getSymbol(i + 1);
      if (std::error_code ec = aux.getError())
        return ec;
      _auxSymbol[*sym] = *aux;
    }
  next:
    i += sym->getNumberOfAuxSymbols();
  }
  return std::error_code();
}

/// Create atoms for the absolute symbols.
void FileCOFF::createAbsoluteAtoms(const SymbolVectorT &symbols,
                                   std::vector<const AbsoluteAtom *> &result) {
  for (llvm::object::COFFSymbolRef sym : symbols) {
    if (sym.getSectionNumber() != llvm::COFF::IMAGE_SYM_ABSOLUTE)
      continue;
    auto *atom = new (_alloc) COFFAbsoluteAtom(*this, _symbolName[sym],
                                               getScope(sym), sym.getValue());

    result.push_back(atom);
    _symbolAtom[sym] = atom;
  }
}

/// Create atoms for the undefined symbols. This code is bit complicated
/// because it supports "weak externals" mechanism of COFF. If an undefined
/// symbol (sym1) has auxiliary data, the data contains a symbol table index
/// at which the "second symbol" (sym2) for sym1 exists. If sym1 is resolved,
/// it's linked normally. If not, sym1 is resolved as if it has sym2's
/// name. This relationship between sym1 and sym2 is represented using
/// fallback mechanism of undefined symbol.
std::error_code
FileCOFF::createUndefinedAtoms(const SymbolVectorT &symbols,
                               std::vector<const UndefinedAtom *> &result) {
  // Sort out undefined symbols from all symbols.
  std::set<llvm::object::COFFSymbolRef> undefines;
  std::map<llvm::object::COFFSymbolRef, llvm::object::COFFSymbolRef>
      weakExternal;
  for (llvm::object::COFFSymbolRef sym : symbols) {
    if (sym.getSectionNumber() != llvm::COFF::IMAGE_SYM_UNDEFINED)
      continue;
    undefines.insert(sym);

    // Create a mapping from sym1 to sym2, if the undefined symbol has
    // auxiliary data.
    auto iter = _auxSymbol.find(sym);
    if (iter == _auxSymbol.end())
      continue;
    const coff_aux_weak_external *aux =
        reinterpret_cast<const coff_aux_weak_external *>(
            iter->second.getRawPtr());
    ErrorOr<llvm::object::COFFSymbolRef> sym2 = _obj->getSymbol(aux->TagIndex);
    if (std::error_code ec = sym2.getError())
      return ec;
    weakExternal[sym] = *sym2;
  }

  // Sort out sym1s from sym2s. Sym2s shouldn't be added to the undefined atom
  // list because they shouldn't be resolved unless sym1 is failed to
  // be resolved.
  for (auto i : weakExternal)
    undefines.erase(i.second);

  // Create atoms for the undefined symbols.
  for (llvm::object::COFFSymbolRef sym : undefines) {
    // If the symbol has sym2, create an undefiend atom for sym2, so that we
    // can pass it as a fallback atom.
    UndefinedAtom *fallback = nullptr;
    auto iter = weakExternal.find(sym);
    if (iter != weakExternal.end()) {
      llvm::object::COFFSymbolRef sym2 = iter->second;
      fallback = new (_alloc) COFFUndefinedAtom(*this, _symbolName[sym2]);
      _symbolAtom[sym2] = fallback;
    }

    // Create an atom for the symbol.
    auto *atom =
        new (_alloc) COFFUndefinedAtom(*this, _symbolName[sym], fallback);
    result.push_back(atom);
    _symbolAtom[sym] = atom;
  }
  return std::error_code();
}

/// Create atoms for the defined symbols. This pass is a bit complicated than
/// the other two, because in order to create the atom for the defined symbol
/// we need to know the adjacent symbols.
std::error_code
FileCOFF::createDefinedSymbols(const SymbolVectorT &symbols,
                               std::vector<const DefinedAtom *> &result) {
  // A defined atom can be merged if its section attribute allows its contents
  // to be merged. In COFF, it's not very easy to get the section attribute
  // for the symbol, so scan all sections in advance and cache the attributes
  // for later use.
  if (std::error_code ec = cacheSectionAttributes())
    return ec;

  // Filter non-defined atoms, and group defined atoms by its section.
  SectionToSymbolsT definedSymbols;
  for (llvm::object::COFFSymbolRef sym : symbols) {
    // A symbol with section number 0 and non-zero value represents a common
    // symbol. The MS COFF spec did not give a definition of what the common
    // symbol is. We should probably follow ELF's definition shown below.
    //
    // - If one object file has a common symbol and another has a definition,
    //   the common symbol is treated as an undefined reference.
    // - If there is no definition for a common symbol, the program linker
    //   acts as though it saw a definition initialized to zero of the
    //   appropriate size.
    // - Two object files may have common symbols of
    //   different sizes, in which case the program linker will use the
    //   largest size.
    //
    // FIXME: We are currently treating the common symbol as a normal
    // mergeable atom. Implement the above semantcis.
    if (sym.getSectionNumber() == llvm::COFF::IMAGE_SYM_UNDEFINED &&
        sym.getValue() > 0) {
      StringRef name = _symbolName[sym];
      uint32_t size = sym.getValue();
      auto *atom = new (_alloc)
          COFFBSSAtom(*this, name, getScope(sym), DefinedAtom::permRW_,
                      DefinedAtom::mergeAsWeakAndAddressUsed, size, _ordinal++);

      // Common symbols should be aligned on natural boundaries with the maximum
      // of 32 byte. It's not documented anywhere, but it's what MSVC link.exe
      // seems to be doing.
      uint64_t alignment = std::min((uint64_t)32, llvm::NextPowerOf2(size));
      atom->setAlignment(
          DefinedAtom::Alignment(llvm::countTrailingZeros(alignment)));
      result.push_back(atom);
      continue;
    }

    // Skip if it's not for defined atom.
    if (sym.getSectionNumber() == llvm::COFF::IMAGE_SYM_DEBUG ||
        sym.getSectionNumber() == llvm::COFF::IMAGE_SYM_ABSOLUTE ||
        sym.getSectionNumber() == llvm::COFF::IMAGE_SYM_UNDEFINED)
      continue;

    const coff_section *sec;
    if (std::error_code ec = _obj->getSection(sym.getSectionNumber(), sec))
      return ec;
    assert(sec && "SectionIndex > 0, Sec must be non-null!");

    // Skip if it's a section symbol for a COMDAT section. A section symbol
    // has the name of the section and value 0. A translation unit may contain
    // multiple COMDAT sections whose section name are the same. We don't want
    // to make atoms for them as they would become duplicate symbols.
    StringRef sectionName;
    if (std::error_code ec = _obj->getSectionName(sec, sectionName))
      return ec;
    if (_symbolName[sym] == sectionName && sym.getValue() == 0 &&
        _merge[sec] != DefinedAtom::mergeNo)
      continue;

    uint8_t sc = sym.getStorageClass();
    if (sc != llvm::COFF::IMAGE_SYM_CLASS_EXTERNAL &&
        sc != llvm::COFF::IMAGE_SYM_CLASS_STATIC &&
        sc != llvm::COFF::IMAGE_SYM_CLASS_FUNCTION &&
        sc != llvm::COFF::IMAGE_SYM_CLASS_LABEL) {
      llvm::errs() << "Unable to create atom for: " << _symbolName[sym] << " ("
                   << static_cast<int>(sc) << ")\n";
      return llvm::object::object_error::parse_failed;
    }

    definedSymbols[sec].push_back(sym);
  }

  // Atomize the defined symbols.
  if (std::error_code ec = AtomizeDefinedSymbols(definedSymbols, result))
    return ec;

  return std::error_code();
}

// Cache the COMDAT attributes, which indicate whether the symbols in the
// section can be merged or not.
std::error_code FileCOFF::cacheSectionAttributes() {
  // The COMDAT section attribute is not an attribute of coff_section, but is
  // stored in the auxiliary symbol for the first symbol referring a COMDAT
  // section. It feels to me that it's unnecessarily complicated, but this is
  // how COFF works.
  for (auto i : _auxSymbol) {
    // Read a section from the file
    llvm::object::COFFSymbolRef sym = i.first;
    if (sym.getSectionNumber() == llvm::COFF::IMAGE_SYM_ABSOLUTE ||
        sym.getSectionNumber() == llvm::COFF::IMAGE_SYM_UNDEFINED)
      continue;

    const coff_section *sec;
    if (std::error_code ec = _obj->getSection(sym.getSectionNumber(), sec))
      return ec;
    const coff_aux_section_definition *aux =
        reinterpret_cast<const coff_aux_section_definition *>(
            i.second.getRawPtr());

    if (sec->Characteristics & llvm::COFF::IMAGE_SCN_LNK_COMDAT) {
      // Read aux symbol data.
      _comdatSections.insert(sec);
      _merge[sec] = getMerge(aux);
    }

    // Handle associative sections.
    if (aux->Selection == llvm::COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) {
      const coff_section *parent;
      if (std::error_code ec =
              _obj->getSection(aux->getNumber(sym.isBigObj()), parent))
        return ec;
      _association[parent].insert(sec);
    }
  }

  // The sections that does not have auxiliary symbol are regular sections, in
  // which symbols are not allowed to be merged.
  for (const auto &section : _obj->sections()) {
    const coff_section *sec = _obj->getCOFFSection(section);
    if (!_merge.count(sec))
      _merge[sec] = DefinedAtom::mergeNo;
  }
  return std::error_code();
}

/// Atomize \p symbols and append the results to \p atoms. The symbols are
/// assumed to have been defined in the \p section.
std::error_code FileCOFF::AtomizeDefinedSymbolsInSection(
    const coff_section *section, SymbolVectorT &symbols,
    std::vector<COFFDefinedFileAtom *> &atoms) {
  // Sort symbols by position.
  std::stable_sort(
      symbols.begin(), symbols.end(),
      // For some reason MSVC fails to allow the lambda in this context with a
      // "illegal use of local type in type instantiation". MSVC is clearly
      // wrong here. Force a conversion to function pointer to work around.
      static_cast<bool (*)(llvm::object::COFFSymbolRef,
                           llvm::object::COFFSymbolRef)>(
          [](llvm::object::COFFSymbolRef a, llvm::object::COFFSymbolRef b)
              -> bool { return a.getValue() < b.getValue(); }));

  StringRef sectionName;
  if (std::error_code ec = _obj->getSectionName(section, sectionName))
    return ec;

  // BSS section does not have contents. If this is the BSS section, create
  // COFFBSSAtom instead of COFFDefinedAtom.
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) {
    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      llvm::object::COFFSymbolRef sym = *si;
      uint32_t size = (si + 1 == se) ? section->SizeOfRawData - sym.getValue()
                                     : si[1].getValue() - sym.getValue();
      auto *atom = new (_alloc) COFFBSSAtom(
          *this, _symbolName[sym], getScope(sym), getPermissions(section),
          DefinedAtom::mergeAsWeakAndAddressUsed, size, _ordinal++);
      atoms.push_back(atom);
      _symbolAtom[sym] = atom;
    }
    return std::error_code();
  }

  ArrayRef<uint8_t> secData;
  if (std::error_code ec = _obj->getSectionContents(section, secData))
    return ec;

  // A section with IMAGE_SCN_LNK_{INFO,REMOVE} attribute will never become
  // a part of the output image. That's what the COFF spec says.
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_LNK_INFO ||
      section->Characteristics & llvm::COFF::IMAGE_SCN_LNK_REMOVE)
    return std::error_code();

  // Supporting debug info needs more work than just linking and combining
  // .debug sections. We don't support it yet. Let's discard .debug sections at
  // the very beginning of the process so that we don't spend time on linking
  // blobs that nobody would understand.
  if ((section->Characteristics & llvm::COFF::IMAGE_SCN_MEM_DISCARDABLE) &&
      (sectionName == ".debug" || sectionName.startswith(".debug$"))) {
    return std::error_code();
  }

  DefinedAtom::ContentType type = getContentType(section);
  DefinedAtom::ContentPermissions perms = getPermissions(section);
  bool isComdat = (_comdatSections.count(section) == 1);

  // Create an atom for the entire section.
  if (symbols.empty()) {
    ArrayRef<uint8_t> data(secData.data(), secData.size());
    auto *atom = new (_alloc) COFFDefinedAtom(
        *this, "", sectionName, Atom::scopeTranslationUnit, type, isComdat,
        perms, _merge[section], data, _ordinal++);
    atoms.push_back(atom);
    _definedAtomLocations[section][0].push_back(atom);
    return std::error_code();
  }

  // Create an unnamed atom if the first atom isn't at the start of the
  // section.
  if (symbols[0].getValue() != 0) {
    uint64_t size = symbols[0].getValue();
    ArrayRef<uint8_t> data(secData.data(), size);
    auto *atom = new (_alloc) COFFDefinedAtom(
        *this, "", sectionName, Atom::scopeTranslationUnit, type, isComdat,
        perms, _merge[section], data, _ordinal++);
    atoms.push_back(atom);
    _definedAtomLocations[section][0].push_back(atom);
  }

  for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
    const uint8_t *start = secData.data() + si->getValue();
    // if this is the last symbol, take up the remaining data.
    const uint8_t *end = (si + 1 == se) ? secData.data() + secData.size()
                                        : secData.data() + (si + 1)->getValue();
    ArrayRef<uint8_t> data(start, end);
    auto *atom = new (_alloc) COFFDefinedAtom(
        *this, _symbolName[*si], sectionName, getScope(*si), type, isComdat,
        perms, _merge[section], data, _ordinal++);
    atoms.push_back(atom);
    _symbolAtom[*si] = atom;
    _definedAtomLocations[section][si->getValue()].push_back(atom);
  }
  return std::error_code();
}

std::error_code FileCOFF::AtomizeDefinedSymbols(
    SectionToSymbolsT &definedSymbols,
    std::vector<const DefinedAtom *> &definedAtoms) {
  // For each section, make atoms for all the symbols defined in the
  // section, and append the atoms to the result objects.
  for (auto &i : definedSymbols) {
    const coff_section *section = i.first;
    SymbolVectorT &symbols = i.second;
    std::vector<COFFDefinedFileAtom *> atoms;
    if (std::error_code ec =
            AtomizeDefinedSymbolsInSection(section, symbols, atoms))
      return ec;

    // Set alignment to the first atom so that the section contents
    // will be aligned as specified by the object section header.
    if (atoms.size() > 0)
      atoms[0]->setAlignment(getAlignment(section));

    // Connect atoms with layout-before/layout-after edges.
    connectAtomsWithLayoutEdge(atoms);

    for (COFFDefinedFileAtom *atom : atoms) {
      _sectionAtoms[section].push_back(atom);
      definedAtoms.push_back(atom);
    }
  }

  // A COMDAT section with SELECT_ASSOCIATIVE attribute refer to other
  // section. If the referred section is linked to a binary, the
  // referring section needs to be linked too. A typical use case of
  // this attribute is a static initializer; a parent is a comdat BSS
  // section, and a child is a static initializer code for the data.
  //
  // We add referring section contents to the referred section's
  // associate list, so that Resolver takes care of them.
  for (auto i : _association) {
    const coff_section *parent = i.first;
    const std::set<const coff_section *> &childSections = i.second;
    assert(_sectionAtoms[parent].size() > 0);

    COFFDefinedFileAtom *p = _sectionAtoms[parent][0];
    for (const coff_section *sec : childSections) {
      if (_sectionAtoms.count(sec)) {
        assert(_sectionAtoms[sec].size() > 0);
        p->addAssociate(_sectionAtoms[sec][0]);
      }
    }
  }

  return std::error_code();
}

/// Find the atom that is at \p targetAddress in \p section.
std::error_code FileCOFF::findAtomAt(const coff_section *section,
                                     uint32_t targetAddress,
                                     COFFDefinedFileAtom *&result,
                                     uint32_t &offsetInAtom) {
  for (auto i : _definedAtomLocations[section]) {
    uint32_t atomAddress = i.first;
    std::vector<COFFDefinedAtom *> &atomsAtSameLocation = i.second;
    COFFDefinedAtom *atom = atomsAtSameLocation.back();
    if (atomAddress <= targetAddress &&
        targetAddress < atomAddress + atom->size()) {
      result = atom;
      offsetInAtom = targetAddress - atomAddress;
      return std::error_code();
    }
  }
  // Relocation target is out of range
  return llvm::object::object_error::parse_failed;
}

/// Find the atom for the symbol that was at the \p index in the symbol
/// table.
std::error_code FileCOFF::getAtomBySymbolIndex(uint32_t index, Atom *&ret) {
  ErrorOr<llvm::object::COFFSymbolRef> symbol = _obj->getSymbol(index);
  if (std::error_code ec = symbol.getError())
    return ec;
  ret = _symbolAtom[*symbol];
  assert(ret);
  return std::error_code();
}

/// Add relocation information to an atom based on \p rel. \p rel is an
/// relocation entry for the \p section, and \p atoms are all the atoms
/// defined in the \p section.
std::error_code FileCOFF::addRelocationReference(
    const coff_relocation *rel, const coff_section *section,
    const std::vector<COFFDefinedFileAtom *> &atoms) {
  assert(atoms.size() > 0);
  // The address of the item which relocation is applied. Section's
  // VirtualAddress needs to be added for historical reasons, but the value
  // is usually just zero, so adding it is usually no-op.
  uint32_t itemAddress = rel->VirtualAddress + section->VirtualAddress;

  Atom *targetAtom = nullptr;
  if (std::error_code ec =
          getAtomBySymbolIndex(rel->SymbolTableIndex, targetAtom))
    return ec;

  COFFDefinedFileAtom *atom;
  uint32_t offsetInAtom;
  if (std::error_code ec = findAtomAt(section, itemAddress, atom, offsetInAtom))
    return ec;
  atom->addReference(std::unique_ptr<COFFReference>(
      new COFFReference(targetAtom, offsetInAtom, rel->Type, _referenceArch)));
  return std::error_code();
}

// Read section contents.
std::error_code FileCOFF::getSectionContents(StringRef sectionName,
                                             ArrayRef<uint8_t> &result) {
  const coff_section *section = nullptr;
  if (std::error_code ec = findSection(sectionName, section))
    return ec;
  if (!section)
    return std::error_code();
  if (std::error_code ec = _obj->getSectionContents(section, result))
    return ec;
  return std::error_code();
}

/// Returns the target machine type of the current object file.
std::error_code FileCOFF::getReferenceArch(Reference::KindArch &result) {
  switch (_obj->getMachine()) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    result = Reference::KindArch::x86;
    return std::error_code();
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    result = Reference::KindArch::x86_64;
    return std::error_code();
  case llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN:
    result = Reference::KindArch::all;
    return std::error_code();
  }
  llvm::errs() << "Unsupported machine type: 0x"
               << llvm::utohexstr(_obj->getMachine()) << '\n';
  return llvm::object::object_error::parse_failed;
}

/// Add relocation information to atoms.
std::error_code FileCOFF::addRelocationReferenceToAtoms() {
  // Relocation entries are defined for each section.
  for (const auto &sec : _obj->sections()) {
    const coff_section *section = _obj->getCOFFSection(sec);

    // Skip there's no atom for the section. Currently we do not create any
    // atoms for some sections, such as "debug$S", and such sections need to
    // be skipped here too.
    if (_sectionAtoms.find(section) == _sectionAtoms.end())
      continue;

    for (const auto &reloc : sec.relocations()) {
      const coff_relocation *rel = _obj->getCOFFRelocation(reloc);
      if (auto ec =
              addRelocationReference(rel, section, _sectionAtoms[section]))
        return ec;
    }
  }
  return std::error_code();
}

// Read .sxdata section if exists. .sxdata is a x86-only section that contains a
// vector of symbol offsets. The symbols pointed by this section are SEH handler
// functions contained in the same object file. The linker needs to construct a
// SEH table and emit it to executable.
//
// On x86, exception handler addresses are in stack, so they are vulnerable to
// stack overflow attack. In order to protect against it, Windows runtime uses
// the SEH table to check if a SEH handler address in stack is a real address of
// a handler created by compiler.
//
// What we want to emit from the linker is a vector of SEH handler VAs, but here
// we have a vector of offsets to the symbol table. So we convert the latter to
// the former.
std::error_code FileCOFF::maybeCreateSXDataAtoms() {
  ArrayRef<uint8_t> sxdata;
  if (std::error_code ec = getSectionContents(".sxdata", sxdata))
    return ec;
  if (sxdata.empty())
    return std::error_code();

  std::vector<uint8_t> atomContent =
      *new (_alloc) std::vector<uint8_t>((size_t)sxdata.size());
  auto *atom = new (_alloc) COFFDefinedAtom(
      *this, "", ".sxdata", Atom::scopeTranslationUnit, DefinedAtom::typeData,
      false /*isComdat*/, DefinedAtom::permR__, DefinedAtom::mergeNo,
      atomContent, _ordinal++);

  const ulittle32_t *symbolIndex =
      reinterpret_cast<const ulittle32_t *>(sxdata.data());
  int numSymbols = sxdata.size() / sizeof(uint32_t);

  for (int i = 0; i < numSymbols; ++i) {
    Atom *handlerFunc;
    if (std::error_code ec = getAtomBySymbolIndex(symbolIndex[i], handlerFunc))
      return ec;
    int offsetInAtom = i * sizeof(uint32_t);

    uint16_t rtype;
    switch (_obj->getMachine()) {
    case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
      rtype = llvm::COFF::IMAGE_REL_AMD64_ADDR32;
      break;
    case llvm::COFF::IMAGE_FILE_MACHINE_I386:
      rtype = llvm::COFF::IMAGE_REL_I386_DIR32;
      break;
    default:
      llvm_unreachable("unsupported machine type");
    }

    atom->addReference(std::unique_ptr<COFFReference>(new COFFReference(
        handlerFunc, offsetInAtom, rtype, _referenceArch)));
  }

  _definedAtoms._atoms.push_back(atom);
  return std::error_code();
}

/// Find a section by name.
std::error_code FileCOFF::findSection(StringRef name,
                                      const coff_section *&result) {
  for (const auto &sec : _obj->sections()) {
    const coff_section *section = _obj->getCOFFSection(sec);
    StringRef sectionName;
    if (auto ec = _obj->getSectionName(section, sectionName))
      return ec;
    if (sectionName == name) {
      result = section;
      return std::error_code();
    }
  }
  // Section was not found, but it's not an error. This method returns
  // an error only when there's a read error.
  return std::error_code();
}

// Convert ArrayRef<uint8_t> to std::string. The array contains a string which
// may not be terminated by NUL.
StringRef FileCOFF::ArrayRefToString(ArrayRef<uint8_t> array) {
  // Skip the UTF-8 byte marker if exists. The contents of .drectve section
  // is, according to the Microsoft PE/COFF spec, encoded as ANSI or UTF-8
  // with the BOM marker.
  //
  // FIXME: I think "ANSI" in the spec means Windows-1252 encoding, which is a
  // superset of ASCII. We need to convert it to UTF-8.
  if (array.size() >= 3 && array[0] == 0xEF && array[1] == 0xBB &&
      array[2] == 0xBF) {
    array = array.slice(3);
  }

  if (array.empty())
    return "";

  size_t len = 0;
  size_t e = array.size();
  while (len < e && array[len] != '\0')
    ++len;
  std::string *contents =
      new (_alloc) std::string(reinterpret_cast<const char *>(&array[0]), len);
  return StringRef(*contents).trim();
}

class COFFObjectReader : public Reader {
public:
  COFFObjectReader(PECOFFLinkingContext &ctx) : _ctx(ctx) {}

  bool canParse(file_magic magic, StringRef ext,
                const MemoryBuffer &) const override {
    return magic == llvm::sys::fs::file_magic::coff_object;
  }

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const Registry &registry,
            std::vector<std::unique_ptr<File>> &result) const override {
    // Parse the memory buffer as PECOFF file.
    const char *mbName = mb->getBufferIdentifier();
    std::error_code ec;
    std::unique_ptr<FileCOFF> file(new FileCOFF(std::move(mb), ec));
    if (ec)
      return ec;

    // The set to contain the symbols specified as arguments of
    // /INCLUDE option.
    std::set<StringRef> undefinedSymbols;

    // Interpret .drectve section if the section has contents.
    StringRef directives = file->getLinkerDirectives();
    if (!directives.empty())
      if (std::error_code ec = handleDirectiveSection(
              directives, &undefinedSymbols))
        return ec;

    if (std::error_code ec = file->parse())
      return ec;

    // Check for /SAFESEH.
    if (_ctx.requireSEH() && !file->isCompatibleWithSEH()) {
      llvm::errs() << "/SAFESEH is specified, but " << mbName
                   << " is not compatible with SEH.\n";
      return llvm::object::object_error::parse_failed;
    }

    // Add /INCLUDE'ed symbols to the file as if they existed in the
    // file as undefined symbols.
    for (StringRef sym : undefinedSymbols)
      file->addUndefinedSymbol(sym);

    // One can define alias symbols using /alternatename:<sym>=<sym> option.
    // The mapping for /alternatename is in the context object. This helper
    // function iterate over defined atoms and create alias atoms if needed.
    createAlternateNameAtoms(*file);

    // Acquire the mutex to mutate _ctx.
    std::lock_guard<std::recursive_mutex> lock(_ctx.getMutex());

    // In order to emit SEH table, all input files need to be compatible with
    // SEH. Disable SEH if the file being read is not compatible.
    if (!file->isCompatibleWithSEH())
      _ctx.setSafeSEH(false);

    if (_ctx.deadStrip())
      for (StringRef sym : undefinedSymbols)
        _ctx.addDeadStripRoot(sym);

    result.push_back(std::move(file));
    return std::error_code();
  }

private:
  // Interpret the contents of .drectve section. If exists, the section contains
  // a string containing command line options. The linker is expected to
  // interpret the options as if they were given via the command line.
  //
  // The section mainly contains /defaultlib (-l in Unix), but can contain any
  // options as long as they are valid.
  std::error_code handleDirectiveSection(StringRef directives,
                                         std::set<StringRef> *undefinedSymbols) const {
    DEBUG(llvm::dbgs() << ".drectve: " << directives << "\n");

    // Split the string into tokens, as the shell would do for argv.
    SmallVector<const char *, 16> tokens;
    tokens.push_back("link"); // argv[0] is the command name. Will be ignored.
    llvm::cl::TokenizeWindowsCommandLine(directives, _stringSaver, tokens);
    tokens.push_back(nullptr);

    // Calls the command line parser to interpret the token string as if they
    // were given via the command line.
    int argc = tokens.size() - 1;
    const char **argv = &tokens[0];
    std::string errorMessage;
    llvm::raw_string_ostream stream(errorMessage);
    bool parseFailed = !WinLinkDriver::parse(argc, argv, _ctx, stream,
                                             /*isDirective*/ true,
                                             undefinedSymbols);
    stream.flush();
    // Print error message if error.
    if (parseFailed) {
      auto msg = Twine("Failed to parse '") + directives + "'\n"
        + "Reason: " + errorMessage;
      return make_dynamic_error_code(msg);
    }
    if (!errorMessage.empty()) {
      llvm::errs() << "lld warning: " << errorMessage << "\n";
    }
    return std::error_code();
  }

  AliasAtom *createAlias(FileCOFF &file, StringRef name,
                         const DefinedAtom *target) const {
    AliasAtom *alias = new (file._alloc) AliasAtom(file, name);
    alias->addReference(Reference::KindNamespace::all, Reference::KindArch::all,
                        Reference::kindLayoutAfter, 0, target, 0);
    alias->setMerge(DefinedAtom::mergeAsWeak);
    if (target->contentType() == DefinedAtom::typeCode)
      alias->setDeadStrip(DefinedAtom::deadStripNever);
    return alias;
  }

  // Iterates over defined atoms and create alias atoms if needed.
  void createAlternateNameAtoms(FileCOFF &file) const {
    std::vector<AliasAtom *> aliases;
    for (const DefinedAtom *atom : file.defined()) {
      auto it = _ctx.alternateNames().find(atom->name());
      if (it != _ctx.alternateNames().end())
        aliases.push_back(createAlias(file, it->second, atom));
    }
    for (AliasAtom *alias : aliases) {
      file.addDefinedAtom(alias);
    }
  }

  PECOFFLinkingContext &_ctx;
  mutable BumpPtrStringSaver _stringSaver;
};

using namespace llvm::COFF;

const Registry::KindStrings kindStringsI386[] = {
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_ABSOLUTE),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_DIR16),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_REL16),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_DIR32),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_DIR32NB),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_SEG12),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_SECTION),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_SECREL),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_TOKEN),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_SECREL7),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_I386_REL32),
    LLD_KIND_STRING_END};

const Registry::KindStrings kindStringsAMD64[] = {
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_ABSOLUTE),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_ADDR64),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_ADDR32),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_ADDR32NB),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_REL32),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_REL32_1),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_REL32_2),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_REL32_3),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_REL32_4),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_REL32_5),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_SECTION),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_SECREL),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_SECREL7),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_TOKEN),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_SREL32),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_PAIR),
    LLD_KIND_STRING_ENTRY(IMAGE_REL_AMD64_SSPAN32),
    LLD_KIND_STRING_END};

} // end namespace anonymous

namespace lld {

void Registry::addSupportCOFFObjects(PECOFFLinkingContext &ctx) {
  add(std::unique_ptr<Reader>(new COFFObjectReader(ctx)));
  addKindTable(Reference::KindNamespace::COFF, Reference::KindArch::x86,
               kindStringsI386);
  addKindTable(Reference::KindNamespace::COFF, Reference::KindArch::x86_64,
               kindStringsAMD64);
}

}
