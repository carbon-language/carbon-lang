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
#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <map>
#include <set>
#include <vector>

using std::vector;
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

using namespace lld;

namespace {

class FileCOFF : public File {
private:
  typedef vector<const coff_symbol *> SymbolVectorT;
  typedef std::map<const coff_section *, SymbolVectorT> SectionToSymbolsT;
  typedef std::map<const StringRef, Atom *> SymbolNameToAtomT;
  typedef std::map<const coff_section *, vector<COFFDefinedFileAtom *>>
  SectionToAtomsT;

public:
  typedef const std::map<std::string, std::string> StringMap;

  FileCOFF(std::unique_ptr<MemoryBuffer> mb, error_code &ec);

  error_code parse(StringMap &altNames);
  StringRef getLinkerDirectives() const { return _directives; }
  bool isCompatibleWithSEH() const { return _compatibleWithSEH; }

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

private:
  error_code readSymbolTable(vector<const coff_symbol *> &result);

  void createAbsoluteAtoms(const SymbolVectorT &symbols,
                           vector<const AbsoluteAtom *> &result);

  error_code createUndefinedAtoms(const SymbolVectorT &symbols,
                                  vector<const UndefinedAtom *> &result);

  error_code createDefinedSymbols(const SymbolVectorT &symbols,
                                  StringMap &altNames,
                                  vector<const DefinedAtom *> &result);

  error_code cacheSectionAttributes();

  error_code
  AtomizeDefinedSymbolsInSection(const coff_section *section,
                                 StringMap &altNames,
                                 vector<const coff_symbol *> &symbols,
                                 vector<COFFDefinedFileAtom *> &atoms);

  error_code AtomizeDefinedSymbols(SectionToSymbolsT &definedSymbols,
                                   StringMap &altNames,
                                   vector<const DefinedAtom *> &definedAtoms);

  error_code findAtomAt(const coff_section *section, uint32_t targetAddress,
                        COFFDefinedFileAtom *&result, uint32_t &offsetInAtom);

  error_code getAtomBySymbolIndex(uint32_t index, Atom *&ret);

  error_code addRelocationReference(const coff_relocation *rel,
                                    const coff_section *section,
                                    const vector<COFFDefinedFileAtom *> &atoms);

  error_code getReferenceArch(Reference::KindArch &result);
  error_code addRelocationReferenceToAtoms();
  error_code findSection(StringRef name, const coff_section *&result);
  StringRef ArrayRefToString(ArrayRef<uint8_t> array);

  std::unique_ptr<const llvm::object::COFFObjectFile> _obj;
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
  std::map<const coff_symbol *, StringRef> _symbolName;

  // A map from symbol to its resultant atom.
  std::map<const coff_symbol *, Atom *> _symbolAtom;

  // A map from symbol to its aux symbol.
  std::map<const coff_symbol *, const coff_symbol *> _auxSymbol;

  // A map from section to its atoms.
  std::map<const coff_section *, vector<COFFDefinedFileAtom *> > _sectionAtoms;

  // A set of COMDAT sections.
  std::set<const coff_section *> _comdatSections;

  // A map to get whether the section allows its contents to be merged or not.
  std::map<const coff_section *, DefinedAtom::Merge> _merge;

  // A sorted map to find an atom from a section and an offset within
  // the section.
  std::map<const coff_section *,
           std::map<uint32_t, std::vector<COFFDefinedAtom *>>>
  _definedAtomLocations;

  mutable llvm::BumpPtrAllocator _alloc;
  uint64_t _ordinal;
};

class BumpPtrStringSaver : public llvm::cl::StringSaver {
public:
  virtual const char *SaveString(const char *str) {
    size_t len = strlen(str);
    char *copy = _alloc.Allocate<char>(len + 1);
    memcpy(copy, str, len + 1);
    return copy;
  }

private:
  llvm::BumpPtrAllocator _alloc;
};

// Converts the COFF symbol attribute to the LLD's atom attribute.
Atom::Scope getScope(const coff_symbol *symbol) {
  switch (symbol->StorageClass) {
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
  case llvm::COFF::IMAGE_COMDAT_SELECT_SAME_SIZE:
  case llvm::COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
  case llvm::COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE:
  case llvm::COFF::IMAGE_COMDAT_SELECT_LARGEST:
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

FileCOFF::FileCOFF(std::unique_ptr<MemoryBuffer> mb, error_code &ec)
    : File(mb->getBufferIdentifier(), kindObject), _compatibleWithSEH(false),
      _ordinal(0) {
  auto binaryOrErr = llvm::object::createBinary(mb.release());
  if ((ec = binaryOrErr.getError()))
    return;
  OwningPtr<llvm::object::Binary> bin(binaryOrErr.get());

  _obj.reset(dyn_cast<const llvm::object::COFFObjectFile>(bin.get()));
  if (!_obj) {
    ec = make_error_code(llvm::object::object_error::invalid_file_type);
    return;
  }
  bin.take();

  // Read .drectve section if exists.
  const coff_section *section = nullptr;
  if ((ec = findSection(".drectve", section)))
    return;
  if (section != nullptr) {
    ArrayRef<uint8_t> contents;
    if ((ec = _obj->getSectionContents(section, contents)))
      return;
    _directives = ArrayRefToString(contents);
  }
}

error_code FileCOFF::parse(StringMap &altNames) {
  if (error_code ec = getReferenceArch(_referenceArch))
    return ec;

  // Read the symbol table and atomize them if possible. Defined atoms
  // cannot be atomized in one pass, so they will be not be atomized but
  // added to symbolToAtom.
  SymbolVectorT symbols;
  if (error_code ec = readSymbolTable(symbols))
    return ec;

  createAbsoluteAtoms(symbols, _absoluteAtoms._atoms);
  if (error_code ec = createUndefinedAtoms(symbols, _undefinedAtoms._atoms))
    return ec;
  if (error_code ec = createDefinedSymbols(symbols, altNames, _definedAtoms._atoms))
    return ec;
  if (error_code ec = addRelocationReferenceToAtoms())
    return ec;
  return error_code::success();
}

/// Iterate over the symbol table to retrieve all symbols.
error_code FileCOFF::readSymbolTable(vector<const coff_symbol *> &result) {
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

    StringRef name;
    if (error_code ec = _obj->getSymbolName(sym, name))
      return ec;

    // Existence of the symbol @feat.00 indicates that object file is compatible
    // with Safe Exception Handling.
    if (name == "@feat.00") {
      _compatibleWithSEH = true;
      continue;
    }

    // Cache the name.
    _symbolName[sym] = name;

    // Symbol may be followed by auxiliary symbol table records. The aux
    // record can be in any format, but the size is always the same as the
    // regular symbol. The aux record supplies additional information for the
    // standard symbol. We do not interpret the aux record here, but just
    // store it to _auxSymbol.
    if (sym->NumberOfAuxSymbols > 0) {
      const coff_symbol *aux = nullptr;
      if (error_code ec = _obj->getAuxSymbol(i + 1, aux))
        return ec;
      _auxSymbol[sym] = aux;
      i += sym->NumberOfAuxSymbols;
    }
  }
  return error_code::success();
}

/// Create atoms for the absolute symbols.
void FileCOFF::createAbsoluteAtoms(const SymbolVectorT &symbols,
                                   vector<const AbsoluteAtom *> &result) {
  for (const coff_symbol *sym : symbols) {
    if (sym->SectionNumber != llvm::COFF::IMAGE_SYM_ABSOLUTE)
      continue;
    auto *atom = new (_alloc)
        COFFAbsoluteAtom(*this, _symbolName[sym], getScope(sym), sym->Value);

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
error_code
FileCOFF::createUndefinedAtoms(const SymbolVectorT &symbols,
                               vector<const UndefinedAtom *> &result) {
  // Sort out undefined symbols from all symbols.
  std::set<const coff_symbol *> undefines;
  std::map<const coff_symbol *, const coff_symbol *> weakExternal;
  for (const coff_symbol *sym : symbols) {
    if (sym->SectionNumber != llvm::COFF::IMAGE_SYM_UNDEFINED)
      continue;
    undefines.insert(sym);

    // Create a mapping from sym1 to sym2, if the undefined symbol has
    // auxiliary data.
    auto iter = _auxSymbol.find(sym);
    if (iter == _auxSymbol.end())
      continue;
    const coff_aux_weak_external *aux =
        reinterpret_cast<const coff_aux_weak_external *>(iter->second);
    const coff_symbol *sym2;
    if (error_code ec = _obj->getSymbol(aux->TagIndex, sym2))
      return ec;
    weakExternal[sym] = sym2;
  }

  // Sort out sym1s from sym2s. Sym2s shouldn't be added to the undefined atom
  // list because they shouldn't be resolved unless sym1 is failed to
  // be resolved.
  for (auto i : weakExternal)
    undefines.erase(i.second);

  // Create atoms for the undefined symbols.
  for (const coff_symbol *sym : undefines) {
    // If the symbol has sym2, create an undefiend atom for sym2, so that we
    // can pass it as a fallback atom.
    UndefinedAtom *fallback = nullptr;
    auto iter = weakExternal.find(sym);
    if (iter != weakExternal.end()) {
      const coff_symbol *sym2 = iter->second;
      fallback = new (_alloc) COFFUndefinedAtom(*this, _symbolName[sym2]);
      _symbolAtom[sym2] = fallback;
    }

    // Create an atom for the symbol.
    auto *atom =
        new (_alloc) COFFUndefinedAtom(*this, _symbolName[sym], fallback);
    result.push_back(atom);
    _symbolAtom[sym] = atom;
  }
  return error_code::success();
}

/// Create atoms for the defined symbols. This pass is a bit complicated than
/// the other two, because in order to create the atom for the defined symbol
/// we need to know the adjacent symbols.
error_code FileCOFF::createDefinedSymbols(const SymbolVectorT &symbols,
                                          StringMap &altNames,
                                          vector<const DefinedAtom *> &result) {
  // A defined atom can be merged if its section attribute allows its contents
  // to be merged. In COFF, it's not very easy to get the section attribute
  // for the symbol, so scan all sections in advance and cache the attributes
  // for later use.
  if (error_code ec = cacheSectionAttributes())
    return ec;

  // Filter non-defined atoms, and group defined atoms by its section.
  SectionToSymbolsT definedSymbols;
  for (const coff_symbol *sym : symbols) {
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
    if (sym->SectionNumber == llvm::COFF::IMAGE_SYM_UNDEFINED &&
        sym->Value > 0) {
      StringRef name = _symbolName[sym];
      uint32_t size = sym->Value;
      auto *atom = new (_alloc)
          COFFBSSAtom(*this, name, getScope(sym), DefinedAtom::permRW_,
                      DefinedAtom::mergeAsWeakAndAddressUsed, size, _ordinal++);
      result.push_back(atom);
      continue;
    }

    // Skip if it's not for defined atom.
    if (sym->SectionNumber == llvm::COFF::IMAGE_SYM_ABSOLUTE ||
        sym->SectionNumber == llvm::COFF::IMAGE_SYM_UNDEFINED)
      continue;

    const coff_section *sec;
    if (error_code ec = _obj->getSection(sym->SectionNumber, sec))
      return ec;
    assert(sec && "SectionIndex > 0, Sec must be non-null!");

    // Skip if it's a section symbol for a COMDAT section. A section symbol
    // has the name of the section and value 0. A translation unit may contain
    // multiple COMDAT sections whose section name are the same. We don't want
    // to make atoms for them as they would become duplicate symbols.
    StringRef sectionName;
    if (error_code ec = _obj->getSectionName(sec, sectionName))
      return ec;
    if (_symbolName[sym] == sectionName && sym->Value == 0 &&
        _merge[sec] != DefinedAtom::mergeNo)
      continue;

    uint8_t sc = sym->StorageClass;
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
  if (error_code ec = AtomizeDefinedSymbols(definedSymbols, altNames, result))
    return ec;

  return error_code::success();
}

// Cache the COMDAT attributes, which indicate whether the symbols in the
// section can be merged or not.
error_code FileCOFF::cacheSectionAttributes() {
  // The COMDAT section attribute is not an attribute of coff_section, but is
  // stored in the auxiliary symbol for the first symbol referring a COMDAT
  // section. It feels to me that it's unnecessarily complicated, but this is
  // how COFF works.
  for (auto i : _auxSymbol) {
    const coff_symbol *sym = i.first;
    if (sym->SectionNumber == llvm::COFF::IMAGE_SYM_ABSOLUTE ||
        sym->SectionNumber == llvm::COFF::IMAGE_SYM_UNDEFINED)
      continue;

    const coff_section *sec;
    if (error_code ec = _obj->getSection(sym->SectionNumber, sec))
      return ec;

    if (_merge.count(sec))
      continue;
    if (!(sec->Characteristics & llvm::COFF::IMAGE_SCN_LNK_COMDAT))
      continue;

    _comdatSections.insert(sec);

    if (sym->NumberOfAuxSymbols == 0)
      return llvm::object::object_error::parse_failed;
    const coff_aux_section_definition *aux =
        reinterpret_cast<const coff_aux_section_definition *>(i.second);
    _merge[sec] = getMerge(aux);
  }

  // The sections that does not have auxiliary symbol are regular sections, in
  // which symbols are not allowed to be merged.
  for (auto si = _obj->section_begin(), se = _obj->section_end(); si != se;
       ++si) {
    const coff_section *sec = _obj->getCOFFSection(si);
    if (!_merge.count(sec))
      _merge[sec] = DefinedAtom::mergeNo;
  }
  return error_code::success();
}

/// Atomize \p symbols and append the results to \p atoms. The symbols are
/// assumed to have been defined in the \p section.
error_code
FileCOFF::AtomizeDefinedSymbolsInSection(const coff_section *section,
                                         StringMap &altNames,
                                         vector<const coff_symbol *> &symbols,
                                         vector<COFFDefinedFileAtom *> &atoms) {
  // Sort symbols by position.
  std::stable_sort(
      symbols.begin(), symbols.end(),
      // For some reason MSVC fails to allow the lambda in this context with a
      // "illegal use of local type in type instantiation". MSVC is clearly
      // wrong here. Force a conversion to function pointer to work around.
      static_cast<bool (*)(const coff_symbol *, const coff_symbol *)>([](
          const coff_symbol * a,
          const coff_symbol * b)->bool { return a->Value < b->Value; }));

  StringRef sectionName;
  if (error_code ec = _obj->getSectionName(section, sectionName))
    return ec;

  // BSS section does not have contents. If this is the BSS section, create
  // COFFBSSAtom instead of COFFDefinedAtom.
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) {
    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      const coff_symbol *sym = *si;
      uint32_t size = (si + 1 == se) ? section->SizeOfRawData - sym->Value
                                     : si[1]->Value - sym->Value;
      auto *atom = new (_alloc) COFFBSSAtom(
          *this, _symbolName[sym], getScope(sym), getPermissions(section),
          DefinedAtom::mergeAsWeakAndAddressUsed, size, _ordinal++);
      atoms.push_back(atom);
      _symbolAtom[sym] = atom;
    }
    return error_code::success();
  }

  ArrayRef<uint8_t> secData;
  if (error_code ec = _obj->getSectionContents(section, secData))
    return ec;

  // A section with IMAGE_SCN_LNK_{INFO,REMOVE} attribute will never become
  // a part of the output image. That's what the COFF spec says.
  if (section->Characteristics & llvm::COFF::IMAGE_SCN_LNK_INFO ||
      section->Characteristics & llvm::COFF::IMAGE_SCN_LNK_REMOVE)
    return error_code::success();

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
    return error_code::success();
  }

  // Create an unnamed atom if the first atom isn't at the start of the
  // section.
  if (symbols[0]->Value != 0) {
    uint64_t size = symbols[0]->Value;
    ArrayRef<uint8_t> data(secData.data(), size);
    auto *atom = new (_alloc) COFFDefinedAtom(
        *this, "", sectionName, Atom::scopeTranslationUnit, type, isComdat,
        perms, _merge[section], data, _ordinal++);
    atoms.push_back(atom);
    _definedAtomLocations[section][0].push_back(atom);
  }

  for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
    const uint8_t *start = secData.data() + (*si)->Value;
    // if this is the last symbol, take up the remaining data.
    const uint8_t *end = (si + 1 == se) ? secData.data() + secData.size()
                                        : secData.data() + (*(si + 1))->Value;
    auto pos = altNames.find(_symbolName[*si]);
    if (pos != altNames.end()) {
      auto *atom = new (_alloc) COFFDefinedAtom(
          *this, pos->second, sectionName, getScope(*si), type, isComdat, perms,
          DefinedAtom::mergeAsWeak, ArrayRef<uint8_t>(), _ordinal++);
      atoms.push_back(atom);
      _symbolAtom[*si] = atom;
      _definedAtomLocations[section][(*si)->Value].push_back(atom);
    }

    ArrayRef<uint8_t> data(start, end);
    auto *atom = new (_alloc) COFFDefinedAtom(
        *this, _symbolName[*si], sectionName, getScope(*si), type, isComdat,
        perms, _merge[section], data, _ordinal++);
    atoms.push_back(atom);
    _symbolAtom[*si] = atom;
    _definedAtomLocations[section][(*si)->Value].push_back(atom);
  }

  // Finally, set alignment to the first atom so that the section contents
  // will be aligned as specified by the object section header.
  _definedAtomLocations[section][0][0]->setAlignment(getAlignment(section));
  return error_code::success();
}

error_code
FileCOFF::AtomizeDefinedSymbols(SectionToSymbolsT &definedSymbols,
                                StringMap &altNames,
                                vector<const DefinedAtom *> &definedAtoms) {
  // For each section, make atoms for all the symbols defined in the
  // section, and append the atoms to the result objects.
  for (auto &i : definedSymbols) {
    const coff_section *section = i.first;
    vector<const coff_symbol *> &symbols = i.second;
    vector<COFFDefinedFileAtom *> atoms;
    if (error_code ec =
            AtomizeDefinedSymbolsInSection(section, altNames, symbols, atoms))
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

/// Find the atom that is at \p targetAddress in \p section.
error_code FileCOFF::findAtomAt(const coff_section *section,
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
      return error_code::success();
    }
  }
  // Relocation target is out of range
  return llvm::object::object_error::parse_failed;
}

/// Find the atom for the symbol that was at the \p index in the symbol
/// table.
error_code FileCOFF::getAtomBySymbolIndex(uint32_t index, Atom *&ret) {
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
FileCOFF::addRelocationReference(const coff_relocation *rel,
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

  COFFDefinedFileAtom *atom;
  uint32_t offsetInAtom;
  if (error_code ec = findAtomAt(section, itemAddress, atom, offsetInAtom))
    return ec;
  atom->addReference(std::unique_ptr<COFFReference>(
      new COFFReference(targetAtom, offsetInAtom, rel->Type,
                        Reference::KindNamespace::COFF,
                        _referenceArch)));
  return error_code::success();
}

/// Returns the target machine type of the current object file.
error_code FileCOFF::getReferenceArch(Reference::KindArch &result) {
  const llvm::object::coff_file_header *header = nullptr;
  if (error_code ec = _obj->getHeader(header))
    return ec;
  switch (header->Machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    result = Reference::KindArch::x86;
    return error_code::success();
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    result = Reference::KindArch::x86_64;
    return error_code::success();
  }
  llvm::errs() << "Unsupported machine type: " << header->Machine << "\n";
  return llvm::object::object_error::parse_failed;
}

/// Add relocation information to atoms.
error_code FileCOFF::addRelocationReferenceToAtoms() {
  // Relocation entries are defined for each section.
  for (auto si = _obj->section_begin(), se = _obj->section_end(); si != se;
       ++si) {
    const coff_section *section = _obj->getCOFFSection(si);

    // Skip there's no atom for the section. Currently we do not create any
    // atoms for some sections, such as "debug$S", and such sections need to
    // be skipped here too.
    if (_sectionAtoms.find(section) == _sectionAtoms.end())
      continue;

    for (auto ri = si->relocation_begin(), re = si->relocation_end();
         ri != re; ++ri) {
      const coff_relocation *rel = _obj->getCOFFRelocation(ri);
      if (auto ec =
              addRelocationReference(rel, section, _sectionAtoms[section]))
        return ec;
    }
  }
  return error_code::success();
}

/// Find a section by name.
error_code FileCOFF::findSection(StringRef name, const coff_section *&result) {
  for (auto si = _obj->section_begin(), se = _obj->section_end(); si != se;
       ++si) {
    const coff_section *section = _obj->getCOFFSection(si);
    StringRef sectionName;
    if (auto ec = _obj->getSectionName(section, sectionName))
      return ec;
    if (sectionName == name) {
      result = section;
      return error_code::success();
    }
  }
  // Section was not found, but it's not an error. This method returns
  // an error only when there's a read error.
  return error_code::success();
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
  std::string *contents = new (_alloc)
    std::string(reinterpret_cast<const char *>(&array[0]), len);
  return StringRef(*contents).trim();
}

// Convert .res file to .coff file and then parse it. Resource file is a file
// containing various types of data, such as icons, translation texts,
// etc. "cvtres.exe" command reads an RC file to create a COFF file which
// encapsulates resource data into rsrc$N sections, where N is an integer.
//
// The linker is not capable to handle RC files directly. Instead, it runs
// cvtres.exe on RC files and then then link its outputs.
class ResourceFileReader : public Reader {
public:
  virtual bool canParse(file_magic magic, StringRef ext,
                        const MemoryBuffer &) const {
    return (magic == llvm::sys::fs::file_magic::windows_resource);
  }

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const {
    // Convert RC file to COFF
    ErrorOr<std::string> coffPath = convertResourceFileToCOFF(std::move(mb));
    if (error_code ec = coffPath.getError())
      return ec;
    llvm::FileRemover coffFileRemover(*coffPath);

    // Read and parse the COFF
    OwningPtr<MemoryBuffer> opmb;
    if (error_code ec = MemoryBuffer::getFile(*coffPath, opmb))
      return ec;
    std::unique_ptr<MemoryBuffer> newmb(opmb.take());
    error_code ec;
    std::unique_ptr<FileCOFF> file(new FileCOFF(std::move(newmb), ec));
    if (ec)
      return ec;
    FileCOFF::StringMap emptyMap;
    if (error_code ec = file->parse(emptyMap))
      return ec;
    result.push_back(std::move(file));
    return error_code::success();
  }

private:
  static ErrorOr<std::string>
  writeResToTemporaryFile(std::unique_ptr<MemoryBuffer> mb) {
    // Get a temporary file path for .res file.
    SmallString<128> tempFilePath;
    if (error_code ec =
            llvm::sys::fs::createTemporaryFile("tmp", "res", tempFilePath))
      return ec;

    // Write the memory buffer contents to .res file, so that we can run
    // cvtres.exe on it.
    OwningPtr<llvm::FileOutputBuffer> buffer;
    if (error_code ec = llvm::FileOutputBuffer::create(
            tempFilePath.str(), mb->getBufferSize(), buffer))
      return ec;
    memcpy(buffer->getBufferStart(), mb->getBufferStart(), mb->getBufferSize());
    if (error_code ec = buffer->commit())
      return ec;

    // Convert SmallString -> StringRef -> std::string.
    return tempFilePath.str().str();
  }

  static ErrorOr<std::string>
  convertResourceFileToCOFF(std::unique_ptr<MemoryBuffer> mb) {
    // Write the resource file to a temporary file.
    ErrorOr<std::string> inFilePath = writeResToTemporaryFile(std::move(mb));
    if (error_code ec = inFilePath.getError())
      return ec;
    llvm::FileRemover inFileRemover(*inFilePath);

    // Create an output file path.
    SmallString<128> outFilePath;
    if (error_code ec =
            llvm::sys::fs::createTemporaryFile("tmp", "obj", outFilePath))
      return ec;
    std::string outFileArg = ("/out:" + outFilePath).str();

    // Construct CVTRES.EXE command line and execute it.
    std::string program = "cvtres.exe";
    std::string programPath = llvm::sys::FindProgramByName(program);
    if (programPath.empty()) {
      llvm::errs() << "Unable to find " << program << " in PATH\n";
      return llvm::errc::broken_pipe;
    }
    std::vector<const char *> args;
    args.push_back(programPath.c_str());
    args.push_back("/machine:x86");
    args.push_back("/readonly");
    args.push_back("/nologo");
    args.push_back(outFileArg.c_str());
    args.push_back(inFilePath->c_str());
    args.push_back(nullptr);

    DEBUG({
      for (const char **p = &args[0]; *p; ++p)
        llvm::dbgs() << *p << " ";
      llvm::dbgs() << "\n";
    });

    if (llvm::sys::ExecuteAndWait(programPath.c_str(), &args[0]) != 0) {
      llvm::errs() << program << " failed\n";
      return llvm::errc::broken_pipe;
    }
    return outFilePath.str().str();
  }
};

class COFFObjectReader : public Reader {
public:
  COFFObjectReader(PECOFFLinkingContext &ctx) : _context(ctx) {}

  virtual bool canParse(file_magic magic, StringRef ext,
                        const MemoryBuffer &) const {
    return (magic == llvm::sys::fs::file_magic::coff_object);
  }

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const Registry &registry,
            std::vector<std::unique_ptr<File>> &result) const {
    // Parse the memory buffer as PECOFF file.
    const char *mbName = mb->getBufferIdentifier();
    error_code ec;
    std::unique_ptr<FileCOFF> file(new FileCOFF(std::move(mb), ec));
    if (ec)
      return ec;

    // Interpret .drectve section if the section has contents.
    StringRef directives = file->getLinkerDirectives();
    if (!directives.empty())
      if (error_code ec = handleDirectiveSection(registry, directives))
        return ec;

    if (error_code ec = file->parse(_context.alternateNames()))
      return ec;

    // Check for /SAFESEH.
    if (_context.requireSEH() && !file->isCompatibleWithSEH()) {
      llvm::errs() << "/SAFESEH is specified, but " << mbName
                   << " is not compatible with SEH.\n";
      return llvm::object::object_error::parse_failed;
    }

    result.push_back(std::move(file));
    return error_code::success();
  }

private:
  // Interpret the contents of .drectve section. If exists, the section contains
  // a string containing command line options. The linker is expected to
  // interpret the options as if they were given via the command line.
  //
  // The section mainly contains /defaultlib (-l in Unix), but can contain any
  // options as long as they are valid.
  error_code handleDirectiveSection(const Registry &registry,
                                    StringRef directives) const {
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
    bool parseFailed = !WinLinkDriver::parse(argc, argv, _context, stream,
                                             /*isDirective*/ true);
    stream.flush();
    // Print error message if error.
    if (parseFailed) {
      llvm::errs() << "Failed to parse '" << directives << "'\n"
                   << "Reason: " << errorMessage;
      return make_error_code(llvm::object::object_error::invalid_file_type);
    }
    if (!errorMessage.empty()) {
      llvm::errs() << "lld warning: " << errorMessage << "\n";
    }
    return error_code::success();
  }

  PECOFFLinkingContext &_context;
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
  LLD_KIND_STRING_END
};

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
  LLD_KIND_STRING_END
};

} // end namespace anonymous

namespace lld {

void Registry::addSupportCOFFObjects(PECOFFLinkingContext &ctx) {
  add(std::unique_ptr<Reader>(new COFFObjectReader(ctx)));
  addKindTable(Reference::KindNamespace::COFF, Reference::KindArch::x86,
               kindStringsI386);
  addKindTable(Reference::KindNamespace::COFF, Reference::KindArch::x86_64,
               kindStringsAMD64);
}

void Registry::addSupportWindowsResourceFiles() {
  add(std::unique_ptr<Reader>(new ResourceFileReader()));
}
}
