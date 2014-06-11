//===- lib/ReaderWriter/MachO/MachONormalizedFileToAtoms.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///
/// \file Converts from in-memory normalized mach-o to in-memory Atoms.
///
///                  +------------+
///                  | normalized |
///                  +------------+
///                        |
///                        |
///                        v
///                    +-------+
///                    | Atoms |
///                    +-------+

#include "MachONormalizedFile.h"
#include "MachONormalizedFileBinaryUtils.h"
#include "File.h"
#include "Atoms.h"

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"

#include "llvm/Support/MachO.h"

using namespace llvm::MachO;
using namespace lld::mach_o::normalized;

namespace lld {
namespace mach_o {


namespace { // anonymous


#define ENTRY(seg, sect, type, atomType) \
  {seg, sect, type, DefinedAtom::atomType }

struct MachORelocatableSectionToAtomType {
  StringRef                 segmentName;
  StringRef                 sectionName;
  SectionType               sectionType;
  DefinedAtom::ContentType  atomType;
};

const MachORelocatableSectionToAtomType sectsToAtomType[] = {
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeCode),
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeResolver),
  ENTRY("__TEXT", "__cstring",        S_CSTRING_LITERALS, typeCString),
  ENTRY("",       "",                 S_CSTRING_LITERALS, typeCString),
  ENTRY("__TEXT", "__ustring",        S_REGULAR,          typeUTF16String),
  ENTRY("__TEXT", "__const",          S_REGULAR,          typeConstant),
  ENTRY("__TEXT", "__eh_frame",       S_COALESCED,        typeCFI),
  ENTRY("__TEXT", "__literal4",       S_4BYTE_LITERALS,   typeLiteral4),
  ENTRY("__TEXT", "__literal8",       S_8BYTE_LITERALS,   typeLiteral8),
  ENTRY("__TEXT", "__literal16",      S_16BYTE_LITERALS,  typeLiteral16),
  ENTRY("__TEXT", "__gcc_except_tab", S_REGULAR,          typeLSDA),
  ENTRY("__DATA", "__data",           S_REGULAR,          typeData),
  ENTRY("__DATA", "__const",          S_REGULAR,          typeConstData),
  ENTRY("__DATA", "__cfstring",       S_REGULAR,          typeCFString),
  ENTRY("__DATA", "__mod_init_func",  S_MOD_INIT_FUNC_POINTERS,
                                                          typeInitializerPtr),
  ENTRY("__DATA", "__mod_term_func",  S_MOD_TERM_FUNC_POINTERS,
                                                          typeTerminatorPtr),
  ENTRY("__DATA", "___got",           S_NON_LAZY_SYMBOL_POINTERS,
                                                          typeGOT),
  ENTRY("__DATA", "___bss",           S_ZEROFILL,         typeZeroFill),
  ENTRY("",       "",                 S_NON_LAZY_SYMBOL_POINTERS,
                                                          typeGOT),
  ENTRY("__LD",   "__compact_unwind", S_REGULAR,
                                                         typeCompactUnwindInfo),
  ENTRY("",       "",                 S_REGULAR,          typeUnknown)
};
#undef ENTRY


/// Figures out ContentType of a mach-o section.
DefinedAtom::ContentType atomTypeFromSection(const Section &section) {
  // First look for match of name and type. Empty names in table are wildcards.
  for (const MachORelocatableSectionToAtomType *p = sectsToAtomType ;
                                 p->atomType != DefinedAtom::typeUnknown; ++p) {
    if (p->sectionType != section.type)
      continue;
    if (!p->segmentName.equals(section.segmentName) && !p->segmentName.empty())
      continue;
    if (!p->sectionName.equals(section.sectionName) && !p->sectionName.empty())
      continue;
    return p->atomType;
  }
  // Look for code denoted by section attributes
  if (section.attributes & S_ATTR_PURE_INSTRUCTIONS)
    return DefinedAtom::typeCode;

  return DefinedAtom::typeUnknown;
}

enum AtomizeModel {
  atomizeAtSymbols,
  atomizeFixedSize,
  atomizePointerSize,
  atomizeUTF8,
  atomizeUTF16,
  atomizeCFI,
  atomizeCU
};

/// Returns info on how to atomize a section of the specified ContentType.
void sectionParseInfo(DefinedAtom::ContentType atomType,
                      unsigned int &sizeMultiple,
                      DefinedAtom::Scope &scope,
                      DefinedAtom::Merge &merge,
                      AtomizeModel &atomizeModel) {
  struct ParseInfo {
    DefinedAtom::ContentType  atomType;
    unsigned int              sizeMultiple;
    DefinedAtom::Scope        scope;
    DefinedAtom::Merge        merge;
    AtomizeModel              atomizeModel;
  };

  #define ENTRY(type, size, scope, merge, model) \
    {DefinedAtom::type, size, DefinedAtom::scope, DefinedAtom::merge, model }

  static const ParseInfo parseInfo[] = {
    ENTRY(typeCode,              1, scopeGlobal,          mergeNo, 
                                                            atomizeAtSymbols),
    ENTRY(typeData,              1, scopeGlobal,          mergeNo, 
                                                            atomizeAtSymbols),
    ENTRY(typeConstData,         1, scopeGlobal,          mergeNo, 
                                                            atomizeAtSymbols),
    ENTRY(typeZeroFill,          1, scopeGlobal,          mergeNo, 
                                                            atomizeAtSymbols),
    ENTRY(typeConstant,          1, scopeGlobal,          mergeNo, 
                                                            atomizeAtSymbols),
    ENTRY(typeCString,           1, scopeLinkageUnit,     mergeByContent, 
                                                            atomizeUTF8),
    ENTRY(typeUTF16String,       1, scopeLinkageUnit,     mergeByContent, 
                                                            atomizeUTF16),
    ENTRY(typeCFI,               1, scopeTranslationUnit, mergeNo, 
                                                            atomizeCFI),
    ENTRY(typeLiteral4,          4, scopeLinkageUnit,     mergeByContent, 
                                                            atomizeFixedSize),
    ENTRY(typeLiteral8,          8, scopeLinkageUnit,     mergeByContent, 
                                                            atomizeFixedSize),
    ENTRY(typeLiteral16,        16, scopeLinkageUnit,     mergeByContent, 
                                                            atomizeFixedSize),
    ENTRY(typeCFString,         16, scopeLinkageUnit,     mergeByContent, 
                                                            atomizeFixedSize),
    ENTRY(typeInitializerPtr,    4, scopeTranslationUnit, mergeNo, 
                                                            atomizePointerSize),
    ENTRY(typeTerminatorPtr,     4, scopeTranslationUnit, mergeNo, 
                                                            atomizePointerSize),
    ENTRY(typeCompactUnwindInfo, 4, scopeTranslationUnit, mergeNo, 
                                                            atomizeCU),
    ENTRY(typeCFI,               4, scopeTranslationUnit, mergeNo, 
                                                            atomizeFixedSize),
    ENTRY(typeGOT,               4, scopeLinkageUnit,     mergeByContent, 
                                                            atomizePointerSize),
    ENTRY(typeUnknown,           1, scopeGlobal,          mergeNo, 
                                                            atomizeAtSymbols)
  };
  #undef ENTRY
  const int tableLen = sizeof(parseInfo) / sizeof(ParseInfo);
  for (int i=0; i < tableLen; ++i) {
    if (parseInfo[i].atomType == atomType) {
      sizeMultiple = parseInfo[i].sizeMultiple;
      scope        = parseInfo[i].scope;
      merge        = parseInfo[i].merge;
      atomizeModel = parseInfo[i].atomizeModel;
      return;
    }
  }

  // Unknown type is atomized by symbols.
  sizeMultiple = 1;
  scope = DefinedAtom::scopeGlobal;
  merge = DefinedAtom::mergeNo;
  atomizeModel = atomizeAtSymbols;
}


Atom::Scope atomScope(uint8_t scope) {
  switch (scope) {
  case N_EXT:
    return Atom::scopeGlobal;
  case N_PEXT | N_EXT:
    return Atom::scopeLinkageUnit;
  case 0:
    return Atom::scopeTranslationUnit;
  }
  llvm_unreachable("unknown scope value!");
}

void appendSymbolsInSection(const std::vector<Symbol> &inSymbols,
                            uint32_t sectionIndex,
                            SmallVector<const Symbol *, 64> &outSyms) {
  for (const Symbol &sym : inSymbols) {
    // Only look at definition symbols.
    if ((sym.type & N_TYPE) != N_SECT)
      continue;
    if (sym.sect != sectionIndex)
      continue;
    outSyms.push_back(&sym);
  }
}

void atomFromSymbol(DefinedAtom::ContentType atomType, const Section &section,
                    MachOFile &file, uint64_t symbolAddr, StringRef symbolName,
                    uint16_t symbolDescFlags, Atom::Scope symbolScope,
                    uint64_t nextSymbolAddr, bool copyRefs) {
  // Mach-O symbol table does have size in it. Instead the size is the
  // difference between this and the next symbol.
  uint64_t size = nextSymbolAddr - symbolAddr;
  if (section.type == llvm::MachO::S_ZEROFILL) {
    file.addZeroFillDefinedAtom(symbolName, symbolScope, size, copyRefs);
  } else {
    uint64_t offset = symbolAddr - section.address;
    ArrayRef<uint8_t> atomContent = section.content.slice(offset, size);
    DefinedAtom::Merge merge = (symbolDescFlags & N_WEAK_DEF) 
                              ? DefinedAtom::mergeAsWeak : DefinedAtom::mergeNo;
    if (atomType == DefinedAtom::typeUnknown) {
      // Mach-O needs a segment and section name.  Concatentate those two
      // with a / seperator (e.g. "seg/sect") to fit into the lld model
      // of just a section name.
      std::string segSectName = section.segmentName.str()
                                + "/" + section.sectionName.str();
      file.addDefinedAtomInCustomSection(symbolName, symbolScope, atomType,
                                         merge, atomContent, segSectName, true);
    } else {
      if ((atomType == lld::DefinedAtom::typeCode) && 
          (symbolDescFlags & N_SYMBOL_RESOLVER)) {
       atomType = lld::DefinedAtom::typeResolver;
      }
      file.addDefinedAtom(symbolName, symbolScope, atomType, merge,
                          atomContent, copyRefs);
    }
  }
}

error_code processSymboledSection(DefinedAtom::ContentType atomType,
                                  const Section &section,
                                  const NormalizedFile &normalizedFile,
                                  MachOFile &file, bool copyRefs) {
  // Find section's index.
  uint32_t sectIndex = 1;
  for (auto &sect : normalizedFile.sections) {
    if (&sect == &section)
      break;
    ++sectIndex;
  }

  // Find all symbols in this section.
  SmallVector<const Symbol *, 64> symbols;
  appendSymbolsInSection(normalizedFile.globalSymbols, sectIndex, symbols);
  appendSymbolsInSection(normalizedFile.localSymbols,  sectIndex, symbols);

  // Sort symbols.
  std::sort(symbols.begin(), symbols.end(),
            [](const Symbol *lhs, const Symbol *rhs) -> bool {
              // First by address.
              if (lhs->value != rhs->value)
                return lhs->value < rhs->value;
              // If same address, one is an alias.  Sort by scope.
              Atom::Scope lScope = atomScope(lhs->scope);
              Atom::Scope rScope = atomScope(rhs->scope);
              if (lScope != rScope)
                return lScope < rScope;
              // If same address and scope, sort by name.   
              return (lhs->name.compare(rhs->name) < 1);
            });

  // Debug logging of symbols.
  //for (const Symbol *sym : symbols)
  //  llvm::errs() << "sym: " << sym->value << ", " << sym->name << "\n";

  // If section has no symbols and no content, there are no atoms.
  if (symbols.empty() && section.content.empty())
    return error_code();

  const uint64_t firstSymbolAddr = symbols.front()->value;
  if (firstSymbolAddr != section.address) {
    // Section has anonymous content before first symbol.
    atomFromSymbol(atomType, section, file, section.address, StringRef(),
                  0, Atom::scopeTranslationUnit, firstSymbolAddr, copyRefs);
  }

  const Symbol *lastSym = nullptr;
  for (const Symbol *sym : symbols) {
    if (lastSym != nullptr) {
      atomFromSymbol(atomType, section, file, lastSym->value, lastSym->name,
                     lastSym->desc, atomScope(lastSym->scope), sym->value, copyRefs);
    }
    lastSym = sym;
  }
  if (lastSym != nullptr) {
    atomFromSymbol(atomType, section, file, lastSym->value, lastSym->name,
                   lastSym->desc, atomScope(lastSym->scope),
                   section.address + section.content.size(), copyRefs);
  }
  return error_code();
}

error_code processSection(DefinedAtom::ContentType atomType,
                          const Section &section,
                          const NormalizedFile &normalizedFile,
                          MachOFile &file, bool copyRefs) {
  const bool is64 = MachOLinkingContext::is64Bit(normalizedFile.arch);
  const bool swap = !MachOLinkingContext::isHostEndian(normalizedFile.arch);

  // Get info on how to atomize section.
  unsigned int       sizeMultiple;
  DefinedAtom::Scope scope;
  DefinedAtom::Merge merge;
  AtomizeModel       atomizeModel;
  sectionParseInfo(atomType, sizeMultiple, scope, merge, atomizeModel);

  // Validate section size.
  if ((section.content.size() % sizeMultiple) != 0)
    return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " has size ("
                                     + Twine(section.content.size())
                                     + ") which is not a multiple of "
                                     + Twine(sizeMultiple) );

  if (atomizeModel == atomizeAtSymbols) {
    // Break section up into atoms each with a fixed size.
    return processSymboledSection(atomType, section, normalizedFile, file,
                                                                      copyRefs);
  } else {
    const uint32_t *cfi;
    unsigned int size;
    for (unsigned int offset = 0, e = section.content.size(); offset != e;) {
      switch (atomizeModel) {
      case atomizeFixedSize:
        // Break section up into atoms each with a fixed size.
        size = sizeMultiple;
        break;
      case atomizePointerSize:
        // Break section up into atoms each the size of a pointer.
        size = is64 ? 8 : 4;;
        break;
      case atomizeUTF8:
        // Break section up into zero terminated c-strings.
        size = 0;
        for (unsigned int i=0; offset+i < e; ++i) {
          if (section.content[i] == 0) {
            size = i+1;
            break;
          }
        }
        break;
      case atomizeUTF16:
        // Break section up into zero terminated UTF16 strings.
        size = 0;
        for (unsigned int i=0; offset+i < e; i += 2) {
          if ((section.content[i] == 0) && (section.content[i+1] == 0)) {
            size = i+2;
            break;
          }
        }
        break;
      case atomizeCFI:
        // Break section up into dwarf unwind CFIs (FDE or CIE).
        cfi = reinterpret_cast<const uint32_t *>(&section.content[offset]);
        size = read32(swap, *cfi) + 4;
        if (offset+size > section.content.size()) {
          return make_dynamic_error_code(Twine(Twine("Section ")
                                         + section.segmentName
                                         + "/" + section.sectionName
                                         + " is malformed.  Size of CFI "
                                         "starting at offset ("
                                         + Twine(offset)
                                         + ") is past end of section."));
        }
        break;
      case atomizeCU:
        // Break section up into compact unwind entries.
        size = is64 ? 32 : 20;
        break;
      case atomizeAtSymbols:
        break;
      }
      if (size == 0) {
        return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " is malformed.  The last atom is "
                                     "not zero terminated.");
      }
      ArrayRef<uint8_t> byteContent = section.content.slice(offset, size);
      file.addDefinedAtom(StringRef(), scope, atomType, merge, byteContent,
                          copyRefs);
      offset += size;
    }
  }
  return error_code();
}

ErrorOr<std::unique_ptr<lld::File>>
normalizedObjectToAtoms(const NormalizedFile &normalizedFile, StringRef path,
                        bool copyRefs) {
  std::unique_ptr<MachOFile> file(new MachOFile(path));
  // Create atoms from each section.
  for (auto &sect : normalizedFile.sections) {
    DefinedAtom::ContentType atomType = atomTypeFromSection(sect);
    if (error_code ec = processSection(atomType, sect, normalizedFile, *file,
                                       copyRefs))
      return ec;
  }
  // Create atoms from undefined symbols.
  for (auto &sym : normalizedFile.undefinedSymbols) {
    // Undefinded symbols with n_value != 0 are actually tentative definitions.
    if (sym.value == Hex64(0)) {
      file->addUndefinedAtom(sym.name, copyRefs);
    } else {
      file->addTentativeDefAtom(sym.name, atomScope(sym.scope), sym.value,
                               DefinedAtom::Alignment(sym.desc >> 8), copyRefs);
    }
  }

  return std::unique_ptr<File>(std::move(file));
}

} // anonymous namespace

namespace normalized {

void relocatableSectionInfoForContentType(DefinedAtom::ContentType atomType,
                                          StringRef &segmentName,
                                          StringRef &sectionName,
                                          SectionType &sectionType,
                                          SectionAttr &sectionAttrs) {

  for (const MachORelocatableSectionToAtomType *p = sectsToAtomType ;
                                 p->atomType != DefinedAtom::typeUnknown; ++p) {
    if (p->atomType != atomType)
      continue;
    // Wild carded entries are ignored for reverse lookups.
    if (p->segmentName.empty() || p->sectionName.empty())
      continue;
    segmentName = p->segmentName;
    sectionName = p->sectionName;
    sectionType = p->sectionType;
    sectionAttrs = 0;
    if (atomType == DefinedAtom::typeCode)
      sectionAttrs = S_ATTR_PURE_INSTRUCTIONS;
    return;
  }
  llvm_unreachable("content type not yet supported");
}

ErrorOr<std::unique_ptr<lld::File>>
normalizedToAtoms(const NormalizedFile &normalizedFile, StringRef path,
                  bool copyRefs) {
  switch (normalizedFile.fileType) {
  case MH_OBJECT:
    return normalizedObjectToAtoms(normalizedFile, path, copyRefs);
  default:
    llvm_unreachable("unhandled MachO file type!");
  }
}

} // namespace normalized
} // namespace mach_o
} // namespace lld
