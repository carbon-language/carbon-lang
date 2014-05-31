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

namespace lld {
namespace mach_o {
namespace normalized {

enum SymbolsInSection {
  symbolsOk,
  symbolsIgnored,
  symbolsIllegal
};

static uint64_t nextSymbolAddress(const NormalizedFile &normalizedFile,
                                  const Symbol &symbol) {
  uint64_t symbolAddr = symbol.value;
  uint8_t symbolSectionIndex = symbol.sect;
  const Section &section = normalizedFile.sections[symbolSectionIndex - 1];
  // If no symbol after this address, use end of section address.
  uint64_t closestAddr = section.address + section.content.size();
  for (const Symbol &s : normalizedFile.globalSymbols) {
    if (s.sect != symbolSectionIndex)
      continue;
    uint64_t sValue = s.value;
    if (sValue <= symbolAddr)
      continue;
    if (sValue < closestAddr)
      closestAddr = s.value;
  }
  for (const Symbol &s : normalizedFile.localSymbols) {
    if (s.sect != symbolSectionIndex)
      continue;
    uint64_t sValue = s.value;
    if (sValue <= symbolAddr)
      continue;
    if (sValue < closestAddr)
      closestAddr = s.value;
  }
  return closestAddr;
}

static Atom::Scope atomScope(uint8_t scope) {
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

static DefinedAtom::ContentType atomTypeFromSection(const Section &section) {
  if (section.attributes & S_ATTR_PURE_INSTRUCTIONS)
    return DefinedAtom::typeCode;
  if (section.segmentName.equals("__TEXT")) {
    if (section.sectionName.equals("__StaticInit"))
      return DefinedAtom::typeCode;
    if (section.sectionName.equals("__gcc_except_tab"))
      return DefinedAtom::typeLSDA;
    if (section.sectionName.startswith("__text"))
      return DefinedAtom::typeCode;
    if (section.sectionName.startswith("__const"))
      return DefinedAtom::typeConstant;
  } else if (section.segmentName.equals("__DATA")) {
    if (section.sectionName.startswith("__data"))
      return DefinedAtom::typeData;
    if (section.sectionName.startswith("__const"))
      return DefinedAtom::typeConstData;
  }

  return DefinedAtom::typeUnknown;
}

static error_code
processSymbol(const NormalizedFile &normalizedFile, MachOFile &file,
              const Symbol &sym, bool copyRefs,
              const SmallVector<SymbolsInSection, 32> symbolsInSect) {
  if (sym.sect > normalizedFile.sections.size()) {
    int sectionIndex = sym.sect;
    return make_dynamic_error_code(Twine("Symbol '") + sym.name
                                 + "' has n_sect ("
                                 + Twine(sectionIndex)
                                 + ") which is too large");
  }
  const Section &section = normalizedFile.sections[sym.sect - 1];
  switch (symbolsInSect[sym.sect-1]) {
  case symbolsOk:
    break;
  case symbolsIgnored:
    return error_code();
    break;
  case symbolsIllegal:
    return make_dynamic_error_code(Twine("Symbol '") + sym.name
                                 + "' is not legal in section "
                                 + section.segmentName + "/"
                                 + section.sectionName);
    break;
  }

  uint64_t offset = sym.value - section.address;
  // Mach-O symbol table does have size in it, so need to scan ahead
  // to find symbol with next highest address.
  uint64_t size = nextSymbolAddress(normalizedFile, sym) - sym.value;
  if (section.type == llvm::MachO::S_ZEROFILL) {
    file.addZeroFillDefinedAtom(sym.name, atomScope(sym.scope), size, copyRefs);
  } else {
    ArrayRef<uint8_t> atomContent = section.content.slice(offset, size);
    DefinedAtom::Merge m = DefinedAtom::mergeNo;
    if (sym.desc & N_WEAK_DEF)
      m = DefinedAtom::mergeAsWeak;
    DefinedAtom::ContentType type = atomTypeFromSection(section);
    if (type == DefinedAtom::typeUnknown) {
      // Mach-O needs a segment and section name.  Concatentate those two
      // with a / seperator (e.g. "seg/sect") to fit into the lld model
      // of just a section name.
      std::string segSectName = section.segmentName.str() 
                                + "/" + section.sectionName.str();
      file.addDefinedAtomInCustomSection(sym.name, atomScope(sym.scope), type, 
                                         m, atomContent, segSectName, true);
    } else {
      file.addDefinedAtom(sym.name, atomScope(sym.scope), type, m, atomContent, 
                        copyRefs);
    }
  }
  return error_code();
}


static void processUndefindeSymbol(MachOFile &file, const Symbol &sym,
                                  bool copyRefs) {
  // Undefinded symbols with n_value!=0 are actually tentative definitions.
  if (sym.value == Hex64(0)) {
    file.addUndefinedAtom(sym.name, copyRefs);
  } else {
    file.addTentativeDefAtom(sym.name, atomScope(sym.scope), sym.value,
                              DefinedAtom::Alignment(sym.desc >> 8), copyRefs);
  }
}

// A __TEXT/__ustring section contains UTF16 strings.  Atom boundaries are
// determined by finding the terminating 0x0000 in each string.
static error_code processUTF16Section(MachOFile &file, const Section &section,
                                      bool is64, bool copyRefs) {
  if ((section.content.size() % 4) != 0)
    return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                 + "/" + section.sectionName
                                 + " has a size that is not even");
  unsigned offset = 0;
  for (size_t i = 0, e = section.content.size(); i != e; i +=2) {
    if ((section.content[i] == 0) && (section.content[i+1] == 0)) {
      unsigned size = i - offset + 2;
      ArrayRef<uint8_t> utf16Content = section.content.slice(offset, size);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                          DefinedAtom::typeUTF16String,
                          DefinedAtom::mergeByContent, utf16Content,
                          copyRefs);
      offset = i + 2;
    }
  }
  if (offset != section.content.size()) {
    return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                   + "/" + section.sectionName
                                   + " is supposed to contain 0x0000 "
                                   "terminated UTF16 strings, but the "
                                   "last string in the section is not zero "
                                   "terminated.");
  }
  return error_code();
}

// A __DATA/__cfstring section contain NS/CFString objects. Atom boundaries
// are determined because each object is known to be 4 pointers in size.
static error_code processCFStringSection(MachOFile &file,const Section &section,
                                      bool is64, bool copyRefs) {
  const uint32_t cfsObjSize = (is64 ? 32 : 16);
  if ((section.content.size() % cfsObjSize) != 0) {
    return make_dynamic_error_code(Twine("Section __DATA/__cfstring has a size "
                                   "(" + Twine(section.content.size())
                                   + ") that is not a multiple of "
                                   + Twine(cfsObjSize)));
  }
  unsigned offset = 0;
  for (size_t i = 0, e = section.content.size(); i != e; i += cfsObjSize) {
    ArrayRef<uint8_t> byteContent = section.content.slice(offset, cfsObjSize);
    file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                        DefinedAtom::typeCFString,
                        DefinedAtom::mergeByContent, byteContent, copyRefs);
    offset += cfsObjSize;
  }
  return error_code();
}


// A __TEXT/__eh_frame section contains dwarf unwind CFIs (either CIE or FDE).
// Atom boundaries are determined by looking at the length content header
// in each CFI.
static error_code processCFISection(MachOFile &file, const Section &section,
                                      bool is64, bool swap, bool copyRefs) {
  const unsigned char* buffer = section.content.data();
  for (size_t offset = 0, end = section.content.size(); offset < end; ) {
    size_t remaining = end - offset;
    if (remaining < 16) {
      return make_dynamic_error_code(Twine("Section __TEXT/__eh_frame is "
                                     "malformed.  Not enough room left for "
                                     "a CFI starting at offset ("
                                     + Twine(offset)
                                     + ")"));
    }
    const uint32_t *cfi = reinterpret_cast<const uint32_t *>(&buffer[offset]);
    uint32_t len = read32(swap, *cfi) + 4;
    if (offset+len > end) {
      return make_dynamic_error_code(Twine("Section __TEXT/__eh_frame is "
                                     "malformed.  Size of CFI starting at "
                                     "at offset ("
                                     + Twine(offset)
                                     + ") is past end of section."));
    }
    ArrayRef<uint8_t> bytes = section.content.slice(offset, len);
    file.addDefinedAtom(StringRef(), DefinedAtom::scopeTranslationUnit,
                        DefinedAtom::typeCFI, DefinedAtom::mergeNo,
                        bytes, copyRefs);
    offset += len;
  }
  return error_code();
}

static error_code 
processCompactUnwindSection(MachOFile &file, const Section &section,
                            bool is64, bool copyRefs) {
  const uint32_t cuObjSize = (is64 ? 32 : 20);
  if ((section.content.size() % cuObjSize) != 0) {
    return make_dynamic_error_code(Twine("Section __LD/__compact_unwind has a "
                                   "size (" + Twine(section.content.size())
                                   + ") that is not a multiple of "
                                   + Twine(cuObjSize)));
  }
  unsigned offset = 0;
  for (size_t i = 0, e = section.content.size(); i != e; i += cuObjSize) {
    ArrayRef<uint8_t> byteContent = section.content.slice(offset, cuObjSize);
    file.addDefinedAtom(StringRef(), DefinedAtom::scopeTranslationUnit,
                        DefinedAtom::typeCompactUnwindInfo,
                        DefinedAtom::mergeNo, byteContent, copyRefs);
    offset += cuObjSize;
  }
  return error_code();
}

static error_code processSection(MachOFile &file, const Section &section,
                                 bool is64, bool swap, bool copyRefs,
                                 SymbolsInSection &symbolsInSect) {
  unsigned offset = 0;
  const unsigned pointerSize = (is64 ? 8 : 4);
  switch (section.type) {
  case llvm::MachO::S_REGULAR:
    if (section.segmentName.equals("__TEXT") &&
        section.sectionName.equals("__ustring")) {
      symbolsInSect = symbolsIgnored;
      return processUTF16Section(file, section, is64, copyRefs);
    } else if (section.segmentName.equals("__DATA") &&
             section.sectionName.equals("__cfstring")) {
      symbolsInSect = symbolsIllegal;
      return processCFStringSection(file, section, is64, copyRefs);
    } else if (section.segmentName.equals("__LD") &&
             section.sectionName.equals("__compact_unwind")) {
      symbolsInSect = symbolsIllegal;
      return processCompactUnwindSection(file, section, is64, copyRefs);
    }
    break;
  case llvm::MachO::S_COALESCED:
    if (section.segmentName.equals("__TEXT") &&
        section.sectionName.equals("__eh_frame")) {
      symbolsInSect = symbolsIgnored;
      return processCFISection(file, section, is64, swap, copyRefs);
    }
  case llvm::MachO::S_ZEROFILL:
    // These sections are broken into atoms based on symbols.
    break;
  case S_MOD_INIT_FUNC_POINTERS:
    if ((section.content.size() % pointerSize) != 0) {
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " has type S_MOD_INIT_FUNC_POINTERS but "
                                     "its size ("
                                     + Twine(section.content.size())
                                     + ") is not a multiple of "
                                     + Twine(pointerSize));
    }
    for (size_t i = 0, e = section.content.size(); i != e; i += pointerSize) {
      ArrayRef<uint8_t> bytes = section.content.slice(offset, pointerSize);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeTranslationUnit,
                          DefinedAtom::typeInitializerPtr, DefinedAtom::mergeNo,
                          bytes, copyRefs);
      offset += pointerSize;
    }
    symbolsInSect = symbolsIllegal;
    break;
  case S_MOD_TERM_FUNC_POINTERS:
    if ((section.content.size() % pointerSize) != 0) {
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " has type S_MOD_TERM_FUNC_POINTERS but "
                                     "its size ("
                                     + Twine(section.content.size())
                                     + ") is not a multiple of "
                                     + Twine(pointerSize));
    }
    for (size_t i = 0, e = section.content.size(); i != e; i += pointerSize) {
      ArrayRef<uint8_t> bytes = section.content.slice(offset, pointerSize);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeTranslationUnit,
                          DefinedAtom::typeTerminatorPtr, DefinedAtom::mergeNo,
                          bytes, copyRefs);
      offset += pointerSize;
    }
    symbolsInSect = symbolsIllegal;
    break;
  case S_NON_LAZY_SYMBOL_POINTERS:
    if ((section.content.size() % pointerSize) != 0) {
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " has type S_NON_LAZY_SYMBOL_POINTERS "
                                     "but its size ("
                                     + Twine(section.content.size())
                                     + ") is not a multiple of "
                                     + Twine(pointerSize));
    }
    for (size_t i = 0, e = section.content.size(); i != e; i += pointerSize) {
      ArrayRef<uint8_t> bytes = section.content.slice(offset, pointerSize);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                          DefinedAtom::typeGOT, DefinedAtom::mergeByContent,
                          bytes, copyRefs);
      offset += pointerSize;
    }
    symbolsInSect = symbolsIllegal;
    break;
  case llvm::MachO::S_CSTRING_LITERALS:
    for (size_t i = 0, e = section.content.size(); i != e; ++i) {
      if (section.content[i] == 0) {
        unsigned size = i - offset + 1;
        ArrayRef<uint8_t> strContent = section.content.slice(offset, size);
        file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                            DefinedAtom::typeCString,
                            DefinedAtom::mergeByContent, strContent, copyRefs);
        offset = i + 1;
      }
    }
    if (offset != section.content.size()) {
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName 
                                     + " has type S_CSTRING_LITERALS but the "
                                     "last string in the section is not zero "
                                     "terminated."); 
    }
    symbolsInSect = symbolsIgnored;
    break;
  case llvm::MachO::S_4BYTE_LITERALS:
    if ((section.content.size() % 4) != 0)
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName 
                                     + " has type S_4BYTE_LITERALS but its "
                                     "size (" + Twine(section.content.size()) 
                                     + ") is not a multiple of 4"); 
    for (size_t i = 0, e = section.content.size(); i != e; i += 4) {
      ArrayRef<uint8_t> byteContent = section.content.slice(offset, 4);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                          DefinedAtom::typeLiteral4,
                          DefinedAtom::mergeByContent, byteContent, copyRefs);
      offset += 4;
    }
    symbolsInSect = symbolsIllegal;
    break;
  case llvm::MachO::S_8BYTE_LITERALS:
    if ((section.content.size() % 8) != 0)
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName 
                                     + " has type S_8YTE_LITERALS but its "
                                     "size (" + Twine(section.content.size()) 
                                     + ") is not a multiple of 8"); 
    for (size_t i = 0, e = section.content.size(); i != e; i += 8) {
      ArrayRef<uint8_t> byteContent = section.content.slice(offset, 8);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                          DefinedAtom::typeLiteral8,
                          DefinedAtom::mergeByContent, byteContent, copyRefs);
      offset += 8;
    }
    symbolsInSect = symbolsIllegal;
    break;
  case llvm::MachO::S_16BYTE_LITERALS:
    if ((section.content.size() % 16) != 0)
      return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName 
                                     + " has type S_16BYTE_LITERALS but its "
                                     "size (" + Twine(section.content.size()) 
                                     + ") is not a multiple of 16"); 
    for (size_t i = 0, e = section.content.size(); i != e; i += 16) {
      ArrayRef<uint8_t> byteContent = section.content.slice(offset, 16);
      file.addDefinedAtom(StringRef(), DefinedAtom::scopeLinkageUnit,
                          DefinedAtom::typeLiteral16,
                          DefinedAtom::mergeByContent, byteContent, copyRefs);
      offset += 16;
    }
    symbolsInSect = symbolsIllegal;
    break;
  default:
    llvm_unreachable("mach-o section type not supported yet");
    break;
  }
  return error_code();
}

static ErrorOr<std::unique_ptr<lld::File>>
normalizedObjectToAtoms(const NormalizedFile &normalizedFile, StringRef path,
                        bool copyRefs) {
  std::unique_ptr<MachOFile> file(new MachOFile(path));

  // Create atoms from sections that don't have symbols.
  bool is64 = MachOLinkingContext::is64Bit(normalizedFile.arch);
  bool swap = !MachOLinkingContext::isHostEndian(normalizedFile.arch);
  SmallVector<SymbolsInSection, 32> symbolsInSect;
  for (auto &sect : normalizedFile.sections) {
    symbolsInSect.push_back(symbolsOk);
    if (error_code ec = processSection(*file, sect, is64, swap, copyRefs,
                                       symbolsInSect.back()))
      return ec;
  }
  // Create atoms from global symbols.
  for (const Symbol &sym : normalizedFile.globalSymbols) {
    if (error_code ec = processSymbol(normalizedFile, *file, sym, copyRefs,
                                      symbolsInSect))
      return ec;
  }
  // Create atoms from local symbols.
  for (const Symbol &sym : normalizedFile.localSymbols) {
    if (error_code ec = processSymbol(normalizedFile, *file, sym, copyRefs,
                                      symbolsInSect))
      return ec;
  }
  // Create atoms from undefined symbols.
  for (auto &sym : normalizedFile.undefinedSymbols) {
    processUndefindeSymbol(*file, sym, copyRefs);
  }

  return std::unique_ptr<File>(std::move(file));
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
