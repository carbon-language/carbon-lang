//===- lib/ReaderWriter/MachO/MachONormalizedFileFromAtoms.cpp ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///
/// \file Converts from in-memory Atoms to in-memory normalized mach-o.
///
///                  +------------+
///                  | normalized |
///                  +------------+
///                        ^
///                        |
///                        |
///                    +-------+
///                    | Atoms |
///                    +-------+

#include "MachONormalizedFile.h"

#include "ArchHandler.h"
#include "MachONormalizedFileBinaryUtils.h"

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MachO.h"
#include <map>
#include <system_error>

using llvm::StringRef;
using llvm::isa;
using namespace llvm::MachO;
using namespace lld::mach_o::normalized;
using namespace lld;

namespace {

struct AtomInfo {
  const DefinedAtom  *atom;
  uint64_t            offsetInSection;
};

struct SectionInfo {
  SectionInfo(StringRef seg, StringRef sect, SectionType type,
              const MachOLinkingContext &ctxt, uint32_t attr=0);

  StringRef                 segmentName;
  StringRef                 sectionName;
  SectionType               type;
  uint32_t                  attributes;
  uint64_t                  address;
  uint64_t                  size;
  uint32_t                  alignment;
  std::vector<AtomInfo>     atomsAndOffsets;
  uint32_t                  normalizedSectionIndex;
  uint32_t                  finalSectionIndex;
};

SectionInfo::SectionInfo(StringRef sg, StringRef sct, SectionType t,
                         const MachOLinkingContext &ctxt, uint32_t attrs)
 : segmentName(sg), sectionName(sct), type(t), attributes(attrs),
                 address(0), size(0), alignment(0),
                 normalizedSectionIndex(0), finalSectionIndex(0) {
  uint8_t align;
  if (ctxt.sectionAligned(segmentName, sectionName, align)) {
    alignment = align;
  }
}

struct SegmentInfo {
  SegmentInfo(StringRef name);

  StringRef                  name;
  uint64_t                   address;
  uint64_t                   size;
  uint32_t                   access;
  std::vector<SectionInfo*>  sections;
  uint32_t                   normalizedSegmentIndex;
};

SegmentInfo::SegmentInfo(StringRef n)
 : name(n), address(0), size(0), access(0), normalizedSegmentIndex(0) {
}


class Util {
public:
  Util(const MachOLinkingContext &ctxt) : _context(ctxt), 
    _archHandler(ctxt.archHandler()), _entryAtom(nullptr) {}

  void      assignAtomsToSections(const lld::File &atomFile);
  void      organizeSections();
  void      assignAddressesToSections(const NormalizedFile &file);
  uint32_t  fileFlags();
  void      copySegmentInfo(NormalizedFile &file);
  void      copySectionInfo(NormalizedFile &file);
  void      updateSectionInfo(NormalizedFile &file);
  void      buildAtomToAddressMap();
  std::error_code addSymbols(const lld::File &atomFile, NormalizedFile &file);
  void      addIndirectSymbols(const lld::File &atomFile, NormalizedFile &file);
  void      addRebaseAndBindingInfo(const lld::File &, NormalizedFile &file);
  void      addSectionRelocs(const lld::File &, NormalizedFile &file);
  void      buildDataInCodeArray(const lld::File &, NormalizedFile &file);
  void      addDependentDylibs(const lld::File &, NormalizedFile &file);
  void      copyEntryPointAddress(NormalizedFile &file);
  void      copySectionContent(NormalizedFile &file);

private:
  typedef std::map<DefinedAtom::ContentType, SectionInfo*> TypeToSection;
  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;

  struct DylibInfo { int ordinal; bool hasWeak; bool hasNonWeak; };
  typedef llvm::StringMap<DylibInfo> DylibPathToInfo;

  SectionInfo *sectionForAtom(const DefinedAtom*);
  SectionInfo *getRelocatableSection(DefinedAtom::ContentType type);
  SectionInfo *getFinalSection(DefinedAtom::ContentType type);
  void         appendAtom(SectionInfo *sect, const DefinedAtom *atom);
  SegmentInfo *segmentForName(StringRef segName);
  void         layoutSectionsInSegment(SegmentInfo *seg, uint64_t &addr);
  void         layoutSectionsInTextSegment(size_t, SegmentInfo *, uint64_t &);
  void         copySectionContent(SectionInfo *si, ContentBytes &content);
  uint16_t     descBits(const DefinedAtom* atom);
  int          dylibOrdinal(const SharedLibraryAtom *sa);
  void         segIndexForSection(const SectionInfo *sect,
                             uint8_t &segmentIndex, uint64_t &segmentStartAddr);
  const Atom  *targetOfLazyPointer(const DefinedAtom *lpAtom);
  const Atom  *targetOfStub(const DefinedAtom *stubAtom);
  std::error_code getSymbolTableRegion(const DefinedAtom* atom,
                                       bool &inGlobalsRegion,
                                       SymbolScope &symbolScope);
  void         appendSection(SectionInfo *si, NormalizedFile &file);
  uint32_t     sectionIndexForAtom(const Atom *atom);

  static uint64_t alignTo(uint64_t value, uint8_t align2);
  typedef llvm::DenseMap<const Atom*, uint32_t> AtomToIndex;
  struct AtomAndIndex { const Atom *atom; uint32_t index; SymbolScope scope; };
  struct AtomSorter {
    bool operator()(const AtomAndIndex &left, const AtomAndIndex &right);
  };
  struct SegmentSorter {
    bool operator()(const SegmentInfo *left, const SegmentInfo *right);
    static unsigned weight(const SegmentInfo *);
  };
  struct TextSectionSorter {
    bool operator()(const SectionInfo *left, const SectionInfo *right);
    static unsigned weight(const SectionInfo *);
  };

  const MachOLinkingContext    &_context;
  mach_o::ArchHandler          &_archHandler;
  llvm::BumpPtrAllocator        _allocator;
  std::vector<SectionInfo*>     _sectionInfos;
  std::vector<SegmentInfo*>     _segmentInfos;
  TypeToSection                 _sectionMap;
  std::vector<SectionInfo*>     _customSections;
  AtomToAddress                 _atomToAddress;
  DylibPathToInfo               _dylibInfo;
  const DefinedAtom            *_entryAtom;
  AtomToIndex                   _atomToSymbolIndex;
};


SectionInfo *Util::getRelocatableSection(DefinedAtom::ContentType type) {
  StringRef segmentName;
  StringRef sectionName;
  SectionType sectionType;
  SectionAttr sectionAttrs;

  // Use same table used by when parsing .o files.
  relocatableSectionInfoForContentType(type, segmentName, sectionName,
                                       sectionType, sectionAttrs);
  // If we already have a SectionInfo with this name, re-use it.
  // This can happen if two ContentType map to the same mach-o section.
  for (auto sect : _sectionMap) {
    if (sect.second->sectionName.equals(sectionName) &&
        sect.second->segmentName.equals(segmentName)) {
      return sect.second;
    }
  }
  // Otherwise allocate new SectionInfo object.
  SectionInfo *sect = new (_allocator) SectionInfo(segmentName, sectionName, 
                                                   sectionType, _context,
                                                   sectionAttrs);
  _sectionInfos.push_back(sect);
  _sectionMap[type] = sect;
  return sect;
}

#define ENTRY(seg, sect, type, atomType) \
  {seg, sect, type, DefinedAtom::atomType }

struct MachOFinalSectionFromAtomType {
  StringRef                 segmentName;
  StringRef                 sectionName;
  SectionType               sectionType;
  DefinedAtom::ContentType  atomType;
};

const MachOFinalSectionFromAtomType sectsToAtomType[] = {
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeCode),
  ENTRY("__TEXT", "__cstring",        S_CSTRING_LITERALS, typeCString),
  ENTRY("__TEXT", "__ustring",        S_REGULAR,          typeUTF16String),
  ENTRY("__TEXT", "__const",          S_REGULAR,          typeConstant),
  ENTRY("__TEXT", "__const",          S_4BYTE_LITERALS,   typeLiteral4),
  ENTRY("__TEXT", "__const",          S_8BYTE_LITERALS,   typeLiteral8),
  ENTRY("__TEXT", "__const",          S_16BYTE_LITERALS,  typeLiteral16),
  ENTRY("__TEXT", "__stubs",          S_SYMBOL_STUBS,     typeStub),
  ENTRY("__TEXT", "__stub_helper",    S_REGULAR,          typeStubHelper),
  ENTRY("__TEXT", "__gcc_except_tab", S_REGULAR,          typeLSDA),
  ENTRY("__TEXT", "__eh_frame",       S_COALESCED,        typeCFI),
  ENTRY("__DATA", "__data",           S_REGULAR,          typeData),
  ENTRY("__DATA", "__const",          S_REGULAR,          typeConstData),
  ENTRY("__DATA", "__cfstring",       S_REGULAR,          typeCFString),
  ENTRY("__DATA", "__la_symbol_ptr",  S_LAZY_SYMBOL_POINTERS,
                                                          typeLazyPointer),
  ENTRY("__DATA", "__mod_init_func",  S_MOD_INIT_FUNC_POINTERS,
                                                          typeInitializerPtr),
  ENTRY("__DATA", "__mod_term_func",  S_MOD_TERM_FUNC_POINTERS,
                                                          typeTerminatorPtr),
  ENTRY("__DATA", "___got",           S_NON_LAZY_SYMBOL_POINTERS,
                                                          typeGOT),
  ENTRY("__DATA", "___bss",           S_ZEROFILL,         typeZeroFill),

  // FIXME: __compact_unwind actually needs to be processed by a pass and put
  // into __TEXT,__unwind_info. For now, forwarding it back to
  // __LD,__compact_unwind is harmless (it's ignored by the unwinder, which then
  // proceeds to process __TEXT,__eh_frame for its instructions).
  ENTRY("__LD",   "__compact_unwind", S_REGULAR,         typeCompactUnwindInfo),
};
#undef ENTRY


SectionInfo *Util::getFinalSection(DefinedAtom::ContentType atomType) {
  for (auto &p : sectsToAtomType) {
    if (p.atomType != atomType)
      continue;
    SectionAttr sectionAttrs = 0;
    switch (atomType) {
    case DefinedAtom::typeCode:
    case DefinedAtom::typeStub:
      sectionAttrs = S_ATTR_PURE_INSTRUCTIONS;
      break;
    default:
      break;
    }
    // If we already have a SectionInfo with this name, re-use it.
    // This can happen if two ContentType map to the same mach-o section.
    for (auto sect : _sectionMap) {
      if (sect.second->sectionName.equals(p.sectionName) &&
          sect.second->segmentName.equals(p.segmentName)) {
        return sect.second;
      }
    }
    // Otherwise allocate new SectionInfo object.
    SectionInfo *sect = new (_allocator) SectionInfo(p.segmentName,
                                                     p.sectionName,
                                                     p.sectionType,
                                                     _context,
                                                     sectionAttrs);
    _sectionInfos.push_back(sect);
    _sectionMap[atomType] = sect;
    return sect;
  }
  llvm_unreachable("content type not yet supported");
}



SectionInfo *Util::sectionForAtom(const DefinedAtom *atom) {
  if (atom->sectionChoice() == DefinedAtom::sectionBasedOnContent) {
    // Section for this atom is derived from content type.
    DefinedAtom::ContentType type = atom->contentType();
    auto pos = _sectionMap.find(type);
    if ( pos != _sectionMap.end() )
      return pos->second;
    bool rMode = (_context.outputMachOType() == llvm::MachO::MH_OBJECT);
    return rMode ? getRelocatableSection(type) : getFinalSection(type);
  } else {
    // This atom needs to be in a custom section.
    StringRef customName = atom->customSectionName();
    // Look to see if we have already allocated the needed custom section.
    for(SectionInfo *sect : _customSections) {
      const DefinedAtom *firstAtom = sect->atomsAndOffsets.front().atom;
      if (firstAtom->customSectionName().equals(customName)) {
        return sect;
      }
    }
    // Not found, so need to create a new custom section.
    size_t seperatorIndex = customName.find('/');
    assert(seperatorIndex != StringRef::npos);
    StringRef segName = customName.slice(0, seperatorIndex);
    StringRef sectName = customName.drop_front(seperatorIndex + 1);
    SectionInfo *sect = new (_allocator) SectionInfo(segName, sectName,
                                                    S_REGULAR, _context);
    _customSections.push_back(sect);
    _sectionInfos.push_back(sect);
    return sect;
  }
}


void Util::appendAtom(SectionInfo *sect, const DefinedAtom *atom) {
  // Figure out offset for atom in this section given alignment constraints.
  uint64_t offset = sect->size;
  DefinedAtom::Alignment atomAlign = atom->alignment();
  uint64_t align2 = 1 << atomAlign.powerOf2;
  uint64_t requiredModulus = atomAlign.modulus;
  uint64_t currentModulus = (offset % align2);
  if ( currentModulus != requiredModulus ) {
    if ( requiredModulus > currentModulus )
      offset += requiredModulus-currentModulus;
    else
      offset += align2+requiredModulus-currentModulus;
  }
  // Record max alignment of any atom in this section.
  if ( atomAlign.powerOf2 > sect->alignment )
    sect->alignment = atomAlign.powerOf2;
  // Assign atom to this section with this offset.
  AtomInfo ai = {atom, offset};
  sect->atomsAndOffsets.push_back(ai);
  // Update section size to include this atom.
  sect->size = offset + atom->size();
}

void Util::assignAtomsToSections(const lld::File &atomFile) {
  for (const DefinedAtom *atom : atomFile.defined()) {
    appendAtom(sectionForAtom(atom), atom);
  }
}

SegmentInfo *Util::segmentForName(StringRef segName) {
  for (SegmentInfo *si : _segmentInfos) {
    if ( si->name.equals(segName) )
      return si;
  }
  SegmentInfo *info = new (_allocator) SegmentInfo(segName);
  if (segName.equals("__TEXT"))
    info->access = VM_PROT_READ | VM_PROT_EXECUTE;
  else if (segName.equals("__DATA"))
    info->access = VM_PROT_READ | VM_PROT_WRITE;
  else if (segName.equals("__PAGEZERO"))
    info->access = 0;
  _segmentInfos.push_back(info);
  return info;
}

unsigned Util::SegmentSorter::weight(const SegmentInfo *seg) {
 return llvm::StringSwitch<unsigned>(seg->name)
    .Case("__PAGEZERO",  1)
    .Case("__TEXT",      2)
    .Case("__DATA",      3)
    .Default(100);
}

bool Util::SegmentSorter::operator()(const SegmentInfo *left,
                                  const SegmentInfo *right) {
  return (weight(left) < weight(right));
}

unsigned Util::TextSectionSorter::weight(const SectionInfo *sect) {
 return llvm::StringSwitch<unsigned>(sect->sectionName)
    .Case("__text",         1)
    .Case("__stubs",        2)
    .Case("__stub_helper",  3)
    .Case("__const",        4)
    .Case("__cstring",      5)
    .Case("__unwind_info",  98)
    .Case("__eh_frame",     99)
    .Default(10);
}

bool Util::TextSectionSorter::operator()(const SectionInfo *left,
                                         const SectionInfo *right) {
  return (weight(left) < weight(right));
}


void Util::organizeSections() {
  if (_context.outputMachOType() == llvm::MachO::MH_OBJECT) {
    // Leave sections ordered as normalized file specified.
    uint32_t sectionIndex = 1;
    for (SectionInfo *si : _sectionInfos) {
      si->finalSectionIndex = sectionIndex++;
    }
  } else {
    // Main executables, need a zero-page segment
    if (_context.outputMachOType() == llvm::MachO::MH_EXECUTE)
      segmentForName("__PAGEZERO");
    // Group sections into segments.
    for (SectionInfo *si : _sectionInfos) {
      SegmentInfo *seg = segmentForName(si->segmentName);
      seg->sections.push_back(si);
    }
    // Sort segments.
    std::sort(_segmentInfos.begin(), _segmentInfos.end(), SegmentSorter());

    // Sort sections within segments.
    for (SegmentInfo *seg : _segmentInfos) {
      if (seg->name.equals("__TEXT")) {
        std::sort(seg->sections.begin(), seg->sections.end(),
                                                          TextSectionSorter());
      }
    }

    // Record final section indexes.
    uint32_t segmentIndex = 0;
    uint32_t sectionIndex = 1;
    for (SegmentInfo *seg : _segmentInfos) {
      seg->normalizedSegmentIndex = segmentIndex++;
      for (SectionInfo *sect : seg->sections) {
        sect->finalSectionIndex = sectionIndex++;
      }
    }
  }

}

uint64_t Util::alignTo(uint64_t value, uint8_t align2) {
  return llvm::RoundUpToAlignment(value, 1 << align2);
}


void Util::layoutSectionsInSegment(SegmentInfo *seg, uint64_t &addr) {
  seg->address = addr;
  for (SectionInfo *sect : seg->sections) {
    sect->address = alignTo(addr, sect->alignment);
    addr += sect->size;
  }
  seg->size = llvm::RoundUpToAlignment(addr - seg->address,_context.pageSize());
}


// __TEXT segment lays out backwards so padding is at front after load commands.
void Util::layoutSectionsInTextSegment(size_t hlcSize, SegmentInfo *seg,
                                                               uint64_t &addr) {
  seg->address = addr;
  // Walks sections starting at end to calculate padding for start.
  int64_t taddr = 0;
  for (auto it = seg->sections.rbegin(); it != seg->sections.rend(); ++it) {
    SectionInfo *sect = *it;
    taddr -= sect->size;
    taddr = taddr & (0 - (1 << sect->alignment));
  }
  int64_t padding = taddr - hlcSize;
  while (padding < 0)
    padding += _context.pageSize();
  // Start assigning section address starting at padded offset.
  addr += (padding + hlcSize);
  for (SectionInfo *sect : seg->sections) {
    sect->address = alignTo(addr, sect->alignment);
    addr = sect->address + sect->size;
  }
  seg->size = llvm::RoundUpToAlignment(addr - seg->address,_context.pageSize());
}


void Util::assignAddressesToSections(const NormalizedFile &file) {
  size_t hlcSize = headerAndLoadCommandsSize(file);
  uint64_t address = 0;  // FIXME
  if (_context.outputMachOType() != llvm::MachO::MH_OBJECT) {
    for (SegmentInfo *seg : _segmentInfos) {
      if (seg->name.equals("__PAGEZERO")) {
        seg->size = _context.pageZeroSize();
        address += seg->size;
      }
      else if (seg->name.equals("__TEXT"))
        layoutSectionsInTextSegment(hlcSize, seg, address);
      else
        layoutSectionsInSegment(seg, address);

      address = llvm::RoundUpToAlignment(address, _context.pageSize());
    }
    DEBUG_WITH_TYPE("WriterMachO-norm",
      llvm::dbgs() << "assignAddressesToSections()\n";
      for (SegmentInfo *sgi : _segmentInfos) {
        llvm::dbgs()  << "   address=" << llvm::format("0x%08llX", sgi->address)
                      << ", size="  << llvm::format("0x%08llX", sgi->size)
                      << ", segment-name='" << sgi->name
                      << "'\n";
        for (SectionInfo *si : sgi->sections) {
          llvm::dbgs()<< "      addr="  << llvm::format("0x%08llX", si->address)
                      << ", size="  << llvm::format("0x%08llX", si->size)
                      << ", section-name='" << si->sectionName
                      << "\n";
        }
      }
    );
  } else {
    for (SectionInfo *sect : _sectionInfos) {
      sect->address = alignTo(address, sect->alignment);
      address = sect->address + sect->size;
    }
    DEBUG_WITH_TYPE("WriterMachO-norm",
      llvm::dbgs() << "assignAddressesToSections()\n";
      for (SectionInfo *si : _sectionInfos) {
        llvm::dbgs()  << "      section=" << si->sectionName
                      << " address= "  << llvm::format("0x%08X", si->address)
                      << " size= "  << llvm::format("0x%08X", si->size)
                      << "\n";
      }
    );
  }
}


void Util::copySegmentInfo(NormalizedFile &file) {
  for (SegmentInfo *sgi : _segmentInfos) {
    Segment seg;
    seg.name    = sgi->name;
    seg.address = sgi->address;
    seg.size    = sgi->size;
    seg.access  = sgi->access;
    file.segments.push_back(seg);
  }
}

void Util::appendSection(SectionInfo *si, NormalizedFile &file) {
   // Add new empty section to end of file.sections.
  Section temp;
  file.sections.push_back(std::move(temp));
  Section* normSect = &file.sections.back();
  // Copy fields to normalized section.
  normSect->segmentName   = si->segmentName;
  normSect->sectionName   = si->sectionName;
  normSect->type          = si->type;
  normSect->attributes    = si->attributes;
  normSect->address       = si->address;
  normSect->alignment     = si->alignment;
  // Record where normalized section is.
  si->normalizedSectionIndex = file.sections.size()-1;
}

void Util::copySectionContent(NormalizedFile &file) {
  const bool r = (_context.outputMachOType() == llvm::MachO::MH_OBJECT);

  // Utility function for ArchHandler to find address of atom in output file.
  auto addrForAtom = [&] (const Atom &atom) -> uint64_t {
    auto pos = _atomToAddress.find(&atom);
    assert(pos != _atomToAddress.end());
    return pos->second;
  };

  for (SectionInfo *si : _sectionInfos) {
    if (si->type == llvm::MachO::S_ZEROFILL)
      continue;
    // Copy content from atoms to content buffer for section.
    uint8_t *sectionContent = file.ownedAllocations.Allocate<uint8_t>(si->size);
    Section *normSect = &file.sections[si->normalizedSectionIndex];
    normSect->content = llvm::makeArrayRef(sectionContent, si->size);
    for (AtomInfo &ai : si->atomsAndOffsets) {
      uint8_t *atomContent = reinterpret_cast<uint8_t*>
                                          (&sectionContent[ai.offsetInSection]);
      _archHandler.generateAtomContent(*ai.atom, r, addrForAtom, atomContent);
    }
  }
}


void Util::copySectionInfo(NormalizedFile &file) {
  file.sections.reserve(_sectionInfos.size());
  // For final linked images, write sections grouped by segment.
  if (_context.outputMachOType() != llvm::MachO::MH_OBJECT) {
    for (SegmentInfo *sgi : _segmentInfos) {
      for (SectionInfo *si : sgi->sections) {
        appendSection(si, file);
      }
    }
  } else {
    // Object files write sections in default order.
    for (SectionInfo *si : _sectionInfos) {
      appendSection(si, file);
    }
  }
}

void Util::updateSectionInfo(NormalizedFile &file) {
  file.sections.reserve(_sectionInfos.size());
  if (_context.outputMachOType() != llvm::MachO::MH_OBJECT) {
    // For final linked images, sections grouped by segment.
    for (SegmentInfo *sgi : _segmentInfos) {
      Segment *normSeg = &file.segments[sgi->normalizedSegmentIndex];
      normSeg->address = sgi->address;
      normSeg->size = sgi->size;
      for (SectionInfo *si : sgi->sections) {
        Section *normSect = &file.sections[si->normalizedSectionIndex];
        normSect->address = si->address;
      }
    }
  } else {
    // Object files write sections in default order.
    for (SectionInfo *si : _sectionInfos) {
      Section *normSect = &file.sections[si->normalizedSectionIndex];
      normSect->address = si->address;
    }
  }
}

void Util::copyEntryPointAddress(NormalizedFile &nFile) {
  if (_context.outputTypeHasEntry()) {
    if (_archHandler.isThumbFunction(*_entryAtom))
      nFile.entryAddress = (_atomToAddress[_entryAtom] | 1);
    else
      nFile.entryAddress = _atomToAddress[_entryAtom];
  }
}

void Util::buildAtomToAddressMap() {
  DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
                   << "assign atom addresses:\n");
  const bool lookForEntry = _context.outputTypeHasEntry();
  for (SectionInfo *sect : _sectionInfos) {
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      _atomToAddress[info.atom] = sect->address + info.offsetInSection;
      if (lookForEntry && (info.atom->contentType() == DefinedAtom::typeCode) &&
          (info.atom->size() != 0) &&
          info.atom->name() == _context.entrySymbolName()) {
        _entryAtom = info.atom;
      }
      DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
              << "   address="
              << llvm::format("0x%016X", _atomToAddress[info.atom])
              << " atom=" << info.atom
              << " name=" << info.atom->name() << "\n");
    }
  }
}

uint16_t Util::descBits(const DefinedAtom* atom) {
  uint16_t desc = 0;
  switch (atom->merge()) {
  case lld::DefinedAtom::mergeNo:
  case lld::DefinedAtom::mergeAsTentative:
    break;
  case lld::DefinedAtom::mergeAsWeak:
  case lld::DefinedAtom::mergeAsWeakAndAddressUsed:
    desc |= N_WEAK_DEF;
    break;
  case lld::DefinedAtom::mergeSameNameAndSize:
  case lld::DefinedAtom::mergeByLargestSection:
  case lld::DefinedAtom::mergeByContent:
    llvm_unreachable("Unsupported DefinedAtom::merge()");
    break;
  }
  if (atom->contentType() == lld::DefinedAtom::typeResolver)
    desc |= N_SYMBOL_RESOLVER;
  if (_archHandler.isThumbFunction(*atom))
    desc |= N_ARM_THUMB_DEF;
  if (atom->deadStrip() == DefinedAtom::deadStripNever) {
    if ((atom->contentType() != DefinedAtom::typeInitializerPtr)
     && (atom->contentType() != DefinedAtom::typeTerminatorPtr))
    desc |= N_NO_DEAD_STRIP;
  }
  return desc;
}


bool Util::AtomSorter::operator()(const AtomAndIndex &left,
                                  const AtomAndIndex &right) {
  return (left.atom->name().compare(right.atom->name()) < 0);
}


std::error_code Util::getSymbolTableRegion(const DefinedAtom* atom,
                                           bool &inGlobalsRegion,
                                           SymbolScope &scope) {
  bool rMode = (_context.outputMachOType() == llvm::MachO::MH_OBJECT);
  switch (atom->scope()) {
  case Atom::scopeTranslationUnit:
    scope = 0;
    inGlobalsRegion = false;
    return std::error_code();
  case Atom::scopeLinkageUnit:
    if ((_context.exportMode() == MachOLinkingContext::ExportMode::whiteList)
        && _context.exportSymbolNamed(atom->name())) {
      return make_dynamic_error_code(Twine("cannot export hidden symbol ")
                                    + atom->name());
    }
    if (rMode) {
      if (_context.keepPrivateExterns()) {
        // -keep_private_externs means keep in globals region as N_PEXT.
        scope = N_PEXT | N_EXT;
        inGlobalsRegion = true;
        return std::error_code();
      }
    }
    // scopeLinkageUnit symbols are no longer global once linked.
    scope = N_PEXT;
    inGlobalsRegion = false;
    return std::error_code();
  case Atom::scopeGlobal:
    if (_context.exportRestrictMode()) {
      if (_context.exportSymbolNamed(atom->name())) {
        scope = N_EXT;
        inGlobalsRegion = true;
        return std::error_code();
      } else {
        scope = N_PEXT;
        inGlobalsRegion = false;
        return std::error_code();
      }
    } else {
      scope = N_EXT;
      inGlobalsRegion = true;
      return std::error_code();
    }
    break;
  }
}

std::error_code Util::addSymbols(const lld::File &atomFile,
                                 NormalizedFile &file) {
  bool rMode = (_context.outputMachOType() == llvm::MachO::MH_OBJECT);
  // Mach-O symbol table has three regions: locals, globals, undefs.

  // Add all local (non-global) symbols in address order
  std::vector<AtomAndIndex> globals;
  globals.reserve(512);
  for (SectionInfo *sect : _sectionInfos) {
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      if (!atom->name().empty()) {
        SymbolScope symbolScope;
        bool inGlobalsRegion;
        if (auto ec = getSymbolTableRegion(atom, inGlobalsRegion, symbolScope)){
          return ec;
        }
        if (inGlobalsRegion) {
          AtomAndIndex ai = { atom, sect->finalSectionIndex, symbolScope };
          globals.push_back(ai);
        } else {
          Symbol sym;
          sym.name  = atom->name();
          sym.type  = N_SECT;
          sym.scope = symbolScope;
          sym.sect  = sect->finalSectionIndex;
          sym.desc  = descBits(atom);
          sym.value = _atomToAddress[atom];
          _atomToSymbolIndex[atom] = file.localSymbols.size();
          file.localSymbols.push_back(sym);
        }
      } else if (rMode && _archHandler.needsLocalSymbolInRelocatableFile(atom)){
        // Create 'Lxxx' labels for anonymous atoms if archHandler says so.
        static unsigned tempNum = 1;
        char tmpName[16];
        sprintf(tmpName, "L%04u", tempNum++);
        StringRef tempRef(tmpName);
        Symbol sym;
        sym.name  = tempRef.copy(file.ownedAllocations);
        sym.type  = N_SECT;
        sym.scope = 0;
        sym.sect  = sect->finalSectionIndex;
        sym.desc  = 0;
        sym.value = _atomToAddress[atom];
        _atomToSymbolIndex[atom] = file.localSymbols.size();
        file.localSymbols.push_back(sym);
      }
    }
  }

  // Sort global symbol alphabetically, then add to symbol table.
  std::sort(globals.begin(), globals.end(), AtomSorter());
  const uint32_t globalStartIndex = file.localSymbols.size();
  for (AtomAndIndex &ai : globals) {
    Symbol sym;
    sym.name  = ai.atom->name();
    sym.type  = N_SECT;
    sym.scope = ai.scope;
    sym.sect  = ai.index;
    sym.desc  = descBits(static_cast<const DefinedAtom*>(ai.atom));
    sym.value = _atomToAddress[ai.atom];
    _atomToSymbolIndex[ai.atom] = globalStartIndex + file.globalSymbols.size();
    file.globalSymbols.push_back(sym);
  }


  // Sort undefined symbol alphabetically, then add to symbol table.
  std::vector<AtomAndIndex> undefs;
  undefs.reserve(128);
  for (const UndefinedAtom *atom : atomFile.undefined()) {
    AtomAndIndex ai = { atom, 0, N_EXT };
    undefs.push_back(ai);
  }
  for (const SharedLibraryAtom *atom : atomFile.sharedLibrary()) {
    AtomAndIndex ai = { atom, 0, N_EXT };
    undefs.push_back(ai);
  }
  std::sort(undefs.begin(), undefs.end(), AtomSorter());
  const uint32_t start = file.globalSymbols.size() + file.localSymbols.size();
  for (AtomAndIndex &ai : undefs) {
    Symbol sym;
    uint16_t desc = 0;
    if (!rMode) {
      uint8_t ordinal = dylibOrdinal(dyn_cast<SharedLibraryAtom>(ai.atom));
      llvm::MachO::SET_LIBRARY_ORDINAL(desc, ordinal);
    }
    sym.name  = ai.atom->name();
    sym.type  = N_UNDF;
    sym.scope = ai.scope;
    sym.sect  = 0;
    sym.desc  = desc;
    sym.value = 0;
    _atomToSymbolIndex[ai.atom] = file.undefinedSymbols.size() + start;
    file.undefinedSymbols.push_back(sym);
  }

  return std::error_code();
}

const Atom *Util::targetOfLazyPointer(const DefinedAtom *lpAtom) {
  for (const Reference *ref : *lpAtom) {
    if (_archHandler.isLazyPointer(*ref)) {
      return ref->target();
    }
  }
  return nullptr;
}

const Atom *Util::targetOfStub(const DefinedAtom *stubAtom) {
  for (const Reference *ref : *stubAtom) {
    if (const Atom *ta = ref->target()) {
      if (const DefinedAtom *lpAtom = dyn_cast<DefinedAtom>(ta)) {
        const Atom *target = targetOfLazyPointer(lpAtom);
        if (target)
          return target;
      }
    }
  }
  return nullptr;
}


void Util::addIndirectSymbols(const lld::File &atomFile, NormalizedFile &file) {
  for (SectionInfo *si : _sectionInfos) {
    Section &normSect = file.sections[si->normalizedSectionIndex];
    switch (si->type) {
    case llvm::MachO::S_NON_LAZY_SYMBOL_POINTERS:
      for (const AtomInfo &info : si->atomsAndOffsets) {
        bool foundTarget = false;
        for (const Reference *ref : *info.atom) {
          const Atom *target = ref->target();
          if (target) {
            if (isa<const SharedLibraryAtom>(target)) {
              uint32_t index = _atomToSymbolIndex[target];
              normSect.indirectSymbols.push_back(index);
              foundTarget = true;
            } else {
              normSect.indirectSymbols.push_back(
                                            llvm::MachO::INDIRECT_SYMBOL_LOCAL);
            }
          }
        }
        if (!foundTarget) {
          normSect.indirectSymbols.push_back(
                                             llvm::MachO::INDIRECT_SYMBOL_ABS);
        }
      }
      break;
    case llvm::MachO::S_LAZY_SYMBOL_POINTERS:
      for (const AtomInfo &info : si->atomsAndOffsets) {
        const Atom *target = targetOfLazyPointer(info.atom);
        if (target) {
          uint32_t index = _atomToSymbolIndex[target];
          normSect.indirectSymbols.push_back(index);
        }
      }
      break;
    case llvm::MachO::S_SYMBOL_STUBS:
      for (const AtomInfo &info : si->atomsAndOffsets) {
        const Atom *target = targetOfStub(info.atom);
        if (target) {
          uint32_t index = _atomToSymbolIndex[target];
          normSect.indirectSymbols.push_back(index);
        }
      }
      break;
    default:
      break;
    }
  }

}

void Util::addDependentDylibs(const lld::File &atomFile,NormalizedFile &nFile) {
  // Scan all imported symbols and build up list of dylibs they are from.
  int ordinal = 1;
  for (const SharedLibraryAtom *slAtom : atomFile.sharedLibrary()) {
    StringRef loadPath = slAtom->loadName();
    DylibPathToInfo::iterator pos = _dylibInfo.find(loadPath);
    if (pos == _dylibInfo.end()) {
      DylibInfo info;
      info.ordinal = ordinal++;
      info.hasWeak = slAtom->canBeNullAtRuntime();
      info.hasNonWeak = !info.hasWeak;
      _dylibInfo[loadPath] = info;
      DependentDylib depInfo;
      depInfo.path = loadPath;
      depInfo.kind = llvm::MachO::LC_LOAD_DYLIB;
      nFile.dependentDylibs.push_back(depInfo);
    } else {
      if ( slAtom->canBeNullAtRuntime() )
        pos->second.hasWeak = true;
      else
        pos->second.hasNonWeak = true;
    }
  }
  // Automatically weak link dylib in which all symbols are weak (canBeNull).
  for (DependentDylib &dep : nFile.dependentDylibs) {
    DylibInfo &info = _dylibInfo[dep.path];
    if (info.hasWeak && !info.hasNonWeak)
      dep.kind = llvm::MachO::LC_LOAD_WEAK_DYLIB;
  }
}


int Util::dylibOrdinal(const SharedLibraryAtom *sa) {
  return _dylibInfo[sa->loadName()].ordinal;
}

void Util::segIndexForSection(const SectionInfo *sect, uint8_t &segmentIndex,
                                                  uint64_t &segmentStartAddr) {
  segmentIndex = 0;
  for (const SegmentInfo *seg : _segmentInfos) {
    if ((seg->address <= sect->address)
      && (seg->address+seg->size >= sect->address+sect->size)) {
      segmentStartAddr = seg->address;
      return;
    }
    ++segmentIndex;
  }
  llvm_unreachable("section not in any segment");
}


uint32_t Util::sectionIndexForAtom(const Atom *atom) {
  uint64_t address = _atomToAddress[atom];
  uint32_t index = 1;
  for (const SectionInfo *si : _sectionInfos) {
    if ((si->address <= address) && (address < si->address+si->size))
      return index;
    ++index;
  }
  llvm_unreachable("atom not in any section");
}

void Util::addSectionRelocs(const lld::File &, NormalizedFile &file) {
  if (_context.outputMachOType() != llvm::MachO::MH_OBJECT)
    return;


  // Utility function for ArchHandler to find symbol index for an atom.
  auto symIndexForAtom = [&] (const Atom &atom) -> uint32_t {
    auto pos = _atomToSymbolIndex.find(&atom);
    assert(pos != _atomToSymbolIndex.end());
    return pos->second;
  };

  // Utility function for ArchHandler to find section index for an atom.
  auto sectIndexForAtom = [&] (const Atom &atom) -> uint32_t {
    return sectionIndexForAtom(&atom);
  };

  // Utility function for ArchHandler to find address of atom in output file.
  auto addressForAtom = [&] (const Atom &atom) -> uint64_t {
    auto pos = _atomToAddress.find(&atom);
    assert(pos != _atomToAddress.end());
    return pos->second;
  };

  for (SectionInfo *si : _sectionInfos) {
    Section &normSect = file.sections[si->normalizedSectionIndex];
    for (const AtomInfo &info : si->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      for (const Reference *ref : *atom) {
        _archHandler.appendSectionRelocations(*atom, info.offsetInSection, *ref,
                                              symIndexForAtom,
                                              sectIndexForAtom,
                                              addressForAtom,
                                              normSect.relocations);
      }
    }
  }
}

void Util::buildDataInCodeArray(const lld::File &, NormalizedFile &file) {
  for (SectionInfo *si : _sectionInfos) {
    for (const AtomInfo &info : si->atomsAndOffsets) {
      // Atoms that contain data-in-code have "transition" references
      // which mark a point where the embedded data starts of ends.
      // This needs to be converted to the mach-o format which is an array
      // of data-in-code ranges.
      uint32_t startOffset = 0;
      DataRegionType mode = DataRegionType(0);
      for (const Reference *ref : *info.atom) {
        if (ref->kindNamespace() != Reference::KindNamespace::mach_o)
          continue;
        if (_archHandler.isDataInCodeTransition(ref->kindValue())) {
          DataRegionType nextMode = (DataRegionType)ref->addend();
          if (mode != nextMode) {
            if (mode != 0) {
              // Found end data range, so make range entry.
              DataInCode entry;
              entry.offset = si->address + info.offsetInSection + startOffset;
              entry.length = ref->offsetInAtom() - startOffset;
              entry.kind   = mode;
              file.dataInCode.push_back(entry);
            }
          }
          mode = nextMode;
          startOffset = ref->offsetInAtom();
        }
      }
      if (mode != 0) {
        // Function ends with data (no end transition).
        DataInCode entry;
        entry.offset = si->address + info.offsetInSection + startOffset;
        entry.length = info.atom->size() - startOffset;
        entry.kind   = mode;
        file.dataInCode.push_back(entry);
      }
    }
  }
}

void Util::addRebaseAndBindingInfo(const lld::File &atomFile,
                                                        NormalizedFile &nFile) {
  if (_context.outputMachOType() == llvm::MachO::MH_OBJECT)
    return;

  uint8_t segmentIndex;
  uint64_t segmentStartAddr;
  for (SectionInfo *sect : _sectionInfos) {
    segIndexForSection(sect, segmentIndex, segmentStartAddr);
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      for (const Reference *ref : *atom) {
        uint64_t segmentOffset = _atomToAddress[atom] + ref->offsetInAtom()
                                - segmentStartAddr;
        const Atom* targ = ref->target();
        if (_archHandler.isPointer(*ref)) {
          // A pointer to a DefinedAtom requires rebasing.
          if (dyn_cast<DefinedAtom>(targ)) {
            RebaseLocation rebase;
            rebase.segIndex = segmentIndex;
            rebase.segOffset = segmentOffset;
            rebase.kind = llvm::MachO::REBASE_TYPE_POINTER;
            nFile.rebasingInfo.push_back(rebase);
          }
          // A pointer to an SharedLibraryAtom requires binding.
          if (const SharedLibraryAtom *sa = dyn_cast<SharedLibraryAtom>(targ)) {
            BindLocation bind;
            bind.segIndex = segmentIndex;
            bind.segOffset = segmentOffset;
            bind.kind = llvm::MachO::BIND_TYPE_POINTER;
            bind.canBeNull = sa->canBeNullAtRuntime();
            bind.ordinal = dylibOrdinal(sa);
            bind.symbolName = targ->name();
            bind.addend = ref->addend();
            nFile.bindingInfo.push_back(bind);
          }
        }
        if (_archHandler.isLazyPointer(*ref)) {
            BindLocation bind;
            bind.segIndex = segmentIndex;
            bind.segOffset = segmentOffset;
            bind.kind = llvm::MachO::BIND_TYPE_POINTER;
            bind.canBeNull = false; //sa->canBeNullAtRuntime();
            bind.ordinal = 1; // FIXME
            bind.symbolName = targ->name();
            bind.addend = ref->addend();
            nFile.lazyBindingInfo.push_back(bind);
        }
      }
    }
  }
}

uint32_t Util::fileFlags() {
  // FIXME: these need to determined at runtime.
  if (_context.outputMachOType() == MH_OBJECT) {
    return MH_SUBSECTIONS_VIA_SYMBOLS;
  } else {
    return MH_DYLDLINK | MH_NOUNDEFS | MH_TWOLEVEL;
  }
}

} // end anonymous namespace


namespace lld {
namespace mach_o {
namespace normalized {

/// Convert a set of Atoms into a normalized mach-o file.
ErrorOr<std::unique_ptr<NormalizedFile>>
normalizedFromAtoms(const lld::File &atomFile,
                                           const MachOLinkingContext &context) {
  // The util object buffers info until the normalized file can be made.
  Util util(context);
  util.assignAtomsToSections(atomFile);
  util.organizeSections();

  std::unique_ptr<NormalizedFile> f(new NormalizedFile());
  NormalizedFile &normFile = *f.get();
  normFile.arch = context.arch();
  normFile.fileType = context.outputMachOType();
  normFile.flags = util.fileFlags();
  normFile.installName = context.installName();
  util.addDependentDylibs(atomFile, normFile);
  util.copySegmentInfo(normFile);
  util.copySectionInfo(normFile);
  util.assignAddressesToSections(normFile);
  util.buildAtomToAddressMap();
  util.updateSectionInfo(normFile);
  util.copySectionContent(normFile);
  if (auto ec = util.addSymbols(atomFile, normFile)) {
    return ec;
  }
  util.addIndirectSymbols(atomFile, normFile);
  util.addRebaseAndBindingInfo(atomFile, normFile);
  util.addSectionRelocs(atomFile, normFile);
  util.buildDataInCodeArray(atomFile, normFile);
  util.copyEntryPointAddress(normFile);

  return std::move(f);
}


} // namespace normalized
} // namespace mach_o
} // namespace lld

