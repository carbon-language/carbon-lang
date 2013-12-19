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
#include "ReferenceKinds.h"

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/system_error.h"

#include <map>

using llvm::StringRef;
using llvm::dyn_cast;
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
  SectionInfo(StringRef seg, StringRef sect, SectionType type, uint32_t attr=0);
  
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

SectionInfo::SectionInfo(StringRef sg, StringRef sct, SectionType t, uint32_t a) 
 : segmentName(sg), sectionName(sct), type(t), attributes(a), 
                 address(0), size(0), alignment(0), 
                 normalizedSectionIndex(0), finalSectionIndex(0) {
}

struct SegmentInfo {
  SegmentInfo(StringRef name);
  
  StringRef                  name;
  uint64_t                   address;
  uint64_t                   size;
  uint32_t                   access;
  std::vector<SectionInfo*>  sections;
};

SegmentInfo::SegmentInfo(StringRef n) 
 : name(n), address(0), size(0), access(0) {
}


class Util {
public:
  Util(const MachOLinkingContext &ctxt) : _context(ctxt), _entryAtom(nullptr) {}

  void      assignAtomsToSections(const lld::File &atomFile);
  void      organizeSections();
  void      assignAddressesToSections();
  uint32_t  fileFlags();
  void      copySegmentInfo(NormalizedFile &file);
  void      copySections(NormalizedFile &file);
  void      buildAtomToAddressMap();
  void      addSymbols(const lld::File &atomFile, NormalizedFile &file);
  void      addIndirectSymbols(const lld::File &atomFile, NormalizedFile &file);
  void      addRebaseAndBindingInfo(const lld::File &, NormalizedFile &file);
  void      addSectionRelocs(const lld::File &, NormalizedFile &file);
  void      addDependentDylibs(const lld::File &, NormalizedFile &file);
  void      copyEntryPointAddress(NormalizedFile &file);

private:
  typedef std::map<DefinedAtom::ContentType, SectionInfo*> TypeToSection;
  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;
  
  struct DylibInfo { int ordinal; bool hasWeak; bool hasNonWeak; };
  typedef llvm::StringMap<DylibInfo> DylibPathToInfo;
  
  SectionInfo *sectionForAtom(const DefinedAtom*);
  SectionInfo *makeSection(DefinedAtom::ContentType);
  void         appendAtom(SectionInfo *sect, const DefinedAtom *atom);
  SegmentInfo *segmentForName(StringRef segName);
  void         layoutSectionsInSegment(SegmentInfo *seg, uint64_t &addr);
  void         layoutSectionsInTextSegment(SegmentInfo *seg, uint64_t &addr);
  void         copySectionContent(SectionInfo *si, ContentBytes &content);
  uint8_t      scopeBits(const DefinedAtom* atom);
  int          dylibOrdinal(const SharedLibraryAtom *sa);
  void         segIndexForSection(const SectionInfo *sect, 
                             uint8_t &segmentIndex, uint64_t &segmentStartAddr);
  const Atom  *targetOfLazyPointer(const DefinedAtom *lpAtom);
  const Atom  *targetOfStub(const DefinedAtom *stubAtom);
  bool         belongsInGlobalSymbolsSection(const DefinedAtom* atom);
  void         appendSection(SectionInfo *si, NormalizedFile &file);
  void         appendReloc(const DefinedAtom *atom, const Reference *ref, 
                                                      Relocations &relocations);
  
  static uint64_t alignTo(uint64_t value, uint8_t align2);
  typedef llvm::DenseMap<const Atom*, uint32_t> AtomToIndex;
  struct AtomAndIndex { const Atom *atom; uint32_t index; };
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
  llvm::BumpPtrAllocator        _allocator;
  std::vector<SectionInfo*>     _sectionInfos;
  std::vector<SegmentInfo*>     _segmentInfos;
  TypeToSection                 _sectionMap;
  AtomToAddress                 _atomToAddress;
  DylibPathToInfo               _dylibInfo;
  const DefinedAtom            *_entryAtom;
  AtomToIndex                   _atomToSymbolIndex;
};

SectionInfo *Util::makeSection(DefinedAtom::ContentType type) {
  switch ( type ) {
  case DefinedAtom::typeCode:
    return new (_allocator) SectionInfo("__TEXT", "__text",
                             S_REGULAR, S_ATTR_PURE_INSTRUCTIONS 
                                      | S_ATTR_SOME_INSTRUCTIONS);
  case DefinedAtom::typeCString:
     return new (_allocator) SectionInfo("__TEXT", "__cstring",
                             S_CSTRING_LITERALS);
  case DefinedAtom::typeStub:
    return new (_allocator) SectionInfo("__TEXT", "__stubs",
                            S_SYMBOL_STUBS, S_ATTR_PURE_INSTRUCTIONS);
  case DefinedAtom::typeStubHelper:
    return new (_allocator) SectionInfo("__TEXT", "__stub_helper",
                            S_REGULAR, S_ATTR_PURE_INSTRUCTIONS);
  case DefinedAtom::typeLazyPointer:
    return new (_allocator) SectionInfo("__DATA", "__la_symbol_ptr",
                            S_LAZY_SYMBOL_POINTERS);
  case DefinedAtom::typeGOT:
     return new (_allocator) SectionInfo("__DATA", "__got",
                            S_NON_LAZY_SYMBOL_POINTERS);
  default:
    llvm_unreachable("TO DO: add support for more sections");
    break;
  }
}



SectionInfo *Util::sectionForAtom(const DefinedAtom *atom) {
  DefinedAtom::ContentType type = atom->contentType();
  auto pos = _sectionMap.find(type);
  if ( pos != _sectionMap.end() )
    return pos->second;
  SectionInfo *si = makeSection(type);
  _sectionInfos.push_back(si);
  _sectionMap[type] = si;
  return si;
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
  if (_context.outputFileType() == llvm::MachO::MH_OBJECT) {
    // Leave sections ordered as normalized file specified.
    uint32_t sectionIndex = 1;
    for (SectionInfo *si : _sectionInfos) {
      si->finalSectionIndex = sectionIndex++;
    }
  } else {
    // Main executables, need a zero-page segment
    if (_context.outputFileType() == llvm::MachO::MH_EXECUTE)
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
    uint32_t sectionIndex = 1;
    for (SegmentInfo *seg : _segmentInfos) {
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
void Util::layoutSectionsInTextSegment(SegmentInfo *seg, uint64_t &addr) {
  seg->address = addr;
  // Walks sections starting at end to calculate padding for start.
  int64_t taddr = 0;
  for (auto it = seg->sections.rbegin(); it != seg->sections.rend(); ++it) { 
    SectionInfo *sect = *it;
    taddr -= sect->size;
    taddr = taddr & (0 - (1 << sect->alignment));
  }
  int64_t padding = taddr;
  while (padding < 0)
    padding += _context.pageSize();
  // Start assigning section address starting at padded offset.
  addr += padding;
  for (SectionInfo *sect : seg->sections) {
    sect->address = alignTo(addr, sect->alignment);
    addr = sect->address + sect->size;
  }
  seg->size = llvm::RoundUpToAlignment(addr - seg->address,_context.pageSize());
}


void Util::assignAddressesToSections() {
  uint64_t address = 0;  // FIXME
  if (_context.outputFileType() != llvm::MachO::MH_OBJECT) {
    for (SegmentInfo *seg : _segmentInfos) {
      if (seg->name.equals("__PAGEZERO")) {
        seg->size = _context.pageZeroSize();
        address += seg->size;
      }
      else if (seg->name.equals("__TEXT"))
        layoutSectionsInTextSegment(seg, address);
      else
        layoutSectionsInSegment(seg, address);
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
  // Copy content from atoms to content buffer for section.
  // FIXME: zerofill atoms/sections should not take up content space.
  normSect->content.resize(si->size);
  Hex8 *sectionContent = normSect->content.data();
  for (AtomInfo &ai : si->atomsAndOffsets) {
    // Copy raw bytes.
    uint8_t *atomContent = reinterpret_cast<uint8_t*>
                                          (&sectionContent[ai.offsetInSection]);
    memcpy(atomContent, ai.atom->rawContent().data(), ai.atom->size());
    // Apply fix-ups.
    for (const Reference *ref : *ai.atom) {
      uint32_t offset = ref->offsetInAtom();
      uint64_t targetAddress = 0;
      if ( ref->target() != nullptr )
        targetAddress = _atomToAddress[ref->target()];
      uint64_t fixupAddress = _atomToAddress[ai.atom] + offset;
      _context.kindHandler().applyFixup(ref->kindNamespace(), ref->kindArch(), 
                                       ref->kindValue(), ref->addend(),
                                       &atomContent[offset], fixupAddress,
                                       targetAddress);
    }
  }
}

void Util::copySections(NormalizedFile &file) {
  file.sections.reserve(_sectionInfos.size());
  // For final linked images, write sections grouped by segment.
  if (_context.outputFileType() != llvm::MachO::MH_OBJECT) {
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

void Util::copyEntryPointAddress(NormalizedFile &nFile) {
  if (_context.outputTypeHasEntry()) {
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

uint8_t Util::scopeBits(const DefinedAtom* atom) {
  switch (atom->scope()) {
  case Atom::scopeTranslationUnit:
    return 0;
  case Atom::scopeLinkageUnit:
    return N_PEXT | N_EXT;
  case Atom::scopeGlobal:
    return N_EXT;
  }
  llvm_unreachable("Unknown scope");
}

bool Util::AtomSorter::operator()(const AtomAndIndex &left, 
                                  const AtomAndIndex &right) {
  return (left.atom->name().compare(right.atom->name()) < 0);
}

 
bool Util::belongsInGlobalSymbolsSection(const DefinedAtom* atom) {
  return (atom->scope() == Atom::scopeGlobal);
}

void Util::addSymbols(const lld::File &atomFile, NormalizedFile &file) {
  // Mach-O symbol table has three regions: locals, globals, undefs.
  
  // Add all local (non-global) symbols in address order
  std::vector<AtomAndIndex> globals;
  globals.reserve(512);
  for (SectionInfo *sect : _sectionInfos) {
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      if (!atom->name().empty()) {
        if (belongsInGlobalSymbolsSection(atom)) {
          AtomAndIndex ai = { atom, sect->finalSectionIndex };
          globals.push_back(ai);
        } else {
          Symbol sym;
          sym.name  = atom->name();
          sym.type  = N_SECT; 
          sym.scope = scopeBits(atom);
          sym.sect  = sect->finalSectionIndex;
          sym.desc  = 0;
          sym.value = _atomToAddress[atom];
          file.localSymbols.push_back(sym);
        }
      }
    }
  }
  
  // Sort global symbol alphabetically, then add to symbol table.
  std::sort(globals.begin(), globals.end(), AtomSorter());
  for (AtomAndIndex &ai : globals) {
    Symbol sym;
    sym.name  = ai.atom->name();
    sym.type  = N_SECT; 
    sym.scope = scopeBits(static_cast<const DefinedAtom*>(ai.atom));
    sym.sect  = ai.index;
    sym.desc  = 0;
    sym.value = _atomToAddress[ai.atom];
    file.globalSymbols.push_back(sym);
  }
  
  
  // Sort undefined symbol alphabetically, then add to symbol table.
  std::vector<AtomAndIndex> undefs;
  undefs.reserve(128);
  for (const UndefinedAtom *atom : atomFile.undefined()) {
    AtomAndIndex ai = { atom, 0 };
    undefs.push_back(ai);
  }
  for (const SharedLibraryAtom *atom : atomFile.sharedLibrary()) {
    AtomAndIndex ai = { atom, 0 };
    undefs.push_back(ai);
  }
  std::sort(undefs.begin(), undefs.end(), AtomSorter());
  const uint32_t start = file.globalSymbols.size() + file.localSymbols.size();
  for (AtomAndIndex &ai : undefs) {
    Symbol sym;
    sym.name  = ai.atom->name();
    sym.type  = N_UNDF; 
    sym.scope = N_EXT;
    sym.sect  = 0;
    sym.desc  = 0;
    sym.value = 0;
    _atomToSymbolIndex[ai.atom] = file.undefinedSymbols.size() + start;
    file.undefinedSymbols.push_back(sym);
  }
}

const Atom *Util::targetOfLazyPointer(const DefinedAtom *lpAtom) {
  for (const Reference *ref : *lpAtom) {
    if (_context.kindHandler().isLazyTarget(*ref)) {
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


void Util::appendReloc(const DefinedAtom *atom, const Reference *ref, 
                                                     Relocations &relocations) {
  // TODO: convert Reference to normalized relocation
}

void Util::addSectionRelocs(const lld::File &, NormalizedFile &file) {
  if (_context.outputFileType() != llvm::MachO::MH_OBJECT)
    return;
  
  for (SectionInfo *si : _sectionInfos) {
    Section &normSect = file.sections[si->normalizedSectionIndex];
    for (const AtomInfo &info : si->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      for (const Reference *ref : *atom) {
        appendReloc(atom, ref, normSect.relocations);
      }
    }
  }
}

void Util::addRebaseAndBindingInfo(const lld::File &atomFile, 
                                                        NormalizedFile &nFile) {
  if (_context.outputFileType() == llvm::MachO::MH_OBJECT)
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
        if (_context.kindHandler().isPointer(*ref)) {
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
        if (_context.kindHandler().isLazyTarget(*ref)) {
            BindLocation bind;
            bind.segIndex = segmentIndex;
            bind.segOffset = segmentOffset;
            bind.kind = llvm::MachO::BIND_TYPE_POINTER;
            bind.canBeNull = false; //sa->canBeNullAtRuntime();
            bind.ordinal = 1;
            bind.symbolName = targ->name(); 
            bind.addend = ref->addend();
            nFile.lazyBindingInfo.push_back(bind);
        }
      }
    }
  }
}

uint32_t Util::fileFlags() {
  return 0;  //FIX ME
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
  util.assignAddressesToSections();
  util.buildAtomToAddressMap();
  
  std::unique_ptr<NormalizedFile> f(new NormalizedFile());
  NormalizedFile &normFile = *f.get();
  f->arch = context.arch();
  f->fileType = context.outputFileType();
  f->flags = util.fileFlags();
  util.copySegmentInfo(normFile);
  util.copySections(normFile);
  util.addDependentDylibs(atomFile, normFile);
  util.addSymbols(atomFile, normFile);
  util.addIndirectSymbols(atomFile, normFile);
  util.addRebaseAndBindingInfo(atomFile, normFile);
  util.addSectionRelocs(atomFile, normFile);
  util.copyEntryPointAddress(normFile);
 
  return std::move(f);
}


} // namespace normalized
} // namespace mach_o
} // namespace lld

