//===-- RuntimeDyld.cpp - Run-time dynamic linker for MC-JIT ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "JITRegistrar.h"
#include "ObjectImageCommon.h"
#include "RuntimeDyldELF.h"
#include "RuntimeDyldImpl.h"
#include "RuntimeDyldMachO.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MutexGuard.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "dyld"

// Empty out-of-line virtual destructor as the key function.
RuntimeDyldImpl::~RuntimeDyldImpl() {}

// Pin the JITRegistrar's and ObjectImage*'s vtables to this file.
void JITRegistrar::anchor() {}
void ObjectImage::anchor() {}
void ObjectImageCommon::anchor() {}

namespace llvm {

void RuntimeDyldImpl::registerEHFrames() {}

void RuntimeDyldImpl::deregisterEHFrames() {}

// Resolve the relocations for all symbols we currently know about.
void RuntimeDyldImpl::resolveRelocations() {
  MutexGuard locked(lock);

  // First, resolve relocations associated with external symbols.
  resolveExternalSymbols();

  // Just iterate over the sections we have and resolve all the relocations
  // in them. Gross overkill, but it gets the job done.
  for (int i = 0, e = Sections.size(); i != e; ++i) {
    // The Section here (Sections[i]) refers to the section in which the
    // symbol for the relocation is located.  The SectionID in the relocation
    // entry provides the section to which the relocation will be applied.
    uint64_t Addr = Sections[i].LoadAddress;
    DEBUG(dbgs() << "Resolving relocations Section #" << i << "\t"
                 << format("%p", (uint8_t *)Addr) << "\n");
    resolveRelocationList(Relocations[i], Addr);
    Relocations.erase(i);
  }
}

void RuntimeDyldImpl::mapSectionAddress(const void *LocalAddress,
                                        uint64_t TargetAddress) {
  MutexGuard locked(lock);
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    if (Sections[i].Address == LocalAddress) {
      reassignSectionAddress(i, TargetAddress);
      return;
    }
  }
  llvm_unreachable("Attempting to remap address of unknown section!");
}

static error_code getOffset(const SymbolRef &Sym, uint64_t &Result) {
  uint64_t Address;
  if (error_code EC = Sym.getAddress(Address))
    return EC;

  if (Address == UnknownAddressOrSize) {
    Result = UnknownAddressOrSize;
    return object_error::success;
  }

  const ObjectFile *Obj = Sym.getObject();
  section_iterator SecI(Obj->section_begin());
  if (error_code EC = Sym.getSection(SecI))
    return EC;

 if (SecI == Obj->section_end()) {
   Result = UnknownAddressOrSize;
   return object_error::success;
 }

  uint64_t SectionAddress;
  if (error_code EC = SecI->getAddress(SectionAddress))
    return EC;

  Result = Address - SectionAddress;
  return object_error::success;
}

ObjectImage *RuntimeDyldImpl::loadObject(ObjectImage *InputObject) {
  MutexGuard locked(lock);

  std::unique_ptr<ObjectImage> Obj(InputObject);
  if (!Obj)
    return nullptr;

  // Save information about our target
  Arch = (Triple::ArchType)Obj->getArch();
  IsTargetLittleEndian = Obj->getObjectFile()->isLittleEndian();

  // Compute the memory size required to load all sections to be loaded
  // and pass this information to the memory manager
  if (MemMgr->needsToReserveAllocationSpace()) {
    uint64_t CodeSize = 0, DataSizeRO = 0, DataSizeRW = 0;
    computeTotalAllocSize(*Obj, CodeSize, DataSizeRO, DataSizeRW);
    MemMgr->reserveAllocationSpace(CodeSize, DataSizeRO, DataSizeRW);
  }

  // Symbols found in this object
  StringMap<SymbolLoc> LocalSymbols;
  // Used sections from the object file
  ObjSectionToIDMap LocalSections;

  // Common symbols requiring allocation, with their sizes and alignments
  CommonSymbolMap CommonSymbols;
  // Maximum required total memory to allocate all common symbols
  uint64_t CommonSize = 0;

  // Parse symbols
  DEBUG(dbgs() << "Parse symbols:\n");
  for (symbol_iterator I = Obj->begin_symbols(), E = Obj->end_symbols(); I != E;
       ++I) {
    object::SymbolRef::Type SymType;
    StringRef Name;
    Check(I->getType(SymType));
    Check(I->getName(Name));

    uint32_t Flags = I->getFlags();

    bool IsCommon = Flags & SymbolRef::SF_Common;
    if (IsCommon) {
      // Add the common symbols to a list.  We'll allocate them all below.
      if (!GlobalSymbolTable.count(Name)) {
        uint32_t Align;
        Check(I->getAlignment(Align));
        uint64_t Size = 0;
        Check(I->getSize(Size));
        CommonSize += Size + Align;
        CommonSymbols[*I] = CommonSymbolInfo(Size, Align);
      }
    } else {
      if (SymType == object::SymbolRef::ST_Function ||
          SymType == object::SymbolRef::ST_Data ||
          SymType == object::SymbolRef::ST_Unknown) {
        uint64_t SectOffset;
        StringRef SectionData;
        bool IsCode;
        section_iterator SI = Obj->end_sections();
        Check(getOffset(*I, SectOffset));
        Check(I->getSection(SI));
        if (SI == Obj->end_sections())
          continue;
        Check(SI->getContents(SectionData));
        Check(SI->isText(IsCode));
        unsigned SectionID =
            findOrEmitSection(*Obj, *SI, IsCode, LocalSections);
        LocalSymbols[Name.data()] = SymbolLoc(SectionID, SectOffset);
        DEBUG(dbgs() << "\tOffset: " << format("%p", (uintptr_t)SectOffset)
                     << " flags: " << Flags << " SID: " << SectionID);
        GlobalSymbolTable[Name] = SymbolLoc(SectionID, SectOffset);
      }
    }
    DEBUG(dbgs() << "\tType: " << SymType << " Name: " << Name << "\n");
  }

  // Allocate common symbols
  if (CommonSize != 0)
    emitCommonSymbols(*Obj, CommonSymbols, CommonSize, GlobalSymbolTable);

  // Parse and process relocations
  DEBUG(dbgs() << "Parse relocations:\n");
  for (section_iterator SI = Obj->begin_sections(), SE = Obj->end_sections();
       SI != SE; ++SI) {
    unsigned SectionID = 0;
    StubMap Stubs;
    section_iterator RelocatedSection = SI->getRelocatedSection();

    relocation_iterator I = SI->relocation_begin();
    relocation_iterator E = SI->relocation_end();

    if (I == E && !ProcessAllSections)
      continue;

    bool IsCode = false;
    Check(RelocatedSection->isText(IsCode));
    SectionID =
        findOrEmitSection(*Obj, *RelocatedSection, IsCode, LocalSections);
    DEBUG(dbgs() << "\tSectionID: " << SectionID << "\n");

    for (; I != E;)
      I = processRelocationRef(SectionID, I, *Obj, LocalSections, LocalSymbols,
                               Stubs);
  }

  // Give the subclasses a chance to tie-up any loose ends.
  finalizeLoad(*Obj, LocalSections);

  return Obj.release();
}

// A helper method for computeTotalAllocSize.
// Computes the memory size required to allocate sections with the given sizes,
// assuming that all sections are allocated with the given alignment
static uint64_t
computeAllocationSizeForSections(std::vector<uint64_t> &SectionSizes,
                                 uint64_t Alignment) {
  uint64_t TotalSize = 0;
  for (size_t Idx = 0, Cnt = SectionSizes.size(); Idx < Cnt; Idx++) {
    uint64_t AlignedSize =
        (SectionSizes[Idx] + Alignment - 1) / Alignment * Alignment;
    TotalSize += AlignedSize;
  }
  return TotalSize;
}

// Compute an upper bound of the memory size that is required to load all
// sections
void RuntimeDyldImpl::computeTotalAllocSize(ObjectImage &Obj,
                                            uint64_t &CodeSize,
                                            uint64_t &DataSizeRO,
                                            uint64_t &DataSizeRW) {
  // Compute the size of all sections required for execution
  std::vector<uint64_t> CodeSectionSizes;
  std::vector<uint64_t> ROSectionSizes;
  std::vector<uint64_t> RWSectionSizes;
  uint64_t MaxAlignment = sizeof(void *);

  // Collect sizes of all sections to be loaded;
  // also determine the max alignment of all sections
  for (section_iterator SI = Obj.begin_sections(), SE = Obj.end_sections();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;

    bool IsRequired;
    Check(Section.isRequiredForExecution(IsRequired));

    // Consider only the sections that are required to be loaded for execution
    if (IsRequired) {
      uint64_t DataSize = 0;
      uint64_t Alignment64 = 0;
      bool IsCode = false;
      bool IsReadOnly = false;
      StringRef Name;
      Check(Section.getSize(DataSize));
      Check(Section.getAlignment(Alignment64));
      Check(Section.isText(IsCode));
      Check(Section.isReadOnlyData(IsReadOnly));
      Check(Section.getName(Name));
      unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;

      uint64_t StubBufSize = computeSectionStubBufSize(Obj, Section);
      uint64_t SectionSize = DataSize + StubBufSize;

      // The .eh_frame section (at least on Linux) needs an extra four bytes
      // padded
      // with zeroes added at the end.  For MachO objects, this section has a
      // slightly different name, so this won't have any effect for MachO
      // objects.
      if (Name == ".eh_frame")
        SectionSize += 4;

      if (SectionSize > 0) {
        // save the total size of the section
        if (IsCode) {
          CodeSectionSizes.push_back(SectionSize);
        } else if (IsReadOnly) {
          ROSectionSizes.push_back(SectionSize);
        } else {
          RWSectionSizes.push_back(SectionSize);
        }
        // update the max alignment
        if (Alignment > MaxAlignment) {
          MaxAlignment = Alignment;
        }
      }
    }
  }

  // Compute the size of all common symbols
  uint64_t CommonSize = 0;
  for (symbol_iterator I = Obj.begin_symbols(), E = Obj.end_symbols(); I != E;
       ++I) {
    uint32_t Flags = I->getFlags();
    if (Flags & SymbolRef::SF_Common) {
      // Add the common symbols to a list.  We'll allocate them all below.
      uint64_t Size = 0;
      Check(I->getSize(Size));
      CommonSize += Size;
    }
  }
  if (CommonSize != 0) {
    RWSectionSizes.push_back(CommonSize);
  }

  // Compute the required allocation space for each different type of sections
  // (code, read-only data, read-write data) assuming that all sections are
  // allocated with the max alignment. Note that we cannot compute with the
  // individual alignments of the sections, because then the required size
  // depends on the order, in which the sections are allocated.
  CodeSize = computeAllocationSizeForSections(CodeSectionSizes, MaxAlignment);
  DataSizeRO = computeAllocationSizeForSections(ROSectionSizes, MaxAlignment);
  DataSizeRW = computeAllocationSizeForSections(RWSectionSizes, MaxAlignment);
}

// compute stub buffer size for the given section
unsigned RuntimeDyldImpl::computeSectionStubBufSize(ObjectImage &Obj,
                                                    const SectionRef &Section) {
  unsigned StubSize = getMaxStubSize();
  if (StubSize == 0) {
    return 0;
  }
  // FIXME: this is an inefficient way to handle this. We should computed the
  // necessary section allocation size in loadObject by walking all the sections
  // once.
  unsigned StubBufSize = 0;
  for (section_iterator SI = Obj.begin_sections(), SE = Obj.end_sections();
       SI != SE; ++SI) {
    section_iterator RelSecI = SI->getRelocatedSection();
    if (!(RelSecI == Section))
      continue;

    for (const RelocationRef &Reloc : SI->relocations()) {
      (void)Reloc;
      StubBufSize += StubSize;
    }
  }

  // Get section data size and alignment
  uint64_t Alignment64;
  uint64_t DataSize;
  Check(Section.getSize(DataSize));
  Check(Section.getAlignment(Alignment64));

  // Add stubbuf size alignment
  unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;
  unsigned StubAlignment = getStubAlignment();
  unsigned EndAlignment = (DataSize | Alignment) & -(DataSize | Alignment);
  if (StubAlignment > EndAlignment)
    StubBufSize += StubAlignment - EndAlignment;
  return StubBufSize;
}

void RuntimeDyldImpl::emitCommonSymbols(ObjectImage &Obj,
                                        const CommonSymbolMap &CommonSymbols,
                                        uint64_t TotalSize,
                                        SymbolTableMap &SymbolTable) {
  // Allocate memory for the section
  unsigned SectionID = Sections.size();
  uint8_t *Addr = MemMgr->allocateDataSection(TotalSize, sizeof(void *),
                                              SectionID, StringRef(), false);
  if (!Addr)
    report_fatal_error("Unable to allocate memory for common symbols!");
  uint64_t Offset = 0;
  Sections.push_back(SectionEntry(StringRef(), Addr, TotalSize, 0));
  memset(Addr, 0, TotalSize);

  DEBUG(dbgs() << "emitCommonSection SectionID: " << SectionID << " new addr: "
               << format("%p", Addr) << " DataSize: " << TotalSize << "\n");

  // Assign the address of each symbol
  for (CommonSymbolMap::const_iterator it = CommonSymbols.begin(),
       itEnd = CommonSymbols.end(); it != itEnd; ++it) {
    uint64_t Size = it->second.first;
    uint64_t Align = it->second.second;
    StringRef Name;
    it->first.getName(Name);
    if (Align) {
      // This symbol has an alignment requirement.
      uint64_t AlignOffset = OffsetToAlignment((uint64_t)Addr, Align);
      Addr += AlignOffset;
      Offset += AlignOffset;
      DEBUG(dbgs() << "Allocating common symbol " << Name << " address "
                   << format("%p\n", Addr));
    }
    Obj.updateSymbolAddress(it->first, (uint64_t)Addr);
    SymbolTable[Name.data()] = SymbolLoc(SectionID, Offset);
    Offset += Size;
    Addr += Size;
  }
}

unsigned RuntimeDyldImpl::emitSection(ObjectImage &Obj,
                                      const SectionRef &Section, bool IsCode) {

  StringRef data;
  uint64_t Alignment64;
  Check(Section.getContents(data));
  Check(Section.getAlignment(Alignment64));

  unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;
  bool IsRequired;
  bool IsVirtual;
  bool IsZeroInit;
  bool IsReadOnly;
  uint64_t DataSize;
  unsigned PaddingSize = 0;
  unsigned StubBufSize = 0;
  StringRef Name;
  Check(Section.isRequiredForExecution(IsRequired));
  Check(Section.isVirtual(IsVirtual));
  Check(Section.isZeroInit(IsZeroInit));
  Check(Section.isReadOnlyData(IsReadOnly));
  Check(Section.getSize(DataSize));
  Check(Section.getName(Name));

  StubBufSize = computeSectionStubBufSize(Obj, Section);

  // The .eh_frame section (at least on Linux) needs an extra four bytes padded
  // with zeroes added at the end.  For MachO objects, this section has a
  // slightly different name, so this won't have any effect for MachO objects.
  if (Name == ".eh_frame")
    PaddingSize = 4;

  uintptr_t Allocate;
  unsigned SectionID = Sections.size();
  uint8_t *Addr;
  const char *pData = nullptr;

  // Some sections, such as debug info, don't need to be loaded for execution.
  // Leave those where they are.
  if (IsRequired) {
    Allocate = DataSize + PaddingSize + StubBufSize;
    Addr = IsCode ? MemMgr->allocateCodeSection(Allocate, Alignment, SectionID,
                                                Name)
                  : MemMgr->allocateDataSection(Allocate, Alignment, SectionID,
                                                Name, IsReadOnly);
    if (!Addr)
      report_fatal_error("Unable to allocate section memory!");

    // Virtual sections have no data in the object image, so leave pData = 0
    if (!IsVirtual)
      pData = data.data();

    // Zero-initialize or copy the data from the image
    if (IsZeroInit || IsVirtual)
      memset(Addr, 0, DataSize);
    else
      memcpy(Addr, pData, DataSize);

    // Fill in any extra bytes we allocated for padding
    if (PaddingSize != 0) {
      memset(Addr + DataSize, 0, PaddingSize);
      // Update the DataSize variable so that the stub offset is set correctly.
      DataSize += PaddingSize;
    }

    DEBUG(dbgs() << "emitSection SectionID: " << SectionID << " Name: " << Name
                 << " obj addr: " << format("%p", pData)
                 << " new addr: " << format("%p", Addr)
                 << " DataSize: " << DataSize << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate << "\n");
    Obj.updateSectionAddress(Section, (uint64_t)Addr);
  } else {
    // Even if we didn't load the section, we need to record an entry for it
    // to handle later processing (and by 'handle' I mean don't do anything
    // with these sections).
    Allocate = 0;
    Addr = nullptr;
    DEBUG(dbgs() << "emitSection SectionID: " << SectionID << " Name: " << Name
                 << " obj addr: " << format("%p", data.data()) << " new addr: 0"
                 << " DataSize: " << DataSize << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate << "\n");
  }

  Sections.push_back(SectionEntry(Name, Addr, DataSize, (uintptr_t)pData));
  return SectionID;
}

unsigned RuntimeDyldImpl::findOrEmitSection(ObjectImage &Obj,
                                            const SectionRef &Section,
                                            bool IsCode,
                                            ObjSectionToIDMap &LocalSections) {

  unsigned SectionID = 0;
  ObjSectionToIDMap::iterator i = LocalSections.find(Section);
  if (i != LocalSections.end())
    SectionID = i->second;
  else {
    SectionID = emitSection(Obj, Section, IsCode);
    LocalSections[Section] = SectionID;
  }
  return SectionID;
}

void RuntimeDyldImpl::addRelocationForSection(const RelocationEntry &RE,
                                              unsigned SectionID) {
  Relocations[SectionID].push_back(RE);
}

void RuntimeDyldImpl::addRelocationForSymbol(const RelocationEntry &RE,
                                             StringRef SymbolName) {
  // Relocation by symbol.  If the symbol is found in the global symbol table,
  // create an appropriate section relocation.  Otherwise, add it to
  // ExternalSymbolRelocations.
  SymbolTableMap::const_iterator Loc = GlobalSymbolTable.find(SymbolName);
  if (Loc == GlobalSymbolTable.end()) {
    ExternalSymbolRelocations[SymbolName].push_back(RE);
  } else {
    // Copy the RE since we want to modify its addend.
    RelocationEntry RECopy = RE;
    RECopy.Addend += Loc->second.second;
    Relocations[Loc->second.first].push_back(RECopy);
  }
}

uint8_t *RuntimeDyldImpl::createStubFunction(uint8_t *Addr) {
  if (Arch == Triple::aarch64 || Arch == Triple::aarch64_be ||
      Arch == Triple::arm64 || Arch == Triple::arm64_be) {
    // This stub has to be able to access the full address space,
    // since symbol lookup won't necessarily find a handy, in-range,
    // PLT stub for functions which could be anywhere.
    uint32_t *StubAddr = (uint32_t *)Addr;

    // Stub can use ip0 (== x16) to calculate address
    *StubAddr = 0xd2e00010; // movz ip0, #:abs_g3:<addr>
    StubAddr++;
    *StubAddr = 0xf2c00010; // movk ip0, #:abs_g2_nc:<addr>
    StubAddr++;
    *StubAddr = 0xf2a00010; // movk ip0, #:abs_g1_nc:<addr>
    StubAddr++;
    *StubAddr = 0xf2800010; // movk ip0, #:abs_g0_nc:<addr>
    StubAddr++;
    *StubAddr = 0xd61f0200; // br ip0

    return Addr;
  } else if (Arch == Triple::arm || Arch == Triple::armeb) {
    // TODO: There is only ARM far stub now. We should add the Thumb stub,
    // and stubs for branches Thumb - ARM and ARM - Thumb.
    uint32_t *StubAddr = (uint32_t *)Addr;
    *StubAddr = 0xe51ff004; // ldr pc,<label>
    return (uint8_t *)++StubAddr;
  } else if (Arch == Triple::mipsel || Arch == Triple::mips) {
    uint32_t *StubAddr = (uint32_t *)Addr;
    // 0:   3c190000        lui     t9,%hi(addr).
    // 4:   27390000        addiu   t9,t9,%lo(addr).
    // 8:   03200008        jr      t9.
    // c:   00000000        nop.
    const unsigned LuiT9Instr = 0x3c190000, AdduiT9Instr = 0x27390000;
    const unsigned JrT9Instr = 0x03200008, NopInstr = 0x0;

    *StubAddr = LuiT9Instr;
    StubAddr++;
    *StubAddr = AdduiT9Instr;
    StubAddr++;
    *StubAddr = JrT9Instr;
    StubAddr++;
    *StubAddr = NopInstr;
    return Addr;
  } else if (Arch == Triple::ppc64 || Arch == Triple::ppc64le) {
    // PowerPC64 stub: the address points to a function descriptor
    // instead of the function itself. Load the function address
    // on r11 and sets it to control register. Also loads the function
    // TOC in r2 and environment pointer to r11.
    writeInt32BE(Addr,    0x3D800000); // lis   r12, highest(addr)
    writeInt32BE(Addr+4,  0x618C0000); // ori   r12, higher(addr)
    writeInt32BE(Addr+8,  0x798C07C6); // sldi  r12, r12, 32
    writeInt32BE(Addr+12, 0x658C0000); // oris  r12, r12, h(addr)
    writeInt32BE(Addr+16, 0x618C0000); // ori   r12, r12, l(addr)
    writeInt32BE(Addr+20, 0xF8410028); // std   r2,  40(r1)
    writeInt32BE(Addr+24, 0xE96C0000); // ld    r11, 0(r12)
    writeInt32BE(Addr+28, 0xE84C0008); // ld    r2,  0(r12)
    writeInt32BE(Addr+32, 0x7D6903A6); // mtctr r11
    writeInt32BE(Addr+36, 0xE96C0010); // ld    r11, 16(r2)
    writeInt32BE(Addr+40, 0x4E800420); // bctr

    return Addr;
  } else if (Arch == Triple::systemz) {
    writeInt16BE(Addr,    0xC418);     // lgrl %r1,.+8
    writeInt16BE(Addr+2,  0x0000);
    writeInt16BE(Addr+4,  0x0004);
    writeInt16BE(Addr+6,  0x07F1);     // brc 15,%r1
    // 8-byte address stored at Addr + 8
    return Addr;
  } else if (Arch == Triple::x86_64) {
    *Addr      = 0xFF; // jmp
    *(Addr+1)  = 0x25; // rip
    // 32-bit PC-relative address of the GOT entry will be stored at Addr+2
  } else if (Arch == Triple::x86) {
    *Addr      = 0xE9; // 32-bit pc-relative jump.
  }
  return Addr;
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldImpl::reassignSectionAddress(unsigned SectionID,
                                             uint64_t Addr) {
  // The address to use for relocation resolution is not
  // the address of the local section buffer. We must be doing
  // a remote execution environment of some sort. Relocations can't
  // be applied until all the sections have been moved.  The client must
  // trigger this with a call to MCJIT::finalize() or
  // RuntimeDyld::resolveRelocations().
  //
  // Addr is a uint64_t because we can't assume the pointer width
  // of the target is the same as that of the host. Just use a generic
  // "big enough" type.
  Sections[SectionID].LoadAddress = Addr;
}

void RuntimeDyldImpl::resolveRelocationList(const RelocationList &Relocs,
                                            uint64_t Value) {
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    const RelocationEntry &RE = Relocs[i];
    // Ignore relocations for sections that were not loaded
    if (Sections[RE.SectionID].Address == nullptr)
      continue;
    resolveRelocation(RE, Value);
  }
}

void RuntimeDyldImpl::resolveExternalSymbols() {
  while (!ExternalSymbolRelocations.empty()) {
    StringMap<RelocationList>::iterator i = ExternalSymbolRelocations.begin();

    StringRef Name = i->first();
    if (Name.size() == 0) {
      // This is an absolute symbol, use an address of zero.
      DEBUG(dbgs() << "Resolving absolute relocations."
                   << "\n");
      RelocationList &Relocs = i->second;
      resolveRelocationList(Relocs, 0);
    } else {
      uint64_t Addr = 0;
      SymbolTableMap::const_iterator Loc = GlobalSymbolTable.find(Name);
      if (Loc == GlobalSymbolTable.end()) {
        // This is an external symbol, try to get its address from
        // MemoryManager.
        Addr = MemMgr->getSymbolAddress(Name.data());
        // The call to getSymbolAddress may have caused additional modules to
        // be loaded, which may have added new entries to the
        // ExternalSymbolRelocations map.  Consquently, we need to update our
        // iterator.  This is also why retrieval of the relocation list
        // associated with this symbol is deferred until below this point.
        // New entries may have been added to the relocation list.
        i = ExternalSymbolRelocations.find(Name);
      } else {
        // We found the symbol in our global table.  It was probably in a
        // Module that we loaded previously.
        SymbolLoc SymLoc = Loc->second;
        Addr = getSectionLoadAddress(SymLoc.first) + SymLoc.second;
      }

      // FIXME: Implement error handling that doesn't kill the host program!
      if (!Addr)
        report_fatal_error("Program used external function '" + Name +
                           "' which could not be resolved!");

      updateGOTEntries(Name, Addr);
      DEBUG(dbgs() << "Resolving relocations Name: " << Name << "\t"
                   << format("0x%lx", Addr) << "\n");
      // This list may have been updated when we called getSymbolAddress, so
      // don't change this code to get the list earlier.
      RelocationList &Relocs = i->second;
      resolveRelocationList(Relocs, Addr);
    }

    ExternalSymbolRelocations.erase(i);
  }
}

//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation
RuntimeDyld::RuntimeDyld(RTDyldMemoryManager *mm) {
  // FIXME: There's a potential issue lurking here if a single instance of
  // RuntimeDyld is used to load multiple objects.  The current implementation
  // associates a single memory manager with a RuntimeDyld instance.  Even
  // though the public class spawns a new 'impl' instance for each load,
  // they share a single memory manager.  This can become a problem when page
  // permissions are applied.
  Dyld = nullptr;
  MM = mm;
  ProcessAllSections = false;
}

RuntimeDyld::~RuntimeDyld() { delete Dyld; }

static std::unique_ptr<RuntimeDyldELF>
createRuntimeDyldELF(RTDyldMemoryManager *MM, bool ProcessAllSections) {
  std::unique_ptr<RuntimeDyldELF> Dyld(new RuntimeDyldELF(MM));
  Dyld->setProcessAllSections(ProcessAllSections);
  return Dyld;
}

static std::unique_ptr<RuntimeDyldMachO>
createRuntimeDyldMachO(RTDyldMemoryManager *MM, bool ProcessAllSections) {
  std::unique_ptr<RuntimeDyldMachO> Dyld(new RuntimeDyldMachO(MM));
  Dyld->setProcessAllSections(ProcessAllSections);
  return Dyld;
}

ObjectImage *RuntimeDyld::loadObject(std::unique_ptr<ObjectFile> InputObject) {
  std::unique_ptr<ObjectImage> InputImage;

  ObjectFile &Obj = *InputObject;

  if (InputObject->isELF()) {
    InputImage.reset(RuntimeDyldELF::createObjectImageFromFile(std::move(InputObject)));
    if (!Dyld)
      Dyld = createRuntimeDyldELF(MM, ProcessAllSections).release();
  } else if (InputObject->isMachO()) {
    InputImage.reset(RuntimeDyldMachO::createObjectImageFromFile(std::move(InputObject)));
    if (!Dyld)
      Dyld = createRuntimeDyldMachO(MM, ProcessAllSections).release();
  } else
    report_fatal_error("Incompatible object format!");

  if (!Dyld->isCompatibleFile(&Obj))
    report_fatal_error("Incompatible object format!");

  Dyld->loadObject(InputImage.get());
  return InputImage.release();
}

ObjectImage *RuntimeDyld::loadObject(ObjectBuffer *InputBuffer) {
  std::unique_ptr<ObjectImage> InputImage;
  sys::fs::file_magic Type = sys::fs::identify_magic(InputBuffer->getBuffer());

  switch (Type) {
  case sys::fs::file_magic::elf_relocatable:
  case sys::fs::file_magic::elf_executable:
  case sys::fs::file_magic::elf_shared_object:
  case sys::fs::file_magic::elf_core:
    InputImage.reset(RuntimeDyldELF::createObjectImage(InputBuffer));
    if (!Dyld)
      Dyld = createRuntimeDyldELF(MM, ProcessAllSections).release();
    break;
  case sys::fs::file_magic::macho_object:
  case sys::fs::file_magic::macho_executable:
  case sys::fs::file_magic::macho_fixed_virtual_memory_shared_lib:
  case sys::fs::file_magic::macho_core:
  case sys::fs::file_magic::macho_preload_executable:
  case sys::fs::file_magic::macho_dynamically_linked_shared_lib:
  case sys::fs::file_magic::macho_dynamic_linker:
  case sys::fs::file_magic::macho_bundle:
  case sys::fs::file_magic::macho_dynamically_linked_shared_lib_stub:
  case sys::fs::file_magic::macho_dsym_companion:
    InputImage.reset(RuntimeDyldMachO::createObjectImage(InputBuffer));
    if (!Dyld)
      Dyld = createRuntimeDyldMachO(MM, ProcessAllSections).release();
    break;
  case sys::fs::file_magic::unknown:
  case sys::fs::file_magic::bitcode:
  case sys::fs::file_magic::archive:
  case sys::fs::file_magic::coff_object:
  case sys::fs::file_magic::coff_import_library:
  case sys::fs::file_magic::pecoff_executable:
  case sys::fs::file_magic::macho_universal_binary:
  case sys::fs::file_magic::windows_resource:
    report_fatal_error("Incompatible object format!");
  }

  if (!Dyld->isCompatibleFormat(InputBuffer))
    report_fatal_error("Incompatible object format!");

  Dyld->loadObject(InputImage.get());
  return InputImage.release();
}

void *RuntimeDyld::getSymbolAddress(StringRef Name) {
  if (!Dyld)
    return nullptr;
  return Dyld->getSymbolAddress(Name);
}

uint64_t RuntimeDyld::getSymbolLoadAddress(StringRef Name) {
  if (!Dyld)
    return 0;
  return Dyld->getSymbolLoadAddress(Name);
}

void RuntimeDyld::resolveRelocations() { Dyld->resolveRelocations(); }

void RuntimeDyld::reassignSectionAddress(unsigned SectionID, uint64_t Addr) {
  Dyld->reassignSectionAddress(SectionID, Addr);
}

void RuntimeDyld::mapSectionAddress(const void *LocalAddress,
                                    uint64_t TargetAddress) {
  Dyld->mapSectionAddress(LocalAddress, TargetAddress);
}

bool RuntimeDyld::hasError() { return Dyld->hasError(); }

StringRef RuntimeDyld::getErrorString() { return Dyld->getErrorString(); }

void RuntimeDyld::registerEHFrames() {
  if (Dyld)
    Dyld->registerEHFrames();
}

void RuntimeDyld::deregisterEHFrames() {
  if (Dyld)
    Dyld->deregisterEHFrames();
}

} // end namespace llvm
