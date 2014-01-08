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

#define DEBUG_TYPE "dyld"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "JITRegistrar.h"
#include "ObjectImageCommon.h"
#include "RuntimeDyldELF.h"
#include "RuntimeDyldImpl.h"
#include "RuntimeDyldMachO.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MutexGuard.h"

using namespace llvm;
using namespace llvm::object;

// Empty out-of-line virtual destructor as the key function.
RuntimeDyldImpl::~RuntimeDyldImpl() {}

// Pin the JITRegistrar's and ObjectImage*'s vtables to this file.
void JITRegistrar::anchor() {}
void ObjectImage::anchor() {}
void ObjectImageCommon::anchor() {}

namespace llvm {

void RuntimeDyldImpl::registerEHFrames() {
}

void RuntimeDyldImpl::deregisterEHFrames() {
}

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
    DEBUG(dbgs() << "Resolving relocations Section #" << i
            << "\t" << format("%p", (uint8_t *)Addr)
            << "\n");
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

// Subclasses can implement this method to create specialized image instances.
// The caller owns the pointer that is returned.
ObjectImage *RuntimeDyldImpl::createObjectImage(ObjectBuffer *InputBuffer) {
  return new ObjectImageCommon(InputBuffer);
}

ObjectImage *RuntimeDyldImpl::createObjectImageFromFile(ObjectFile *InputObject) {
  return new ObjectImageCommon(InputObject);
}

ObjectImage *RuntimeDyldImpl::loadObject(ObjectFile *InputObject) {
  return loadObject(createObjectImageFromFile(InputObject));
}

ObjectImage *RuntimeDyldImpl::loadObject(ObjectBuffer *InputBuffer) {
  return loadObject(createObjectImage(InputBuffer));
} 

ObjectImage *RuntimeDyldImpl::loadObject(ObjectImage *InputObject) {
  MutexGuard locked(lock);

  OwningPtr<ObjectImage> obj(InputObject);
  if (!obj)
    return NULL;

  // Save information about our target
  Arch = (Triple::ArchType)obj->getArch();
  IsTargetLittleEndian = obj->getObjectFile()->isLittleEndian();

  // Symbols found in this object
  StringMap<SymbolLoc> LocalSymbols;
  // Used sections from the object file
  ObjSectionToIDMap LocalSections;

  // Common symbols requiring allocation, with their sizes and alignments
  CommonSymbolMap CommonSymbols;
  // Maximum required total memory to allocate all common symbols
  uint64_t CommonSize = 0;

  error_code err;
  // Parse symbols
  DEBUG(dbgs() << "Parse symbols:\n");
  for (symbol_iterator i = obj->begin_symbols(), e = obj->end_symbols();
       i != e; i.increment(err)) {
    Check(err);
    object::SymbolRef::Type SymType;
    StringRef Name;
    Check(i->getType(SymType));
    Check(i->getName(Name));

    uint32_t flags;
    Check(i->getFlags(flags));

    bool isCommon = flags & SymbolRef::SF_Common;
    if (isCommon) {
      // Add the common symbols to a list.  We'll allocate them all below.
      uint32_t Align;
      Check(i->getAlignment(Align));
      uint64_t Size = 0;
      Check(i->getSize(Size));
      CommonSize += Size + Align;
      CommonSymbols[*i] = CommonSymbolInfo(Size, Align);
    } else {
      if (SymType == object::SymbolRef::ST_Function ||
          SymType == object::SymbolRef::ST_Data ||
          SymType == object::SymbolRef::ST_Unknown) {
        uint64_t FileOffset;
        StringRef SectionData;
        bool IsCode;
        section_iterator si = obj->end_sections();
        Check(i->getFileOffset(FileOffset));
        Check(i->getSection(si));
        if (si == obj->end_sections()) continue;
        Check(si->getContents(SectionData));
        Check(si->isText(IsCode));
        const uint8_t* SymPtr = (const uint8_t*)InputObject->getData().data() +
                                (uintptr_t)FileOffset;
        uintptr_t SectOffset = (uintptr_t)(SymPtr -
                                           (const uint8_t*)SectionData.begin());
        unsigned SectionID = findOrEmitSection(*obj, *si, IsCode, LocalSections);
        LocalSymbols[Name.data()] = SymbolLoc(SectionID, SectOffset);
        DEBUG(dbgs() << "\tFileOffset: " << format("%p", (uintptr_t)FileOffset)
                     << " flags: " << flags
                     << " SID: " << SectionID
                     << " Offset: " << format("%p", SectOffset));
        GlobalSymbolTable[Name] = SymbolLoc(SectionID, SectOffset);
      }
    }
    DEBUG(dbgs() << "\tType: " << SymType << " Name: " << Name << "\n");
  }

  // Allocate common symbols
  if (CommonSize != 0)
    emitCommonSymbols(*obj, CommonSymbols, CommonSize, LocalSymbols);

  // Parse and process relocations
  DEBUG(dbgs() << "Parse relocations:\n");
  for (section_iterator si = obj->begin_sections(),
       se = obj->end_sections(); si != se; si.increment(err)) {
    Check(err);
    bool isFirstRelocation = true;
    unsigned SectionID = 0;
    StubMap Stubs;
    section_iterator RelocatedSection = si->getRelocatedSection();

    for (relocation_iterator i = si->begin_relocations(),
         e = si->end_relocations(); i != e; i.increment(err)) {
      Check(err);

      // If it's the first relocation in this section, find its SectionID
      if (isFirstRelocation) {
        SectionID =
            findOrEmitSection(*obj, *RelocatedSection, true, LocalSections);
        DEBUG(dbgs() << "\tSectionID: " << SectionID << "\n");
        isFirstRelocation = false;
      }

      processRelocationRef(SectionID, *i, *obj, LocalSections, LocalSymbols,
                           Stubs);
    }
  }

  // Give the subclasses a chance to tie-up any loose ends.
  finalizeLoad(LocalSections);

  return obj.take();
}

void RuntimeDyldImpl::emitCommonSymbols(ObjectImage &Obj,
                                        const CommonSymbolMap &CommonSymbols,
                                        uint64_t TotalSize,
                                        SymbolTableMap &SymbolTable) {
  // Allocate memory for the section
  unsigned SectionID = Sections.size();
  uint8_t *Addr = MemMgr->allocateDataSection(
    TotalSize, sizeof(void*), SectionID, StringRef(), false);
  if (!Addr)
    report_fatal_error("Unable to allocate memory for common symbols!");
  uint64_t Offset = 0;
  Sections.push_back(SectionEntry(StringRef(), Addr, TotalSize, 0));
  memset(Addr, 0, TotalSize);

  DEBUG(dbgs() << "emitCommonSection SectionID: " << SectionID
               << " new addr: " << format("%p", Addr)
               << " DataSize: " << TotalSize
               << "\n");

  // Assign the address of each symbol
  for (CommonSymbolMap::const_iterator it = CommonSymbols.begin(),
       itEnd = CommonSymbols.end(); it != itEnd; it++) {
    uint64_t Size = it->second.first;
    uint64_t Align = it->second.second;
    StringRef Name;
    it->first.getName(Name);
    if (Align) {
      // This symbol has an alignment requirement.
      uint64_t AlignOffset = OffsetToAlignment((uint64_t)Addr, Align);
      Addr += AlignOffset;
      Offset += AlignOffset;
      DEBUG(dbgs() << "Allocating common symbol " << Name << " address " <<
                      format("%p\n", Addr));
    }
    Obj.updateSymbolAddress(it->first, (uint64_t)Addr);
    SymbolTable[Name.data()] = SymbolLoc(SectionID, Offset);
    Offset += Size;
    Addr += Size;
  }
}

unsigned RuntimeDyldImpl::emitSection(ObjectImage &Obj,
                                      const SectionRef &Section,
                                      bool IsCode) {

  unsigned StubBufSize = 0,
           StubSize = getMaxStubSize();
  error_code err;
  const ObjectFile *ObjFile = Obj.getObjectFile();
  // FIXME: this is an inefficient way to handle this. We should computed the
  // necessary section allocation size in loadObject by walking all the sections
  // once.
  if (StubSize > 0) {
    for (section_iterator SI = ObjFile->begin_sections(),
           SE = ObjFile->end_sections();
         SI != SE; SI.increment(err), Check(err)) {
      section_iterator RelSecI = SI->getRelocatedSection();
      if (!(RelSecI == Section))
        continue;

      for (relocation_iterator I = SI->begin_relocations(),
             E = SI->end_relocations(); I != E; I.increment(err), Check(err)) {
        StubBufSize += StubSize;
      }
    }
  }

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
  StringRef Name;
  Check(Section.isRequiredForExecution(IsRequired));
  Check(Section.isVirtual(IsVirtual));
  Check(Section.isZeroInit(IsZeroInit));
  Check(Section.isReadOnlyData(IsReadOnly));
  Check(Section.getSize(DataSize));
  Check(Section.getName(Name));
  if (StubSize > 0) {
    unsigned StubAlignment = getStubAlignment();
    unsigned EndAlignment = (DataSize | Alignment) & -(DataSize | Alignment);
    if (StubAlignment > EndAlignment)
      StubBufSize += StubAlignment - EndAlignment;
  }

  // The .eh_frame section (at least on Linux) needs an extra four bytes padded
  // with zeroes added at the end.  For MachO objects, this section has a
  // slightly different name, so this won't have any effect for MachO objects.
  if (Name == ".eh_frame")
    PaddingSize = 4;

  unsigned Allocate;
  unsigned SectionID = Sections.size();
  uint8_t *Addr;
  const char *pData = 0;

  // Some sections, such as debug info, don't need to be loaded for execution.
  // Leave those where they are.
  if (IsRequired) {
    Allocate = DataSize + PaddingSize + StubBufSize;
    Addr = IsCode
      ? MemMgr->allocateCodeSection(Allocate, Alignment, SectionID, Name)
      : MemMgr->allocateDataSection(Allocate, Alignment, SectionID, Name,
                                    IsReadOnly);
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

    DEBUG(dbgs() << "emitSection SectionID: " << SectionID
                 << " Name: " << Name
                 << " obj addr: " << format("%p", pData)
                 << " new addr: " << format("%p", Addr)
                 << " DataSize: " << DataSize
                 << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate
                 << "\n");
    Obj.updateSectionAddress(Section, (uint64_t)Addr);
  }
  else {
    // Even if we didn't load the section, we need to record an entry for it
    // to handle later processing (and by 'handle' I mean don't do anything
    // with these sections).
    Allocate = 0;
    Addr = 0;
    DEBUG(dbgs() << "emitSection SectionID: " << SectionID
                 << " Name: " << Name
                 << " obj addr: " << format("%p", data.data())
                 << " new addr: 0"
                 << " DataSize: " << DataSize
                 << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate
                 << "\n");
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
  SymbolTableMap::const_iterator Loc =
      GlobalSymbolTable.find(SymbolName);
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
  if (Arch == Triple::aarch64) {
    // This stub has to be able to access the full address space,
    // since symbol lookup won't necessarily find a handy, in-range,
    // PLT stub for functions which could be anywhere.
    uint32_t *StubAddr = (uint32_t*)Addr;

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
  } else if (Arch == Triple::arm) {
    // TODO: There is only ARM far stub now. We should add the Thumb stub,
    // and stubs for branches Thumb - ARM and ARM - Thumb.
    uint32_t *StubAddr = (uint32_t*)Addr;
    *StubAddr = 0xe51ff004; // ldr pc,<label>
    return (uint8_t*)++StubAddr;
  } else if (Arch == Triple::mipsel || Arch == Triple::mips) {
    uint32_t *StubAddr = (uint32_t*)Addr;
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
    if (Sections[RE.SectionID].Address == 0)
      continue;
    resolveRelocation(RE, Value);
  }
}

void RuntimeDyldImpl::resolveExternalSymbols() {
  while(!ExternalSymbolRelocations.empty()) {
    StringMap<RelocationList>::iterator i = ExternalSymbolRelocations.begin();

    StringRef Name = i->first();
    if (Name.size() == 0) {
      // This is an absolute symbol, use an address of zero.
      DEBUG(dbgs() << "Resolving absolute relocations." << "\n");
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
      DEBUG(dbgs() << "Resolving relocations Name: " << Name
              << "\t" << format("0x%lx", Addr)
              << "\n");
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
  Dyld = 0;
  MM = mm;
}

RuntimeDyld::~RuntimeDyld() {
  delete Dyld;
}

ObjectImage *RuntimeDyld::loadObject(ObjectFile *InputObject) {
  if (!Dyld) {
    if (InputObject->isELF())
      Dyld = new RuntimeDyldELF(MM);
    else if (InputObject->isMachO())
      Dyld = new RuntimeDyldMachO(MM);
    else
      report_fatal_error("Incompatible object format!");
  } else {
    if (!Dyld->isCompatibleFile(InputObject))
      report_fatal_error("Incompatible object format!");
  }

  return Dyld->loadObject(InputObject);
}

ObjectImage *RuntimeDyld::loadObject(ObjectBuffer *InputBuffer) {
  if (!Dyld) {
    sys::fs::file_magic Type =
        sys::fs::identify_magic(InputBuffer->getBuffer());
    switch (Type) {
    case sys::fs::file_magic::elf_relocatable:
    case sys::fs::file_magic::elf_executable:
    case sys::fs::file_magic::elf_shared_object:
    case sys::fs::file_magic::elf_core:
      Dyld = new RuntimeDyldELF(MM);
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
      Dyld = new RuntimeDyldMachO(MM);
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
  } else {
    if (!Dyld->isCompatibleFormat(InputBuffer))
      report_fatal_error("Incompatible object format!");
  }

  return Dyld->loadObject(InputBuffer);
}

void *RuntimeDyld::getSymbolAddress(StringRef Name) {
  if (!Dyld)
    return NULL;
  return Dyld->getSymbolAddress(Name);
}

uint64_t RuntimeDyld::getSymbolLoadAddress(StringRef Name) {
  if (!Dyld)
    return 0;
  return Dyld->getSymbolLoadAddress(Name);
}

void RuntimeDyld::resolveRelocations() {
  Dyld->resolveRelocations();
}

void RuntimeDyld::reassignSectionAddress(unsigned SectionID,
                                         uint64_t Addr) {
  Dyld->reassignSectionAddress(SectionID, Addr);
}

void RuntimeDyld::mapSectionAddress(const void *LocalAddress,
                                    uint64_t TargetAddress) {
  Dyld->mapSectionAddress(LocalAddress, TargetAddress);
}

StringRef RuntimeDyld::getErrorString() {
  return Dyld->getErrorString();
}

void RuntimeDyld::registerEHFrames() {
  if (Dyld)
    Dyld->registerEHFrames();
}

void RuntimeDyld::deregisterEHFrames() {
  if (Dyld)
    Dyld->deregisterEHFrames();
}

} // end namespace llvm
