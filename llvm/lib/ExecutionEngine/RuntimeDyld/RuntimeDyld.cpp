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
#include "RuntimeDyldImpl.h"
#include "RuntimeDyldELF.h"
#include "RuntimeDyldMachO.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

// Empty out-of-line virtual destructor as the key function.
RTDyldMemoryManager::~RTDyldMemoryManager() {}
RuntimeDyldImpl::~RuntimeDyldImpl() {}

namespace llvm {



// Resolve the relocations for all symbols we currently know about.
void RuntimeDyldImpl::resolveRelocations() {
  // First, resolve relocations assotiated with external symbols.
  resolveSymbols();

  // Just iterate over the sections we have and resolve all the relocations
  // in them. Gross overkill, but it gets the job done.
  for (int i = 0, e = Sections.size(); i != e; ++i) {
    reassignSectionAddress(i, Sections[i].LoadAddress);
  }
}

void RuntimeDyldImpl::mapSectionAddress(void *LocalAddress,
                                        uint64_t TargetAddress) {
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    if (Sections[i].Address == LocalAddress) {
      reassignSectionAddress(i, TargetAddress);
      return;
    }
  }
  llvm_unreachable("Attempting to remap address of unknown section!");
}

bool RuntimeDyldImpl::loadObject(const MemoryBuffer *InputBuffer) {
  // FIXME: ObjectFile don't modify MemoryBuffer.
  //        It should use const MemoryBuffer as parameter.
  ObjectFile *obj = ObjectFile::
                      createObjectFile(const_cast<MemoryBuffer*>(InputBuffer));

  Arch = (Triple::ArchType)obj->getArch();

  LocalSymbolMap LocalSymbols;     // Functions and data symbols from the
                                   // object file.
  ObjSectionToIDMap LocalSections; // Used sections from the object file

  error_code err;


  // Parse symbols
  DEBUG(dbgs() << "Parse symbols:\n");
  for (symbol_iterator it = obj->begin_symbols(), itEnd = obj->end_symbols();
       it != itEnd; it.increment(err)) {
    if (err) break;
    object::SymbolRef::Type SymType;
    StringRef Name;
    if ((bool)(err = it->getType(SymType))) break;
    if ((bool)(err = it->getName(Name))) break;

    if (SymType == object::SymbolRef::ST_Function ||
        SymType == object::SymbolRef::ST_Data) {
      uint64_t FileOffset;
      uint32_t flags;
      StringRef sData;
      section_iterator sIt = obj->end_sections();
      if ((bool)(err = it->getFileOffset(FileOffset))) break;
      if ((bool)(err = it->getFlags(flags))) break;
      if ((bool)(err = it->getSection(sIt))) break;
      if (sIt == obj->end_sections()) continue;
      if ((bool)(err = sIt->getContents(sData))) break;
      const uint8_t* SymPtr = (const uint8_t*)InputBuffer->getBufferStart() +
                              (uintptr_t)FileOffset;
      uintptr_t SectOffset = (uintptr_t)(SymPtr - (const uint8_t*)sData.begin());
      unsigned SectionID =
        findOrEmitSection(*sIt,
                          SymType == object::SymbolRef::ST_Function,
                          LocalSections);
      bool isGlobal = flags & SymbolRef::SF_Global;
      LocalSymbols[Name.data()] = SymbolLoc(SectionID, SectOffset);
      DEBUG(dbgs() << "\tFileOffset: " << format("%p", (uintptr_t)FileOffset)
                   << " flags: " << flags
                   << " SID: " << SectionID
                   << " Offset: " << format("%p", SectOffset));
      if (isGlobal)
        SymbolTable[Name] = SymbolLoc(SectionID, SectOffset);
    }
    DEBUG(dbgs() << "\tType: " << SymType << " Name: " << Name << "\n");
  }
  if (err) {
    report_fatal_error(err.message());
  }

  // Parse and proccess relocations
  DEBUG(dbgs() << "Parse relocations:\n");
  for (section_iterator sIt = obj->begin_sections(),
       sItEnd = obj->end_sections(); sIt != sItEnd; sIt.increment(err)) {
    if (err) break;
    bool isFirstRelocation = true;
    unsigned SectionID = 0;
    StubMap Stubs;

    for (relocation_iterator it = sIt->begin_relocations(),
         itEnd = sIt->end_relocations(); it != itEnd; it.increment(err)) {
      if (err) break;

      // If it's first relocation in this section, find its SectionID
      if (isFirstRelocation) {
        SectionID = findOrEmitSection(*sIt, true, LocalSections);
        DEBUG(dbgs() << "\tSectionID: " << SectionID << "\n");
        isFirstRelocation = false;
      }

      ObjRelocationInfo RI;
      RI.SectionID = SectionID;
      if ((bool)(err = it->getAdditionalInfo(RI.AdditionalInfo))) break;
      if ((bool)(err = it->getOffset(RI.Offset))) break;
      if ((bool)(err = it->getSymbol(RI.Symbol))) break;
      if ((bool)(err = it->getType(RI.Type))) break;

      DEBUG(dbgs() << "\t\tAddend: " << RI.AdditionalInfo
                   << " Offset: " << format("%p", (uintptr_t)RI.Offset)
                   << " Type: " << (uint32_t)(RI.Type & 0xffffffffL)
                   << "\n");
      processRelocationRef(RI, *obj, LocalSections, LocalSymbols, Stubs);
    }
    if (err) {
      report_fatal_error(err.message());
    }
  }
  return false;
}

unsigned RuntimeDyldImpl::emitSection(const SectionRef &Section,
                                      bool IsCode) {

  unsigned StubBufSize = 0,
           StubSize = getMaxStubSize();
  error_code err;
  if (StubSize > 0) {
    for (relocation_iterator it = Section.begin_relocations(),
         itEnd = Section.end_relocations(); it != itEnd; it.increment(err))
      StubBufSize += StubSize;
  }
  StringRef data;
  uint64_t Alignment64;
  if ((bool)(err = Section.getContents(data))) report_fatal_error(err.message());
  if ((bool)(err = Section.getAlignment(Alignment64)))
    report_fatal_error(err.message());

  unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;
  unsigned DataSize = data.size();
  unsigned Allocate = DataSize + StubBufSize;
  unsigned SectionID = Sections.size();
  const char *pData = data.data();
  uint8_t *Addr = IsCode
    ? MemMgr->allocateCodeSection(Allocate, Alignment, SectionID)
    : MemMgr->allocateDataSection(Allocate, Alignment, SectionID);

  memcpy(Addr, pData, DataSize);
  DEBUG(dbgs() << "emitSection SectionID: " << SectionID
               << " obj addr: " << format("%p", pData)
               << " new addr: " << format("%p", Addr)
               << " DataSize: " << DataSize
               << " StubBufSize: " << StubBufSize
               << " Allocate: " << Allocate
               << "\n");
  Sections.push_back(SectionEntry(Addr, Allocate, DataSize,(uintptr_t)pData));
  return SectionID;
}

unsigned RuntimeDyldImpl::
findOrEmitSection(const SectionRef &Section, bool IsCode,
                  ObjSectionToIDMap &LocalSections) {

  unsigned SectionID = 0;
  ObjSectionToIDMap::iterator sIDIt = LocalSections.find(Section);
  if (sIDIt != LocalSections.end())
    SectionID = sIDIt->second;
  else {
    SectionID = emitSection(Section, IsCode);
    LocalSections[Section] = SectionID;
  }
  return SectionID;
}

void RuntimeDyldImpl::AddRelocation(const RelocationValueRef &Value,
                                   unsigned SectionID, uintptr_t Offset,
                                   uint32_t RelType) {
  DEBUG(dbgs() << "AddRelocation SymNamePtr: " << format("%p", Value.SymbolName)
               << " SID: " << Value.SectionID
               << " Addend: " << format("%p", Value.Addend)
               << " Offset: " << format("%p", Offset)
               << " RelType: " << format("%x", RelType)
               << "\n");

  if (Value.SymbolName == 0) {
    Relocations[Value.SectionID].push_back(RelocationEntry(
      SectionID,
      Offset,
      RelType,
      Value.Addend));
  } else
    SymbolRelocations[Value.SymbolName].push_back(RelocationEntry(
      SectionID,
      Offset,
      RelType,
      Value.Addend));
}

uint8_t *RuntimeDyldImpl::createStubFunction(uint8_t *Addr) {
  // TODO: There is only ARM far stub now. We should add the Thumb stub,
  // and stubs for branches Thumb - ARM and ARM - Thumb.
  if (Arch == Triple::arm) {
    uint32_t *StubAddr = (uint32_t*)Addr;
    *StubAddr = 0xe51ff004; // ldr pc,<label>
    return (uint8_t*)++StubAddr;
  }
  else
    return Addr;
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldImpl::reassignSectionAddress(unsigned SectionID,
                                             uint64_t Addr) {
  // The address to use for relocation resolution is not
  // the address of the local section buffer. We must be doing
  // a remote execution environment of some sort. Re-apply any
  // relocations referencing this section with the given address.
  //
  // Addr is a uint64_t because we can't assume the pointer width
  // of the target is the same as that of the host. Just use a generic
  // "big enough" type.
  Sections[SectionID].LoadAddress = Addr;
  DEBUG(dbgs() << "Resolving relocations Section #" << SectionID
          << "\t" << format("%p", (uint8_t *)Addr)
          << "\n");
  resolveRelocationList(Relocations[SectionID], Addr);
}

void RuntimeDyldImpl::resolveRelocationEntry(const RelocationEntry &RE,
                                             uint64_t Value) {
    uint8_t *Target = Sections[RE.SectionID].Address + RE.Offset;
    DEBUG(dbgs() << "\tSectionID: " << RE.SectionID
          << " + " << RE.Offset << " (" << format("%p", Target) << ")"
          << " Data: " << RE.Data
          << " Addend: " << RE.Addend
          << "\n");

    resolveRelocation(Target, Sections[RE.SectionID].LoadAddress + RE.Offset,
                      Value, RE.Data, RE.Addend);
}

void RuntimeDyldImpl::resolveRelocationList(const RelocationList &Relocs,
                                            uint64_t Value) {
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    resolveRelocationEntry(Relocs[i], Value);
  }
}

// resolveSymbols - Resolve any relocations to the specified symbols if
// we know where it lives.
void RuntimeDyldImpl::resolveSymbols() {
  StringMap<RelocationList>::iterator it = SymbolRelocations.begin(),
                                      itEnd = SymbolRelocations.end();
  for (; it != itEnd; it++) {
    StringRef Name = it->first();
    RelocationList &Relocs = it->second;
    StringMap<SymbolLoc>::const_iterator Loc = SymbolTable.find(Name);
    if (Loc == SymbolTable.end()) {
      // This is an external symbol, try to get it address from
      // MemoryManager.
      uint8_t *Addr = (uint8_t*) MemMgr->getPointerToNamedFunction(Name.data(),
                                                                   true);
      DEBUG(dbgs() << "Resolving relocations Name: " << Name
              << "\t" << format("%p", Addr)
              << "\n");
      resolveRelocationList(Relocs, (uintptr_t)Addr);
    } else {
      // Change the relocation to be section relative rather than symbol
      // relative and move it to the resolved relocation list.
      DEBUG(dbgs() << "Resolving symbol '" << Name << "'\n");
      for (int i = 0, e = Relocs.size(); i != e; ++i) {
        RelocationEntry Entry = Relocs[i];
        Entry.Addend += Loc->second.second;
        Relocations[Loc->second.first].push_back(Entry);
      }
      Relocs.clear();
    }
  }
}


//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation
RuntimeDyld::RuntimeDyld(RTDyldMemoryManager *mm) {
  Dyld = 0;
  MM = mm;
}

RuntimeDyld::~RuntimeDyld() {
  delete Dyld;
}

bool RuntimeDyld::loadObject(MemoryBuffer *InputBuffer) {
  if (!Dyld) {
    sys::LLVMFileType type = sys::IdentifyFileType(
            InputBuffer->getBufferStart(),
            static_cast<unsigned>(InputBuffer->getBufferSize()));
    switch (type) {
      case sys::ELF_Relocatable_FileType:
      case sys::ELF_Executable_FileType:
      case sys::ELF_SharedObject_FileType:
      case sys::ELF_Core_FileType:
        Dyld = new RuntimeDyldELF(MM);
        break;
      case sys::Mach_O_Object_FileType:
      case sys::Mach_O_Executable_FileType:
      case sys::Mach_O_FixedVirtualMemorySharedLib_FileType:
      case sys::Mach_O_Core_FileType:
      case sys::Mach_O_PreloadExecutable_FileType:
      case sys::Mach_O_DynamicallyLinkedSharedLib_FileType:
      case sys::Mach_O_DynamicLinker_FileType:
      case sys::Mach_O_Bundle_FileType:
      case sys::Mach_O_DynamicallyLinkedSharedLibStub_FileType:
      case sys::Mach_O_DSYMCompanion_FileType:
        Dyld = new RuntimeDyldMachO(MM);
        break;
      case sys::Unknown_FileType:
      case sys::Bitcode_FileType:
      case sys::Archive_FileType:
      case sys::COFF_FileType:
        report_fatal_error("Incompatible object format!");
    }
  } else {
    if (!Dyld->isCompatibleFormat(InputBuffer))
      report_fatal_error("Incompatible object format!");
  }

  return Dyld->loadObject(InputBuffer);
}

void *RuntimeDyld::getSymbolAddress(StringRef Name) {
  return Dyld->getSymbolAddress(Name);
}

void RuntimeDyld::resolveRelocations() {
  Dyld->resolveRelocations();
}

void RuntimeDyld::reassignSectionAddress(unsigned SectionID,
                                         uint64_t Addr) {
  Dyld->reassignSectionAddress(SectionID, Addr);
}

void RuntimeDyld::mapSectionAddress(void *LocalAddress,
                                    uint64_t TargetAddress) {
  Dyld->mapSectionAddress(LocalAddress, TargetAddress);
}

StringRef RuntimeDyld::getErrorString() {
  return Dyld->getErrorString();
}

} // end namespace llvm
