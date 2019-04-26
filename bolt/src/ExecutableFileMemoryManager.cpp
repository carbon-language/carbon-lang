//===--- ExecutableFileMemoryManager.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "ExecutableFileMemoryManager.h"
#include "RewriteInstance.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace llvm {

namespace bolt {

uint8_t *ExecutableFileMemoryManager::allocateSection(intptr_t Size,
                                                      unsigned Alignment,
                                                      unsigned SectionID,
                                                      StringRef SectionName,
                                                      bool IsCode,
                                                      bool IsReadOnly) {
  // Register a debug section as a note section.
  if (RewriteInstance::isDebugSection(SectionName)) {
    uint8_t *DataCopy = new uint8_t[Size];
    auto &Section = BC.registerOrUpdateNoteSection(SectionName,
                                                   DataCopy,
                                                   Size,
                                                   Alignment);
    Section.setSectionID(SectionID);
    assert(!Section.isAllocatable() && "note sections cannot be allocatable");
    return DataCopy;
  }

  uint8_t *Ret;
  if (IsCode) {
    Ret = SectionMemoryManager::allocateCodeSection(Size, Alignment,
                                                    SectionID, SectionName);
  } else {
    Ret = SectionMemoryManager::allocateDataSection(Size, Alignment,
                                                    SectionID, SectionName,
                                                    IsReadOnly);
  }

  const auto Flags = BinarySection::getFlags(IsReadOnly, IsCode, true);
  auto &Section = BC.registerOrUpdateSection(SectionName,
                                             ELF::SHT_PROGBITS,
                                             Flags,
                                             Ret,
                                             Size,
                                             Alignment);
  Section.setSectionID(SectionID);
  assert(Section.isAllocatable() &&
         "verify that allocatable is marked as allocatable");

  DEBUG(dbgs() << "BOLT: allocating " << (Section.isLocal() ? "local " : "")
               << (IsCode ? "code" : (IsReadOnly ? "read-only data" : "data"))
               << " section : " << SectionName
               << " with size " << Size << ", alignment " << Alignment
               << " at 0x" << Ret << ", ID = " << SectionID << "\n");

  return Ret;
}

/// Notifier for non-allocatable (note) section.
uint8_t *ExecutableFileMemoryManager::recordNoteSection(
    const uint8_t *Data,
    uintptr_t Size,
    unsigned Alignment,
    unsigned SectionID,
    StringRef SectionName) {
  DEBUG(dbgs() << "BOLT: note section "
               << SectionName
               << " with size " << Size << ", alignment " << Alignment
               << " at 0x"
               << Twine::utohexstr(reinterpret_cast<uint64_t>(Data)) << '\n');
  auto &Section = BC.registerOrUpdateNoteSection(SectionName,
                                                 copyByteArray(Data, Size),
                                                 Size,
                                                 Alignment);
  Section.setSectionID(SectionID);
  assert(!Section.isAllocatable() && "note sections cannot be allocatable");
  return Section.getOutputData();
}

bool ExecutableFileMemoryManager::finalizeMemory(std::string *ErrMsg) {
  DEBUG(dbgs() << "BOLT: finalizeMemory()\n");
  return SectionMemoryManager::finalizeMemory(ErrMsg);
}

ExecutableFileMemoryManager::~ExecutableFileMemoryManager() { }

}

}
