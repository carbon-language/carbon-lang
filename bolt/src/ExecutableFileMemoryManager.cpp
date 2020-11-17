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
#define DEBUG_TYPE "efmm"

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
  if (!ObjectsLoaded && RewriteInstance::isDebugSection(SectionName)) {
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
    Ret = SectionMemoryManager::allocateDataSection(Size, Alignment, SectionID,
                                                    SectionName, IsReadOnly);
  }

  SmallVector<char, 256> Buf;
  if (ObjectsLoaded > 0) {
    if (BC.isELF()) {
      SectionName = (Twine(SectionName) + ".bolt.extra." + Twine(ObjectsLoaded))
                        .toStringRef(Buf);
    } else if (BC.isMachO()) {
      assert((SectionName == "__text" || SectionName == "__data" ||
              SectionName == "__setup" || SectionName == "__cstring") &&
             "Unexpected section in the instrumentation library");
      SectionName = ("I" + Twine(SectionName)).toStringRef(Buf);
    }
  }

  auto &Section = BC.registerOrUpdateSection(
      SectionName, ELF::SHT_PROGBITS,
      BinarySection::getFlags(IsReadOnly, IsCode, true), Ret, Size, Alignment);
  Section.setSectionID(SectionID);
  assert(Section.isAllocatable() &&
         "verify that allocatable is marked as allocatable");

  DEBUG(dbgs() << "BOLT: allocating "
               << (IsCode ? "code" : (IsReadOnly ? "read-only data" : "data"))
               << " section : " << SectionName
               << " with size " << Size << ", alignment " << Alignment
               << " at 0x" << Ret << ", ID = " << SectionID << "\n");
  return Ret;
}

bool ExecutableFileMemoryManager::finalizeMemory(std::string *ErrMsg) {
  DEBUG(dbgs() << "BOLT: finalizeMemory()\n");
  ++ObjectsLoaded;
  return SectionMemoryManager::finalizeMemory(ErrMsg);
}

ExecutableFileMemoryManager::~ExecutableFileMemoryManager() { }

}

}
