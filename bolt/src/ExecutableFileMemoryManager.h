//===--- ExecutableFileMemoryManager.h ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_EXECUTABLE_FILE_MEMORY_MANAGER_H
#define LLVM_TOOLS_LLVM_BOLT_EXECUTABLE_FILE_MEMORY_MANAGER_H

#include "BinaryContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace bolt {

struct SegmentInfo {
  uint64_t Address;           /// Address of the segment in memory.
  uint64_t Size;              /// Size of the segment in memory.
  uint64_t FileOffset;        /// Offset in the file.
  uint64_t FileSize;          /// Size in file.

  void print(raw_ostream &OS) const {
    OS << "SegmentInfo { Address: 0x"
       << Twine::utohexstr(Address) << ", Size: 0x"
       << Twine::utohexstr(Size) << ", FileOffset: 0x"
       << Twine::utohexstr(FileOffset) << ", FileSize: 0x"
       << Twine::utohexstr(FileSize) << "}";
  };
};

inline raw_ostream &operator<<(raw_ostream &OS, const SegmentInfo &SegInfo) {
  SegInfo.print(OS);
  return OS;
}

/// Class responsible for allocating and managing code and data sections.
class ExecutableFileMemoryManager : public SectionMemoryManager {
private:
  uint8_t *allocateSection(intptr_t Size,
                           unsigned Alignment,
                           unsigned SectionID,
                           StringRef SectionName,
                           bool IsCode,
                           bool IsReadOnly);
  BinaryContext &BC;
  bool AllowStubs;

public:
  /// [start memory address] -> [segment info] mapping.
  std::map<uint64_t, SegmentInfo> SegmentMapInfo;

  ExecutableFileMemoryManager(BinaryContext &BC, bool AllowStubs)
    : BC(BC), AllowStubs(AllowStubs) {}

  ~ExecutableFileMemoryManager();

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    return allocateSection(Size, Alignment, SectionID, SectionName,
                           /*IsCode=*/true, true);
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {
    return allocateSection(Size, Alignment, SectionID, SectionName,
                           /*IsCode=*/false, IsReadOnly);
  }

  uint8_t *recordNoteSection(const uint8_t *Data, uintptr_t Size,
                             unsigned Alignment, unsigned SectionID,
                             StringRef SectionName) override;

  bool allowStubAllocation() const override { return AllowStubs; }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override;
};

} // namespace bolt
} // namespace llvm

#endif
