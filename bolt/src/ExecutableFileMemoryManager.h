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
  // Our linker's main purpose is to handle a single object file, created
  // by RewriteInstance after reading the input binary and reordering it.
  // After objects finish loading, we increment this. Therefore, whenever
  // this is greater than zero, we are dealing with additional objects that
  // will not be managed by BinaryContext but only exist to support linking
  // user-supplied objects into the main input executable.
  uint32_t ObjectsLoaded{0};

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

  bool allowStubAllocation() const override { return AllowStubs; }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override;
};

} // namespace bolt
} // namespace llvm

#endif
