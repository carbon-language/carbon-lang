//===- LinkerScript.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_LINKER_SCRIPT_H
#define LLD_ELF_LINKER_SCRIPT_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace elf2 {

class ScriptParser;

class LinkerScript {
  friend class ScriptParser;

public:
  // Parses a linker script. Calling this function may update
  // this object and Config.
  void read(MemoryBufferRef MB);

  StringRef getOutputSection(StringRef InputSection);
  bool isDiscarded(StringRef InputSection);
  int compareSections(StringRef A, StringRef B);

private:
  // A map for SECTIONS command. The key is input section name
  // and the value is the corresponding output section name.
  llvm::DenseMap<StringRef, StringRef> Sections;

  // Output sections are sorted by this order.
  std::vector<StringRef> SectionOrder;

  llvm::BumpPtrAllocator Alloc;
};

extern LinkerScript *Script;

} // namespace elf2
} // namespace lld

#endif
