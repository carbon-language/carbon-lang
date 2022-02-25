//===- PDB.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_PDB_H
#define LLD_COFF_PDB_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace codeview {
union DebugInfo;
}
}

namespace lld {
class Timer;

namespace coff {
class OutputSection;
class SectionChunk;
class SymbolTable;

void createPDB(SymbolTable *symtab,
               llvm::ArrayRef<OutputSection *> outputSections,
               llvm::ArrayRef<uint8_t> sectionTable,
               llvm::codeview::DebugInfo *buildId);

llvm::Optional<std::pair<llvm::StringRef, uint32_t>>
getFileLineCodeView(const SectionChunk *c, uint32_t addr);

extern Timer loadGHashTimer;
extern Timer mergeGHashTimer;

} // namespace coff
} // namespace lld

#endif
