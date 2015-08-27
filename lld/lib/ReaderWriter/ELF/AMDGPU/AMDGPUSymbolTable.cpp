//===--------- lib/ReaderWriter/ELF/AMDGPU/AMDGPUSymbolTable.cpp ----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSymbolTable.h"
#include "ELFFile.h"
#include "Atoms.h"
#include "SectionChunks.h"

using namespace lld;
using namespace lld::elf;

AMDGPUSymbolTable::AMDGPUSymbolTable(const ELFLinkingContext &ctx)
    : SymbolTable(ctx, ".symtab", TargetLayout<ELF64LE>::ORDER_SYMBOL_TABLE) {}

void AMDGPUSymbolTable::addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                                       int64_t addr) {
  SymbolTable::addDefinedAtom(sym, da, addr);

  // FIXME: Only do this for kernel functions.
  sym.setType(STT_AMDGPU_HSA_KERNEL);

  // Make st_value section relative.
  // FIXME: This is hack to give kernel symbols a section relative offset.
  // Because of this hack only on kernel can be included in a binary file.
  sym.st_value = 0;
}
