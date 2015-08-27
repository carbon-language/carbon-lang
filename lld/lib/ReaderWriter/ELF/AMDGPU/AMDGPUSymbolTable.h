//===--------- lib/ReaderWriter/ELF/AMDGPU/AMDGPUSymbolTable.h ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_SYMBOL_TABLE_H
#define LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_SYMBOL_TABLE_H

#include "TargetLayout.h"

namespace lld {
namespace elf {

/// \brief The SymbolTable class represents the symbol table in a ELF file
class AMDGPUSymbolTable : public SymbolTable<ELF64LE> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELF64LE> Elf_Sym;

  AMDGPUSymbolTable(const ELFLinkingContext &ctx);

  void addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                      int64_t addr) override;
};

} // elf
} // lld

#endif
