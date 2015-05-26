//===--------- lib/ReaderWriter/ELF/ARM/ARMSymbolTable.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_SYMBOL_TABLE_H
#define LLD_READER_WRITER_ELF_ARM_ARM_SYMBOL_TABLE_H

#include "SectionChunks.h"
#include "TargetLayout.h"
#include "ARMELFFile.h"

namespace lld {
namespace elf {

/// \brief The SymbolTable class represents the symbol table in a ELF file
class ARMSymbolTable : public SymbolTable<ELF32LE> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELF32LE> Elf_Sym;

  ARMSymbolTable(const ELFLinkingContext &ctx);

  void addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                      int64_t addr) override;
};

ARMSymbolTable::ARMSymbolTable(const ELFLinkingContext &ctx)
    : SymbolTable(ctx, ".symtab", TargetLayout<ELF32LE>::ORDER_SYMBOL_TABLE) {}

void ARMSymbolTable::addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                                    int64_t addr) {
  SymbolTable::addDefinedAtom(sym, da, addr);

  if ((ARMELFDefinedAtom::ARMContentType)da->contentType() ==
      ARMELFDefinedAtom::typeARMExidx)
    sym.st_value = addr;

  // Set zero bit to distinguish real symbols addressing Thumb instructions.
  // Don't care about mapping symbols like $t and others.
  if (DefinedAtom::codeARMThumb == da->codeModel())
    sym.st_value = static_cast<int64_t>(sym.st_value) | 0x1;

  // Mapping symbols should have special values of binding, type and size set.
  if ((DefinedAtom::codeARM_a == da->codeModel()) ||
      (DefinedAtom::codeARM_d == da->codeModel()) ||
      (DefinedAtom::codeARM_t == da->codeModel())) {
    sym.setBindingAndType(llvm::ELF::STB_LOCAL, llvm::ELF::STT_NOTYPE);
    sym.st_size = 0;
  }
}

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_SYMBOL_TABLE_H
