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

namespace lld {
namespace elf {

/// \brief The SymbolTable class represents the symbol table in a ELF file
template<class ELFT>
class ARMSymbolTable : public SymbolTable<ELFT> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  ARMSymbolTable(const ELFLinkingContext &context);

  void addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                      int64_t addr) override;
};

template <class ELFT>
ARMSymbolTable<ELFT>::ARMSymbolTable(const ELFLinkingContext &context)
    : SymbolTable<ELFT>(context, ".symtab",
                        DefaultLayout<ELFT>::ORDER_SYMBOL_TABLE) {}

template <class ELFT>
void ARMSymbolTable<ELFT>::addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                                          int64_t addr) {
  SymbolTable<ELFT>::addDefinedAtom(sym, da, addr);

  // Set zero bit to distinguish symbols addressing Thumb instructions
  if (DefinedAtom::codeARMThumb == da->codeModel())
    sym.st_value = static_cast<int64_t>(sym.st_value) | 0x1;
}

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_SYMBOL_TABLE_H
