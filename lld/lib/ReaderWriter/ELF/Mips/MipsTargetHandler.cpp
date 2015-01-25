//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetHandler.h"

using namespace lld;
using namespace elf;

void MipsRelocationStringTable::registerTable(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::Mips, kindStrings);
}

#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),

const Registry::KindStrings MipsRelocationStringTable::kindStrings[] = {
#include "llvm/Support/ELFRelocs/Mips.def"
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_GOT),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_32_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_26),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_LO16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_STO_PLT),
  LLD_KIND_STRING_ENTRY(LLD_R_MICROMIPS_GLOBAL_26_S1),
  LLD_KIND_STRING_END
};

#undef ELF_RELOC
