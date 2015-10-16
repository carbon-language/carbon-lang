//===- InputSection.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "Config.h"
#include "Error.h"
#include "InputFiles.h"
#include "OutputSections.h"
#include "Target.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
InputSection<ELFT>::InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header)
    : File(F), Header(Header) {}

template <class ELFT>
void InputSection<ELFT>::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                     const Elf_Rel &Rel, uint32_t Type,
                                     uintX_t BaseAddr, uintX_t SymVA) {
  Target->relocateOne(Buf, BufEnd, reinterpret_cast<const void *>(&Rel), Type,
                      BaseAddr, SymVA);
}

template <class ELFT>
void InputSection<ELFT>::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                     const Elf_Rela &Rel, uint32_t Type,
                                     uintX_t BaseAddr, uintX_t SymVA) {
  Target->relocateOne(Buf, BufEnd, reinterpret_cast<const void *>(&Rel), Type,
                      BaseAddr, SymVA + Rel.r_addend);
}

template <class ELFT>
template <bool isRela>
void InputSection<ELFT>::relocate(
    uint8_t *Buf, uint8_t *BufEnd,
    iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels,
    const ObjectFile<ELFT> &File, uintX_t BaseAddr) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(Config->Mips64EL);
    uint32_t Type = RI.getType(Config->Mips64EL);

    // Handle relocations for local symbols -- they never get
    // resolved so we don't allocate a SymbolBody.
    const Elf_Shdr *SymTab = File.getSymbolTable();
    if (SymIndex < SymTab->sh_info) {
      uintX_t SymVA = getLocalRelTarget(File, RI);
      relocateOne(Buf, BufEnd, RI, Type, BaseAddr, SymVA);
      continue;
    }

    SymbolBody &Body = *File.getSymbolBody(SymIndex)->repl();
    uintX_t SymVA = getSymVA<ELFT>(Body);
    if (Target->relocNeedsPlt(Type, Body)) {
      SymVA = Out<ELFT>::Plt->getEntryAddr(Body);
      Type = Target->getPLTRefReloc(Type);
    } else if (Target->relocNeedsGot(Type, Body)) {
      SymVA = Out<ELFT>::Got->getEntryAddr(Body);
      Type = Target->getGotRefReloc();
    } else if (Target->relocPointsToGot(Type)) {
      SymVA = Out<ELFT>::Got->getVA();
      Type = Target->getPCRelReloc();
    } else if (isa<SharedSymbol<ELFT>>(Body)) {
      continue;
    }
    relocateOne(Buf, BufEnd, RI, Type, BaseAddr, SymVA);
  }
}

template <class ELFT> void InputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = *File->getObj().getSectionContents(Header);
  memcpy(Buf + OutSecOff, Data.data(), Data.size());

  ELFFile<ELFT> &EObj = File->getObj();
  uint8_t *Base = Buf + OutSecOff;
  uintX_t BaseAddr = OutSec->getVA() + OutSecOff;
  // Iterate over all relocation sections that apply to this section.
  for (const Elf_Shdr *RelSec : RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      relocate(Base, Base + Data.size(), EObj.relas(RelSec), *File, BaseAddr);
    else
      relocate(Base, Base + Data.size(), EObj.rels(RelSec), *File, BaseAddr);
  }
}

template <class ELFT> StringRef InputSection<ELFT>::getSectionName() const {
  ErrorOr<StringRef> Name = File->getObj().getSectionName(Header);
  error(Name);
  return *Name;
}

namespace lld {
namespace elf2 {
template class InputSection<object::ELF32LE>;
template class InputSection<object::ELF32BE>;
template class InputSection<object::ELF64LE>;
template class InputSection<object::ELF64BE>;
}
}
