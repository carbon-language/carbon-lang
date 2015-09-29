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
template <bool isRela>
void InputSection<ELFT>::relocate(
    uint8_t *Buf, iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels,
    const ObjectFile<ELFT> &File, uintX_t BaseAddr,
    const OutputSection<ELFT> &BssSec, const PltSection<ELFT> &PltSec,
    const GotSection<ELFT> &GotSec) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  bool IsMips64EL = File.getObj().isMips64EL();
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    uint32_t Type = RI.getType(IsMips64EL);
    uintX_t GotVA = GotSec.getVA();
    uintX_t SymVA;

    // Handle relocations for local symbols -- they never get
    // resolved so we don't allocate a SymbolBody.
    const Elf_Shdr *SymTab = File.getSymbolTable();
    if (SymIndex < SymTab->sh_info) {
      const Elf_Sym *Sym = File.getObj().getRelocationSymbol(&RI, SymTab);
      if (!Sym)
        continue;
      SymVA = getLocalSymVA(Sym, File);
    } else {
      const auto &Body =
          *cast<ELFSymbolBody<ELFT>>(File.getSymbolBody(SymIndex));
      SymVA = getSymVA<ELFT>(Body, BssSec);
      if (Target->relocNeedsPlt(Type, Body)) {
        SymVA = PltSec.getEntryAddr(Body);
        Type = Target->getPCRelReloc();
      } else if (Target->relocNeedsGot(Type, Body)) {
        SymVA = GotSec.getEntryAddr(Body);
        Type = Target->getGotRefReloc();
      } else if (Target->relocPointsToGot(Type)) {
        SymVA = GotVA;
        Type = Target->getPCRelReloc();
      } else if (isa<SharedSymbol<ELFT>>(Body)) {
        continue;
      }
    }

    Target->relocateOne(Buf, reinterpret_cast<const void *>(&RI), Type,
                        BaseAddr, SymVA, GotVA);
  }
}

template <class ELFT>
void InputSection<ELFT>::writeTo(uint8_t *Buf,
                                 const OutputSection<ELFT> &BssSec,
                                 const PltSection<ELFT> &PltSec,
                                 const GotSection<ELFT> &GotSec) {
  if (Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = *File->getObj().getSectionContents(Header);
  memcpy(Buf + OutputSectionOff, Data.data(), Data.size());

  ObjectFile<ELFT> *File = getFile();
  ELFFile<ELFT> &EObj = File->getObj();
  uint8_t *Base = Buf + getOutputSectionOff();
  uintX_t BaseAddr = Out->getVA() + getOutputSectionOff();
  // Iterate over all relocation sections that apply to this section.
  for (const Elf_Shdr *RelSec : RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      relocate(Base, EObj.relas(RelSec), *File, BaseAddr, BssSec, PltSec,
               GotSec);
    else
      relocate(Base, EObj.rels(RelSec), *File, BaseAddr, BssSec, PltSec,
               GotSec);
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
