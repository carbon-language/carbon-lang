//===- InputSection.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "Error.h"
#include "InputFiles.h"
#include "OutputSections.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
InputSection<ELFT>::InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header)
    : File(F), Header(Header) {}

template <class ELFT>
void InputSection<ELFT>::relocateOne(uint8_t *Buf, const Elf_Rel &Rel,
                                     uint32_t Type, uintX_t BaseAddr,
                                     uintX_t SymVA) {
  uintX_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  uint32_t Addend = *(support::ulittle32_t *)Location;
  switch (Type) {
  case R_386_32:
    support::endian::write32le(Location, SymVA + Addend);
    break;
  default:
    llvm::errs() << Twine("unrecognized reloc ") + Twine(Type) << '\n';
    break;
  }
}

template <class ELFT>
void InputSection<ELFT>::relocateOne(uint8_t *Buf, const Elf_Rela &Rel,
                                     uint32_t Type, uintX_t BaseAddr,
                                     uintX_t SymVA) {
  uintX_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  switch (Type) {
  case R_X86_64_PC32:
    support::endian::write32le(Location,
                               SymVA + Rel.r_addend - (BaseAddr + Offset));
    break;
  case R_X86_64_64:
    support::endian::write64le(Location, SymVA + Rel.r_addend);
    break;
  case R_X86_64_32: {
  case R_X86_64_32S:
    uint64_t VA = SymVA + Rel.r_addend;
    if (Type == R_X86_64_32 && !isUInt<32>(VA))
      error("R_X86_64_32 out of range");
    else if (!isInt<32>(VA))
      error("R_X86_64_32S out of range");

    support::endian::write32le(Location, VA);
    break;
  }
  default:
    llvm::errs() << Twine("unrecognized reloc ") + Twine(Type) << '\n';
    break;
  }
}

template <class ELFT>
template <bool isRela>
void InputSection<ELFT>::relocate(
    uint8_t *Buf, iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels,
    const ObjectFile<ELFT> &File, uintX_t BaseAddr,
    const PltSection<ELFT> &PltSec, const GotSection<ELFT> &GotSec) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  bool IsMips64EL = File.getObj()->isMips64EL();
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    uint32_t Type = RI.getType(IsMips64EL);
    uintX_t SymVA;

    // Handle relocations for local symbols -- they never get
    // resolved so we don't allocate a SymbolBody.
    const Elf_Shdr *SymTab = File.getSymbolTable();
    if (SymIndex < SymTab->sh_info) {
      const Elf_Sym *Sym = File.getObj()->getRelocationSymbol(&RI, SymTab);
      if (!Sym)
        continue;
      SymVA = getLocalSymVA(Sym, File);
    } else {
      const SymbolBody *Body = File.getSymbolBody(SymIndex);
      if (!Body)
        continue;
      switch (Body->kind()) {
      case SymbolBody::DefinedRegularKind:
        SymVA = getSymVA<ELFT>(cast<DefinedRegular<ELFT>>(Body));
        break;
      case SymbolBody::DefinedAbsoluteKind:
        SymVA = cast<DefinedAbsolute<ELFT>>(Body)->Sym.st_value;
        break;
      case SymbolBody::DefinedCommonKind: {
        auto *DC = cast<DefinedCommon<ELFT>>(Body);
        SymVA = DC->OutputSec->getVA() + DC->OffsetInBSS;
        break;
      }
      case SymbolBody::SharedKind:
        if (relocNeedsPLT(Type)) {
          SymVA = PltSec.getEntryAddr(*Body);
          Type = R_X86_64_PC32;
        } else if (relocNeedsGOT(Type)) {
          SymVA = GotSec.getEntryAddr(*Body);
          Type = R_X86_64_PC32;
        } else {
          continue;
        }
        break;
      case SymbolBody::UndefinedKind:
        assert(Body->isWeak() && "Undefined symbol reached writer");
        SymVA = 0;
        break;
      case SymbolBody::LazyKind:
        llvm_unreachable("Lazy symbol reached writer");
      }
    }

    relocateOne(Buf, RI, Type, BaseAddr, SymVA);
  }
}

template <class ELFT>
void InputSection<ELFT>::writeTo(uint8_t *Buf, const PltSection<ELFT> &PltSec,
                                 const GotSection<ELFT> &GotSec) {
  if (Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = *File->getObj()->getSectionContents(Header);
  memcpy(Buf + OutputSectionOff, Data.data(), Data.size());

  const ObjectFile<ELFT> *File = getFile();
  ELFFile<ELFT> *EObj = File->getObj();
  uint8_t *Base = Buf + getOutputSectionOff();
  uintX_t BaseAddr = Out->getVA() + getOutputSectionOff();
  // Iterate over all relocation sections that apply to this section.
  for (const Elf_Shdr *RelSec : RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      relocate(Base, EObj->relas(RelSec), *File, BaseAddr, PltSec, GotSec);
    else
      relocate(Base, EObj->rels(RelSec), *File, BaseAddr, PltSec, GotSec);
  }
}

template <class ELFT> StringRef InputSection<ELFT>::getSectionName() const {
  ErrorOr<StringRef> Name = File->getObj()->getSectionName(Header);
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
