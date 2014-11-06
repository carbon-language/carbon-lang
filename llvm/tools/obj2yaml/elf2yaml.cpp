//===------ utils/elf2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "obj2yaml.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFYAML.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;

namespace {

template <class ELFT>
class ELFDumper {
  typedef typename object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename object::ELFFile<ELFT>::Elf_Sym_Iter Elf_Sym_Iter;

  const object::ELFFile<ELFT> &Obj;

  std::error_code dumpSymbol(Elf_Sym_Iter Sym, ELFYAML::Symbol &S);
  std::error_code dumpCommonSection(const Elf_Shdr *Shdr, ELFYAML::Section &S);
  std::error_code dumpCommonRelocationSection(const Elf_Shdr *Shdr,
                                              ELFYAML::RelocationSection &S);
  template <class RelT>
  std::error_code dumpRelocation(const Elf_Shdr *Shdr, const RelT *Rel,
                                 ELFYAML::Relocation &R);

  ErrorOr<ELFYAML::RelocationSection *> dumpRelSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::RelocationSection *> dumpRelaSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::RawContentSection *>
  dumpContentSection(const Elf_Shdr *Shdr);

public:
  ELFDumper(const object::ELFFile<ELFT> &O);
  ErrorOr<ELFYAML::Object *> dump();
};

}

template <class ELFT>
ELFDumper<ELFT>::ELFDumper(const object::ELFFile<ELFT> &O)
    : Obj(O) {}

template <class ELFT>
ErrorOr<ELFYAML::Object *> ELFDumper<ELFT>::dump() {
  auto Y = make_unique<ELFYAML::Object>();

  // Dump header
  Y->Header.Class = ELFYAML::ELF_ELFCLASS(Obj.getHeader()->getFileClass());
  Y->Header.Data = ELFYAML::ELF_ELFDATA(Obj.getHeader()->getDataEncoding());
  Y->Header.OSABI = Obj.getHeader()->e_ident[ELF::EI_OSABI];
  Y->Header.Type = Obj.getHeader()->e_type;
  Y->Header.Machine = Obj.getHeader()->e_machine;
  Y->Header.Flags = Obj.getHeader()->e_flags;
  Y->Header.Entry = Obj.getHeader()->e_entry;

  // Dump sections
  for (const Elf_Shdr &Sec : Obj.sections()) {
    switch (Sec.sh_type) {
    case ELF::SHT_NULL:
    case ELF::SHT_SYMTAB:
    case ELF::SHT_DYNSYM:
    case ELF::SHT_STRTAB:
      // Do not dump these sections.
      break;
    case ELF::SHT_RELA: {
      ErrorOr<ELFYAML::RelocationSection *> S = dumpRelaSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    case ELF::SHT_REL: {
      ErrorOr<ELFYAML::RelocationSection *> S = dumpRelSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    // FIXME: Support SHT_GROUP section format.
    default: {
      ErrorOr<ELFYAML::RawContentSection *> S = dumpContentSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
    }
    }
  }

  // Dump symbols
  bool IsFirstSym = true;
  for (auto SI = Obj.begin_symbols(), SE = Obj.end_symbols(); SI != SE; ++SI) {
    if (IsFirstSym) {
      IsFirstSym = false;
      continue;
    }

    ELFYAML::Symbol S;
    if (std::error_code EC = ELFDumper<ELFT>::dumpSymbol(SI, S))
      return EC;

    switch (SI->getBinding())
    {
    case ELF::STB_LOCAL:
      Y->Symbols.Local.push_back(S);
      break;
    case ELF::STB_GLOBAL:
      Y->Symbols.Global.push_back(S);
      break;
    case ELF::STB_WEAK:
      Y->Symbols.Weak.push_back(S);
      break;
    default:
      llvm_unreachable("Unknown ELF symbol binding");
    }
  }

  return Y.release();
}

template <class ELFT>
std::error_code ELFDumper<ELFT>::dumpSymbol(Elf_Sym_Iter Sym,
                                            ELFYAML::Symbol &S) {
  S.Type = Sym->getType();
  S.Value = Sym->st_value;
  S.Size = Sym->st_size;
  S.Other = Sym->st_other;

  ErrorOr<StringRef> NameOrErr = Obj.getSymbolName(Sym);
  if (std::error_code EC = NameOrErr.getError())
    return EC;
  S.Name = NameOrErr.get();

  const Elf_Shdr *Shdr = Obj.getSection(&*Sym);
  if (!Shdr)
    return obj2yaml_error::success;

  NameOrErr = Obj.getSectionName(Shdr);
  if (std::error_code EC = NameOrErr.getError())
    return EC;
  S.Section = NameOrErr.get();

  return obj2yaml_error::success;
}

template <class ELFT>
template <class RelT>
std::error_code ELFDumper<ELFT>::dumpRelocation(const Elf_Shdr *Shdr,
                                                const RelT *Rel,
                                                ELFYAML::Relocation &R) {
  R.Type = Rel->getType(Obj.isMips64EL());
  R.Offset = Rel->r_offset;
  R.Addend = 0;

  auto NamePair = Obj.getRelocationSymbol(Shdr, Rel);
  if (!NamePair.first)
    return obj2yaml_error::success;

  ErrorOr<StringRef> NameOrErr =
      Obj.getSymbolName(NamePair.first, NamePair.second);
  if (std::error_code EC = NameOrErr.getError())
    return EC;
  R.Symbol = NameOrErr.get();

  return obj2yaml_error::success;
}

template <class ELFT>
std::error_code ELFDumper<ELFT>::dumpCommonSection(const Elf_Shdr *Shdr,
                                                   ELFYAML::Section &S) {
  S.Type = Shdr->sh_type;
  S.Flags = Shdr->sh_flags;
  S.Address = Shdr->sh_addr;
  S.AddressAlign = Shdr->sh_addralign;

  ErrorOr<StringRef> NameOrErr = Obj.getSectionName(Shdr);
  if (std::error_code EC = NameOrErr.getError())
    return EC;
  S.Name = NameOrErr.get();

  if (Shdr->sh_link != ELF::SHN_UNDEF) {
    if (const Elf_Shdr *LinkSection = Obj.getSection(Shdr->sh_link)) {
      NameOrErr = Obj.getSectionName(LinkSection);
      if (std::error_code EC = NameOrErr.getError())
        return EC;
      S.Link = NameOrErr.get();
    }
  }

  return obj2yaml_error::success;
}

template <class ELFT>
std::error_code
ELFDumper<ELFT>::dumpCommonRelocationSection(const Elf_Shdr *Shdr,
                                             ELFYAML::RelocationSection &S) {
  if (std::error_code EC = dumpCommonSection(Shdr, S))
    return EC;

  if (const Elf_Shdr *InfoSection = Obj.getSection(Shdr->sh_info)) {
    ErrorOr<StringRef> NameOrErr = Obj.getSectionName(InfoSection);
    if (std::error_code EC = NameOrErr.getError())
      return EC;
    S.Info = NameOrErr.get();
  }

  return obj2yaml_error::success;
}

template <class ELFT>
ErrorOr<ELFYAML::RelocationSection *>
ELFDumper<ELFT>::dumpRelSection(const Elf_Shdr *Shdr) {
  assert(Shdr->sh_type == ELF::SHT_REL && "Section type is not SHT_REL");
  auto S = make_unique<ELFYAML::RelocationSection>();

  if (std::error_code EC = dumpCommonRelocationSection(Shdr, *S))
    return EC;

  for (auto RI = Obj.begin_rel(Shdr), RE = Obj.end_rel(Shdr); RI != RE;
       ++RI) {
    ELFYAML::Relocation R;
    if (std::error_code EC = dumpRelocation(Shdr, &*RI, R))
      return EC;
    S->Relocations.push_back(R);
  }

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::RelocationSection *>
ELFDumper<ELFT>::dumpRelaSection(const Elf_Shdr *Shdr) {
  assert(Shdr->sh_type == ELF::SHT_RELA && "Section type is not SHT_RELA");
  auto S = make_unique<ELFYAML::RelocationSection>();

  if (std::error_code EC = dumpCommonRelocationSection(Shdr, *S))
    return EC;

  for (auto RI = Obj.begin_rela(Shdr), RE = Obj.end_rela(Shdr); RI != RE;
       ++RI) {
    ELFYAML::Relocation R;
    if (std::error_code EC = dumpRelocation(Shdr, &*RI, R))
      return EC;
    R.Addend = RI->r_addend;
    S->Relocations.push_back(R);
  }

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::RawContentSection *>
ELFDumper<ELFT>::dumpContentSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::RawContentSection>();

  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;

  ErrorOr<ArrayRef<uint8_t>> ContentOrErr = Obj.getSectionContents(Shdr);
  if (std::error_code EC = ContentOrErr.getError())
    return EC;
  S->Content = yaml::BinaryRef(ContentOrErr.get());
  S->Size = S->Content.binary_size();

  return S.release();
}

template <class ELFT>
static std::error_code elf2yaml(raw_ostream &Out,
                                const object::ELFFile<ELFT> &Obj) {
  ELFDumper<ELFT> Dumper(Obj);
  ErrorOr<ELFYAML::Object *> YAMLOrErr = Dumper.dump();
  if (std::error_code EC = YAMLOrErr.getError())
    return EC;

  std::unique_ptr<ELFYAML::Object> YAML(YAMLOrErr.get());
  yaml::Output Yout(Out);
  Yout << *YAML;

  return object::object_error::success;
}

std::error_code elf2yaml(raw_ostream &Out, const object::ObjectFile &Obj) {
  if (const auto *ELFObj = dyn_cast<object::ELF32LEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF32BEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF64LEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF64BEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  return obj2yaml_error::unsupported_obj_file_format;
}
