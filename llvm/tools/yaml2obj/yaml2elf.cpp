//===- yaml2elf - Convert YAML to a ELF object file -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The ELF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFYAML.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

template <class ELFT>
static void writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  const ELFYAML::Header &Hdr = Doc.Header;
  using namespace llvm::ELF;
  using namespace llvm::object;
  typename ELFObjectFile<ELFT>::Elf_Ehdr Header;
  memset(&Header, 0, sizeof(Header));
  Header.e_ident[EI_MAG0] = 0x7f;
  Header.e_ident[EI_MAG1] = 'E';
  Header.e_ident[EI_MAG2] = 'L';
  Header.e_ident[EI_MAG3] = 'F';
  Header.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  bool IsLittleEndian = ELFT::TargetEndianness == support::little;
  Header.e_ident[EI_DATA] = IsLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;

  Header.e_ident[EI_VERSION] = EV_CURRENT;

  // TODO: Implement ELF_ELFOSABI enum.
  Header.e_ident[EI_OSABI] = ELFOSABI_NONE;
  // TODO: Implement ELF_ABIVERSION enum.
  Header.e_ident[EI_ABIVERSION] = 0;
  Header.e_type = Hdr.Type;
  Header.e_machine = Hdr.Machine;
  Header.e_version = EV_CURRENT;
  Header.e_entry = Hdr.Entry;
  Header.e_ehsize = sizeof(Header);

  // TODO: Section headers and program headers.

  OS.write((const char *)&Header, sizeof(Header));
}

int yaml2elf(llvm::raw_ostream &Out, llvm::MemoryBuffer *Buf) {
  yaml::Input YIn(Buf->getBuffer());
  ELFYAML::Object Doc;
  YIn >> Doc;
  if (YIn.error()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }
  if (Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64)) {
    if (Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB))
      writeELF<object::ELFType<support::little, 8, true> >(outs(), Doc);
    else
      writeELF<object::ELFType<support::big, 8, true> >(outs(), Doc);
  } else {
    if (Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB))
      writeELF<object::ELFType<support::little, 4, false> >(outs(), Doc);
    else
      writeELF<object::ELFType<support::big, 4, false> >(outs(), Doc);
  }

  return 0;
}
