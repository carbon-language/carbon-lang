//===- ELFYAML.cpp - ELF YAMLIO implementation ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of ELF.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFYAML.h"

namespace llvm {
namespace yaml {

void
ScalarEnumerationTraits<ELFYAML::ELF_ET>::enumeration(IO &IO,
                                                      ELFYAML::ELF_ET &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(ET_NONE)
  ECase(ET_REL)
  ECase(ET_EXEC)
  ECase(ET_DYN)
  ECase(ET_CORE)
#undef ECase
}

void
ScalarEnumerationTraits<ELFYAML::ELF_EM>::enumeration(IO &IO,
                                                      ELFYAML::ELF_EM &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(EM_NONE)
  ECase(EM_M32)
  ECase(EM_SPARC)
  ECase(EM_386)
  ECase(EM_68K)
  ECase(EM_88K)
  ECase(EM_486)
  ECase(EM_860)
  ECase(EM_MIPS)
  ECase(EM_S370)
  ECase(EM_MIPS_RS3_LE)
  ECase(EM_PARISC)
  ECase(EM_VPP500)
  ECase(EM_SPARC32PLUS)
  ECase(EM_960)
  ECase(EM_PPC)
  ECase(EM_PPC64)
  ECase(EM_S390)
  ECase(EM_SPU)
  ECase(EM_V800)
  ECase(EM_FR20)
  ECase(EM_RH32)
  ECase(EM_RCE)
  ECase(EM_ARM)
  ECase(EM_ALPHA)
  ECase(EM_SH)
  ECase(EM_SPARCV9)
  ECase(EM_TRICORE)
  ECase(EM_ARC)
  ECase(EM_H8_300)
  ECase(EM_H8_300H)
  ECase(EM_H8S)
  ECase(EM_H8_500)
  ECase(EM_IA_64)
  ECase(EM_MIPS_X)
  ECase(EM_COLDFIRE)
  ECase(EM_68HC12)
  ECase(EM_MMA)
  ECase(EM_PCP)
  ECase(EM_NCPU)
  ECase(EM_NDR1)
  ECase(EM_STARCORE)
  ECase(EM_ME16)
  ECase(EM_ST100)
  ECase(EM_TINYJ)
  ECase(EM_X86_64)
  ECase(EM_PDSP)
  ECase(EM_PDP10)
  ECase(EM_PDP11)
  ECase(EM_FX66)
  ECase(EM_ST9PLUS)
  ECase(EM_ST7)
  ECase(EM_68HC16)
  ECase(EM_68HC11)
  ECase(EM_68HC08)
  ECase(EM_68HC05)
  ECase(EM_SVX)
  ECase(EM_ST19)
  ECase(EM_VAX)
  ECase(EM_CRIS)
  ECase(EM_JAVELIN)
  ECase(EM_FIREPATH)
  ECase(EM_ZSP)
  ECase(EM_MMIX)
  ECase(EM_HUANY)
  ECase(EM_PRISM)
  ECase(EM_AVR)
  ECase(EM_FR30)
  ECase(EM_D10V)
  ECase(EM_D30V)
  ECase(EM_V850)
  ECase(EM_M32R)
  ECase(EM_MN10300)
  ECase(EM_MN10200)
  ECase(EM_PJ)
  ECase(EM_OPENRISC)
  ECase(EM_ARC_COMPACT)
  ECase(EM_XTENSA)
  ECase(EM_VIDEOCORE)
  ECase(EM_TMM_GPP)
  ECase(EM_NS32K)
  ECase(EM_TPC)
  ECase(EM_SNP1K)
  ECase(EM_ST200)
  ECase(EM_IP2K)
  ECase(EM_MAX)
  ECase(EM_CR)
  ECase(EM_F2MC16)
  ECase(EM_MSP430)
  ECase(EM_BLACKFIN)
  ECase(EM_SE_C33)
  ECase(EM_SEP)
  ECase(EM_ARCA)
  ECase(EM_UNICORE)
  ECase(EM_EXCESS)
  ECase(EM_DXP)
  ECase(EM_ALTERA_NIOS2)
  ECase(EM_CRX)
  ECase(EM_XGATE)
  ECase(EM_C166)
  ECase(EM_M16C)
  ECase(EM_DSPIC30F)
  ECase(EM_CE)
  ECase(EM_M32C)
  ECase(EM_TSK3000)
  ECase(EM_RS08)
  ECase(EM_SHARC)
  ECase(EM_ECOG2)
  ECase(EM_SCORE7)
  ECase(EM_DSP24)
  ECase(EM_VIDEOCORE3)
  ECase(EM_LATTICEMICO32)
  ECase(EM_SE_C17)
  ECase(EM_TI_C6000)
  ECase(EM_TI_C2000)
  ECase(EM_TI_C5500)
  ECase(EM_MMDSP_PLUS)
  ECase(EM_CYPRESS_M8C)
  ECase(EM_R32C)
  ECase(EM_TRIMEDIA)
  ECase(EM_HEXAGON)
  ECase(EM_8051)
  ECase(EM_STXP7X)
  ECase(EM_NDS32)
  ECase(EM_ECOG1)
  ECase(EM_ECOG1X)
  ECase(EM_MAXQ30)
  ECase(EM_XIMO16)
  ECase(EM_MANIK)
  ECase(EM_CRAYNV2)
  ECase(EM_RX)
  ECase(EM_METAG)
  ECase(EM_MCST_ELBRUS)
  ECase(EM_ECOG16)
  ECase(EM_CR16)
  ECase(EM_ETPU)
  ECase(EM_SLE9X)
  ECase(EM_L10M)
  ECase(EM_K10M)
  ECase(EM_AARCH64)
  ECase(EM_AVR32)
  ECase(EM_STM8)
  ECase(EM_TILE64)
  ECase(EM_TILEPRO)
  ECase(EM_MICROBLAZE)
  ECase(EM_CUDA)
  ECase(EM_TILEGX)
  ECase(EM_CLOUDSHIELD)
  ECase(EM_COREA_1ST)
  ECase(EM_COREA_2ND)
  ECase(EM_ARC_COMPACT2)
  ECase(EM_OPEN8)
  ECase(EM_RL78)
  ECase(EM_VIDEOCORE5)
  ECase(EM_78KOR)
  ECase(EM_56800EX)
  ECase(EM_MBLAZE)
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_ELFCLASS>::enumeration(
    IO &IO, ELFYAML::ELF_ELFCLASS &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  // Since the semantics of ELFCLASSNONE is "invalid", just don't accept it
  // here.
  ECase(ELFCLASS32)
  ECase(ELFCLASS64)
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_ELFDATA>::enumeration(
    IO &IO, ELFYAML::ELF_ELFDATA &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  // Since the semantics of ELFDATANONE is "invalid", just don't accept it
  // here.
  ECase(ELFDATA2LSB)
  ECase(ELFDATA2MSB)
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_ELFOSABI>::enumeration(
    IO &IO, ELFYAML::ELF_ELFOSABI &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(ELFOSABI_NONE)
  ECase(ELFOSABI_HPUX)
  ECase(ELFOSABI_NETBSD)
  ECase(ELFOSABI_GNU)
  ECase(ELFOSABI_GNU)
  ECase(ELFOSABI_HURD)
  ECase(ELFOSABI_SOLARIS)
  ECase(ELFOSABI_AIX)
  ECase(ELFOSABI_IRIX)
  ECase(ELFOSABI_FREEBSD)
  ECase(ELFOSABI_TRU64)
  ECase(ELFOSABI_MODESTO)
  ECase(ELFOSABI_OPENBSD)
  ECase(ELFOSABI_OPENVMS)
  ECase(ELFOSABI_NSK)
  ECase(ELFOSABI_AROS)
  ECase(ELFOSABI_FENIXOS)
  ECase(ELFOSABI_C6000_ELFABI)
  ECase(ELFOSABI_C6000_LINUX)
  ECase(ELFOSABI_ARM)
  ECase(ELFOSABI_STANDALONE)
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_SHT>::enumeration(
    IO &IO, ELFYAML::ELF_SHT &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(SHT_NULL)
  ECase(SHT_PROGBITS)
  ECase(SHT_SYMTAB)
  ECase(SHT_STRTAB)
  ECase(SHT_RELA)
  ECase(SHT_HASH)
  ECase(SHT_DYNAMIC)
  ECase(SHT_NOTE)
  ECase(SHT_NOBITS)
  ECase(SHT_REL)
  ECase(SHT_SHLIB)
  ECase(SHT_DYNSYM)
  ECase(SHT_INIT_ARRAY)
  ECase(SHT_FINI_ARRAY)
  ECase(SHT_PREINIT_ARRAY)
  ECase(SHT_GROUP)
  ECase(SHT_SYMTAB_SHNDX)
#undef ECase
}

void ScalarBitSetTraits<ELFYAML::ELF_SHF>::bitset(IO &IO,
                                                  ELFYAML::ELF_SHF &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, ELF::X);
  BCase(SHF_WRITE)
  BCase(SHF_ALLOC)
  BCase(SHF_EXECINSTR)
  BCase(SHF_MERGE)
  BCase(SHF_STRINGS)
  BCase(SHF_INFO_LINK)
  BCase(SHF_LINK_ORDER)
  BCase(SHF_OS_NONCONFORMING)
  BCase(SHF_GROUP)
  BCase(SHF_TLS)
#undef BCase
}

void ScalarEnumerationTraits<ELFYAML::ELF_STB>::enumeration(
    IO &IO, ELFYAML::ELF_STB &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(STB_LOCAL)
  ECase(STB_GLOBAL)
  ECase(STB_WEAK)
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_STT>::enumeration(
    IO &IO, ELFYAML::ELF_STT &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(STT_NOTYPE)
  ECase(STT_OBJECT)
  ECase(STT_FUNC)
  ECase(STT_SECTION)
  ECase(STT_FILE)
  ECase(STT_COMMON)
  ECase(STT_TLS)
  ECase(STT_GNU_IFUNC)
#undef ECase
}

void MappingTraits<ELFYAML::FileHeader>::mapping(IO &IO,
                                                 ELFYAML::FileHeader &FileHdr) {
  IO.mapRequired("Class", FileHdr.Class);
  IO.mapRequired("Data", FileHdr.Data);
  IO.mapOptional("OSABI", FileHdr.OSABI, ELFYAML::ELF_ELFOSABI(0));
  IO.mapRequired("Type", FileHdr.Type);
  IO.mapRequired("Machine", FileHdr.Machine);
  IO.mapOptional("Entry", FileHdr.Entry, Hex64(0));
}

void MappingTraits<ELFYAML::Symbol>::mapping(IO &IO, ELFYAML::Symbol &Symbol) {
  IO.mapOptional("Name", Symbol.Name, StringRef());
  IO.mapOptional("Binding", Symbol.Binding, ELFYAML::ELF_STB(0));
  IO.mapOptional("Type", Symbol.Type, ELFYAML::ELF_STT(0));
  IO.mapOptional("Section", Symbol.Section, StringRef());
}

void MappingTraits<ELFYAML::Section>::mapping(IO &IO,
                                              ELFYAML::Section &Section) {
  IO.mapOptional("Name", Section.Name, StringRef());
  IO.mapRequired("Type", Section.Type);
  IO.mapOptional("Flags", Section.Flags, ELFYAML::ELF_SHF(0));
  IO.mapOptional("Address", Section.Address, Hex64(0));
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Link", Section.Link);
  IO.mapOptional("AddressAlign", Section.AddressAlign, Hex64(0));
  // TODO: Error if `Type` is SHT_SYMTAB and this is not present, or if
  // `Type` is *not* SHT_SYMTAB and this *is* present. (By SHT_SYMTAB I
  // also mean SHT_DYNSYM, but for simplicity right now we just do
  // SHT_SYMTAB). Want to be able to share the predicate with consumers of
  // this structure.
  IO.mapOptional("Symbols", Section.Symbols);
}

void MappingTraits<ELFYAML::Object>::mapping(IO &IO, ELFYAML::Object &Object) {
  IO.mapRequired("FileHeader", Object.Header);
  IO.mapOptional("Sections", Object.Sections);
}

} // end namespace yaml
} // end namespace llvm
