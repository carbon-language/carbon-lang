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
#include "llvm/Support/Casting.h"

namespace llvm {

ELFYAML::Section::~Section() {}

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

void ScalarBitSetTraits<ELFYAML::ELF_EF>::bitset(IO &IO,
                                                 ELFYAML::ELF_EF &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");
#define BCase(X) IO.bitSetCase(Value, #X, ELF::X);
  switch (Object->Header.Machine) {
  case ELF::EM_ARM:
    BCase(EF_ARM_SOFT_FLOAT)
    BCase(EF_ARM_VFP_FLOAT)
    BCase(EF_ARM_EABI_UNKNOWN)
    BCase(EF_ARM_EABI_VER1)
    BCase(EF_ARM_EABI_VER2)
    BCase(EF_ARM_EABI_VER3)
    BCase(EF_ARM_EABI_VER4)
    BCase(EF_ARM_EABI_VER5)
    break;
  case ELF::EM_MIPS:
    BCase(EF_MIPS_NOREORDER)
    BCase(EF_MIPS_PIC)
    BCase(EF_MIPS_CPIC)
    BCase(EF_MIPS_ABI2)
    BCase(EF_MIPS_32BITMODE)
    BCase(EF_MIPS_ABI_O32)
    BCase(EF_MIPS_MICROMIPS)
    BCase(EF_MIPS_ARCH_ASE_M16)
    BCase(EF_MIPS_ARCH_1)
    BCase(EF_MIPS_ARCH_2)
    BCase(EF_MIPS_ARCH_3)
    BCase(EF_MIPS_ARCH_4)
    BCase(EF_MIPS_ARCH_5)
    BCase(EF_MIPS_ARCH_32)
    BCase(EF_MIPS_ARCH_64)
    BCase(EF_MIPS_ARCH_32R2)
    BCase(EF_MIPS_ARCH_64R2)
    break;
  case ELF::EM_HEXAGON:
    BCase(EF_HEXAGON_MACH_V2)
    BCase(EF_HEXAGON_MACH_V3)
    BCase(EF_HEXAGON_MACH_V4)
    BCase(EF_HEXAGON_MACH_V5)
    BCase(EF_HEXAGON_ISA_V2)
    BCase(EF_HEXAGON_ISA_V3)
    BCase(EF_HEXAGON_ISA_V4)
    BCase(EF_HEXAGON_ISA_V5)
    break;
  default:
    llvm_unreachable("Unsupported architecture");
  }
#undef BCase
}

void ScalarEnumerationTraits<ELFYAML::ELF_SHT>::enumeration(
    IO &IO, ELFYAML::ELF_SHT &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  ECase(SHT_NULL)
  ECase(SHT_PROGBITS)
  // No SHT_SYMTAB. Use the top-level `Symbols` key instead.
  // FIXME: Issue a diagnostic with this information.
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
  ECase(SHT_LOOS)
  ECase(SHT_GNU_ATTRIBUTES)
  ECase(SHT_GNU_HASH)
  ECase(SHT_GNU_verdef)
  ECase(SHT_GNU_verneed)
  ECase(SHT_GNU_versym)
  ECase(SHT_HIOS)
  ECase(SHT_LOPROC)
  ECase(SHT_ARM_EXIDX)
  ECase(SHT_ARM_PREEMPTMAP)
  ECase(SHT_ARM_ATTRIBUTES)
  ECase(SHT_ARM_DEBUGOVERLAY)
  ECase(SHT_ARM_OVERLAYSECTION)
  ECase(SHT_HEX_ORDERED)
  ECase(SHT_X86_64_UNWIND)
  ECase(SHT_MIPS_REGINFO)
  ECase(SHT_MIPS_OPTIONS)
#undef ECase
}

void ScalarBitSetTraits<ELFYAML::ELF_SHF>::bitset(IO &IO,
                                                  ELFYAML::ELF_SHF &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, ELF::X);
  BCase(SHF_WRITE)
  BCase(SHF_ALLOC)
  BCase(SHF_EXCLUDE)
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

void ScalarEnumerationTraits<ELFYAML::ELF_REL>::enumeration(
    IO &IO, ELFYAML::ELF_REL &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");
#define ECase(X) IO.enumCase(Value, #X, ELF::X);
  switch (Object->Header.Machine) {
  case ELF::EM_X86_64:
    ECase(R_X86_64_NONE)
    ECase(R_X86_64_64)
    ECase(R_X86_64_PC32)
    ECase(R_X86_64_GOT32)
    ECase(R_X86_64_PLT32)
    ECase(R_X86_64_COPY)
    ECase(R_X86_64_GLOB_DAT)
    ECase(R_X86_64_JUMP_SLOT)
    ECase(R_X86_64_RELATIVE)
    ECase(R_X86_64_GOTPCREL)
    ECase(R_X86_64_32)
    ECase(R_X86_64_32S)
    ECase(R_X86_64_16)
    ECase(R_X86_64_PC16)
    ECase(R_X86_64_8)
    ECase(R_X86_64_PC8)
    ECase(R_X86_64_DTPMOD64)
    ECase(R_X86_64_DTPOFF64)
    ECase(R_X86_64_TPOFF64)
    ECase(R_X86_64_TLSGD)
    ECase(R_X86_64_TLSLD)
    ECase(R_X86_64_DTPOFF32)
    ECase(R_X86_64_GOTTPOFF)
    ECase(R_X86_64_TPOFF32)
    ECase(R_X86_64_PC64)
    ECase(R_X86_64_GOTOFF64)
    ECase(R_X86_64_GOTPC32)
    ECase(R_X86_64_GOT64)
    ECase(R_X86_64_GOTPCREL64)
    ECase(R_X86_64_GOTPC64)
    ECase(R_X86_64_GOTPLT64)
    ECase(R_X86_64_PLTOFF64)
    ECase(R_X86_64_SIZE32)
    ECase(R_X86_64_SIZE64)
    ECase(R_X86_64_GOTPC32_TLSDESC)
    ECase(R_X86_64_TLSDESC_CALL)
    ECase(R_X86_64_TLSDESC)
    ECase(R_X86_64_IRELATIVE)
    break;
  case ELF::EM_MIPS:
    ECase(R_MIPS_NONE)
    ECase(R_MIPS_16)
    ECase(R_MIPS_32)
    ECase(R_MIPS_REL32)
    ECase(R_MIPS_26)
    ECase(R_MIPS_HI16)
    ECase(R_MIPS_LO16)
    ECase(R_MIPS_GPREL16)
    ECase(R_MIPS_LITERAL)
    ECase(R_MIPS_GOT16)
    ECase(R_MIPS_PC16)
    ECase(R_MIPS_CALL16)
    ECase(R_MIPS_GPREL32)
    ECase(R_MIPS_UNUSED1)
    ECase(R_MIPS_UNUSED2)
    ECase(R_MIPS_SHIFT5)
    ECase(R_MIPS_SHIFT6)
    ECase(R_MIPS_64)
    ECase(R_MIPS_GOT_DISP)
    ECase(R_MIPS_GOT_PAGE)
    ECase(R_MIPS_GOT_OFST)
    ECase(R_MIPS_GOT_HI16)
    ECase(R_MIPS_GOT_LO16)
    ECase(R_MIPS_SUB)
    ECase(R_MIPS_INSERT_A)
    ECase(R_MIPS_INSERT_B)
    ECase(R_MIPS_DELETE)
    ECase(R_MIPS_HIGHER)
    ECase(R_MIPS_HIGHEST)
    ECase(R_MIPS_CALL_HI16)
    ECase(R_MIPS_CALL_LO16)
    ECase(R_MIPS_SCN_DISP)
    ECase(R_MIPS_REL16)
    ECase(R_MIPS_ADD_IMMEDIATE)
    ECase(R_MIPS_PJUMP)
    ECase(R_MIPS_RELGOT)
    ECase(R_MIPS_JALR)
    ECase(R_MIPS_TLS_DTPMOD32)
    ECase(R_MIPS_TLS_DTPREL32)
    ECase(R_MIPS_TLS_DTPMOD64)
    ECase(R_MIPS_TLS_DTPREL64)
    ECase(R_MIPS_TLS_GD)
    ECase(R_MIPS_TLS_LDM)
    ECase(R_MIPS_TLS_DTPREL_HI16)
    ECase(R_MIPS_TLS_DTPREL_LO16)
    ECase(R_MIPS_TLS_GOTTPREL)
    ECase(R_MIPS_TLS_TPREL32)
    ECase(R_MIPS_TLS_TPREL64)
    ECase(R_MIPS_TLS_TPREL_HI16)
    ECase(R_MIPS_TLS_TPREL_LO16)
    ECase(R_MIPS_GLOB_DAT)
    ECase(R_MIPS_COPY)
    ECase(R_MIPS_JUMP_SLOT)
    ECase(R_MICROMIPS_26_S1)
    ECase(R_MICROMIPS_HI16)
    ECase(R_MICROMIPS_LO16)
    ECase(R_MICROMIPS_GOT16)
    ECase(R_MICROMIPS_PC16_S1)
    ECase(R_MICROMIPS_CALL16)
    ECase(R_MICROMIPS_GOT_DISP)
    ECase(R_MICROMIPS_GOT_PAGE)
    ECase(R_MICROMIPS_GOT_OFST)
    ECase(R_MICROMIPS_TLS_GD)
    ECase(R_MICROMIPS_TLS_LDM)
    ECase(R_MICROMIPS_TLS_DTPREL_HI16)
    ECase(R_MICROMIPS_TLS_DTPREL_LO16)
    ECase(R_MICROMIPS_TLS_TPREL_HI16)
    ECase(R_MICROMIPS_TLS_TPREL_LO16)
    ECase(R_MIPS_NUM)
    ECase(R_MIPS_PC32)
    break;
  case ELF::EM_HEXAGON:
    ECase(R_HEX_NONE)
    ECase(R_HEX_B22_PCREL)
    ECase(R_HEX_B15_PCREL)
    ECase(R_HEX_B7_PCREL)
    ECase(R_HEX_LO16)
    ECase(R_HEX_HI16)
    ECase(R_HEX_32)
    ECase(R_HEX_16)
    ECase(R_HEX_8)
    ECase(R_HEX_GPREL16_0)
    ECase(R_HEX_GPREL16_1)
    ECase(R_HEX_GPREL16_2)
    ECase(R_HEX_GPREL16_3)
    ECase(R_HEX_HL16)
    ECase(R_HEX_B13_PCREL)
    ECase(R_HEX_B9_PCREL)
    ECase(R_HEX_B32_PCREL_X)
    ECase(R_HEX_32_6_X)
    ECase(R_HEX_B22_PCREL_X)
    ECase(R_HEX_B15_PCREL_X)
    ECase(R_HEX_B13_PCREL_X)
    ECase(R_HEX_B9_PCREL_X)
    ECase(R_HEX_B7_PCREL_X)
    ECase(R_HEX_16_X)
    ECase(R_HEX_12_X)
    ECase(R_HEX_11_X)
    ECase(R_HEX_10_X)
    ECase(R_HEX_9_X)
    ECase(R_HEX_8_X)
    ECase(R_HEX_7_X)
    ECase(R_HEX_6_X)
    ECase(R_HEX_32_PCREL)
    ECase(R_HEX_COPY)
    ECase(R_HEX_GLOB_DAT)
    ECase(R_HEX_JMP_SLOT)
    ECase(R_HEX_RELATIVE)
    ECase(R_HEX_PLT_B22_PCREL)
    ECase(R_HEX_GOTREL_LO16)
    ECase(R_HEX_GOTREL_HI16)
    ECase(R_HEX_GOTREL_32)
    ECase(R_HEX_GOT_LO16)
    ECase(R_HEX_GOT_HI16)
    ECase(R_HEX_GOT_32)
    ECase(R_HEX_GOT_16)
    ECase(R_HEX_DTPMOD_32)
    ECase(R_HEX_DTPREL_LO16)
    ECase(R_HEX_DTPREL_HI16)
    ECase(R_HEX_DTPREL_32)
    ECase(R_HEX_DTPREL_16)
    ECase(R_HEX_GD_PLT_B22_PCREL)
    ECase(R_HEX_GD_GOT_LO16)
    ECase(R_HEX_GD_GOT_HI16)
    ECase(R_HEX_GD_GOT_32)
    ECase(R_HEX_GD_GOT_16)
    ECase(R_HEX_IE_LO16)
    ECase(R_HEX_IE_HI16)
    ECase(R_HEX_IE_32)
    ECase(R_HEX_IE_GOT_LO16)
    ECase(R_HEX_IE_GOT_HI16)
    ECase(R_HEX_IE_GOT_32)
    ECase(R_HEX_IE_GOT_16)
    ECase(R_HEX_TPREL_LO16)
    ECase(R_HEX_TPREL_HI16)
    ECase(R_HEX_TPREL_32)
    ECase(R_HEX_TPREL_16)
    ECase(R_HEX_6_PCREL_X)
    ECase(R_HEX_GOTREL_32_6_X)
    ECase(R_HEX_GOTREL_16_X)
    ECase(R_HEX_GOTREL_11_X)
    ECase(R_HEX_GOT_32_6_X)
    ECase(R_HEX_GOT_16_X)
    ECase(R_HEX_GOT_11_X)
    ECase(R_HEX_DTPREL_32_6_X)
    ECase(R_HEX_DTPREL_16_X)
    ECase(R_HEX_DTPREL_11_X)
    ECase(R_HEX_GD_GOT_32_6_X)
    ECase(R_HEX_GD_GOT_16_X)
    ECase(R_HEX_GD_GOT_11_X)
    ECase(R_HEX_IE_32_6_X)
    ECase(R_HEX_IE_16_X)
    ECase(R_HEX_IE_GOT_32_6_X)
    ECase(R_HEX_IE_GOT_16_X)
    ECase(R_HEX_IE_GOT_11_X)
    ECase(R_HEX_TPREL_32_6_X)
    ECase(R_HEX_TPREL_16_X)
    ECase(R_HEX_TPREL_11_X)
    break;
  default:
    llvm_unreachable("Unsupported architecture");
  }
#undef ECase
}

void MappingTraits<ELFYAML::FileHeader>::mapping(IO &IO,
                                                 ELFYAML::FileHeader &FileHdr) {
  IO.mapRequired("Class", FileHdr.Class);
  IO.mapRequired("Data", FileHdr.Data);
  IO.mapOptional("OSABI", FileHdr.OSABI, ELFYAML::ELF_ELFOSABI(0));
  IO.mapRequired("Type", FileHdr.Type);
  IO.mapRequired("Machine", FileHdr.Machine);
  IO.mapOptional("Flags", FileHdr.Flags, ELFYAML::ELF_EF(0));
  IO.mapOptional("Entry", FileHdr.Entry, Hex64(0));
}

void MappingTraits<ELFYAML::Symbol>::mapping(IO &IO, ELFYAML::Symbol &Symbol) {
  IO.mapOptional("Name", Symbol.Name, StringRef());
  IO.mapOptional("Type", Symbol.Type, ELFYAML::ELF_STT(0));
  IO.mapOptional("Section", Symbol.Section, StringRef());
  IO.mapOptional("Value", Symbol.Value, Hex64(0));
  IO.mapOptional("Size", Symbol.Size, Hex64(0));
}

void MappingTraits<ELFYAML::LocalGlobalWeakSymbols>::mapping(
    IO &IO, ELFYAML::LocalGlobalWeakSymbols &Symbols) {
  IO.mapOptional("Local", Symbols.Local);
  IO.mapOptional("Global", Symbols.Global);
  IO.mapOptional("Weak", Symbols.Weak);
}

static void commonSectionMapping(IO &IO, ELFYAML::Section &Section) {
  IO.mapOptional("Name", Section.Name, StringRef());
  IO.mapRequired("Type", Section.Type);
  IO.mapOptional("Flags", Section.Flags, ELFYAML::ELF_SHF(0));
  IO.mapOptional("Address", Section.Address, Hex64(0));
  IO.mapOptional("Link", Section.Link);
  IO.mapOptional("Info", Section.Info);
  IO.mapOptional("AddressAlign", Section.AddressAlign, Hex64(0));
}

static void sectionMapping(IO &IO, ELFYAML::RawContentSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
}

static void sectionMapping(IO &IO, ELFYAML::RelocationSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Relocations", Section.Relocations);
}

void MappingTraits<std::unique_ptr<ELFYAML::Section>>::mapping(
    IO &IO, std::unique_ptr<ELFYAML::Section> &Section) {
  ELFYAML::ELF_SHT sectionType;
  if (IO.outputting())
    sectionType = Section->Type;
  IO.mapRequired("Type", sectionType);

  switch (sectionType) {
  case ELF::SHT_REL:
  case ELF::SHT_RELA:
    if (!IO.outputting())
      Section.reset(new ELFYAML::RelocationSection());
    sectionMapping(IO, *cast<ELFYAML::RelocationSection>(Section.get()));
    break;
  default:
    if (!IO.outputting())
      Section.reset(new ELFYAML::RawContentSection());
    sectionMapping(IO, *cast<ELFYAML::RawContentSection>(Section.get()));
  }
}

void MappingTraits<ELFYAML::Relocation>::mapping(IO &IO,
                                                 ELFYAML::Relocation &Rel) {
  IO.mapRequired("Offset", Rel.Offset);
  IO.mapRequired("Symbol", Rel.Symbol);
  IO.mapRequired("Type", Rel.Type);
  IO.mapOptional("Addend", Rel.Addend);
}

void MappingTraits<ELFYAML::Object>::mapping(IO &IO, ELFYAML::Object &Object) {
  assert(!IO.getContext() && "The IO context is initialized already");
  IO.setContext(&Object);
  IO.mapRequired("FileHeader", Object.Header);
  IO.mapOptional("Sections", Object.Sections);
  IO.mapOptional("Symbols", Object.Symbols);
  IO.setContext(nullptr);
}

} // end namespace yaml
} // end namespace llvm
