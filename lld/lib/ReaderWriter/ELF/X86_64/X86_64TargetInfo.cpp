//===- lib/ReaderWriter/ELF/X86_64/X86_64ELFTargetInfo.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64ELFTargetInfo.h"

#include "llvm/ADT/StringSwitch.h"

using namespace lld;

#define LLD_CASE(name) .Case(#name, llvm::ELF::name)

ErrorOr<int32_t> elf::X86_64ELFTargetInfo::relocKindFromString(StringRef str) const {
  int32_t ret = llvm::StringSwitch<int32_t>(str)
    LLD_CASE(R_X86_64_NONE)
    LLD_CASE(R_X86_64_64)
    LLD_CASE(R_X86_64_PC32)
    LLD_CASE(R_X86_64_GOT32)
    LLD_CASE(R_X86_64_PLT32)
    LLD_CASE(R_X86_64_COPY)
    LLD_CASE(R_X86_64_GLOB_DAT)
    LLD_CASE(R_X86_64_JUMP_SLOT)
    LLD_CASE(R_X86_64_RELATIVE)
    LLD_CASE(R_X86_64_GOTPCREL)
    LLD_CASE(R_X86_64_32)
    LLD_CASE(R_X86_64_32S)
    LLD_CASE(R_X86_64_16)
    LLD_CASE(R_X86_64_PC16)
    LLD_CASE(R_X86_64_8)
    LLD_CASE(R_X86_64_PC8)
    LLD_CASE(R_X86_64_DTPMOD64)
    LLD_CASE(R_X86_64_DTPOFF64)
    LLD_CASE(R_X86_64_TPOFF64)
    LLD_CASE(R_X86_64_TLSGD)
    LLD_CASE(R_X86_64_TLSLD)
    LLD_CASE(R_X86_64_DTPOFF32)
    LLD_CASE(R_X86_64_GOTTPOFF)
    LLD_CASE(R_X86_64_TPOFF32)
    LLD_CASE(R_X86_64_PC64)
    LLD_CASE(R_X86_64_GOTOFF64)
    LLD_CASE(R_X86_64_GOTPC32)
    LLD_CASE(R_X86_64_GOT64)
    LLD_CASE(R_X86_64_GOTPCREL64)
    LLD_CASE(R_X86_64_GOTPC64)
    LLD_CASE(R_X86_64_GOTPLT64)
    LLD_CASE(R_X86_64_PLTOFF64)
    LLD_CASE(R_X86_64_SIZE32)
    LLD_CASE(R_X86_64_SIZE64)
    LLD_CASE(R_X86_64_GOTPC32_TLSDESC)
    LLD_CASE(R_X86_64_TLSDESC_CALL)
    LLD_CASE(R_X86_64_TLSDESC)
    LLD_CASE(R_X86_64_IRELATIVE)
    .Default(-1);

  if (ret == -1)
    return make_error_code(yaml_reader_error::illegal_value);
  return ret;
}

#undef LLD_CASE

#define LLD_CASE(name) case llvm::ELF::name: return std::string(#name);

ErrorOr<std::string> elf::X86_64ELFTargetInfo::stringFromRelocKind(int32_t kind) const {
  switch (kind) {
  LLD_CASE(R_X86_64_NONE)
  LLD_CASE(R_X86_64_64)
  LLD_CASE(R_X86_64_PC32)
  LLD_CASE(R_X86_64_GOT32)
  LLD_CASE(R_X86_64_PLT32)
  LLD_CASE(R_X86_64_COPY)
  LLD_CASE(R_X86_64_GLOB_DAT)
  LLD_CASE(R_X86_64_JUMP_SLOT)
  LLD_CASE(R_X86_64_RELATIVE)
  LLD_CASE(R_X86_64_GOTPCREL)
  LLD_CASE(R_X86_64_32)
  LLD_CASE(R_X86_64_32S)
  LLD_CASE(R_X86_64_16)
  LLD_CASE(R_X86_64_PC16)
  LLD_CASE(R_X86_64_8)
  LLD_CASE(R_X86_64_PC8)
  LLD_CASE(R_X86_64_DTPMOD64)
  LLD_CASE(R_X86_64_DTPOFF64)
  LLD_CASE(R_X86_64_TPOFF64)
  LLD_CASE(R_X86_64_TLSGD)
  LLD_CASE(R_X86_64_TLSLD)
  LLD_CASE(R_X86_64_DTPOFF32)
  LLD_CASE(R_X86_64_GOTTPOFF)
  LLD_CASE(R_X86_64_TPOFF32)
  LLD_CASE(R_X86_64_PC64)
  LLD_CASE(R_X86_64_GOTOFF64)
  LLD_CASE(R_X86_64_GOTPC32)
  LLD_CASE(R_X86_64_GOT64)
  LLD_CASE(R_X86_64_GOTPCREL64)
  LLD_CASE(R_X86_64_GOTPC64)
  LLD_CASE(R_X86_64_GOTPLT64)
  LLD_CASE(R_X86_64_PLTOFF64)
  LLD_CASE(R_X86_64_SIZE32)
  LLD_CASE(R_X86_64_SIZE64)
  LLD_CASE(R_X86_64_GOTPC32_TLSDESC)
  LLD_CASE(R_X86_64_TLSDESC_CALL)
  LLD_CASE(R_X86_64_TLSDESC)
  LLD_CASE(R_X86_64_IRELATIVE)
  }

  return make_error_code(yaml_reader_error::illegal_value);
}
