//===- lib/ReaderWriter/ELF/FileCommon.cpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFFile.h"
#include "FileCommon.h"

using namespace llvm::ELF;

namespace lld {
namespace elf {

static const char *elf32_expected = "ELF32 expected, but got ELF64";
static const char *elf64_expected = "ELF64 expected, but got ELF32";
static const char *le_expected =
      "Little endian files are expected, but got a big endian file.";
static const char *be_expected =
      "Big endian files are expected, but got a little endian file.";

template <>
std::error_code checkCompatibility<ELF32LE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS64)
    return make_dynamic_error_code(elf32_expected);
  if (endian == ELFDATA2MSB)
    return make_dynamic_error_code(le_expected);
  return std::error_code();
}

template <>
std::error_code checkCompatibility<ELF32BE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS64)
    return make_dynamic_error_code(elf32_expected);
  if (endian == ELFDATA2LSB)
    return make_dynamic_error_code(be_expected);
  return std::error_code();
}

template <>
std::error_code checkCompatibility<ELF64LE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS32)
    return make_dynamic_error_code(elf64_expected);
  if (endian == ELFDATA2MSB)
    return make_dynamic_error_code(le_expected);
  return std::error_code();
}

template <>
std::error_code checkCompatibility<ELF64BE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS32)
    return make_dynamic_error_code(elf64_expected);
  if (endian == ELFDATA2LSB)
    return make_dynamic_error_code(be_expected);
  return std::error_code();
}

} // end namespace elf
} // end namespace lld
