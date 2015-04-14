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

template <>
std::error_code checkCompatibility<ELF32LE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS64)
    return make_dynamic_error_code("ELF32 expected, but got ELF64");
  if (endian == ELFDATA2MSB)
    return make_dynamic_error_code(
        "Little endian files are expected, but got a big endian file.");
  return std::error_code();
}

template <>
std::error_code checkCompatibility<ELF32BE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS64)
    return make_dynamic_error_code("ELF32 expected, but got ELF64");
  if (endian == ELFDATA2LSB)
    return make_dynamic_error_code(
        "Big endian files are expected, but got a little endian file.");
  return std::error_code();
}

template <>
std::error_code checkCompatibility<ELF64LE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS32)
    return make_dynamic_error_code("ELF64 expected, but got ELF32");
  if (endian == ELFDATA2MSB)
    return make_dynamic_error_code(
        "Little endian files are expected, but got a big endian file.");
  return std::error_code();
}

template <>
std::error_code checkCompatibility<ELF64BE>(unsigned char size,
                                            unsigned char endian) {
  if (size == ELFCLASS32)
    return make_dynamic_error_code("ELF64 expected, but got ELF32");
  if (endian == ELFDATA2LSB)
    return make_dynamic_error_code(
        "Big endian files are expected, but got a little endian file.");
  return std::error_code();
}

} // end namespace elf
} // end namespace lld
