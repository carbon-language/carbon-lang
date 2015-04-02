//===- lib/ReaderWriter/ELF/CreateELF.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides a simple way to create an object templated on
/// ELFType depending on the runtime type needed.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_CREATE_ELF_H
#define LLD_READER_WRITER_ELF_CREATE_ELF_H

#include "llvm/Object/ELF.h"
#include "llvm/Support/Compiler.h"

namespace {
using llvm::object::ELFType;

/// \func createELF
/// \brief Create an object depending on the runtime attributes and alignment
/// of an ELF file.
///
/// \param Traits
/// Traits::result_type must be a type convertable from what create returns.
/// Traits::create must be a template function which takes an ELFType and
/// returns something convertable to Traits::result_type.
///
/// \param ident pair of EI_CLASS and EI_DATA.
/// \param maxAlignment the maximum alignment of the file.
/// \param args arguments forwarded to CreateELFTraits<T>::create.

#define LLVM_CREATE_ELF_CreateELFTraits(endian, align, is64, ...) \
  Traits::template create<ELFType<llvm::support::endian, align, is64>>( \
      __VA_ARGS__);

#if !LLVM_IS_UNALIGNED_ACCESS_FAST
# define LLVM_CREATE_ELF_Create(normal, low, endian, is64, ...) \
  if (maxAlignment >= normal) \
    return LLVM_CREATE_ELF_CreateELFTraits(endian, normal, is64, __VA_ARGS__) \
  else if (maxAlignment >= low) \
    return LLVM_CREATE_ELF_CreateELFTraits(endian, low, is64, __VA_ARGS__) \
  else \
    llvm_unreachable("Invalid alignment for ELF file!");
#else
# define LLVM_CREATE_ELF_Create(normal, low, endian, is64, ...) \
  if (maxAlignment >= low) \
    return LLVM_CREATE_ELF_CreateELFTraits(endian, low, is64, __VA_ARGS__) \
  else \
    llvm_unreachable("Invalid alignment for ELF file!");
#endif

template <class Traits, class ...Args>
typename Traits::result_type createELF(
    std::pair<unsigned char, unsigned char> ident, std::size_t maxAlignment,
    Args &&...args) {
  if (ident.first == llvm::ELF::ELFCLASS32 &&
      ident.second == llvm::ELF::ELFDATA2LSB) {
    LLVM_CREATE_ELF_Create(4, 2, little, false, std::forward<Args>(args)...)
  } else if (ident.first == llvm::ELF::ELFCLASS32 &&
             ident.second == llvm::ELF::ELFDATA2MSB) {
    LLVM_CREATE_ELF_Create(4, 2, big, false, std::forward<Args>(args)...)
  } else if (ident.first == llvm::ELF::ELFCLASS64 &&
             ident.second == llvm::ELF::ELFDATA2MSB) {
    LLVM_CREATE_ELF_Create(8, 2, big, true, std::forward<Args>(args)...)
  } else if (ident.first == llvm::ELF::ELFCLASS64 &&
             ident.second == llvm::ELF::ELFDATA2LSB) {
    LLVM_CREATE_ELF_Create(8, 2, little, true, std::forward<Args>(args)...)
  }
  llvm_unreachable("Invalid ELF type!");
}
} // end anon namespace

#undef LLVM_CREATE_ELF_CreateELFTraits
#undef LLVM_CREATE_ELF_Create

#endif
