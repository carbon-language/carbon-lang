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

#include "lld/Core/File.h"
#include "llvm/Object/ELF.h"
#include "llvm/ADT/STLExtras.h"

namespace {

/// \func createELF
/// \brief Create an object depending on the runtime attributes and alignment
/// of an ELF file.
///
/// \param Traits
/// Traits::create must be a template function which takes an ELFType and
/// returns an ErrorOr<std::unique_ptr<lld::File>>.
///
/// \param ident pair of EI_CLASS and EI_DATA.
/// \param maxAlignment the maximum alignment of the file.
/// \param args arguments forwarded to CreateELFTraits<T>::create.
template <template <typename ELFT> class FileT, class... Args>
llvm::ErrorOr<std::unique_ptr<lld::File>>
createELF(std::pair<unsigned char, unsigned char> ident,
          std::size_t maxAlignment, Args &&... args) {
  using namespace llvm::ELF;
  using namespace llvm::support;
  using llvm::object::ELFType;
  if (maxAlignment < 2)
    llvm_unreachable("Invalid alignment for ELF file!");

  lld::File *file = nullptr;
  unsigned char size = ident.first;
  unsigned char endian = ident.second;
  if (size == ELFCLASS32 && endian == ELFDATA2LSB) {
    file = new FileT<ELFType<little, 2, false>>(std::forward<Args>(args)...);
  } else if (size == ELFCLASS32 && endian == ELFDATA2MSB) {
    file = new FileT<ELFType<big, 2, false>>(std::forward<Args>(args)...);
  } else if (size == ELFCLASS64 && endian == ELFDATA2LSB) {
    file = new FileT<ELFType<little, 2, true>>(std::forward<Args>(args)...);
  } else if (size == ELFCLASS64 && endian == ELFDATA2MSB) {
    file = new FileT<ELFType<big, 2, true>>(std::forward<Args>(args)...);
  }
  if (!file)
    llvm_unreachable("Invalid ELF type!");
  return std::unique_ptr<lld::File>(file);
}

} // end anon namespace

#endif
