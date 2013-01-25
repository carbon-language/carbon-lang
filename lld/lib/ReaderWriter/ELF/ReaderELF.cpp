//===- lib/ReaderWriter/ELF/ReaderELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the ELF Reader and all helper sub classes to consume an ELF
/// file and produces atoms out of it.
///
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Reader.h"

#include "lld/Core/Reference.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"
#include "lld/ReaderWriter/ReaderArchive.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "AtomsELF.h"
#include "FileELF.h"

#include <map>
#include <vector>

using namespace lld;
using llvm::support::endianness;
using namespace llvm::object;

namespace {
/// \brief A reader object that will instantiate correct FileELF by examining the
/// memory buffer for ELF class and bit width
class ReaderELF : public Reader {
public:
  ReaderELF(const ELFTargetInfo &ti, std::function<ReaderFunc> read)
      : Reader(ti), _elfTargetInfo(ti), _readerArchive(ti, read) {
  }

  error_code parseFile(std::unique_ptr<MemoryBuffer> mb,
                       std::vector<std::unique_ptr<File> > &result) {
    using llvm::object::ELFType;
    llvm::sys::LLVMFileType fileType =
        llvm::sys::IdentifyFileType(mb->getBufferStart(),
                                    static_cast<unsigned>(mb->getBufferSize()));

    std::size_t MaxAlignment =
        1ULL << llvm::CountTrailingZeros_64(uintptr_t(mb->getBufferStart()));

    llvm::error_code ec;
    switch (fileType) {
    case llvm::sys::ELF_Relocatable_FileType: {
      std::pair<unsigned char, unsigned char> Ident = getElfArchType(&*mb);
      std::unique_ptr<File> f;
      // Instantiate the correct FileELF template instance based on the Ident
      // pair. Once the File is created we push the file to the vector of files
      // already created during parser's life.
      if (Ident.first == llvm::ELF::ELFCLASS32 &&
          Ident.second == llvm::ELF::ELFDATA2LSB) {
        if (MaxAlignment >= 4)
          f.reset(new FileELF<ELFType<llvm::support::little, 4, false> >(
                          _elfTargetInfo, std::move(mb), ec));
        else if (MaxAlignment >= 2)
          f.reset(new FileELF<ELFType<llvm::support::little, 2, false> >(
                          _elfTargetInfo, std::move(mb), ec));
        else
          llvm_unreachable("Invalid alignment for ELF file!");
      } else if (Ident.first == llvm::ELF::ELFCLASS32 &&
                 Ident.second == llvm::ELF::ELFDATA2MSB) {
        if (MaxAlignment >= 4)
          f.reset(new FileELF<ELFType<llvm::support::big, 4, false> >(
                          _elfTargetInfo, std::move(mb), ec));
        else if (MaxAlignment >= 2)
          f.reset(new FileELF<ELFType<llvm::support::big, 2, false> >(
                          _elfTargetInfo, std::move(mb), ec));
        else
          llvm_unreachable("Invalid alignment for ELF file!");
      } else if (Ident.first == llvm::ELF::ELFCLASS64 &&
                 Ident.second == llvm::ELF::ELFDATA2MSB) {
        if (MaxAlignment >= 8)
          f.reset(new FileELF<ELFType<llvm::support::big, 8, true> >(
                          _elfTargetInfo, std::move(mb), ec));
        else if (MaxAlignment >= 2)
          f.reset(new FileELF<ELFType<llvm::support::big, 2, true> >(
                          _elfTargetInfo, std::move(mb), ec));
        else
          llvm_unreachable("Invalid alignment for ELF file!");
      } else if (Ident.first == llvm::ELF::ELFCLASS64 &&
                 Ident.second == llvm::ELF::ELFDATA2LSB) {
        if (MaxAlignment >= 8)
          f.reset(new FileELF<ELFType<llvm::support::little, 8, true> >(
                          _elfTargetInfo, std::move(mb), ec));
        else if (MaxAlignment >= 2)
          f.reset(new FileELF<ELFType<llvm::support::little, 2, true> >(
                          _elfTargetInfo, std::move(mb), ec));
        else
          llvm_unreachable("Invalid alignment for ELF file!");
      }
      if (!ec)
        result.push_back(std::move(f));
      break;
    }
    case llvm::sys::Archive_FileType:
      ec = _readerArchive.parseFile(std::move(mb), result);
      break;
    default:
      llvm_unreachable("not supported format");
      break;
    }

    if (ec)
      return ec;

    return error_code::success();
  }

private:
  const ELFTargetInfo &_elfTargetInfo;
  ReaderArchive _readerArchive;
};
} // end anon namespace.

namespace lld {
std::unique_ptr<Reader> createReaderELF(const ELFTargetInfo &eti,
                                        std::function<ReaderFunc> read) {
  return std::unique_ptr<Reader>(new ReaderELF(eti, std::move(read)));
}
} // end namespace lld
