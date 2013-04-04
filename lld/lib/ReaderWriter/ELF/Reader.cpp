//===- lib/ReaderWriter/ELF/Reader.cpp ------------------------------------===//
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

#include "Atoms.h"
#include "CreateELF.h"
#include "DynamicFile.h"
#include "File.h"

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
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include <map>
#include <vector>

using llvm::support::endianness;
using namespace llvm::object;

namespace {
struct DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(const lld::ELFTargetInfo &ti,
                            std::unique_ptr<llvm::MemoryBuffer> mb) {
    return lld::elf::DynamicFile<ELFT>::create(ti, std::move(mb));
  }
};

struct ELFFileCreateELFTraits {
  typedef std::unique_ptr<lld::File> result_type;

  template <class ELFT>
  static result_type create(const lld::ELFTargetInfo &ti,
                            std::unique_ptr<llvm::MemoryBuffer> mb,
                            lld::error_code &ec) {
    return std::unique_ptr<lld::File>(
        new lld::elf::ELFFile<ELFT>(ti, std::move(mb), ec));
  }
};
}

namespace lld {
namespace elf {
/// \brief A reader object that will instantiate correct File by examining the
/// memory buffer for ELF class and bit width
class ELFReader : public Reader {
public:
  ELFReader(const ELFTargetInfo &ti)
      : lld::Reader(ti), _elfTargetInfo(ti), _readerArchive(ti, *this) {
  }

  error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                       std::vector<std::unique_ptr<File> > &result) const {
    using llvm::object::ELFType;
    llvm::sys::LLVMFileType fileType =
        llvm::sys::IdentifyFileType(mb->getBufferStart(),
                                    static_cast<unsigned>(mb->getBufferSize()));

    std::size_t MaxAlignment =
        1ULL << llvm::CountTrailingZeros_64(uintptr_t(mb->getBufferStart()));

    llvm::error_code ec;
    switch (fileType) {
    case llvm::sys::ELF_Relocatable_FileType: {
      std::unique_ptr<File> f(createELF<ELFFileCreateELFTraits>(
          getElfArchType(&*mb), MaxAlignment, _elfTargetInfo, std::move(mb),
          ec));
      if (ec)
        return ec;
      result.push_back(std::move(f));
      break;
    }
    case llvm::sys::ELF_SharedObject_FileType: {
      auto f = createELF<DynamicFileCreateELFTraits>(
          getElfArchType(&*mb), MaxAlignment, _elfTargetInfo, std::move(mb));
      if (!f)
        return f;
      result.push_back(std::move(*f));
      break;
    }
    case llvm::sys::Archive_FileType:
      ec = _readerArchive.parseFile(mb, result);
      break;
    default:
      return llvm::make_error_code(llvm::errc::executable_format_error);
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
} // end namespace elf

std::unique_ptr<Reader> createReaderELF(const ELFTargetInfo &targetinfo) {
  return std::unique_ptr<Reader>(new elf::ELFReader(targetinfo));
}
} // end namespace lld
