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
#include "X86/X86TargetHandler.h"
#include "X86_64/X86_64TargetHandler.h"
#include "Hexagon/HexagonTargetHandler.h"

#include "lld/Core/Reference.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"

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

namespace lld {
namespace {

struct DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb, 
                                                            bool useUndefines) {
    return lld::elf::DynamicFile<ELFT>::create(std::move(mb), useUndefines);
  }
};

struct ELFFileCreateELFTraits {
  typedef std::unique_ptr<lld::File> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings,
                            TargetHandlerBase *handler,
                            lld::error_code &ec) {
    return std::unique_ptr<lld::File>(
        new lld::elf::ELFFile<ELFT>(std::move(mb), atomizeStrings, handler,ec));
  }
};

class ELFObjectReader : public Reader {
public:
  ELFObjectReader(bool atomizeStrings, TargetHandlerBase* handler) 
    : _atomizeStrings(atomizeStrings), _handler(handler) { }

  virtual bool canParse(file_magic magic, StringRef,const MemoryBuffer&) const {
    return (magic == llvm::sys::fs::file_magic::elf_relocatable);
  }
  
  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const {
    error_code ec;
    std::size_t maxAlignment =
              1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    std::unique_ptr<File> f(createELF<ELFFileCreateELFTraits>(
                       getElfArchType(&*mb), maxAlignment, std::move(mb), 
                       _atomizeStrings, _handler, ec));
    if (ec)
      return ec;
    result.push_back(std::move(f));
    return error_code::success();
  }
private:
  bool               _atomizeStrings;
  TargetHandlerBase *_handler;
};


class ELFDSOReader : public Reader {
public:
  ELFDSOReader(bool useUndefines) : _useUndefines(useUndefines) { }

  virtual bool canParse(file_magic magic, StringRef, const MemoryBuffer&) const{
    return (magic == llvm::sys::fs::file_magic::elf_shared_object);
  }
  
  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const {
    std::size_t maxAlignment =
              1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<DynamicFileCreateELFTraits>( 
                                            getElfArchType(&*mb), maxAlignment, 
                                            std::move(mb), _useUndefines);
    if (!f)
      return f;   
    result.push_back(std::move(*f));
    return error_code::success();
  }
private:
  bool _useUndefines;
};

} // anonymous


// This dynamic registration of a handler causes support for all ELF 
// architectures to be pulled into the linker.  If we want to support making a 
// linker that only supports one ELF architecture, we'd need to change this
// to have a different registration method for each architecture.
void Registry::addSupportELFObjects(bool atomizeStrings, 
                                                 TargetHandlerBase *handler) {

  // Tell registry about the ELF object file parser.
  add(std::unique_ptr<Reader>(new ELFObjectReader(atomizeStrings, handler)));
  
  // Tell registry about the relocation name to number mapping for this arch.
  handler->registerRelocationNames(*this);
}

void Registry::addSupportELFDynamicSharedObjects(bool useShlibUndefines) {
  add(std::unique_ptr<Reader>(new ELFDSOReader(useShlibUndefines)));
}


} // end namespace lld
