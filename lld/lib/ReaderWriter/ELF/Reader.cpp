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

#include "ELFReader.h"

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
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::ELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class ELFObjectReader : public Reader {
public:
  ELFObjectReader(bool atomizeStrings) : _atomizeStrings(atomizeStrings) {}

  virtual bool canParse(file_magic magic, StringRef,
                        const MemoryBuffer &) const override {
    return (magic == llvm::sys::fs::file_magic::elf_relocatable);
  }

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    error_code ec;
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<ELFFileCreateELFTraits>(
        getElfArchType(&*mb), maxAlignment, std::move(mb), _atomizeStrings);
    if (error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return error_code::success();
  }

private:
  bool _atomizeStrings;
};

class ELFDSOReader : public Reader {
public:
  ELFDSOReader(bool useUndefines) : _useUndefines(useUndefines) {}

  virtual bool canParse(file_magic magic, StringRef,
                        const MemoryBuffer &) const override {
    return (magic == llvm::sys::fs::file_magic::elf_shared_object);
  }

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<DynamicFileCreateELFTraits>(
        getElfArchType(&*mb), maxAlignment, std::move(mb), _useUndefines);
    if (error_code ec = f.getError())
      return ec;
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
  add(std::move(handler->getObjReader(atomizeStrings)));

  // Tell registry about the relocation name to number mapping for this arch.
  handler->registerRelocationNames(*this);
}

void Registry::addSupportELFDynamicSharedObjects(bool useShlibUndefines,
                                                 TargetHandlerBase *handler) {
  // Tell registry about the ELF dynamic shared library file parser.
  add(handler->getDSOReader(useShlibUndefines));
}

} // end namespace lld
