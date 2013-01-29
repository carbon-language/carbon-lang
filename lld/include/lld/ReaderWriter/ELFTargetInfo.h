//===- lld/ReaderWriter/ELFTargetInfo.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_INFO_H
#define LLD_READER_WRITER_ELF_TARGET_INFO_H

#include "lld/Core/LinkerOptions.h"
#include "lld/Core/TargetInfo.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

#include <memory>

namespace lld {
class DefinedAtom;
class Reference;

namespace elf { template <typename ELFT> class ELFTargetHandler; }

class ELFTargetHandlerBase {
public:
  virtual ~ELFTargetHandlerBase() {}
};

class ELFTargetInfo : public TargetInfo {
protected:
  ELFTargetInfo(const LinkerOptions &lo) : TargetInfo(lo) {}

public:
  uint16_t getOutputType() const;
  uint16_t getOutputMachine() const;

  virtual StringRef getEntry() const;
  virtual uint64_t getBaseAddress() const { return _options._baseAddress; }

  virtual bool isRuntimeRelocation(const DefinedAtom &,
                                   const Reference &) const {
    return false;
  }

  static std::unique_ptr<ELFTargetInfo> create(const LinkerOptions &lo);

  template <typename ELFT>
  lld::elf::ELFTargetHandler<ELFT> &getTargetHandler() const {
    return static_cast<
        lld::elf::ELFTargetHandler<ELFT> &>(*_targetHandler.get());
  }

protected:
  std::unique_ptr<ELFTargetHandlerBase> _targetHandler;
};
} // end namespace lld

#endif
