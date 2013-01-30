//===- lib/ReaderWriter/ELF/Hexagon/X86_64TargetInfo.h --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_TARGETINFO_H
#define LLD_READER_WRITER_ELF_X86_64_TARGETINFO_H

#include "X86_64TargetHandler.h"

#include "lld/Core/LinkerOptions.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

class X86_64TargetInfo LLVM_FINAL : public ELFTargetInfo {
public:
  X86_64TargetInfo(const LinkerOptions &lo) : ELFTargetInfo(lo) {
    _targetHandler =
        std::unique_ptr<TargetHandlerBase>(new X86_64TargetHandler(*this));
  }

  virtual uint64_t getPageSize() const { return 0x1000; }

  virtual void addPasses(PassManager &) const;

  virtual uint64_t getBaseAddress() const {
    if (_options._baseAddress == 0)
      return 0x400000;
    return _options._baseAddress;
  }

  virtual bool isRuntimeRelocation(const DefinedAtom &,
                                   const Reference &r) const {
    return r.kind() == llvm::ELF::R_X86_64_IRELATIVE;
  }

  virtual ErrorOr<int32_t> relocKindFromString(StringRef str) const;
  virtual ErrorOr<std::string> stringFromRelocKind(int32_t kind) const;

};
} // end namespace elf
} // end namespace lld

#endif
