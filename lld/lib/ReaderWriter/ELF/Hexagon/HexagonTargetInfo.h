//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetInfo.h -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_TARGETINFO_H
#define LLD_READER_WRITER_ELF_HEXAGON_TARGETINFO_H

#include "HexagonTargetHandler.h"

#include "lld/Core/LinkerOptions.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

class HexagonTargetInfo LLVM_FINAL : public ELFTargetInfo {
public:
  HexagonTargetInfo(const LinkerOptions &lo) : ELFTargetInfo(lo) {
    _targetHandler = std::unique_ptr<TargetHandlerBase>(
        new HexagonTargetHandler(*this));
  }

  virtual uint64_t getPageSize() const { return 0x1000; }

  virtual ErrorOr<int32_t> relocKindFromString(StringRef str) const;
  virtual ErrorOr<std::string> stringFromRelocKind(int32_t kind) const;

  virtual void addPasses(PassManager &) const;

  virtual bool isDynamicRelocation(const DefinedAtom &,
                                   const Reference &r) const {
    switch (r.kind()){
    case llvm::ELF::R_HEX_RELATIVE:
    case llvm::ELF::R_HEX_GLOB_DAT:
      return true;
    default:
      return false;
    }
  }

  virtual bool isPLTRelocation(const DefinedAtom &,
                               const Reference &r) const {
    switch (r.kind()){
    case llvm::ELF::R_HEX_JMP_SLOT:
      return true;
    default:
      return false;
    }
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_TARGETINFO_H
