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
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_TARGETINFO_H
