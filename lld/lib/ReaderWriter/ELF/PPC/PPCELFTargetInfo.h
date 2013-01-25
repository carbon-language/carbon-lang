//===- lib/ReaderWriter/ELF/Hexagon/PPCTargetInfo.h -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_PPC_TARGETINFO_H
#define LLD_READER_WRITER_ELF_PPC_TARGETINFO_H

#include "lld/ReaderWriter/ELFTargetInfo.h"
#include "lld/Core/LinkerOptions.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

#include "DefaultELFTargetHandler.h"

namespace lld {
namespace elf {
class PPCELFTargetInfo LLVM_FINAL : public ELFTargetInfo {
public:
  PPCELFTargetInfo(const LinkerOptions &lo) : ELFTargetInfo(lo) {
    _targetHandler = std::unique_ptr<ELFTargetHandlerBase>(
        new DefaultELFTargetHandler<
                llvm::object::ELFType<llvm::support::big, 4, false> >(*this));
  }

  virtual bool isLittleEndian() const { return false; }

  virtual uint64_t getPageSize() const { return 0x1000; }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_PPC_TARGETINFO_H
