//===- lib/ReaderWriter/ELF/AArch64/AArch64TargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_AARCH64_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_AARCH64_AARCH64_TARGET_HANDLER_H

#include "AArch64ELFReader.h"
#include "AArch64RelocationHandler.h"
#include "TargetLayout.h"
#include "lld/Core/Simple.h"

namespace lld {
namespace elf {
class AArch64LinkingContext;

class AArch64TargetHandler final : public TargetHandler {
public:
  AArch64TargetHandler(AArch64LinkingContext &ctx);

  const AArch64TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<AArch64ELFObjectReader>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<AArch64ELFDSOReader>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  AArch64LinkingContext &_ctx;
  std::unique_ptr<TargetLayout<AArch64ELFType>> _targetLayout;
  std::unique_ptr<AArch64TargetRelocationHandler> _relocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
