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

#include "AArch64RelocationHandler.h"
#include "ELFReader.h"
#include "TargetLayout.h"
#include "lld/Core/Simple.h"

namespace lld {
namespace elf {
class AArch64LinkingContext;

class AArch64TargetLayout final : public TargetLayout<ELF64LE> {
public:
  AArch64TargetLayout(ELFLinkingContext &ctx) : TargetLayout(ctx) {}

  uint64_t getAlignedTLSSize() const {
    return llvm::RoundUpToAlignment(TCB_ALIGNMENT, this->getTLSSize());
  }

private:
  // Alignment requirement for TCB.
  enum { TCB_ALIGNMENT = 0x10 };
};

class AArch64TargetHandler final : public TargetHandler {
public:
  AArch64TargetHandler(AArch64LinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ELFReader<ELFFile<ELF64LE>>>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFReader<DynamicFile<ELF64LE>>>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  AArch64LinkingContext &_ctx;
  std::unique_ptr<AArch64TargetLayout> _targetLayout;
  std::unique_ptr<AArch64TargetRelocationHandler> _relocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
