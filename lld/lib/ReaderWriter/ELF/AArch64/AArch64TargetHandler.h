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

#include "AArch64ELFFile.h"
#include "AArch64ELFReader.h"
#include "AArch64RelocationHandler.h"
#include "DefaultTargetHandler.h"
#include "TargetLayout.h"
#include "lld/Core/Simple.h"

namespace lld {
namespace elf {
class AArch64LinkingContext;

template <class ELFT> class AArch64TargetLayout : public TargetLayout<ELFT> {
public:
  AArch64TargetLayout(AArch64LinkingContext &ctx) : TargetLayout<ELFT>(ctx) {}
};

class AArch64TargetHandler final : public DefaultTargetHandler<AArch64ELFType> {
public:
  AArch64TargetHandler(AArch64LinkingContext &ctx);

  AArch64TargetLayout<AArch64ELFType> &getTargetLayout() override {
    return *_aarch64TargetLayout;
  }

  void registerRelocationNames(Registry &registry) override;

  const AArch64TargetRelocationHandler &getRelocationHandler() const override {
    return *_aarch64RelocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return std::unique_ptr<Reader>(new AArch64ELFObjectReader(_ctx));
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return std::unique_ptr<Reader>(new AArch64ELFDSOReader(_ctx));
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  static const Registry::KindStrings kindStrings[];
  AArch64LinkingContext &_ctx;
  std::unique_ptr<AArch64TargetLayout<AArch64ELFType>> _aarch64TargetLayout;
  std::unique_ptr<AArch64TargetRelocationHandler> _aarch64RelocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
