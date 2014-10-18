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
typedef llvm::object::ELFType<llvm::support::little, 2, true> AArch64ELFType;
class AArch64LinkingContext;

template <class ELFT> class AArch64TargetLayout : public TargetLayout<ELFT> {
public:
  AArch64TargetLayout(AArch64LinkingContext &context)
      : TargetLayout<ELFT>(context) {}
};

class AArch64TargetHandler final : public DefaultTargetHandler<AArch64ELFType> {
public:
  AArch64TargetHandler(AArch64LinkingContext &context);

  AArch64TargetLayout<AArch64ELFType> &getTargetLayout() override {
    return *(_AArch64TargetLayout.get());
  }

  void registerRelocationNames(Registry &registry) override;

  const AArch64TargetRelocationHandler &getRelocationHandler() const override {
    return *(_AArch64RelocationHandler.get());
  }

  std::unique_ptr<Reader> getObjReader(bool atomizeStrings) override {
    return std::unique_ptr<Reader>(new AArch64ELFObjectReader(atomizeStrings));
  }

  std::unique_ptr<Reader> getDSOReader(bool useShlibUndefines) override {
    return std::unique_ptr<Reader>(new AArch64ELFDSOReader(useShlibUndefines));
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  static const Registry::KindStrings kindStrings[];
  AArch64LinkingContext &_context;
  std::unique_ptr<AArch64TargetLayout<AArch64ELFType>> _AArch64TargetLayout;
  std::unique_ptr<AArch64TargetRelocationHandler> _AArch64RelocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
