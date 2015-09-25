//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPUTargetHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_TARGET_HANDLER_H
#define AMDGPU_TARGET_HANDLER_H

#include "ELFFile.h"
#include "ELFReader.h"
#include "AMDGPURelocationHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
class AMDGPULinkingContext;

class HSATextSection : public AtomSection<ELF64LE> {
public:
  HSATextSection(const ELFLinkingContext &ctx);
};

/// \brief TargetLayout for AMDGPU
class AMDGPUTargetLayout final : public TargetLayout<ELF64LE> {
public:
  AMDGPUTargetLayout(AMDGPULinkingContext &ctx) : TargetLayout(ctx) {}

  void assignSectionsToSegments() override;

  /// \brief Gets or creates a section.
  AtomSection<ELF64LE> *
  createSection(StringRef name, int32_t contentType,
                DefinedAtom::ContentPermissions contentPermissions,
                TargetLayout::SectionOrder sectionOrder) override {
    if (name == ".hsatext")
      return new (_allocator) HSATextSection(_ctx);

    if (name == ".note")
      contentType = DefinedAtom::typeRONote;

    return TargetLayout::createSection(name, contentType, contentPermissions,
                                       sectionOrder);
  }
};

/// \brief TargetHandler for AMDGPU
class AMDGPUTargetHandler final : public TargetHandler {
public:
  AMDGPUTargetHandler(AMDGPULinkingContext &targetInfo);

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
  AMDGPULinkingContext &_ctx;
  std::unique_ptr<AMDGPUTargetLayout> _targetLayout;
  std::unique_ptr<AMDGPUTargetRelocationHandler> _relocationHandler;
};

void finalizeAMDGPURuntimeAtomValues(AMDGPUTargetLayout &layout);

} // end namespace elf
} // end namespace lld

#endif
