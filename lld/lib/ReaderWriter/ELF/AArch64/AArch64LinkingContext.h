//===- lib/ReaderWriter/ELF/AArch64/AArch64LinkingContext.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_LINKING_CONTEXT_H
#define LLD_READER_WRITER_ELF_AARCH64_LINKING_CONTEXT_H

#include "AArch64TargetHandler.h"

#include "lld/ReaderWriter/ELFLinkingContext.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

enum {
  /// \brief The offset to add operation for a R_AARCH64_ADR_GOT_PAGE
  ADD_AARCH64_GOTRELINDEX = 0xE000,
};

class AArch64LinkingContext final : public ELFLinkingContext {
public:
  AArch64LinkingContext(llvm::Triple triple)
      : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                                      new AArch64TargetHandler(*this))) {}

  void addPasses(PassManager &) override;

  uint64_t getBaseAddress() const override {
    if (_baseAddress == 0)
      return 0x400000;
    return _baseAddress;
  }

  bool isDynamicRelocation(const DefinedAtom &,
                           const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    switch (r.kindValue()) {
    case llvm::ELF::R_AARCH64_COPY:
    case llvm::ELF::R_AARCH64_GLOB_DAT:
    case llvm::ELF::R_AARCH64_RELATIVE:
    case llvm::ELF::R_AARCH64_TLS_DTPREL64:
    case llvm::ELF::R_AARCH64_TLS_DTPMOD64:
    case llvm::ELF::R_AARCH64_TLS_TPREL64:
    case llvm::ELF::R_AARCH64_TLSDESC:
    case llvm::ELF::R_AARCH64_IRELATIVE:
      return true;
    default:
      return false;
    }
  }

  bool isPLTRelocation(const DefinedAtom &,
                               const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    switch (r.kindValue()) {
    case llvm::ELF::R_AARCH64_JUMP_SLOT:
    case llvm::ELF::R_AARCH64_IRELATIVE:
      return true;
    default:
      return false;
    }
  }

  bool isRelativeReloc(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::AArch64);
    switch (r.kindValue()) {
    case llvm::ELF::R_AARCH64_IRELATIVE:
    case llvm::ELF::R_AARCH64_RELATIVE:
      return true;
    default:
      return false;
    }
  }

  /// \brief Create Internal files for Init/Fini
  void createInternalFiles(std::vector<std::unique_ptr<File>> &) const override;
};
} // end namespace elf
} // end namespace lld

#endif
