//===- lib/ReaderWriter/ELF/X86/X86LinkingContext.h -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_TARGETINFO_H
#define LLD_READER_WRITER_ELF_X86_TARGETINFO_H

#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {
class X86LinkingContext final : public ELFLinkingContext {
public:
  static std::unique_ptr<ELFLinkingContext> create(llvm::Triple);
  int getMachineType() const override { return llvm::ELF::EM_386; }
  X86LinkingContext(llvm::Triple);
  void registerRelocationNames(Registry &r) override;

  /// \brief X86 has only two relative relocation
  /// a) for supporting IFUNC relocs - R_386_IRELATIVE
  /// b) for supporting relative relocs - R_386_RELATIVE
  bool isRelativeReloc(const Reference &r) const override {
    if (r.kindNamespace() != Reference::KindNamespace::ELF)
      return false;
    assert(r.kindArch() == Reference::KindArch::x86);
    switch (r.kindValue()) {
    case llvm::ELF::R_386_IRELATIVE:
    case llvm::ELF::R_386_RELATIVE:
      return true;
    default:
      return false;
    }
  }
};
} // end namespace elf
} // end namespace lld
#endif
