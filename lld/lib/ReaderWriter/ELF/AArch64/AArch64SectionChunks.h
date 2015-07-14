//===- lib/ReaderWriter/ELF/AArch64/AArch64SectionChunks.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_AARCH64_SECTION_CHUNKS_H
#define LLD_READER_WRITER_ELF_AARCH64_AARCH64_SECTION_CHUNKS_H

#include "TargetLayout.h"

namespace lld {
namespace elf {

class AArch64GOTSection : public AtomSection<ELF64LE> {
public:
  AArch64GOTSection(const ELFLinkingContext &ctx, StringRef name,
                    int32_t order);

  bool hasGlobalGOTEntry(const Atom *a) const {
    return _tlsMap.count(a);
  }

  const AtomLayout *appendAtom(const Atom *atom) override;

private:
  /// \brief Map TLS Atoms to their GOT entry index.
  llvm::DenseMap<const Atom *, std::size_t> _tlsMap;
};

} // elf
} // lld

#endif
