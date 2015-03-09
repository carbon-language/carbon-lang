//===- lld/ReaderWriter/ELF/Mips/MipsRelocationHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_RELOCATION_HANDLER_H

#include "TargetHandler.h"
#include "lld/Core/Reference.h"

namespace lld {
namespace elf {

class MipsLinkingContext;
template <class ELFT> class MipsTargetLayout;

template <class ELFT>
class MipsRelocationHandler : public TargetRelocationHandler {
public:
  MipsRelocationHandler(MipsLinkingContext &ctx) : _ctx(ctx) {}

  std::error_code applyRelocation(ELFWriter &writer,
                                  llvm::FileOutputBuffer &buf,
                                  const lld::AtomLayout &atom,
                                  const Reference &ref) const override;

  static Reference::Addend readAddend(Reference::KindValue kind,
                                      const uint8_t *content);

private:
  MipsLinkingContext &_ctx;
};

template <class ELFT>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler(MipsLinkingContext &ctx);

} // elf
} // lld

#endif
