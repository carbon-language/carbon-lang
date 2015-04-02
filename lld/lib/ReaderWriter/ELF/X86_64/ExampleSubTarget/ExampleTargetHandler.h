//===- lib/ReaderWriter/ELF/X86_64/ExampleTarget/ExampleTargetHandler.h ---===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_EXAMPLE_TARGET_EXAMPLE_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_64_EXAMPLE_TARGET_EXAMPLE_TARGET_HANDLER_H

#include "X86_64TargetHandler.h"

namespace lld {
namespace elf {
class ExampleLinkingContext;

class ExampleTargetHandler final : public X86_64TargetHandler {
public:
  ExampleTargetHandler(ExampleLinkingContext &c);

  std::unique_ptr<Writer> getWriter() override;

private:
  ExampleLinkingContext &_ctx;
};
} // end namespace elf
} // end namespace lld

#endif
