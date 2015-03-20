//===- lib/ReaderWriter/ELF/X86_64/ExampleTarget/ExampleLinkingContext.h --===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_EXAMPLE_TARGET_EXAMPLE_LINKING_CONTEXT
#define LLD_READER_WRITER_ELF_X86_64_EXAMPLE_TARGET_EXAMPLE_LINKING_CONTEXT

#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"

namespace lld {
namespace elf {

class ExampleLinkingContext final : public X86_64LinkingContext {
public:
  static std::unique_ptr<ELFLinkingContext> create(llvm::Triple);
  ExampleLinkingContext(llvm::Triple triple);

  StringRef entrySymbolName() const override;
  void addPasses(PassManager &) override;
};

} // end namespace elf
} // end namespace lld

#endif
