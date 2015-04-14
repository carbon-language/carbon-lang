//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_TARGET_HANDLER_H

#include "TargetLayout.h"
#include "ELFReader.h"
#include "X86RelocationHandler.h"

namespace lld {
namespace elf {

class X86LinkingContext;

class X86TargetHandler final : public TargetHandler {
public:
  X86TargetHandler(X86LinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ELFReader<ELFFile<ELF32LE>>>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFReader<DynamicFile<ELF32LE>>>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

protected:
  X86LinkingContext &_ctx;
  std::unique_ptr<TargetLayout<ELF32LE>> _targetLayout;
  std::unique_ptr<X86TargetRelocationHandler> _relocationHandler;
};
} // end namespace elf
} // end namespace lld

#endif
