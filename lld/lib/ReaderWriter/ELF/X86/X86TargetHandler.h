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
#include "X86ELFFile.h"
#include "X86ELFReader.h"
#include "X86RelocationHandler.h"

namespace lld {
namespace elf {

class X86LinkingContext;

class X86TargetHandler final : public TargetHandler {
public:
  X86TargetHandler(X86LinkingContext &ctx);

  void registerRelocationNames(Registry &registry) override;

  const X86TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<X86ELFObjectReader>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<X86ELFDSOReader>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

protected:
  X86LinkingContext &_ctx;
  std::unique_ptr<TargetLayout<X86ELFType>> _targetLayout;
  std::unique_ptr<X86TargetRelocationHandler> _relocationHandler;
};
} // end namespace elf
} // end namespace lld

#endif
