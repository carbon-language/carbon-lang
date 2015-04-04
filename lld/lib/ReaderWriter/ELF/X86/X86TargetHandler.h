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
  typedef llvm::object::ELFType<llvm::support::little, 2, false> ELFTy;
  typedef ELFReader<ELFTy, X86LinkingContext, ELFFile> ObjReader;
  typedef ELFReader<ELFTy, X86LinkingContext, DynamicFile> DSOReader;

public:
  X86TargetHandler(X86LinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ObjReader>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<DSOReader>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

protected:
  X86LinkingContext &_ctx;
  std::unique_ptr<TargetLayout<ELFTy>> _targetLayout;
  std::unique_ptr<X86TargetRelocationHandler> _relocationHandler;
};
} // end namespace elf
} // end namespace lld

#endif
