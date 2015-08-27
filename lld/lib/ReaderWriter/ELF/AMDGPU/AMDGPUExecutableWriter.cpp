//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPUExecutableWriter.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUExecutableWriter.h"

using namespace lld;
using namespace lld::elf;

AMDGPUExecutableWriter::AMDGPUExecutableWriter(AMDGPULinkingContext &ctx,
                                               AMDGPUTargetLayout &layout)
    : ExecutableWriter(ctx, layout), _ctx(ctx) {}

void AMDGPUExecutableWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &Result) {
  // ExecutableWriter::createImplicitFiles() adds C runtime symbols that we
  // don't need, so we use the OutputELFWriter implementation instead.
  OutputELFWriter<ELF64LE>::createImplicitFiles(Result);
}

void AMDGPUExecutableWriter::finalizeDefaultAtomValues() {

  // ExecutableWriter::finalizeDefaultAtomValues() assumes the presence of
  // C runtime symbols.  However, since we skip the call to
  // ExecutableWriter::createImplicitFiles(), these symbols are never added
  // and ExectuableWriter::finalizeDefaultAtomValues() will crash if we call
  // it.
  OutputELFWriter<ELF64LE>::finalizeDefaultAtomValues();
}
