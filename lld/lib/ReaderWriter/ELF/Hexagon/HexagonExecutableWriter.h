//===- lib/ReaderWriter/ELF/Hexagon/HexagonExecutableWriter.h -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_EXECUTABLE_WRITER_H
#define HEXAGON_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "HexagonLinkingContext.h"
#include "HexagonTargetHandler.h"

namespace lld {
namespace elf {

class HexagonTargetLayout;

class HexagonExecutableWriter : public ExecutableWriter<ELF32LE> {
public:
  HexagonExecutableWriter(HexagonLinkingContext &ctx,
                          HexagonTargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;

  std::error_code setELFHeader() override {
    ExecutableWriter::setELFHeader();
    setHexagonELFHeader(*_elfHeader);
    return std::error_code();
  }

private:
  HexagonLinkingContext &_ctx;
  HexagonTargetLayout &_targetLayout;
};

HexagonExecutableWriter::HexagonExecutableWriter(HexagonLinkingContext &ctx,
                                                 HexagonTargetLayout &layout)
    : ExecutableWriter(ctx, layout), _ctx(ctx), _targetLayout(layout) {}

void HexagonExecutableWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter::createImplicitFiles(result);
  // Add the default atoms as defined for hexagon
  auto file =
      llvm::make_unique<RuntimeFile<ELF32LE>>(_ctx, "Hexagon runtime file");
  file->addAbsoluteAtom("_SDA_BASE_");
  if (_ctx.isDynamic()) {
    file->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    file->addAbsoluteAtom("_DYNAMIC");
  }
  result.push_back(std::move(file));
}

void HexagonExecutableWriter::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  ExecutableWriter::finalizeDefaultAtomValues();
  AtomLayout *sdabaseAtom = _targetLayout.findAbsoluteAtom("_SDA_BASE_");
  sdabaseAtom->_virtualAddr = _targetLayout.getSDataSection()->virtualAddr();
  if (_ctx.isDynamic())
    finalizeHexagonRuntimeAtomValues(_targetLayout);
}

} // namespace elf
} // namespace lld

#endif // HEXAGON_EXECUTABLE_WRITER_H
