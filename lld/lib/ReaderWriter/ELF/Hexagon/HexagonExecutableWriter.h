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
#include "HexagonELFWriters.h"
#include "HexagonExecutableAtoms.h"
#include "HexagonLinkingContext.h"

namespace lld {
namespace elf {

template <typename ELFT> class HexagonTargetLayout;

template <class ELFT>
class HexagonExecutableWriter : public ExecutableWriter<ELFT>,
                                public HexagonELFWriter<ELFT> {
public:
  HexagonExecutableWriter(HexagonLinkingContext &context,
                          HexagonTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  virtual void finalizeDefaultAtomValues();

  virtual std::error_code setELFHeader() {
    ExecutableWriter<ELFT>::setELFHeader();
    HexagonELFWriter<ELFT>::setELFHeader(*this->_elfHeader);
    return std::error_code();
  }

private:
  void addDefaultAtoms() {
    _hexagonRuntimeFile->addAbsoluteAtom("_SDA_BASE_");
    if (this->_context.isDynamic()) {
      _hexagonRuntimeFile->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      _hexagonRuntimeFile->addAbsoluteAtom("_DYNAMIC");
    }
  }

  HexagonLinkingContext &_hexagonLinkingContext;
  HexagonTargetLayout<ELFT> &_hexagonTargetLayout;
  std::unique_ptr<HexagonRuntimeFile<ELFT>> _hexagonRuntimeFile;
};

template <class ELFT>
HexagonExecutableWriter<ELFT>::HexagonExecutableWriter(
    HexagonLinkingContext &context, HexagonTargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(context, layout),
      HexagonELFWriter<ELFT>(context, layout), _hexagonLinkingContext(context),
      _hexagonTargetLayout(layout),
      _hexagonRuntimeFile(new HexagonRuntimeFile<ELFT>(context)) {}

template <class ELFT>
bool HexagonExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  // Add the default atoms as defined for hexagon
  addDefaultAtoms();
  result.push_back(std::move(_hexagonRuntimeFile));
  return true;
}

template <class ELFT>
void HexagonExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  auto sdabaseAtomIter = _hexagonTargetLayout.findAbsoluteAtom("_SDA_BASE_");
  (*sdabaseAtomIter)->_virtualAddr =
      _hexagonTargetLayout.getSDataSection()->virtualAddr();
  HexagonELFWriter<ELFT>::finalizeHexagonRuntimeAtomValues();
}

} // namespace elf
} // namespace lld

#endif // HEXAGON_EXECUTABLE_WRITER_H
