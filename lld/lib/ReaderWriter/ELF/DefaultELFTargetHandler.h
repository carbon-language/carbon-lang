//===- lib/ReaderWriter/ELF/DefaultELFTargetHandler.h ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_DEFAULT_ELF_TARGETHANDLER_H
#define LLD_READER_WRITER_DEFAULT_ELF_TARGETHANDLER_H

#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "lld/Core/LinkerOptions.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/ELF.h"

#include "DefaultELFLayout.h"
#include "ELFTargetHandler.h"

namespace lld {
namespace elf {

template <class ELFT>
class DefaultELFTargetHandler : public ELFTargetHandler<ELFT> {

public:
  DefaultELFTargetHandler(ELFTargetInfo &targetInfo)
      : ELFTargetHandler<ELFT>(targetInfo) {
  }

  bool doesOverrideELFHeader() { return false; }

  void setELFHeaderInfo(ELFHeader<ELFT> *elfHeader) {
    llvm_unreachable("Target should provide implementation for function ");
  }

  /// ELFTargetLayout 
  ELFTargetLayout<ELFT> &targetLayout() {
    llvm_unreachable("Target should provide implementation for function ");
  }

  /// ELFTargetAtomHandler
  ELFTargetAtomHandler<ELFT> &targetAtomHandler() {
    llvm_unreachable("Target should provide implementation for function ");
  }

  /// Create a set of Default target sections that a target might needj
  void createDefaultSections() {}

  /// \brief Add a section to the current Layout
  void addSection(Section<ELFT> *section) {}

  /// \brief add new symbol file 
  void addFiles(InputFiles &) {}

  /// \brief Finalize the symbol values
  void finalizeSymbolValues() {}

  /// \brief allocate Commons, some architectures may move small common
  /// symbols over to small data, this would also be used 
  void allocateCommons() {}
};

} // elf
} // lld

#endif
