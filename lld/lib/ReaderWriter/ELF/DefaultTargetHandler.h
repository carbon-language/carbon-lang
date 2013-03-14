//===- lib/ReaderWriter/ELF/DefaultTargetHandler.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_DEFAULT_TARGETHANDLER_H
#define LLD_READER_WRITER_ELF_DEFAULT_TARGETHANDLER_H

#include "DefaultLayout.h"
#include "TargetHandler.h"

#include "lld/Core/LinkerOptions.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {
template <class ELFT>
class DefaultTargetHandler : public TargetHandler<ELFT> {
public:
  DefaultTargetHandler(ELFTargetInfo &targetInfo)
      : TargetHandler<ELFT>(targetInfo) {
  }

  bool doesOverrideHeader() { return false; }

  void setHeaderInfo(Header<ELFT> *Header) {
    llvm_unreachable("Target should provide implementation for function ");
  }

  /// TargetAtomHandler
  TargetAtomHandler<ELFT> &targetAtomHandler() {
    llvm_unreachable("Target should provide implementation for function ");
  }

  const TargetRelocationHandler<ELFT> &getRelocationHandler() const {
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
} // end namespace elf
} // end namespace lld

#endif
