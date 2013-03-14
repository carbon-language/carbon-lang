//===- lib/ReaderWriter/ELF/TargetLayout.h --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_LAYOUT_H
#define LLD_READER_WRITER_ELF_TARGET_LAYOUT_H

#include "DefaultLayout.h"

#include "lld/Core/LLVM.h"

namespace lld {
namespace elf {
/// \brief The target can override certain functions in the DefaultLayout
/// class so that the order, the name of the section and the segment type could
/// be changed in the final layout
template <class ELFT> class TargetLayout : public DefaultLayout<ELFT> {
public:
  TargetLayout(const ELFTargetInfo &targetInfo)
    : DefaultLayout<ELFT>(targetInfo) {}
};
} // end namespace elf
} // end namespace lld

#endif
