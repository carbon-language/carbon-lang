//===- lib/ReaderWriter/ELF/TargetHandler.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief These interfaces provide target specific hooks to change the linker's
/// behaivor.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_TARGET_HANDLER_H

#include "Layout.h"
#include "lld/Core/Atom.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/STDExtras.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>
#include <vector>

namespace lld {
namespace elf {
template <class ELFT> class DynamicTable;
template <class ELFT> class DynamicSymbolTable;
template <class ELFT> class ELFDefinedAtom;
template <class ELFT> class ELFReference;
class ELFWriter;
template <class ELFT> class ELFHeader;
template <class ELFT> class Section;
template <class ELFT> class TargetLayout;

inline std::error_code make_unhandled_reloc_error() {
  return make_dynamic_error_code(Twine("Unhandled reference type"));
}

inline std::error_code make_out_of_range_reloc_error() {
  return make_dynamic_error_code(Twine("Relocation out of range"));
}

} // end namespace elf
} // end namespace lld

#endif
