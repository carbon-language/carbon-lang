//===- lib/ReaderWriter/ELF/TargetHandler.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_TARGET_HANDLER_H

#include "lld/Core/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

namespace lld {
namespace elf {

inline std::error_code make_unhandled_reloc_error() {
  return make_dynamic_error_code("Unhandled reference type");
}

inline std::error_code make_out_of_range_reloc_error() {
  return make_dynamic_error_code("Relocation out of range");
}

inline std::error_code make_unaligned_range_reloc_error() {
  return make_dynamic_error_code("Relocation not aligned");
}

} // end namespace elf
} // end namespace lld

#endif
