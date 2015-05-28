//===- Driver.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DRIVER_H
#define LLD_COFF_DRIVER_H

#include "Memory.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <system_error>
#include <vector>

namespace lld {
namespace coff {

class InputFile;

std::error_code parseDirectives(StringRef S,
                                std::vector<std::unique_ptr<InputFile>> *Res,
                                StringAllocator *Alloc);

} // namespace coff
} // namespace lld

#endif
