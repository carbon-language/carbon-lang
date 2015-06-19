//===- lib/ReaderWriter/ELF/OrderPass.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ORDER_PASS_H
#define LLD_READER_WRITER_ELF_ORDER_PASS_H

#include "lld/Core/Parallel.h"
#include <limits>

namespace lld {
namespace elf {

/// \brief This pass sorts atoms by file and atom ordinals.
class OrderPass : public Pass {
public:
  std::error_code perform(SimpleFile &file) override {
    parallel_sort(file.definedAtoms().begin(), file.definedAtoms().end(),
                  DefinedAtom::compareByPosition);
    return std::error_code();
  }
};
}
}

#endif
