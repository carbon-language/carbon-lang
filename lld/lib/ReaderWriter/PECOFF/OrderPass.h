//===- lib/ReaderWriter/PECOFF/OrderPass.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file \brief This pass sorts atoms by section name, so that they will appear
/// in the correct order in the output.
///
/// In COFF, sections will be merged into one section by the linker if their
/// names are the same after discarding the "$" character and all characters
/// follow it from their names. The characters following the "$" character
/// determines the merge order. Assume there's an object file containing four
/// data sections in the following order.
///
///   - .data$2
///   - .data$3
///   - .data$1
///   - .data
///
/// In this case, the resulting binary should have ".data" section with the
/// contents of ".data", ".data$1", ".data$2" and ".data$3" in that order.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_ORDER_PASS_H
#define LLD_READER_WRITER_PE_COFF_ORDER_PASS_H

#include "Atoms.h"
#include "lld/Core/Parallel.h"
#include "lld/Core/Pass.h"
#include <algorithm>

namespace lld {
namespace pecoff {

static bool compare(const DefinedAtom *lhs, const DefinedAtom *rhs) {
  bool lhsCustom = (lhs->sectionChoice() == DefinedAtom::sectionCustomRequired);
  bool rhsCustom = (rhs->sectionChoice() == DefinedAtom::sectionCustomRequired);
  if (lhsCustom && rhsCustom) {
    int cmp = lhs->customSectionName().compare(rhs->customSectionName());
    if (cmp != 0)
      return cmp < 0;
    return DefinedAtom::compareByPosition(lhs, rhs);
  }
  if (lhsCustom && !rhsCustom)
    return true;
  if (!lhsCustom && rhsCustom)
    return false;
  return DefinedAtom::compareByPosition(lhs, rhs);
}

class OrderPass : public lld::Pass {
public:
  std::error_code perform(SimpleFile &file) override {
    SimpleFile::DefinedAtomRange defined = file.definedAtoms();
    parallel_sort(defined.begin(), defined.end(), compare);
    return std::error_code();
  }
};

} // namespace pecoff
} // namespace lld

#endif
