//===- lib/ReaderWriter/PECOFF/GroupedSectionsPass.h ----------------------===//
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

#ifndef LLD_READER_WRITER_PE_COFF_GROUPED_SECTIONS_PASS_H
#define LLD_READER_WRITER_PE_COFF_GROUPED_SECTIONS_PASS_H

#include "Atoms.h"
#include "lld/Core/Pass.h"

#include <algorithm>

namespace lld {
namespace pecoff {

static bool compare(const DefinedAtom *left, const DefinedAtom *right) {
  if (left->sectionChoice() == DefinedAtom::sectionCustomRequired &&
      right->sectionChoice() == DefinedAtom::sectionCustomRequired) {
    return left->customSectionName().compare(right->customSectionName()) < 0;
  }
  return left->sectionChoice() == DefinedAtom::sectionCustomRequired &&
         right->sectionChoice() != DefinedAtom::sectionCustomRequired;
}

class GroupedSectionsPass : public lld::Pass {
public:
  void perform(std::unique_ptr<MutableFile> &file) override {
    auto definedAtoms = file->definedAtoms();
    std::stable_sort(definedAtoms.begin(), definedAtoms.end(), compare);
  }
};

} // namespace pecoff
} // namespace lld

#endif
