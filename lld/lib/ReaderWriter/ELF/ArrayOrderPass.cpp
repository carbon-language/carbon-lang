//===- lib/ReaderWriter/ELF/ArrayOrderPass.cpp ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ArrayOrderPass.h"
#include <algorithm>
#include <limits>

namespace lld {
namespace elf {
void ArrayOrderPass::perform(std::unique_ptr<MutableFile> &f) {
  auto definedAtoms = f->definedAtoms();

  // Move sections need to be sorted into the separate continious group.
  // That reduces a number of sorting elements and simplifies conditions
  // in the sorting predicate.
  auto last = std::stable_partition(definedAtoms.begin(), definedAtoms.end(),
                                    [](const DefinedAtom *atom) {
    if (atom->sectionChoice() != DefinedAtom::sectionCustomRequired)
      return false;

    StringRef name = atom->customSectionName();
    return name.startswith(".init_array") || name.startswith(".fini_array");
  });

  std::stable_sort(definedAtoms.begin(), last,
                   [](const DefinedAtom *left, const DefinedAtom *right) {
    StringRef leftSec = left->customSectionName();
    StringRef rightSec = right->customSectionName();

    // Drop the front dot from the section name and get
    // an optional section's number starting after the second dot.
    StringRef leftNum = leftSec.drop_front().rsplit('.').second;
    StringRef rightNum = rightSec.drop_front().rsplit('.').second;

    // Sort {.init_array, .fini_array}[.<num>] sections
    // according to their number. Sections without optional
    // numer suffix should go last.

    uint32_t leftPriority = std::numeric_limits<uint32_t>::max();
    if (!leftNum.empty())
      leftNum.getAsInteger(10, leftPriority);

    uint32_t rightPriority = std::numeric_limits<uint32_t>::max();
    if (!rightNum.empty())
      rightNum.getAsInteger(10, rightPriority);

    return leftPriority < rightPriority;
  });
}
} // end namespace elf
} // end namespace lld
