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

namespace lld {
namespace elf {
void ArrayOrderPass::perform(MutableFile &f) {
  auto definedAtoms = f.definedAtoms();
  std::stable_sort(definedAtoms.begin(), definedAtoms.end(),
                   [](const DefinedAtom *left, const DefinedAtom *right) {
    if (left->sectionChoice() != DefinedAtom::sectionCustomRequired ||
        right->sectionChoice() != DefinedAtom::sectionCustomRequired)
      return false;

    StringRef leftSec = left->customSectionName();
    StringRef rightSec = right->customSectionName();

    // Both sections start with the same array type.
    if (!(leftSec.startswith(".init_array") &&
          rightSec.startswith(".init_array")) &&
        !(leftSec.startswith(".fini_array") &&
          rightSec.startswith(".fini_array")))
      return false;

    // Get priortiy
    uint16_t leftPriority = 0;
    leftSec.rsplit('.').second.getAsInteger(10, leftPriority);
    uint16_t rightPriority = 0;
    rightSec.rsplit('.').second.getAsInteger(10, rightPriority);

    return leftPriority < rightPriority;
  });
}
} // end namespace elf
} // end namespace lld
