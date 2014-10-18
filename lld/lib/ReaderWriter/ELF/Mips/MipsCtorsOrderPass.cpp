//===- lib/ReaderWriter/ELF/Mips/Mips/CtorsOrderPass.cpp ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsCtorsOrderPass.h"
#include <algorithm>
#include <climits>

using namespace lld;
using namespace lld::elf;

static bool matchCrtObjName(StringRef objName, StringRef objPath) {
  if (!objPath.endswith(".o"))
    return false;

  // check *<objName> case
  objPath = objPath.drop_back(2);
  if (objPath.endswith(objName))
    return true;

  // check *<objName>? case
  return !objPath.empty() && objPath.drop_back(1).endswith(objName);
}

static int32_t getSectionPriority(StringRef path, StringRef sectionName) {
  // Arrange .ctors/.dtors sections in the following order:
  //   .ctors from crtbegin.o or crtbegin?.o
  //   .ctors from regular object files
  //   .ctors.* (sorted) from regular object files
  //   .ctors from crtend.o or crtend?.o

  if (matchCrtObjName("crtbegin", path))
    return std::numeric_limits<int32_t>::min();
  if (matchCrtObjName("crtend", path))
    return std::numeric_limits<int32_t>::max();

  StringRef num = sectionName.drop_front().rsplit('.').second;

  int32_t priority = std::numeric_limits<int32_t>::min() + 1;
  if (!num.empty())
    num.getAsInteger(10, priority);

  return priority;
}

void MipsCtorsOrderPass::perform(std::unique_ptr<MutableFile> &f) {
  auto definedAtoms = f->definedAtoms();

  auto last = std::stable_partition(definedAtoms.begin(), definedAtoms.end(),
                                    [](const DefinedAtom *atom) {
    if (atom->sectionChoice() != DefinedAtom::sectionCustomRequired)
      return false;

    StringRef name = atom->customSectionName();
    return name.startswith(".ctors") || name.startswith(".dtors");
  });

  std::stable_sort(definedAtoms.begin(), last,
                   [](const DefinedAtom *left, const DefinedAtom *right) {
    StringRef leftSec = left->customSectionName();
    StringRef rightSec = right->customSectionName();

    int32_t leftPriority = getSectionPriority(left->file().path(), leftSec);
    int32_t rightPriority = getSectionPriority(right->file().path(), rightSec);

    return leftPriority < rightPriority;
  });
}
