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
#include <algorithm>
#include <limits>

namespace lld {
namespace elf {

/// \brief This pass sorts atoms by file and atom ordinals.
/// .{init,fini}_array.<priority> sections are handled specially.
class OrderPass : public Pass {
public:
  void perform(std::unique_ptr<MutableFile> &file) override {
    MutableFile::DefinedAtomRange defined = file->definedAtoms();
    auto last = std::partition(defined.begin(), defined.end(), isInitFini);
    parallel_sort(defined.begin(), last, compareInitFini);
    parallel_sort(last, defined.end(), DefinedAtom::compareByPosition);
  }

private:
  static bool isInitFini(const DefinedAtom *atom) {
    if (atom->sectionChoice() != DefinedAtom::sectionCustomRequired)
      return false;
    StringRef name = atom->customSectionName();
    return name.startswith(".init_array") || name.startswith(".fini_array");
  }

  static bool compareInitFini(const DefinedAtom *lhs, const DefinedAtom *rhs) {
    StringRef lhsSec = lhs->customSectionName();
    StringRef rhsSec = rhs->customSectionName();

    // Drop the front dot from the section name and get
    // an optional section's number starting after the second dot.
    StringRef lhsNum = lhsSec.drop_front().rsplit('.').second;
    StringRef rhsNum = rhsSec.drop_front().rsplit('.').second;

    // Sort {.init_array, .fini_array}[.<num>] sections
    // according to their number. Sections without optional
    // numer suffix should go last.

    uint32_t lhsPriority;
    uint32_t rhsPriority;
    if (lhsNum.getAsInteger(10, lhsPriority))
      lhsPriority = std::numeric_limits<uint32_t>::max();
    if (rhsNum.getAsInteger(10, rhsPriority))
      rhsPriority = std::numeric_limits<uint32_t>::max();

    if (lhsPriority != rhsPriority)
      return lhsPriority < rhsPriority;

    // If both atoms have the same priority, fall back to default.
    return DefinedAtom::compareByPosition(lhs, rhs);
  }
};

}
}

#endif
