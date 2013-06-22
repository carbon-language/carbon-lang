//===- lib/ReaderWriter/PECOFF/Atoms.cpp ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"

namespace lld {
namespace coff {

namespace {
void addEdge(COFFDefinedAtom *a, COFFDefinedAtom *b,
             lld::Reference::Kind kind) {
  auto ref = new COFFReference(kind);
  ref->setTarget(b);
  a->addReference(std::unique_ptr<COFFReference>(ref));
}

void connectWithLayoutEdge(COFFDefinedAtom *a, COFFDefinedAtom *b) {
  addEdge(a, b, lld::Reference::kindLayoutAfter);
  addEdge(b, a, lld::Reference::kindLayoutBefore);
}
} // anonymous namespace

/// Connect atoms with layout-{before,after} edges. It usually serves two
/// purposes.
///
///   - To prevent atoms from being GC'ed (aka dead-stripped) if there is a
///     reference to one of the atoms. In that case we want to emit all the
///     atoms appeared in the same section, because the referenced "live" atom
///     may reference other atoms in the same section. If we don't add layout
///     edges between atoms, unreferenced atoms in the same section would be
///     GC'ed.
///   - To preserve the order of atmos. We want to emit the atoms in the
///     same order as they appeared in the input object file.
void connectAtomsWithLayoutEdge(std::vector<COFFDefinedAtom *> atoms) {
  if (atoms.size() < 2)
    return;
  for (auto it = atoms.begin(), e = atoms.end(); it + 1 != e; ++it)
    connectWithLayoutEdge(*it, *(it + 1));
}

} // namespace coff
} // namespace lld
