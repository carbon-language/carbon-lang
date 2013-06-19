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
}

void connectAtomsWithLayoutEdge(COFFDefinedAtom *a, COFFDefinedAtom *b) {
  addEdge(a, b, lld::Reference::kindLayoutAfter);
  addEdge(b, a, lld::Reference::kindLayoutBefore);
}

} // namespace coff
} // namespace lld
