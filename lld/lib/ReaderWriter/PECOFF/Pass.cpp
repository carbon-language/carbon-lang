//===- lib/ReaderWriter/PECOFF/Pass.cpp -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "Pass.h"

#include "lld/Core/File.h"
#include "llvm/Support/COFF.h"

namespace lld {
namespace pecoff {

static void addReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                     size_t offsetInAtom, Reference::KindValue relType) {
  std::unique_ptr<COFFReference> ref(
      new COFFReference(target, offsetInAtom, relType));
  atom->addReference(std::move(ref));
}

void addDir32Reloc(COFFBaseDefinedAtom *atom, const Atom *target,
                   size_t offsetInAtom) {
  addReloc(atom, target, offsetInAtom, llvm::COFF::IMAGE_REL_I386_DIR32);
}

void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                     size_t offsetInAtom) {
  addReloc(atom, target, offsetInAtom, llvm::COFF::IMAGE_REL_I386_DIR32NB);
}

} // end namespace pecoff
} // end namespace lld
