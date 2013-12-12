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

void addDir32NBReloc(coff::COFFBaseDefinedAtom *atom, const Atom *target,
                     size_t offsetInAtom) {
  std::unique_ptr<coff::COFFReference> ref(new coff::COFFReference(
      target, offsetInAtom, llvm::COFF::IMAGE_REL_I386_DIR32NB));
  atom->addReference(std::move(ref));
}

} // end namespace pecoff
} // end namespace lld

