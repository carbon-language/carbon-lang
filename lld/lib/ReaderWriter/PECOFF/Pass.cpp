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
                     size_t offsetInAtom, Reference::KindArch arch,
                     Reference::KindValue relType) {
  auto *ref = new COFFReference(target, offsetInAtom, relType,
                                Reference::KindNamespace::COFF, arch);
  atom->addReference(std::unique_ptr<COFFReference>(ref));
}

void addDir32Reloc(COFFBaseDefinedAtom *atom, const Atom *target, bool is64,
                   size_t offsetInAtom) {
  if (is64) {
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86_64,
             llvm::COFF::IMAGE_REL_AMD64_ADDR32);
  } else {
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86,
             llvm::COFF::IMAGE_REL_I386_DIR32);
  }
}

void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target, bool is64,
                     size_t offsetInAtom) {
  if (is64) {
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86_64,
             llvm::COFF::IMAGE_REL_AMD64_ADDR32NB);
  } else {
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86,
             llvm::COFF::IMAGE_REL_I386_DIR32NB);
  }
}

} // end namespace pecoff
} // end namespace lld
