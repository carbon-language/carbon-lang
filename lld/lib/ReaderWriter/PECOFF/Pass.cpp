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
  atom->addReference(llvm::make_unique<COFFReference>(
      target, offsetInAtom, relType, arch));
}

void addDir64Reloc(COFFBaseDefinedAtom *atom, const Atom *target,
                   llvm::COFF::MachineTypes machine, size_t offsetInAtom) {
  switch (machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86,
             llvm::COFF::IMAGE_REL_I386_DIR32);
    return;
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86_64,
             llvm::COFF::IMAGE_REL_AMD64_ADDR64);
    return;
  default:
    llvm_unreachable("unsupported machine type");
  }
}

void addDir32Reloc(COFFBaseDefinedAtom *atom, const Atom *target,
                   llvm::COFF::MachineTypes machine, size_t offsetInAtom) {
  switch (machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86,
             llvm::COFF::IMAGE_REL_I386_DIR32);
    return;
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86_64,
             llvm::COFF::IMAGE_REL_AMD64_ADDR32);
    return;
  default:
    llvm_unreachable("unsupported machine type");
  }
}

void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                     llvm::COFF::MachineTypes machine, size_t offsetInAtom) {
  switch (machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86,
             llvm::COFF::IMAGE_REL_I386_DIR32NB);
    return;
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86_64,
             llvm::COFF::IMAGE_REL_AMD64_ADDR32NB);
    return;
  default:
    llvm_unreachable("unsupported machine type");
  }
}

void addRel32Reloc(COFFBaseDefinedAtom *atom, const Atom *target,
                   llvm::COFF::MachineTypes machine, size_t offsetInAtom) {
  switch (machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86,
             llvm::COFF::IMAGE_REL_I386_REL32);
    return;
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    addReloc(atom, target, offsetInAtom, Reference::KindArch::x86_64,
             llvm::COFF::IMAGE_REL_AMD64_REL32);
    return;
  default:
    llvm_unreachable("unsupported machine type");
  }
}

} // end namespace pecoff
} // end namespace lld
