//===- Passes/StubsPass.cpp - Adds stubs ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This linker pass updates call-sites which have references to shared library
// atoms to instead have a reference to a stub (PLT entry) for the specified
// symbol.  Each file format defines a subclass of StubsPass which implements
// the abstract methods for creating the file format specific StubAtoms.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Reference.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {

void StubsPass::perform(std::unique_ptr<MutableFile> &mergedFile) {
  // Skip this pass if output format uses text relocations instead of stubs.
  if (!this->noTextRelocs())
    return;

  // Scan all references in all atoms.
  for (const DefinedAtom *atom : mergedFile->defined()) {
    for (const Reference *ref : *atom) {
      // Look at call-sites.
      if (!this->isCallSite(*ref))
        continue;
      const Atom *target = ref->target();
      assert(target != nullptr);
      if (target->definition() == Atom::definitionSharedLibrary) {
        // Calls to shared libraries go through stubs.
        replaceCalleeWithStub(target, ref);
        continue;
      }
      const DefinedAtom *defTarget = dyn_cast<DefinedAtom>(target);
      if (defTarget && defTarget->interposable() != DefinedAtom::interposeNo) {
        // Calls to interposable functions in same linkage unit must also go
        // through a stub.
        assert(defTarget->scope() != DefinedAtom::scopeTranslationUnit);
        replaceCalleeWithStub(target, ref);
      }
    }
  }
  // Add all created stubs and support Atoms.
  this->addStubAtoms(*mergedFile);
}

void StubsPass::replaceCalleeWithStub(const Atom *target,
                                      const Reference *ref) {
  // Make file-format specific stub and other support atoms.
  const DefinedAtom *stub = this->getStub(*target);
  assert(stub != nullptr);
  // Switch call site to reference stub atom instead.
  const_cast<Reference *>(ref)->setTarget(stub);
}

} // end namespace lld
