//===- Passes/StubsPass.cpp - Adds stubs ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This linker pass updates call sites which have references to shared library
// atoms to instead have a reference to a stub (PLT entry) for the specified
// symbol.  The platform object does the work of creating the platform-specific
// StubAtom.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Platform.h"
#include "lld/Core/Reference.h"

#include "llvm/ADT/DenseMap.h"

namespace lld {

void StubsPass::perform() {
  // Skip this pass if output format uses text relocations instead of stubs.
  if ( !_platform.noTextRelocs() )
    return;

  // Use map so all call sites to same shlib symbol use same stub.
  llvm::DenseMap<const Atom*, const DefinedAtom*> targetToStub;

  // Scan all references in all atoms.
  for(auto ait=_file.definedAtomsBegin(), aend=_file.definedAtomsEnd();
                                                      ait != aend; ++ait) {
    const DefinedAtom* atom = *ait;
    for (auto rit=atom->referencesBegin(), rend=atom->referencesEnd();
                                                      rit != rend; ++rit) {
      const Reference* ref = *rit;
      // Look at call-sites.
      if ( _platform.isCallSite(ref->kind()) ) {
        const Atom* target = ref->target();
        assert(target != nullptr);
        bool replaceCalleeWithStub = false;
        if ( target->definition() == Atom::definitionSharedLibrary ) {
          // Calls to shared libraries go through stubs.
          replaceCalleeWithStub = true;
        } else if (const DefinedAtom* defTarget =
                     dyn_cast<DefinedAtom>(target)) {
          if ( defTarget->interposable() != DefinedAtom::interposeNo ) {
            // Calls to interposable functions in same linkage unit
            // must also go through a stub.
            assert(defTarget->scope() != DefinedAtom::scopeTranslationUnit);
            replaceCalleeWithStub = true;
          }
        }
        if ( replaceCalleeWithStub ) {
          // Replace the reference's target with a stub.
          const DefinedAtom* stub;
          auto pos = targetToStub.find(target);
          if ( pos == targetToStub.end() ) {
            // This is no existing stub.  Create a new one.
            stub = _platform.makeStub(*target, _file);
            assert(stub != nullptr);
            assert(stub->contentType() == DefinedAtom::typeStub);
            targetToStub[target] = stub;
          }
          else {
            // Reuse an existing stub.
            stub = pos->second;
            assert(stub != nullptr);
          }
          // Switch call site to reference stub atom.
          (const_cast<Reference*>(ref))->setTarget(stub);
        }
      }
    }
  }

  // add all created stubs to file
  for (auto it=targetToStub.begin(), end=targetToStub.end(); it != end; ++it) {
    _file.addAtom(*it->second);
  }


}



}
