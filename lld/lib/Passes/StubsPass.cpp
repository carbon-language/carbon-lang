//===- Passes/StubsPass.cpp - Adds stubs ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//
// This linker pass is 
//
//
//
//


#include "llvm/ADT/DenseMap.h"

#include "lld/Core/Pass.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Platform/Platform.h"


namespace lld {

void StubsPass::perform() {
  // Skip this pass if output format does not need stubs.
  if ( !_platform.outputUsesStubs() )
    return;
  
  // Use map so all call sites to same shlib symbol use same stub
  llvm::DenseMap<const SharedLibraryAtom*, const Atom*> shlibToStub;
  
  // Scan all references in all atoms.
  for(auto ait=_file.definedAtomsBegin(), aend=_file.definedAtomsEnd(); 
                                                      ait != aend; ++ait) {
    const DefinedAtom* atom = *ait;
    for (auto rit=atom->referencesBegin(), rend=atom->referencesEnd(); 
                                                      rit != rend; ++rit) {
      const Reference* ref = *rit;
      const Atom* target = ref->target();
      assert(target != NULL);
      // If the target of this reference is in a shared library
      if ( const SharedLibraryAtom* shlbTarget = target->sharedLibraryAtom() ) {
        // and this is a call to that shared library symbol.
        if ( _platform.isBranch(ref) ) {
          const Atom* stub;
          // Replace the target with a reference to a stub
          auto pos = shlibToStub.find(shlbTarget);
          if ( pos == shlibToStub.end() ) {
            // This is no existing stub.  Create a new one.
            stub = _platform.makeStub(*shlbTarget, _file);
            shlibToStub[shlbTarget] = stub;
          }
          else {
            // Reuse and existing stub
            stub = pos->second;
          }
          assert(stub != NULL);
          // Switch call site in atom to refrence stub instead of shlib atom.
          (const_cast<Reference*>(ref))->setTarget(stub);
        }
      }
    }
  }
  
  // add all created stubs to file
  for (auto it=shlibToStub.begin(), end=shlibToStub.end(); it != end; ++it) {
    _file.addAtom(*it->second);
  }
  

}



}
