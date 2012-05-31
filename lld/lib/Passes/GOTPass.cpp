//===- Passes/GOTPass.cpp - Adds GOT entries ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This linker pass transforms all GOT kind references to real references.
// That is, in assembly you can write something like:
//     movq foo@GOTPCREL(%rip), %rax 
// which means you want to load a pointer to "foo" out of the GOT (global
// Offsets Table). In the object file, the Atom containing this instruction
// has a Reference whose target is an Atom named "foo" and the Reference
// kind is a GOT load.  The linker needs to instantiate a pointer sized
// GOT entry.  This is done be creating a GOT Atom to represent that pointer
// sized data in this pass, and altering the Atom graph so the Reference now
// points to the GOT Atom entry (corresponding to "foo") and changing the
// Reference Kind to reflect it is now pointing to a GOT entry (rather
// then needing a GOT entry). 
// 
// There is one optimization the linker can do here.  If the target of the GOT
// is in the same linkage unit and does not need to be interposable, and 
// the GOT use is just a load (not some other operation), this pass can 
// transform that load into an LEA (add).  This optimizes away one memory load
// which at runtime that could stall the pipeline.  This optimization only works
// for architectures in which a (GOT) load instruction can be change to an 
// LEA instruction that is the same size.  The method isGOTAccess()
// should only return true for "canBypassGOT" if this optimization is supported.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Reference.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {

void GOTPass::perform(File& mergedFile) {
  // Use map so all pointers to same symbol use same GOT entry.
  llvm::DenseMap<const Atom*, const DefinedAtom*> targetToGOT;
  
  // Scan all references in all atoms.
  for(const DefinedAtom *atom : mergedFile.defined()) {
    for (const Reference *ref : *atom) {
      // Look at instructions accessing the GOT.
      bool canBypassGOT;
      if ( this->isGOTAccess(ref->kind(), canBypassGOT) ) {
        const Atom* target = ref->target();
        assert(target != nullptr);
        const DefinedAtom* defTarget = dyn_cast<DefinedAtom>(target);
        bool replaceTargetWithGOTAtom = false;
        if ( target->definition() == Atom::definitionSharedLibrary ) {
          // Accesses to shared library symbols must go through GOT.
          replaceTargetWithGOTAtom = true;
        }
        else if ( (defTarget != nullptr)
               && (defTarget->interposable() != DefinedAtom::interposeNo) ) {
          // Accesses to interposable symbols in same linkage unit 
          // must also go through GOT.
          assert(defTarget->scope() != DefinedAtom::scopeTranslationUnit);
          replaceTargetWithGOTAtom = true;
        }
        else {
          // Target does not require indirection.  So, if instruction allows
          // GOT to be by-passed, do that optimization and don't create
          // GOT entry.
          replaceTargetWithGOTAtom = !canBypassGOT;
        }
        if ( replaceTargetWithGOTAtom ) {
          // Replace the target with a reference to a GOT entry.
          const DefinedAtom* gotEntry = nullptr;
          auto pos = targetToGOT.find(target);
          if ( pos == targetToGOT.end() ) {
            // This is no existing GOT entry.  Create a new one.
            gotEntry = this->makeGOTEntry(*target);
            assert(gotEntry != nullptr);
            assert(gotEntry->contentType() == DefinedAtom::typeGOT);
            targetToGOT[target] = gotEntry;
          }
          else {
            // Reuse an existing GOT entry.
            gotEntry = pos->second;
            assert(gotEntry != nullptr);
          }
          // Switch reference to GOT atom.
          (const_cast<Reference*>(ref))->setTarget(gotEntry);
        }
        // Update reference kind to reflect
        // that target is now a GOT entry or a direct accesss.
       this->updateReferenceToGOT(ref, replaceTargetWithGOTAtom);
      }
    }
  }
  
  // add all created GOT Atoms to master file
  for (auto it=targetToGOT.begin(), end=targetToGOT.end(); it != end; ++it) {
    mergedFile.addAtom(*it->second);
  }
  

}



}
