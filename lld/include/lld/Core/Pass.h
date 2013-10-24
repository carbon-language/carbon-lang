//===------ Core/Pass.h - Base class for linker passes --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_PASS_H_
#define LLD_CORE_PASS_H_

#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/range.h"
#include "lld/Core/Reference.h"

#include <vector>

namespace lld {
class DefinedAtom;
class MutableFile;

/// Once the core linking is done (which resolves references, coalesces atoms
/// and produces a complete Atom graph), the linker runs a series of passes
/// on the Atom graph. The graph is modeled as a File, which means the pass
/// has access to all the atoms and to File level attributes. Each pass does
/// a particular transformation to the Atom graph or to the File attributes.
///
/// This is the abstract base class for all passes.  A Pass does its
/// actual work in it perform() method.  It can iterator over Atoms in the
/// graph using the *begin()/*end() atom iterator of the File.  It can add
/// new Atoms to the graph using the File's addAtom() method.
class Pass {
public:
  virtual ~Pass() { }

  /// Do the actual work of the Pass.
  virtual void perform(MutableFile &mergedFile) = 0;

protected:
  // Only subclassess can be instantiated.
  Pass() { }
};

/// Pass for adding stubs (PLT entries) for calls to functions
/// outside the linkage unit.  This class is subclassed by each
/// file format Writer which implements the pure virtual methods.
class StubsPass : public Pass {
public:
  StubsPass() : Pass() {}

  /// Scans all Atoms looking for call-site uses of SharedLibraryAtoms
  /// and transfroms the call-site to call a stub instead using the
  /// helper methods below.
  virtual void perform(MutableFile &mergedFile);

  /// If true, the pass should use stubs for references
  /// to shared library symbols. If false, the pass
  /// will generate relocations on the text segment which the
  /// runtime loader will use to patch the program at runtime.
  virtual bool noTextRelocs() = 0;

  /// Returns whether the Reference kind is for a call site.  The pass
  /// uses this to find calls that need to be indirected through a stub.
  virtual bool isCallSite(int32_t) = 0;

  /// Returns a file format specific atom for a stub/PLT entry which contains
  /// instructions which jump to the specified atom.  May be called multiple
  /// times for the same target atom, in which case this method should return
  /// the same stub atom.
  virtual const DefinedAtom *getStub(const Atom &target) = 0;

  /// After the default implementation of perform() is done calling getStub(),
  /// it will call this method to add all the stub (and support) atoms to the
  /// master file object.
  virtual void addStubAtoms(MutableFile &masterFile) = 0;
};

/// Pass for adding GOT entries for pointers to functions/data
/// outside the linkage unit. This class is subclassed by each
/// file format Writer which implements the pure virtual methods.
class GOTPass : public Pass {
public:
  GOTPass() : Pass() {}

  /// Scans all Atoms looking for pointer to SharedLibraryAtoms
  /// and transfroms them to a pointer to a GOT entry using the
  /// helper methods below.
  virtual void perform(MutableFile &mergedFile);

  /// If true, the pass will use GOT entries for references
  /// to shared library symbols. If false, the pass
  /// will generate relocations on the text segment which the
  /// runtime loader will use to patch the program at runtime.
  virtual bool noTextRelocs() = 0;

  /// Returns whether the Reference kind is a pre-instantiated GOT access.
  /// The default implementation of perform() uses this to figure out
  /// what GOT entries to instantiate.
  virtual bool isGOTAccess(int32_t, bool &canBypassGOT) = 0;

  /// The file format Writer needs to alter the reference kind from a
  /// pre-instantiated GOT access to an actual access.  If targetIsNowGOT is
  /// true, the pass has instantiated a GOT atom and altered the reference's
  /// target to point to that atom.  If targetIsNowGOT is false, the pass
  /// determined a GOT entry is not needed because the reference site can
  /// directly access the target.
  virtual void updateReferenceToGOT(const Reference*, bool targetIsNowGOT) = 0;

  /// Returns a file format specific atom for a GOT entry targeting
  /// the specified atom.
  virtual const DefinedAtom *makeGOTEntry(const Atom &target) = 0;
};

} // namespace lld

#endif // LLD_CORE_PASS_H_
