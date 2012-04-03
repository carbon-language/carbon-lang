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
#include "lld/Core/Platform.h"

#include <vector>

namespace lld {


///
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
///
///
class Pass {
public:
    /// Do the actual work of the Pass.
    virtual void perform() = 0;

protected:
  // Only subclassess can be instantiated.
  Pass(File& f, Platform& p) : _file(f), _platform(p) {}


  File&       _file;
  Platform&   _platform;
};


///
/// Pass for adding stubs (PLT entries) for calls to functions
/// outside the linkage unit.
///
class StubsPass : public Pass {
public:
  StubsPass(File& f, Platform& p) : Pass(f, p) {}

  /// Scans all Atoms looking for call-site uses of SharedLibraryAtoms
  /// and transfroms the call-site to call a stub instead.
  virtual void perform();
};


///
/// Pass for adding GOT entries for pointers to functions/data
/// outside the linkage unit.
///
class GOTPass : public Pass {
public:
  GOTPass(File& f, Platform& p) : Pass(f, p) {}

  /// Scans all Atoms looking for pointer to SharedLibraryAtoms
  /// and transfroms them to a pointer to a GOT entry.
  virtual void perform();
};


} // namespace lld

#endif // LLD_CORE_PASS_H_
