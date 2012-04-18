//===- Core/Platform.h - Platform Interface -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_PLATFORM_H_
#define LLD_CORE_PLATFORM_H_

#include "lld/Core/Reference.h"
#include "lld/Core/LLVM.h"
#include <vector>

namespace lld {
class Atom;
class DefinedAtom;
class UndefinedAtom;
class SharedLibraryAtom;
class File;
class InputFiles;


/// The Platform class encapsulated plaform specific linking knowledge.
///
/// Much of what it does is driving by platform specific linker options.
class Platform {
public:
  virtual ~Platform();
  
  virtual void addFiles(InputFiles&) = 0;
  
  /// Converts a reference kind string to a in-memory numeric value.
  /// For use with parsing YAML encoded object files.
  virtual Reference::Kind kindFromString(StringRef) = 0;

  /// Converts an in-memory reference kind value to a string.
  /// For use with writing YAML encoded object files.
  virtual StringRef kindToString(Reference::Kind) = 0;

  /// If true, the linker will use stubs and GOT entries for
  /// references to shared library symbols. If false, the linker
  /// will generate relocations on the text segment which the
  /// runtime loader will use to patch the program at runtime.
  virtual bool noTextRelocs() = 0;

  /// Returns if the Reference kind is for a call site.  The "stubs" Pass uses
  /// this to find calls that need to be indirected through a stub.
  virtual bool isCallSite(Reference::Kind) = 0;

  /// Returns if the Reference kind is a pre-instantiated GOT access.
  /// The "got" Pass uses this to figure out what GOT entries to instantiate.
  virtual bool isGOTAccess(Reference::Kind, bool& canBypassGOT) = 0;

  /// The platform needs to alter the reference kind from a pre-instantiated
  /// GOT access to an actual access.  If targetIsNowGOT is true, the "got"
  /// Pass has instantiated a GOT atom and altered the reference's target
  /// to point to that atom.  If targetIsNowGOT is false, the "got" Pass
  /// determined a GOT entry is not needed because the reference site can
  /// directly access the target.
  virtual void updateReferenceToGOT(const Reference*, bool targetIsNowGOT) = 0;

  /// Returns a platform specific atom for a stub/PLT entry which will
  /// jump to the specified atom.  May be called multiple times for the same
  /// target atom, in which case this method should return the same stub
  /// atom.  The platform needs to maintain a list of all stubs (and 
  /// associated atoms) it has created for use by addStubAtoms().
  virtual const DefinedAtom* getStub(const Atom &target, File&) = 0;

  /// After the stubs Pass is done calling getStub(), the Pass will call
  /// this method to add all the stub (and support) atoms to the master
  /// file object.
  virtual void addStubAtoms(File &file) = 0;

  /// Create a platform specific GOT atom.
  virtual const DefinedAtom* makeGOTEntry(const Atom&, File&) = 0;
  
  /// Write an executable file from the supplied file object to the 
  /// supplied stream.
  virtual void writeExecutable(const lld::File &, raw_ostream &out) = 0;
  
protected:
  Platform();
};



///
/// Creates a platform object for linking as done on Darwin (iOS/OSX).
///
extern Platform *createDarwinPlatform();



} // namespace lld

#endif // LLD_CORE_PLATFORM_H_
