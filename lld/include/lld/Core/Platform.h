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


/// The Platform class encapsulated plaform specific linking knowledge.
///
/// Much of what it does is driving by platform specific linker options.
class Platform {
public:
  virtual ~Platform();

  virtual void initialize() = 0;

  /// @brief tell platform object another file has been added
  virtual void fileAdded(const File &file) = 0;

  /// @brief tell platform object another atom has been added
  virtual void atomAdded(const Atom &file) = 0;

  /// @brief give platform a chance to change each atom's scope
  virtual void adjustScope(const DefinedAtom &atom) = 0;

  /// @brief if specified atom needs alternate names, return AliasAtom(s)
  virtual bool getAliasAtoms(const Atom &atom,
                             std::vector<const DefinedAtom *>&) = 0;

  /// @brief give platform a chance to resolve platform-specific undefs
  virtual bool getPlatformAtoms(StringRef undefined,
                                std::vector<const DefinedAtom *>&) = 0;

  /// @brief resolver should remove unreferenced atoms
  virtual bool deadCodeStripping() = 0;

  /// @brief atom must be kept so should be root of dead-strip graph
  virtual bool isDeadStripRoot(const Atom &atom) = 0;

  /// @brief if target must have some atoms, denote here
  virtual bool getImplicitDeadStripRoots(std::vector<const DefinedAtom *>&) = 0;

  /// @brief return entry point for output file (e.g. "main") or nullptr
  virtual StringRef entryPointName() = 0;

  /// @brief for iterating must-be-defined symbols ("main" or -u command line
  ///        option)
  typedef StringRef const *UndefinesIterator;
  virtual UndefinesIterator  initialUndefinesBegin() const = 0;
  virtual UndefinesIterator  initialUndefinesEnd() const = 0;

  /// @brief if platform wants resolvers to search libraries for overrides
  virtual bool searchArchivesToOverrideTentativeDefinitions() = 0;
  virtual bool searchSharedLibrariesToOverrideTentativeDefinitions() = 0;

  /// @brief if platform allows symbol to remain undefined (e.g. -r)
  virtual bool allowUndefinedSymbol(StringRef name) = 0;

  /// @brief for debugging dead code stripping, -why_live
  virtual bool printWhyLive(StringRef name) = 0;

  /// When core linking finds a duplicate definition, the platform
  /// can either print an error message and terminate or return with
  /// which atom the linker should use.
  virtual const Atom& handleMultipleDefinitions(const Atom& def1,
                                                const Atom& def2) = 0;

  /// @brief print out undefined symbol error messages in platform specific way
  virtual void errorWithUndefines(const std::vector<const Atom *>& undefs,
                                  const std::vector<const Atom *>& all) = 0;

  /// When core linking finds undefined atoms from different object
  /// files that have different canBeNull values, this method is called.
  /// The useUndef2 parameter is set to which canBeNull setting the
  /// linker should use, and can be changed by this method.  Or this
  /// method can emit a warning/error about the mismatch.
  virtual void undefineCanBeNullMismatch(const UndefinedAtom& undef1,
                                         const UndefinedAtom& undef2,
                                         bool& useUndef2) = 0;

  /// When core linking finds shared library atoms from different object
  /// files that have different attribute values, this method is called.
  /// The useShlib2 parameter is set to which atom attributes the
  /// linker should use, and can be changed by this method.  Or this
  /// method can emit a warning/error about the mismatch.
  virtual void sharedLibrarylMismatch(const SharedLibraryAtom& shLib1,
                                      const SharedLibraryAtom& shLib2,
                                      bool& useShlib2) = 0;

  /// @brief last chance for platform to tweak atoms
  virtual void postResolveTweaks(std::vector<const Atom *>& all) = 0;

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
