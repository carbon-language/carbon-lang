//===- Platform/DarwinPlatform.h - Darwin Platform Implementation ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PLATFORM_DARWIN_PLATFORM_H_
#define LLD_PLATFORM_DARWIN_PLATFORM_H_

#include "lld/Core/Platform.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {
namespace darwin {

class DarwinPlatform : public Platform {
public:
                DarwinPlatform();
                
/// @name Platform methods
/// @{
  virtual void initialize();
  virtual void fileAdded(const File &file);
  virtual void atomAdded(const Atom &file);
  virtual void adjustScope(const DefinedAtom &atom);
  virtual bool getAliasAtoms(const Atom &atom,
                             std::vector<const DefinedAtom *>&);
  virtual bool getPlatformAtoms(llvm::StringRef undefined,
                                std::vector<const DefinedAtom *>&);
  virtual bool deadCodeStripping();
  virtual bool isDeadStripRoot(const Atom &atom);
  virtual bool getImplicitDeadStripRoots(std::vector<const DefinedAtom *>&);
  virtual llvm::StringRef entryPointName();
  virtual UndefinesIterator  initialUndefinesBegin() const;
  virtual UndefinesIterator  initialUndefinesEnd() const;
  virtual bool searchArchivesToOverrideTentativeDefinitions();
  virtual bool searchSharedLibrariesToOverrideTentativeDefinitions();
  virtual bool allowUndefinedSymbol(llvm::StringRef name);
  virtual bool printWhyLive(llvm::StringRef name);
  virtual const Atom& handleMultipleDefinitions(const Atom& def1, 
                                                const Atom& def2);
  virtual void errorWithUndefines(const std::vector<const Atom *>& undefs,
                                  const std::vector<const Atom *>& all);
  virtual void undefineCanBeNullMismatch(const UndefinedAtom& undef1,
                                         const UndefinedAtom& undef2,
                                         bool& useUndef2);
  virtual void sharedLibrarylMismatch(const SharedLibraryAtom& shLib1,
                                      const SharedLibraryAtom& shLib2,
                                      bool& useShlib2);
  virtual void postResolveTweaks(std::vector<const Atom *>& all);
  virtual Reference::Kind kindFromString(llvm::StringRef);
  virtual llvm::StringRef kindToString(Reference::Kind);
  virtual bool noTextRelocs();
  virtual bool isCallSite(Reference::Kind);
  virtual bool isGOTAccess(Reference::Kind, bool& canBypassGOT);
  virtual void updateReferenceToGOT(const Reference*, bool targetIsNowGOT);
  virtual const DefinedAtom* getStub(const Atom&, File&);
  virtual void addStubAtoms(File &file);
  virtual const DefinedAtom* makeGOTEntry(const Atom&, File&);
  virtual void applyFixup(Reference::Kind, uint64_t addend, uint8_t*, 
                          uint64_t fixupAddress, uint64_t targetAddress);
  virtual void writeExecutable(const lld::File &, raw_ostream &out);
/// @}
/// @name Darwin specific methods
/// @{
  uint64_t  pageZeroSize();
  void initializeMachHeader(const lld::File& file, class mach_header& mh);
/// @}

private:
  llvm::DenseMap<const Atom*, const DefinedAtom*> _targetToStub;
  std::vector<const DefinedAtom*>                 _lazyPointers;
  std::vector<const DefinedAtom*>                 _stubHelperAtoms;
  const SharedLibraryAtom                        *_stubBinderAtom;
  const DefinedAtom*                              _helperCommonAtom;
  const DefinedAtom*                              _helperCacheAtom;
  const DefinedAtom*                              _helperBinderAtom;
};

} // namespace darwin
} // namespace lld

#endif // LLD_PLATFORM_DARWIN_PLATFORM_H_
