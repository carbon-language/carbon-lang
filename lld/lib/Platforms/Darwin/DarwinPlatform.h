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
  virtual void addFiles(InputFiles&);
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
  const Atom *mainAtom();
/// @}

private:
  llvm::DenseMap<const Atom*, const DefinedAtom*> _targetToStub;
  std::vector<const DefinedAtom*>                 _lazyPointers;
  std::vector<const DefinedAtom*>                 _stubHelperAtoms;
  const SharedLibraryAtom                        *_stubBinderAtom;
  const DefinedAtom*                              _helperCommonAtom;
  const DefinedAtom*                              _helperCacheAtom;
  const DefinedAtom*                              _helperBinderAtom;
  class CRuntimeFile                             *_cRuntimeFile;
};

} // namespace darwin
} // namespace lld

#endif // LLD_PLATFORM_DARWIN_PLATFORM_H_
