//===- Platforms/Darwin/DarwinPlatform.cpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DarwinPlatform.h"
#include "MachOFormat.hpp"
#include "StubAtoms.hpp"
#include "ExecutableAtoms.hpp"
#include "DarwinReferenceKinds.h"
#include "ExecutableWriter.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Core/InputFiles.h"

#include "llvm/Support/ErrorHandling.h"

namespace lld {

Platform *createDarwinPlatform() {
  return new darwin::DarwinPlatform();
}


namespace darwin {

DarwinPlatform::DarwinPlatform()
  : _helperCommonAtom(nullptr), _cRuntimeFile(nullptr) {
}

void DarwinPlatform::addFiles(InputFiles &inputFiles) {
  _cRuntimeFile = new CRuntimeFile();
  inputFiles.prependFile(*_cRuntimeFile);
}

Reference::Kind DarwinPlatform::kindFromString(StringRef kindName) {
  return ReferenceKind::fromString(kindName);
}


StringRef DarwinPlatform::kindToString(Reference::Kind kindValue) {
  return ReferenceKind::toString(kindValue);
}

bool DarwinPlatform::noTextRelocs() {
  return true;
}


bool DarwinPlatform::isCallSite(Reference::Kind kind) {
  return ReferenceKind::isCallSite(kind);
}


bool DarwinPlatform::isGOTAccess(Reference::Kind, bool& canBypassGOT) {
  return false;
}


void DarwinPlatform::updateReferenceToGOT(const Reference*, bool nowGOT) {
}


const DefinedAtom* DarwinPlatform::getStub(const Atom& target, File& file) {
  auto pos = _targetToStub.find(&target);
  if ( pos != _targetToStub.end() ) {
    // Reuse an existing stub.
    assert(pos->second != nullptr);
    return pos->second;
  }
  else {
    // There is no existing stub, so create a new one.
    if ( _helperCommonAtom == nullptr ) {
      // Lazily create common helper code and data.
      _helperCacheAtom = new X86_64NonLazyPointerAtom(file);
      _stubBinderAtom = new StubBinderAtom(file);
      _helperBinderAtom = new X86_64NonLazyPointerAtom(file, *_stubBinderAtom);
      _helperCommonAtom = new X86_64StubHelperCommonAtom(file,
                                        *_helperCacheAtom, *_helperBinderAtom);
    }
    const DefinedAtom* helper = new X86_64StubHelperAtom(file,
                                                          *_helperCommonAtom);
    _stubHelperAtoms.push_back(helper);
    const DefinedAtom* lp = new X86_64LazyPointerAtom(file, *helper, target);
    assert(lp->contentType() == DefinedAtom::typeLazyPointer);
    const DefinedAtom* stub = new X86_64StubAtom(file, *lp);
    assert(stub->contentType() == DefinedAtom::typeStub);
    _targetToStub[&target] = stub;
    _lazyPointers.push_back(lp);
    return stub;
  }
}


void DarwinPlatform::addStubAtoms(File &file) {
  // Add all stubs to master file.
  for (auto it=_targetToStub.begin(), end=_targetToStub.end(); it != end; ++it) {
    file.addAtom(*it->second);
  }
  // Add helper code atoms.
  file.addAtom(*_helperCommonAtom);
  for (const DefinedAtom *lp : _stubHelperAtoms) {
    file.addAtom(*lp);
  }
  // Add GOT slots used for lazy binding.
  file.addAtom(*_helperBinderAtom);
  file.addAtom(*_helperCacheAtom);
  // Add all lazy pointers to master file.
  for (const DefinedAtom *lp : _lazyPointers) {
    file.addAtom(*lp);
  }
  // Add sharedlibrary atom
  file.addAtom(*_stubBinderAtom);
}


const DefinedAtom* DarwinPlatform::makeGOTEntry(const Atom&, File&) {
  return nullptr;
}

void DarwinPlatform::applyFixup(Reference::Kind kind, uint64_t addend,  
                                  uint8_t* location, uint64_t fixupAddress, 
                                                     uint64_t targetAddress) {
  //fprintf(stderr, "applyFixup(kind=%s, addend=0x%0llX, "
  //                "fixupAddress=0x%0llX, targetAddress=0x%0llX\n", 
  //                kindToString(kind).data(), addend, 
  //                fixupAddress, targetAddress);
  if ( ReferenceKind::isRipRel32(kind) ) {
    // compute rip relative value and update.
    int32_t* loc32 = reinterpret_cast<int32_t*>(location);
    *loc32 = (targetAddress - (fixupAddress+4)) + addend;
  }
  else if ( kind == ReferenceKind::pointer64 ) {
    uint64_t* loc64 = reinterpret_cast<uint64_t*>(location);
    *loc64 = targetAddress + addend;
  }
}

void DarwinPlatform::writeExecutable(const lld::File &file, raw_ostream &out) {
  lld::darwin::writeExecutable(file, *this, out);
}


uint64_t DarwinPlatform::pageZeroSize() {
  return 0x100000000;
}


void DarwinPlatform::initializeMachHeader(const lld::File& file, 
                                                   mach_header& mh) {
  // FIXME: Need to get cpu info from file object
  mh.magic      = MAGIC_64;
  mh.cputype    = CPU_TYPE_X86_64;
  mh.cpusubtype = CPU_SUBTYPE_X86_64_ALL;
  mh.filetype   = MH_EXECUTE;
  mh.ncmds      = 0;
  mh.sizeofcmds = 0;
  mh.flags      = 0;
  mh.reserved   = 0;
}

const Atom *DarwinPlatform::mainAtom() {
  assert(_cRuntimeFile != nullptr);
  const Atom *result = _cRuntimeFile->mainAtom();
  assert(result != nullptr);
  if ( result->definition() == Atom::definitionUndefined )
    llvm::report_fatal_error("_main not found");
  return _cRuntimeFile->mainAtom();
}



} // namespace darwin 
} // namespace lld 
