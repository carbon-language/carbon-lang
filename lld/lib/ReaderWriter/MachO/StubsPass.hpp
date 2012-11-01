//===- lib/ReaderWriter/MachO/StubsPass.hpp -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_STUBS_PASS_H_
#define LLD_READER_WRITER_MACHO_STUBS_PASS_H_

#include "llvm/ADT/DenseMap.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Pass.h"

#include "ReferenceKinds.h"
#include "SimpleAtoms.hpp"
#include "StubAtoms.hpp"

namespace lld {
namespace mach_o {


class StubsPass : public lld::StubsPass {
public:
  StubsPass(const WriterOptionsMachO &options) 
    : _options(options), 
      _kindHandler(KindHandler::makeHandler(options.architecture())),
      _helperCommonAtom(nullptr),
      _helperCacheAtom(nullptr),
      _helperBinderAtom(nullptr) {
  }

  virtual bool noTextRelocs() {
    return _options.noTextRelocations();
  }

  virtual bool isCallSite(Reference::Kind kind) {
    return _kindHandler->isCallSite(kind);
  }

  virtual const DefinedAtom* getStub(const Atom& target) {
    auto pos = _targetToStub.find(&target);
    if ( pos != _targetToStub.end() ) {
      // Reuse an existing stub.
      assert(pos->second != nullptr);
      return pos->second;
    }
    else {
      // There is no existing stub, so create a new one.
      return this->makeStub(target);
    }
  }

  const DefinedAtom* makeStub(const Atom& target) {
    switch ( _options.architecture() ) {
      case WriterOptionsMachO::arch_x86_64:
        return makeStub_x86_64(target);

      case WriterOptionsMachO::arch_x86:
        return makeStub_x86(target);

      case WriterOptionsMachO::arch_armv6:
      case WriterOptionsMachO::arch_armv7:
        return makeStub_arm(target);
    }
  }

  const DefinedAtom* makeStub_x86_64(const Atom& target) {
    if ( _helperCommonAtom == nullptr ) {
      // Lazily create common helper code and data.
      _helperCacheAtom = new X86_64NonLazyPointerAtom(_file);
      _binderAtom = new StubBinderAtom(_file);
      _helperBinderAtom = new X86_64NonLazyPointerAtom(_file, *_binderAtom);
      _helperCommonAtom = new X86_64StubHelperCommonAtom(_file,
                                       *_helperCacheAtom, *_helperBinderAtom);
    }
    const DefinedAtom* helper = new X86_64StubHelperAtom(_file,
                                                          *_helperCommonAtom);
    _stubHelperAtoms.push_back(helper);
    const DefinedAtom* lp = new X86_64LazyPointerAtom(_file, *helper, target);
    assert(lp->contentType() == DefinedAtom::typeLazyPointer);
    _lazyPointers.push_back(lp);
    const DefinedAtom* stub = new X86_64StubAtom(_file, *lp);
     assert(stub->contentType() == DefinedAtom::typeStub);
    _targetToStub[&target] = stub;
    return stub;
  }

  const DefinedAtom* makeStub_x86(const Atom& target) {
    if ( _helperCommonAtom == nullptr ) {
      // Lazily create common helper code and data.
      _helperCacheAtom = new X86NonLazyPointerAtom(_file);
      _binderAtom = new StubBinderAtom(_file);
      _helperBinderAtom = new X86NonLazyPointerAtom(_file, *_binderAtom);
      _helperCommonAtom = new X86StubHelperCommonAtom(_file,
                                       *_helperCacheAtom, *_helperBinderAtom);
    }
    const DefinedAtom* helper = new X86StubHelperAtom(_file,
                                                          *_helperCommonAtom);
    _stubHelperAtoms.push_back(helper);
    const DefinedAtom* lp = new X86LazyPointerAtom(_file, *helper, target);
    assert(lp->contentType() == DefinedAtom::typeLazyPointer);
    _lazyPointers.push_back(lp);
    const DefinedAtom* stub = new X86StubAtom(_file, *lp);
     assert(stub->contentType() == DefinedAtom::typeStub);
    _targetToStub[&target] = stub;
    return stub;
  }

  const DefinedAtom* makeStub_arm(const Atom& target) {
    assert(0 && "stubs not yet implemented for arm");
    return nullptr;
  }


  virtual void addStubAtoms(File &mergedFile) {
    // Exit early if no stubs needed.
    if ( _targetToStub.size() == 0 )
      return;
    // Add all stubs to master file.
    for (auto it : _targetToStub) {
      mergedFile.addAtom(*it.second);
    }
    // Add helper code atoms.
    mergedFile.addAtom(*_helperCommonAtom);
    for (const DefinedAtom *lp : _stubHelperAtoms) {
      mergedFile.addAtom(*lp);
    }
    // Add GOT slots used for lazy binding.
    mergedFile.addAtom(*_helperBinderAtom);
    mergedFile.addAtom(*_helperCacheAtom);
    // Add all lazy pointers to master file.
    for (const DefinedAtom *lp : _lazyPointers) {
      mergedFile.addAtom(*lp);
    }
    // Add sharedlibrary atom
    mergedFile.addAtom(*_binderAtom);
  }

private:

  class File : public SimpleFile {
  public:
      File() : SimpleFile("MachO Stubs pass") {
      }
  };

  const WriterOptionsMachO                       &_options;
  KindHandler                                    *_kindHandler;
  File                                            _file;
  llvm::DenseMap<const Atom*, const DefinedAtom*> _targetToStub;
  std::vector<const DefinedAtom*>                 _lazyPointers;
  std::vector<const DefinedAtom*>                 _stubHelperAtoms;
  const SharedLibraryAtom                        *_binderAtom;
  const DefinedAtom*                              _helperCommonAtom;
  const DefinedAtom*                              _helperCacheAtom;
  const DefinedAtom*                              _helperBinderAtom;
};


} // namespace mach_o
} // namespace lld


#endif // LLD_READER_WRITER_MACHO_STUBS_PASS_H_
