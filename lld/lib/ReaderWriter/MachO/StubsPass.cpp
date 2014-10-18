//===- lib/ReaderWriter/MachO/StubsPass.cpp -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This linker pass updates call-sites which have references to shared library
// atoms to instead have a reference to a stub (PLT entry) for the specified
// symbol.  Each file format defines a subclass of StubsPass which implements
// the abstract methods for creating the file format specific StubAtoms.
//
//===----------------------------------------------------------------------===//

#include "ArchHandler.h"
#include "File.h"
#include "MachOPasses.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Simple.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"
#include "llvm/ADT/DenseMap.h"


namespace lld {
namespace mach_o {


//
//  Lazy Pointer Atom created by the stubs pass.
//
class LazyPointerAtom : public SimpleDefinedAtom {
public:
  LazyPointerAtom(const File &file, bool is64) 
    : SimpleDefinedAtom(file), _is64(is64) { }

  ContentType contentType() const override {
    return DefinedAtom::typeLazyPointer;
  }

  Alignment alignment() const override {
    return Alignment(_is64 ? 3 : 2);
  }

  uint64_t size() const override {
    return _is64 ? 8 : 4;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permRW_;
  }

  ArrayRef<uint8_t> rawContent() const override {
    static const uint8_t zeros[] =
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    return llvm::makeArrayRef(zeros, size());
  }

private:
  const bool _is64;
};


//
//  NonLazyPointer (GOT) Atom created by the stubs pass.
//
class NonLazyPointerAtom : public SimpleDefinedAtom {
public:
  NonLazyPointerAtom(const File &file, bool is64) 
    : SimpleDefinedAtom(file), _is64(is64) { }

  ContentType contentType() const override {
    return DefinedAtom::typeGOT;
  }

  Alignment alignment() const override {
    return Alignment(_is64 ? 3 : 2);
  }

  uint64_t size() const override {
    return _is64 ? 8 : 4;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permRW_;
  }

  ArrayRef<uint8_t> rawContent() const override {
    static const uint8_t zeros[] =
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    return llvm::makeArrayRef(zeros, size());
  }

private:
  const bool _is64;
};



//
// Stub Atom created by the stubs pass.
//
class StubAtom : public SimpleDefinedAtom {
public:
  StubAtom(const File &file, const ArchHandler::StubInfo &stubInfo)
      : SimpleDefinedAtom(file), _stubInfo(stubInfo) { }

  ContentType contentType() const override {
    return DefinedAtom::typeStub;
  }

  Alignment alignment() const override {
    return Alignment(_stubInfo.codeAlignment);
  }

  uint64_t size() const override {
    return _stubInfo.stubSize;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permR_X;
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(_stubInfo.stubBytes, _stubInfo.stubSize);
  }

private:
  const ArchHandler::StubInfo   &_stubInfo;
};


//
// Stub Helper Atom created by the stubs pass.
//
class StubHelperAtom : public SimpleDefinedAtom {
public:
  StubHelperAtom(const File &file, const ArchHandler::StubInfo &stubInfo)
      : SimpleDefinedAtom(file), _stubInfo(stubInfo) { }

  ContentType contentType() const override {
    return DefinedAtom::typeStubHelper;
  }

  Alignment alignment() const override {
    return Alignment(_stubInfo.codeAlignment);
  }

  uint64_t size() const override {
    return _stubInfo.stubHelperSize;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permR_X;
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(_stubInfo.stubHelperBytes, 
                              _stubInfo.stubHelperSize);
  }

private:
  const ArchHandler::StubInfo   &_stubInfo;
};


//
// Stub Helper Common Atom created by the stubs pass.
//
class StubHelperCommonAtom : public SimpleDefinedAtom {
public:
  StubHelperCommonAtom(const File &file, const ArchHandler::StubInfo &stubInfo)
      : SimpleDefinedAtom(file), _stubInfo(stubInfo) { }

  ContentType contentType() const override {
    return DefinedAtom::typeStubHelper;
  }

  Alignment alignment() const override {
    return Alignment(_stubInfo.codeAlignment);
  }

  uint64_t size() const override {
    return _stubInfo.stubHelperCommonSize;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permR_X;
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(_stubInfo.stubHelperCommonBytes, 
                        _stubInfo.stubHelperCommonSize);
  }

private:
  const ArchHandler::StubInfo   &_stubInfo;
};


class StubsPass : public Pass {
public:
  StubsPass(const MachOLinkingContext &context)
    : _context(context)
    , _archHandler(_context.archHandler())
    , _stubInfo(_archHandler.stubInfo())
    , _file("<mach-o Stubs pass>")
    , _helperCommonAtom(nullptr)
    , _helperCacheNLPAtom(nullptr)
    , _helperBinderNLPAtom(nullptr) {
  }


  void perform(std::unique_ptr<MutableFile> &mergedFile) override {
    // Skip this pass if output format uses text relocations instead of stubs.
    if (!this->noTextRelocs())
      return;
    
    // Scan all references in all atoms.
    for (const DefinedAtom *atom : mergedFile->defined()) {
      for (const Reference *ref : *atom) {
        // Look at call-sites.
        if (!this->isCallSite(*ref))
          continue;
        const Atom *target = ref->target();
        assert(target != nullptr);
        if (isa<SharedLibraryAtom>(target)) {
          // Calls to shared libraries go through stubs.
          replaceCalleeWithStub(target, ref);
          continue;
        }
        const DefinedAtom *defTarget = dyn_cast<DefinedAtom>(target);
        if (defTarget && defTarget->interposable() != DefinedAtom::interposeNo){
          // Calls to interposable functions in same linkage unit must also go
          // through a stub.
          assert(defTarget->scope() != DefinedAtom::scopeTranslationUnit);
          replaceCalleeWithStub(target, ref);
        }
      }
    }
    // Exit early if no stubs needed.
    if (_targetToStub.empty())
      return;
    
    // Add reference to dyld_stub_binder in libSystem.dylib
    if (_helperBinderNLPAtom) {
      bool found = false;
      for (const SharedLibraryAtom *atom : mergedFile->sharedLibrary()) {
        if (atom->name().equals(_stubInfo.binderSymbolName)) {
          addReference(_helperBinderNLPAtom,  
                       _stubInfo.nonLazyPointerReferenceToBinder, atom);
          found = true;
          break;
        }
      }
      assert(found && "dyld_stub_binder not found");
    }
    
    // Add all stubs to master file.
    for (auto it : _targetToStub) {
      mergedFile->addAtom(*it.second);
    }
    // Add helper code atoms.
    mergedFile->addAtom(*_helperCommonAtom);
    for (const DefinedAtom *lp : _stubHelperAtoms) {
      mergedFile->addAtom(*lp);
    }
    // Add GOT slots used for lazy binding.
    mergedFile->addAtom(*_helperBinderNLPAtom);
    mergedFile->addAtom(*_helperCacheNLPAtom);
    // Add all lazy pointers to master file.
    for (const DefinedAtom *lp : _lazyPointers) {
      mergedFile->addAtom(*lp);
    }
  }


private:

  bool noTextRelocs() {
    return true;
  }

  bool isCallSite(const Reference &ref) {
    return _archHandler.isCallSite(ref);
  }

  void replaceCalleeWithStub(const Atom *target, const Reference *ref) {
    // Make file-format specific stub and other support atoms.
    const DefinedAtom *stub = this->getStub(*target);
    assert(stub != nullptr);
    // Switch call site to reference stub atom instead.
    const_cast<Reference *>(ref)->setTarget(stub);
  }

  const DefinedAtom* getStub(const Atom& target) {
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

  const DefinedAtom* makeStub(const Atom &target) {
    SimpleDefinedAtom* stub   = new (_file.allocator()) 
                                                    StubAtom(_file, _stubInfo);
    SimpleDefinedAtom* lp     = new (_file.allocator()) 
                                    LazyPointerAtom(_file, _context.is64Bit());
    SimpleDefinedAtom* helper = new (_file.allocator()) 
                                              StubHelperAtom(_file, _stubInfo);
   
    addReference(stub, _stubInfo.stubReferenceToLP, lp);
    addOptReference(stub, _stubInfo.stubReferenceToLP,
                    _stubInfo.optStubReferenceToLP, lp);
    addReference(lp, _stubInfo.lazyPointerReferenceToHelper, helper);
    addReference(lp, _stubInfo.lazyPointerReferenceToFinal, &target);
    addReference(helper, _stubInfo.stubHelperReferenceToImm, helper);
    addReference(helper, _stubInfo.stubHelperReferenceToHelperCommon, 
                 helperCommon());
    
    _stubHelperAtoms.push_back(helper);
    _targetToStub[&target] = stub;
    _lazyPointers.push_back(lp);
    
    return stub;
  }
 
  void addReference(SimpleDefinedAtom* atom,
                    const ArchHandler::ReferenceInfo &refInfo,
                    const lld::Atom* target) {
    atom->addReference(Reference::KindNamespace::mach_o,
                      refInfo.arch, refInfo.kind, refInfo.offset,
                      target, refInfo.addend);
  }

   void addOptReference(SimpleDefinedAtom* atom,
                    const ArchHandler::ReferenceInfo &refInfo,
                    const ArchHandler::OptionalRefInfo &optRef,
                    const lld::Atom* target) {
      if (!optRef.used)
        return;
    atom->addReference(Reference::KindNamespace::mach_o,
                      refInfo.arch, optRef.kind, optRef.offset,
                      target, optRef.addend);
  }

  const DefinedAtom* helperCommon() {
    if ( !_helperCommonAtom ) {
      // Lazily create common helper code and data.
      _helperCommonAtom    = new (_file.allocator()) 
                                         StubHelperCommonAtom(_file, _stubInfo);
      _helperCacheNLPAtom  = new (_file.allocator()) 
                                  NonLazyPointerAtom(_file, _context.is64Bit());
      _helperBinderNLPAtom = new (_file.allocator()) 
                                  NonLazyPointerAtom(_file, _context.is64Bit());
      addReference(_helperCommonAtom, 
                   _stubInfo.stubHelperCommonReferenceToCache,
                   _helperCacheNLPAtom);
      addOptReference(_helperCommonAtom,
                      _stubInfo.stubHelperCommonReferenceToCache,
                      _stubInfo.optStubHelperCommonReferenceToCache,
                      _helperCacheNLPAtom);
      addReference(_helperCommonAtom,
                   _stubInfo.stubHelperCommonReferenceToBinder,
                   _helperBinderNLPAtom);
      addOptReference(_helperCommonAtom,
                      _stubInfo.stubHelperCommonReferenceToBinder,
                      _stubInfo.optStubHelperCommonReferenceToBinder,
                      _helperBinderNLPAtom);
    }
    return _helperCommonAtom;
  }


  const MachOLinkingContext                      &_context;
  mach_o::ArchHandler                            &_archHandler;
  const ArchHandler::StubInfo                    &_stubInfo;
  MachOFile                                       _file;
  llvm::DenseMap<const Atom*, const DefinedAtom*> _targetToStub;
  std::vector<const DefinedAtom*>                 _lazyPointers;
  std::vector<const DefinedAtom*>                 _stubHelperAtoms;
  SimpleDefinedAtom                              *_helperCommonAtom;
  SimpleDefinedAtom                              *_helperCacheNLPAtom;
  SimpleDefinedAtom                              *_helperBinderNLPAtom;
};



void addStubsPass(PassManager &pm, const MachOLinkingContext &ctx) {
  pm.add(std::unique_ptr<Pass>(new StubsPass(ctx)));
}

} // end namespace mach_o
} // end namespace lld
