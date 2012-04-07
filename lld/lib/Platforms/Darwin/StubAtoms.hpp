//===- Platforms/Darwin/x86_64StubAtom.hpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PLATFORM_DARWIN_X86_64_STUB_ATOM_H_
#define LLD_PLATFORM_DARWIN_X86_64_STUB_ATOM_H_

#include <vector>

#include "llvm/ADT/ArrayRef.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "DarwinReferenceKinds.h"

namespace lld {
namespace darwin {


//
// Generic Reference
//
class GenericReference : public Reference {
public:
                GenericReference(Reference::Kind k, uint64_t off, 
                                const Atom *t, Reference::Addend a) 
                      : _target(t), _offsetInAtom(off), _addend(a), _kind(k) { }

  virtual uint64_t offsetInAtom() const {
    return _offsetInAtom;
  }

  virtual Kind kind() const {
    return _kind;
  }

  virtual void setKind(Kind k) {
    _kind = k;
  }

  virtual const Atom* target() const {
    return _target;
  }

  virtual Addend addend() const {
    return _addend;
  }

  virtual void setAddend(Addend a) {
    _addend = a;
  }

  virtual void setTarget(const Atom* newAtom) {
    _target = newAtom;
  }
private:
  const Atom*  _target;
  uint64_t     _offsetInAtom;
  Addend       _addend;
  Kind         _kind;
};


//
// Generic Atom base class
//
class BaseAtom : public DefinedAtom {
public:
        BaseAtom(const File &f) : _file(f) {
          static uint32_t lastOrdinal = 0;
          _ordinal = lastOrdinal++; 
        }

  virtual const File& file() const {
    return _file;
  }

  virtual StringRef name() const {
    return StringRef();
  }
  
  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  virtual Scope scope() const {
    return DefinedAtom::scopeLinkageUnit;
  }
  
  virtual Interposable interposable() const {
    return DefinedAtom::interposeNo;
  }
  
  virtual Merge merge() const {
    return DefinedAtom::mergeNo;
  }
  
  virtual Alignment alignment() const {
    return Alignment(0,0);
  }
  
  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }
    
  virtual StringRef customSectionName() const {
    return StringRef();
  }
  virtual DeadStripKind deadStrip() const {
    return DefinedAtom::deadStripNormal;
  }
    
  virtual bool isThumb() const {
    return false;
  }
    
  virtual bool isAlias() const {
    return false;
  }
  
  virtual DefinedAtom::reference_iterator referencesBegin() const {
    uintptr_t index = 0;
    const void* it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  virtual DefinedAtom::reference_iterator referencesEnd() const {
    uintptr_t index = _references.size();
    const void* it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  virtual const Reference* derefIterator(const void* it) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    assert(index < _references.size());
    return &_references[index];
  }

  virtual void incrementIterator(const void*& it) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    ++index;
    it = reinterpret_cast<const void*>(index);
  }
  
  void addReference(Reference::Kind kind, uint64_t offset, const Atom *target, 
                   Reference::Addend addend) {
    _references.push_back(GenericReference(kind, offset, target, addend));
  }
  
private:
  const File&                   _file;
  uint32_t                      _ordinal;
  std::vector<GenericReference> _references;
};



//
// X86_64 Stub Atom created by the stubs pass.
//
class X86_64StubAtom : public BaseAtom {
public:
        X86_64StubAtom(const File &file, const Atom &lazyPointer) 
                       : BaseAtom(file) {
          this->addReference(ReferenceKind::pcRel32, 2, &lazyPointer, 0);
        }

  virtual ContentType contentType() const  {
    return DefinedAtom::typeStub;
  }

  virtual uint64_t size() const {
    return 6;
  }

  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permR_X;
  }
  
  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t instructions[] = 
              { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 }; // jmp *lazyPointer
    assert(sizeof(instructions) == this->size());
    return ArrayRef<uint8_t>(instructions, sizeof(instructions));
  }
  
};


//
// X86_64 Stub Helper Common Atom created by the stubs pass.
//
class X86_64StubHelperCommonAtom : public BaseAtom {
public:
  X86_64StubHelperCommonAtom(const File &file, const Atom &cache,
                                               const Atom &binder)
  : BaseAtom(file) {
    this->addReference(ReferenceKind::pcRel32, 3,  &cache, 0);
    this->addReference(ReferenceKind::pcRel32, 11, &binder, 0);
  }
  
  virtual ContentType contentType() const  {
    return DefinedAtom::typeStubHelper;
  }
  
  virtual uint64_t size() const {
    return 16;
  }
  
  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permR_X;
  }
  
  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t instructions[] = 
    { 0x4C, 0x8D, 0x1D, 0x00, 0x00, 0x00, 0x00,   // leaq cache(%rip),%r11
      0x41, 0x53,                                 // push %r11
      0xFF, 0x25, 0x00, 0x00, 0x00, 0x00,         // jmp *binder(%rip)
      0x90 };                                     // nop
    assert(sizeof(instructions) == this->size());
    return ArrayRef<uint8_t>(instructions, sizeof(instructions));
  }
  
};
  
  

//
// X86_64 Stub Helper Atom created by the stubs pass.
//
class X86_64StubHelperAtom : public BaseAtom {
public:
  X86_64StubHelperAtom(const File &file, const Atom &helperCommon) 
  : BaseAtom(file) {
    this->addReference(ReferenceKind::lazyImm, 1, nullptr, 0);
    this->addReference(ReferenceKind::pcRel32, 6, &helperCommon, 0);
  }
  
  virtual ContentType contentType() const  {
    return DefinedAtom::typeStubHelper;
  }
  
  virtual uint64_t size() const {
    return 10;
  }
  
  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permR_X;
  }
  
  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t instructions[] = 
              { 0x68, 0x00, 0x00, 0x00, 0x00,   // pushq $lazy-info-offset
                0xE9, 0x00, 0x00, 0x00, 0x00 }; // jmp helperhelper
    assert(sizeof(instructions) == this->size());
    return ArrayRef<uint8_t>(instructions, sizeof(instructions));
  }
 
};
  

//
// X86_64 Lazy Pointer Atom created by the stubs pass.
//
class X86_64LazyPointerAtom : public BaseAtom {
public:
        X86_64LazyPointerAtom(const File &file, const Atom &helper,
                                                const Atom &shlib)
              : BaseAtom(file) {
                this->addReference(ReferenceKind::pointer64, 0, &helper, 0);
                this->addReference(ReferenceKind::lazyTarget, 0, &shlib, 0);
        }

  virtual ContentType contentType() const  {
    return DefinedAtom::typeLazyPointer;
  }

  virtual uint64_t size() const {
    return 8;
  }

  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permRW_;
  }
  
  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t bytes[] = 
                            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    return ArrayRef<uint8_t>(bytes, 8);
  }
  
};


//
// X86_64 NonLazy (GOT) Pointer Atom created by the stubs pass.
//
class X86_64NonLazyPointerAtom : public BaseAtom {
public:
  X86_64NonLazyPointerAtom(const File &file)
  : BaseAtom(file) {
  }
  
  X86_64NonLazyPointerAtom(const File &file, const Atom &shlib)
  : BaseAtom(file) {
    this->addReference(ReferenceKind::pointer64, 0, &shlib, 0);
  }
  
  virtual ContentType contentType() const  {
    return DefinedAtom::typeGOT;
  }
  
  virtual uint64_t size() const {
    return 8;
  }
  
  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permRW_;
  }
  
  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t bytes[] = 
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    return ArrayRef<uint8_t>(bytes, 8);
  }
  
};
  
  
//
// StubBinderAtom created by the stubs pass.
//
class StubBinderAtom : public SharedLibraryAtom {
public:
  StubBinderAtom(const File &f) : _file(f) { 
  }
          
  virtual const File& file() const {
    return _file;
  }

  virtual StringRef name() const {
    return StringRef("dyld_stub_binder");
  }

  virtual StringRef loadName() const {
    return StringRef("/usr/lib/libSystem.B.dylib");
  }
  
  virtual bool canBeNullAtRuntime() const {
    return false;
  }
  
private:
  const File  &_file;
};



} // namespace darwin 
} // namespace lld 


#endif // LLD_PLATFORM_DARWIN_X86_64_STUB_ATOM_H_
