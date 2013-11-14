//===- lib/ReaderWriter/MachO/StubAtoms_x86.hpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_STUB_ATOM_X86_H_
#define LLD_READER_WRITER_MACHO_STUB_ATOM_X86_H_

#include "llvm/ADT/ArrayRef.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "ReferenceKinds.h"

using llvm::makeArrayRef;

namespace lld {
namespace mach_o {

//
// X86 Stub Atom created by the stubs pass.
//
class X86StubAtom : public SimpleDefinedAtom {
public:
  X86StubAtom(const File &file, const Atom &lazyPointer)
      : SimpleDefinedAtom(file) {
    this->addReference(KindHandler_x86::abs32, 2, &lazyPointer, 0);
  }

  virtual ContentType contentType() const {
    return DefinedAtom::typeStub;
  }

  virtual uint64_t size() const {
    return 6;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permR_X;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t instructions[] =
        { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 }; // jmp *lazyPointer
    assert(sizeof(instructions) == this->size());
    return makeArrayRef(instructions);
  }
};


//
// X86 Stub Helper Common Atom created by the stubs pass.
//
class X86StubHelperCommonAtom : public SimpleDefinedAtom {
public:
  X86StubHelperCommonAtom(const File &file, const Atom &cache,
                          const Atom &binder)
      : SimpleDefinedAtom(file) {
    this->addReference(KindHandler_x86::abs32, 1, &cache,  0);
    this->addReference(KindHandler_x86::abs32, 7, &binder, 0);
  }

  virtual ContentType contentType() const {
    return DefinedAtom::typeStubHelper;
  }

  virtual uint64_t size() const {
    return 12;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permR_X;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t instructions[] =
        { 0x68, 0x00, 0x00, 0x00, 0x00,         // pushl $dyld_ImageLoaderCache
          0xFF, 0x25, 0x00, 0x00, 0x00, 0x00,   // jmp *_fast_lazy_bind
          0x90 };                               // nop
    assert(sizeof(instructions) == this->size());
    return makeArrayRef(instructions);
  }
};


//
// X86 Stub Helper Atom created by the stubs pass.
//
class X86StubHelperAtom : public SimpleDefinedAtom {
public:
  X86StubHelperAtom(const File &file, const Atom &helperCommon)
      : SimpleDefinedAtom(file) {
    this->addReference(KindHandler_x86::lazyImmediate, 1, this, 0);
    this->addReference(KindHandler_x86::branch32, 6, &helperCommon, 0);
  }

  virtual ContentType contentType() const {
    return DefinedAtom::typeStubHelper;
  }

  virtual uint64_t size() const {
    return 10;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permR_X;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t instructions[] =
        { 0x68, 0x00, 0x00, 0x00, 0x00,   // pushq $lazy-info-offset
          0xE9, 0x00, 0x00, 0x00, 0x00 }; // jmp helperhelper
    assert(sizeof(instructions) == this->size());
    return makeArrayRef(instructions);
  }
};


//
// X86 Lazy Pointer Atom created by the stubs pass.
//
class X86LazyPointerAtom : public SimpleDefinedAtom {
public:
  X86LazyPointerAtom(const File &file, const Atom &helper, const Atom &shlib)
      : SimpleDefinedAtom(file) {
    this->addReference(KindHandler_x86::pointer32,  0, &helper, 0);
    this->addReference(KindHandler_x86::lazyTarget, 0, &shlib,  0);
  }

  virtual ContentType contentType() const {
    return DefinedAtom::typeLazyPointer;
  }

  virtual Alignment alignment() const {
    return Alignment(2);
  }

  virtual uint64_t size() const {
    return 4;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permRW_;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t bytes[] = { 0x00, 0x00, 0x00, 0x00 };
    return makeArrayRef(bytes);
  }
};


//
// X86 NonLazy (GOT) Pointer Atom created by the stubs pass.
//
class X86NonLazyPointerAtom : public SimpleDefinedAtom {
public:
  X86NonLazyPointerAtom(const File &file)
      : SimpleDefinedAtom(file) {}

  X86NonLazyPointerAtom(const File &file, const Atom &shlib)
      : SimpleDefinedAtom(file) {
    this->addReference(KindHandler_x86::pointer32, 0, &shlib, 0);
  }

  virtual ContentType contentType() const {
    return DefinedAtom::typeGOT;
  }

  virtual Alignment alignment() const {
    return Alignment(2);
  }

  virtual uint64_t size() const {
    return 4;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permRW_;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    static const uint8_t bytes[] = { 0x00, 0x00, 0x00, 0x0 };
    return makeArrayRef(bytes);
  }
};

} // namespace mach_o
} // namespace lld


#endif // LLD_READER_WRITER_MACHO_STUB_ATOM_X86_H_
