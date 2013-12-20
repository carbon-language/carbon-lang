//===- lib/FileFormat/MachO/ReferenceKinds.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"

#include "llvm/ADT/Triple.h"

#ifndef LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H
#define LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H

namespace lld {
namespace mach_o {

// Additional Reference Kind values used internally.
enum {
  LLD_X86_64_RELOC_GOT_LOAD_NOW_LEA = 100,
  LLD_X86_64_RELOC_TLV_NOW_LEA      = 101,
  LLD_X86_64_RELOC_LAZY_TARGET      = 102,
  LLD_X86_64_RELOC_LAZY_IMMEDIATE   = 103,
  LLD_X86_64_RELOC_SIGNED_32        = 104,
};
enum {
  LLD_X86_RELOC_BRANCH32       = 100, // CALL or JMP 32-bit pc-rel
  LLD_X86_RELOC_ABS32          = 101, // 32-bit absolute addr in instruction
  LLD_X86_RELOC_FUNC_REL32     = 102, // 32-bit target from start of func
  LLD_X86_RELOC_POINTER32      = 103, // 32-bit data pointer
  LLD_X86_RELOC_LAZY_TARGET    = 104,
  LLD_X86_RELOC_LAZY_IMMEDIATE = 105
};
enum {
  LLD_ARM_RELOC_THUMB_ABS_LO16 = 100, // thumb movw of absolute address
  LLD_ARM_RELOC_THUMB_ABS_HI16 = 101, // thumb movt of absolute address
  LLD_ARM_RELOC_THUMB_REL_LO16 = 102, // thumb movw of (target - pc)
  LLD_ARM_RELOC_THUMB_REL_HI16 = 103, // thumb movt of (target - pc)
  LLD_ARM_RELOC_ABS32          = 104, // 32-bit constant pointer
  LLD_ARM_RELOC_POINTER32      = 105, // 32-bit data pointer
  LLD_ARM_RELOC_LAZY_TARGET    = 106,
  LLD_ARM_RELOC_LAZY_IMMEDIATE = 107,
};

///
/// The KindHandler class is the abstract interface to Reference::Kind
/// values for mach-o files.  Particular Kind values (e.g. 3) has a different
/// meaning for each architecture.
///
class KindHandler {
public:

  static std::unique_ptr<mach_o::KindHandler> create(MachOLinkingContext::Arch);
  virtual ~KindHandler();

  virtual bool isCallSite(const Reference &) = 0;
  virtual bool isPointer(const Reference &) = 0;
  virtual bool isLazyImmediate(const Reference &) = 0;
  virtual bool isLazyTarget(const Reference &) = 0;
  virtual void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress) = 0;

protected:
  KindHandler();
};



class KindHandler_x86_64 : public KindHandler {
public:
  static const Registry::KindStrings kindStrings[];

  virtual ~KindHandler_x86_64();
  virtual bool isCallSite(const Reference &);
  virtual bool isPointer(const Reference &);
  virtual bool isLazyImmediate(const Reference &);
  virtual bool isLazyTarget(const Reference &);
  virtual void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress);
};


class KindHandler_x86 : public KindHandler {
public:
  static const Registry::KindStrings kindStrings[];

  virtual ~KindHandler_x86();
  virtual bool isCallSite(const Reference &);
  virtual bool isPointer(const Reference &);
  virtual bool isLazyImmediate(const Reference &);
  virtual bool isLazyTarget(const Reference &);
  virtual void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress);
};

class KindHandler_arm : public KindHandler {
public:
  static const Registry::KindStrings kindStrings[];

  virtual ~KindHandler_arm();
  virtual bool isCallSite(const Reference &);
  virtual bool isPointer(const Reference &);
  virtual bool isLazyImmediate(const Reference &);
  virtual bool isLazyTarget(const Reference &);
  virtual void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress);
};



} // namespace mach_o
} // namespace lld



#endif // LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H

