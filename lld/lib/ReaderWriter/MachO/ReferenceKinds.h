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

#include "llvm/ADT/Triple.h"

#ifndef LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H_
#define LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H_

namespace lld {
namespace mach_o {


///
/// The KindHandler class is the abstract interface to Reference::Kind
/// values for mach-o files.  Particular Kind values (e.g. 3) has a different
/// meaning for each architecture.
///
class KindHandler {
public:
  typedef Reference::Kind Kind;

  static KindHandler *makeHandler(llvm::Triple::ArchType arch);
  virtual             ~KindHandler();
  virtual Kind        stringToKind(StringRef str) = 0;
  virtual StringRef   kindToString(Kind) = 0;
  virtual bool        isCallSite(Kind) = 0;
  virtual bool        isPointer(Kind) = 0;
  virtual bool        isLazyImmediate(Kind) = 0;
  virtual bool        isLazyTarget(Kind) = 0;
  virtual void        applyFixup(Kind kind, uint64_t addend, uint8_t *location,
                           uint64_t fixupAddress, uint64_t targetAddress) = 0;

protected:
  KindHandler();
};



class KindHandler_x86_64 : public KindHandler {
public:
  enum Kinds {
    invalid,         // used to denote an error creating a Reference
    none,
    branch32,        // CALL or JMP 32-bit pc-rel
    ripRel32,        // RIP-rel access pc-rel to fix up location
    ripRel32_1,      // RIP-rel access pc-rel to fix up location + 1
    ripRel32_2,      // RIP-rel access pc-rel to fix up location + 2
    ripRel32_4,      // RIP-rel access pc-rel to fix up location + 4
    gotLoadRipRel32, // RIP-rel load of GOT slot (can be optimized)
    gotLoadRipRel32NowLea, // RIP-rel movq load of GOT slot optimized to LEA
    gotUseRipRel32,  // RIP-rel non-load of GOT slot (not a movq load of GOT)
    tlvLoadRipRel32, // RIP-rel load of thread local pointer (can be optimized)
    tlvLoadRipRel32NowLea, // RIP-rel movq load of TLV pointer optimized to LEA
    pointer64,       // 64-bit data pointer
    pointerRel32,    // 32-bit pc-rel offset
    lazyTarget,      // Used in lazy pointers to reference ultimate target
    lazyImmediate,   // Location in stub where lazy info offset to be stored
    subordinateFDE,  // Reference to unwind info for this function
    subordinateLSDA  // Reference to excecption info for this function
  };

  virtual ~KindHandler_x86_64();
  virtual Kind stringToKind(StringRef str);
  virtual StringRef kindToString(Kind);
  virtual bool isCallSite(Kind);
  virtual bool isPointer(Kind);
  virtual bool isLazyImmediate(Kind);
  virtual bool isLazyTarget(Kind);
  virtual void applyFixup(Kind kind, uint64_t addend, uint8_t *location,
                  uint64_t fixupAddress, uint64_t targetAddress);

};


class KindHandler_x86 : public KindHandler {
public:
  enum Kinds {
    invalid,         // used to denote an error creating a Reference
    none,
    branch32,        // CALL or JMP 32-bit pc-rel
    abs32,           // 32-bit absolute address embedded in instruction
    funcRel32,       // 32-bit offset to target from start of function
    pointer32,       // 32-bit data pointer
    lazyTarget,      // Used in lazy pointers to reference ultimate target
    lazyImmediate,   // Location in stub where lazy info offset to be stored
    subordinateFDE,  // Reference to unwind info for this function
    subordinateLSDA  // Reference to excecption info for this function
  };

  virtual ~KindHandler_x86();
  virtual Kind stringToKind(StringRef str);
  virtual StringRef kindToString(Kind);
  virtual bool isCallSite(Kind);
  virtual bool isPointer(Kind);
  virtual bool isLazyImmediate(Kind);
  virtual bool isLazyTarget(Kind);
  virtual void applyFixup(Kind kind, uint64_t addend, uint8_t *location,
                  uint64_t fixupAddress, uint64_t targetAddress);

};

class KindHandler_arm : public KindHandler {
public:
  enum Kinds {
    invalid,         // used to denote an error creating a Reference
    none,
    thumbBranch22,   // thumb b or bl with 22/24-bits of displacement
    armBranch24,     // arm b or bl with 24-bits of displacement
    thumbAbsLow16,   // thumb movw of absolute address
    thumbAbsHigh16,  // thumb movt of absolute address
    thumbPcRelLow16, // thumb movw of (target - pc)
    thumbPcRelHigh16,// thumb movt of (target - pc)
    abs32,           // 32-bit absolute address embedded in instructions
    pointer32,       // 32-bit data pointer
    lazyTarget,      // Used in lazy pointers to reference ultimate target
    lazyImmediate,   // Location in stub where lazy info offset to be stored
    subordinateLSDA  // Reference to excecption info for this function
  };

  virtual ~KindHandler_arm();
  virtual Kind stringToKind(StringRef str);
  virtual StringRef kindToString(Kind);
  virtual bool isCallSite(Kind);
  virtual bool isPointer(Kind);
  virtual bool isLazyImmediate(Kind);
  virtual bool isLazyTarget(Kind);
  virtual void applyFixup(Kind kind, uint64_t addend, uint8_t *location,
                  uint64_t fixupAddress, uint64_t targetAddress);

};



} // namespace mach_o
} // namespace lld



#endif // LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H_

