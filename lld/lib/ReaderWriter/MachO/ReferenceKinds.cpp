//===- lib/FileFormat/MachO/ReferenceKinds.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ReferenceKinds.h"

#include "llvm/ADT/StringRef.h"


namespace lld {
namespace mach_o {

  
struct Mapping {
  const char*           string;
  Reference::Kind       value;
  uint32_t              flags;
};

enum {
  flagsNone       = 0x0000,
  flagsIsCallSite = 0x0001,
  flagsUsesGOT    = 0x0002,
  flagsisGOTLoad  = 0x0006,
  flags32RipRel   = 0x1000,
};


static const Mapping sKindMappingsx86_64[] = {
  { "none",           ReferenceKind::x86_64_none,        flagsNone },
  { "call32",         ReferenceKind::x86_64_call32,      flagsIsCallSite | flags32RipRel },
  { "pcrel32",        ReferenceKind::x86_64_pcRel32,     flags32RipRel },
  { "gotLoad32",      ReferenceKind::x86_64_gotLoad32,   flagsisGOTLoad | flags32RipRel },
  { "gotUse32",       ReferenceKind::x86_64_gotUse32,    flagsUsesGOT | flags32RipRel },
  { "lea32wasGot",    ReferenceKind::x86_64_lea32WasGot, flags32RipRel },
  { "lazyTarget",     ReferenceKind::x86_64_lazyTarget,  flagsNone },
  { "lazyImm",        ReferenceKind::x86_64_lazyImm,     flagsNone },
  { "gotTarget",      ReferenceKind::x86_64_gotTarget,   flagsNone },
  { "pointer64",      ReferenceKind::x86_64_pointer64,   flagsNone },
  { NULL,             ReferenceKind::x86_64_none,        flagsNone }
};


Reference::Kind ReferenceKind::fromString(StringRef kindName) {
  for (const Mapping* p = sKindMappingsx86_64; p->string != NULL; ++p) {
    if ( kindName.equals(p->string) )
      return p->value;
  }
  assert(0 && "unknown darwin reference kind");
  return 0;
}

StringRef ReferenceKind::toString(Reference::Kind kindValue) {
  for (const Mapping* p = sKindMappingsx86_64; p->string != NULL; ++p) {
    if ( kindValue == p->value)
      return p->string;
  }
  return StringRef("???");
}

static const Mapping* mappingsForArch(WriterOptionsMachO::Architecture arch) {
 switch ( arch ) {
    case WriterOptionsMachO::arch_x86_64:
      return sKindMappingsx86_64;
    case WriterOptionsMachO::arch_x86:
    case WriterOptionsMachO::arch_arm:
      assert(0 && "references table not yet implemented for arch");
      return nullptr;
  }
}

bool ReferenceKind::isCallSite(WriterOptionsMachO::Architecture arch, 
                                                    Reference::Kind kindValue) {
   for (const Mapping* p = mappingsForArch(arch); p->string != NULL; ++p) {
    if ( kindValue == p->value )
      return (p->flags & flagsIsCallSite);
  }
  return false;
}

bool ReferenceKind::isRipRel32(Reference::Kind kindValue) {
  for (const Mapping* p = sKindMappingsx86_64; p->string != NULL; ++p) {
    if ( kindValue == p->value )
      return (p->flags & flags32RipRel);
  }
  return false;
}





} // namespace mach_o
} // namespace lld



