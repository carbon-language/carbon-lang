//===- Platforms/Darwin/DarwinReferenceKinds.cpp --------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "DarwinReferenceKinds.h"
#include "llvm/ADT/StringRef.h"


namespace lld {
namespace darwin {

  
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


static const Mapping sKindMappings[] = {
  { "call32",         ReferenceKind::call32,      flagsIsCallSite | flags32RipRel },
  { "pcrel32",        ReferenceKind::pcRel32,     flags32RipRel },
  { "gotLoad32",      ReferenceKind::gotLoad32,   flagsisGOTLoad | flags32RipRel },
  { "gotUse32",       ReferenceKind::gotUse32,    flagsUsesGOT | flags32RipRel },
  { "lea32wasGot",    ReferenceKind::lea32WasGot, flags32RipRel },
  { "lazyTarget",     ReferenceKind::lazyTarget,  flagsNone },
  { "lazyImm",        ReferenceKind::lazyImm,     flagsNone },
  { "gotTarget",      ReferenceKind::gotTarget,   flagsNone },
  { "pointer64",      ReferenceKind::pointer64,   flagsNone },
  { NULL,             ReferenceKind::none,        flagsNone }
};


Reference::Kind ReferenceKind::fromString(StringRef kindName) {
  for (const Mapping* p = sKindMappings; p->string != NULL; ++p) {
    if ( kindName.equals(p->string) )
      return p->value;
  }
  assert(0 && "unknown darwin reference kind");
  return ReferenceKind::none;
}

StringRef ReferenceKind::toString(Reference::Kind kindValue) {
  for (const Mapping* p = sKindMappings; p->string != NULL; ++p) {
    if ( kindValue == p->value)
      return p->string;
  }
  return StringRef("???");
}

bool ReferenceKind::isCallSite(Reference::Kind kindValue) {
  for (const Mapping* p = sKindMappings; p->string != NULL; ++p) {
    if ( kindValue == p->value )
      return (p->flags & flagsIsCallSite);
  }
  return false;
}

bool ReferenceKind::isRipRel32(Reference::Kind kindValue) {
  for (const Mapping* p = sKindMappings; p->string != NULL; ++p) {
    if ( kindValue == p->value )
      return (p->flags & flags32RipRel);
  }
  return false;
}





} // namespace darwin
} // namespace lld



