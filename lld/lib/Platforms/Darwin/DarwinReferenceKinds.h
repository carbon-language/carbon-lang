//===- Platforms/Darwin/DarwinReferenceKinds.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"


#ifndef LLD_PLATFORM_DARWIN_REFERENCE_KINDS_H_
#define LLD_PLATFORM_DARWIN_REFERENCE_KINDS_H_

namespace lld {
namespace darwin {


class ReferenceKind {
public:
  enum {
    none        = 0,
    call32      = 1,
    pcRel32     = 2,
    gotLoad32   = 3,
    gotUse32    = 4,
    lea32WasGot = 5,
    lazyTarget  = 6,
    lazyImm     = 7,
    gotTarget   = 8,
    pointer64   = 9,
  };

  static Reference::Kind fromString(StringRef kindName); 
  
  static StringRef toString(Reference::Kind kindValue);
  
  static bool isCallSite(Reference::Kind kindValue);
  
  static bool isRipRel32(Reference::Kind kindValue);
};



} // namespace darwin
} // namespace lld



#endif // LLD_PLATFORM_DARWIN_REFERENCE_KINDS_H_

