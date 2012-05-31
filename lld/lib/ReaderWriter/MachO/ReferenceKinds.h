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
#include "lld/ReaderWriter/WriterMachO.h"

#ifndef LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H_
#define LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H_

namespace lld {
namespace mach_o {


class ReferenceKind {
public:
  // x86_64 Reference Kinds
  enum {
    x86_64_none        = 0,
    x86_64_call32      = 1,
    x86_64_pcRel32     = 2,
    x86_64_gotLoad32   = 3,
    x86_64_gotUse32    = 4,
    x86_64_lea32WasGot = 5,
    x86_64_lazyTarget  = 6,
    x86_64_lazyImm     = 7,
    x86_64_gotTarget   = 8,
    x86_64_pointer64   = 9,
  };

  // x86 Reference Kinds
 enum {
    x86_none        = 0,
    x86_call32      = 1,
    x86_pointer32   = 2,
    x86_lazyTarget  = 3,
    x86_lazyImm     = 4,
    // FIXME
  };

  // ARM Reference Kinds
 enum {
    arm_none        = 0,
    arm_br22        = 1,
    arm_pointer32   = 2,
    arm_lazyTarget  = 3,
    arm_lazyImm     = 4,
    // FIXME
  };

  static bool isCallSite(WriterOptionsMachO::Architecture arch, 
                                                    Reference::Kind kindValue);
  
  static bool isRipRel32(Reference::Kind kindValue);


  static Reference::Kind fromString(StringRef kindName); 
  static StringRef toString(Reference::Kind kindValue);
  
};



} // namespace mach_o
} // namespace lld



#endif // LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H_

