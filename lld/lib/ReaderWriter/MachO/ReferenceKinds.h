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


///
/// The KindHandler class is the abstract interface to Reference::Kind
/// values for mach-o files.  Particular Kind values (e.g. 3) has a different
/// meaning for each architecture.
///
class KindHandler {
public:
  typedef Reference::Kind Kind;
  
  static KindHandler *makeHandler(WriterOptionsMachO::Architecture arch);
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
    none        = 0,
    call32      = 1,
    ripRel32    = 2,
    gotLoad32   = 3,
    gotUse32    = 4,
    lea32WasGot = 5,
    lazyTarget  = 6,
    lazyImm     = 7,
    gotTarget   = 8,
    pointer64   = 9
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
    none        = 0,
    call32      = 1,
    abs32       = 2,
    pointer32   = 3,
    lazyTarget  = 4,
    lazyImm     = 5
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
    none        = 0,
    br22        = 1,
    pointer32   = 2,
    lazyTarget  = 3,
    lazyImm     = 4
    // FIXME
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

