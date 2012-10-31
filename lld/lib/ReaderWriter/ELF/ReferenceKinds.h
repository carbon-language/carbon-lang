//===- lib/ReaderWriter/ELF/ReferenceKinds.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"

#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/ReaderWriter/WriterELF.h"

#include <functional>
#include <map>
#include <memory>

#ifndef LLD_READER_WRITER_ELF_REFERENCE_KINDS_H_
#define LLD_READER_WRITER_ELF_REFERENCE_KINDS_H_

namespace lld {
namespace elf {


///
/// The KindHandler class is the abstract interface to Reference::Kind
/// values for ELF files.  Particular Kind values (e.g. 3) has a different
/// meaning for each architecture.
/// TODO: Needs to be updated for ELF, stubs for now.
///
class KindHandler {
public:
  typedef Reference::Kind Kind;
  
  static std::unique_ptr<KindHandler> makeHandler(uint16_t arch,
                                      llvm::support::endianness endian);
  virtual             ~KindHandler();
  virtual Kind        stringToKind(StringRef str) = 0;
  virtual StringRef   kindToString(Kind) = 0;
  virtual bool        isCallSite(Kind) = 0;
  virtual bool        isPointer(Kind) = 0; 
  virtual bool        isLazyImmediate(Kind) = 0; 
  virtual bool        isLazyTarget(Kind) = 0; 
  virtual void        applyFixup(int32_t reloc, uint64_t addend,
                                 uint8_t *location,
                                 uint64_t fixupAddress,
                                 uint64_t targetAddress) = 0;
  
protected:
  KindHandler();
};


class HexagonKindHandler : public KindHandler {
public:

// Note: Reference::Kinds are a another representation of
// relocation types, using negative values to represent architecture
// independent reference type.
// The positive values are the same ones defined in ELF.h and that
// is what we are using.
  enum Kinds {
    none            = llvm::ELF::R_HEX_NONE,
    invalid=255,         // used to denote an error creating a Reference
  };

  enum RelocationError {
    NoError,
    Overflow
  };

  virtual ~HexagonKindHandler();
  HexagonKindHandler();
  virtual Kind stringToKind(StringRef str);
  virtual StringRef kindToString(Kind);
  virtual bool isCallSite(Kind);
  virtual bool isPointer(Kind); 
  virtual bool isLazyImmediate(Kind); 
  virtual bool isLazyTarget(Kind); 
  virtual void applyFixup(int32_t reloc, uint64_t addend,
                          uint8_t *location,
                          uint64_t fixupAddress, uint64_t targetAddress);

// A map is used here and in the other handlers but if performace overhead
// becomes an issue this could be implemented as an array of function pointers.
private:
  llvm::DenseMap<int32_t,
           std::function <int (uint8_t *location, uint64_t fixupAddress,
                      uint64_t targetAddress, uint64_t addend)> > _fixupHandler;

};


class X86KindHandler : public KindHandler {
public:
  enum Kinds {
    invalid,         // used to denote an error creating a Reference
    none,
  };

  enum RelocationError {
    NoError,
  };

  virtual ~X86KindHandler();
  X86KindHandler();
  virtual Kind stringToKind(StringRef str);
  virtual StringRef kindToString(Kind);
  virtual bool isCallSite(Kind);
  virtual bool isPointer(Kind); 
  virtual bool isLazyImmediate(Kind); 
  virtual bool isLazyTarget(Kind); 
  virtual void applyFixup(int32_t reloc, uint64_t addend, uint8_t *location,
                          uint64_t fixupAddress, uint64_t targetAddress);

private:
  llvm::DenseMap<int32_t,
           std::function <int (uint8_t *location, uint64_t fixupAddress,
                      uint64_t targetAddress, uint64_t addend)> > _fixupHandler;
};

class PPCKindHandler : public KindHandler {
public:

// Note: Reference::Kinds are a another representation of
// relocation types, using negative values to represent architecture
// independent reference type.
// The positive values are the same ones defined in ELF.h and that
// is what we are using.
  enum Kinds {
    none            = llvm::ELF::R_PPC_NONE,
    invalid=255,         // used to denote an error creating a Reference
  };

  enum RelocationError {
    NoError,
    Overflow
  };

  virtual ~PPCKindHandler();
  PPCKindHandler(llvm::support::endianness endian);
  virtual Kind stringToKind(StringRef str);
  virtual StringRef kindToString(Kind);
  virtual bool isCallSite(Kind);
  virtual bool isPointer(Kind); 
  virtual bool isLazyImmediate(Kind); 
  virtual bool isLazyTarget(Kind); 
  virtual void applyFixup(int32_t reloc, uint64_t addend,
                          uint8_t *location,
                          uint64_t fixupAddress, uint64_t targetAddress);

private:
  llvm::DenseMap<int32_t,
           std::function <int (uint8_t *location, uint64_t fixupAddress,
                      uint64_t targetAddress, uint64_t addend)> > _fixupHandler;

};

} // namespace elf
} // namespace lld



#endif // LLD_READER_WRITER_ELF_REFERENCE_KINDS_H_

