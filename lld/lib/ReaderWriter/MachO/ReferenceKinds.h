//===- lib/FileFormat/MachO/ReferenceKinds.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "MachONormalizedFile.h"

#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"

#include "llvm/ADT/Triple.h"

#ifndef LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H
#define LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H

namespace lld {
namespace mach_o {


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
  
  /// Returns true if the specified relocation is paired to the next relocation. 
  virtual bool isPairedReloc(const normalized::Relocation &) = 0;
  
  /// Prototype for a helper function.  Given a sectionIndex and address, 
  /// finds the atom and offset with that atom of that address. 
  typedef std::function<std::error_code (uint32_t sectionIndex, uint64_t addr, 
                        const lld::Atom **, Reference::Addend *)> 
                        FindAtomBySectionAndAddress;
  
  /// Prototype for a helper function.  Given a symbolIndex, finds the atom
  /// representing that symbol.
  typedef std::function<std::error_code (uint32_t symbolIndex, 
                        const lld::Atom **)> FindAtomBySymbolIndex;
  
  /// Analyzes a relocation from a .o file and returns the info
  /// (kind, target, addend) needed to instantiate a Reference.
  /// Two helper functions are passed as parameters to find the target atom
  /// given a symbol index or address.
  virtual std::error_code 
          getReferenceInfo(const normalized::Relocation &reloc,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind, 
                           const lld::Atom **target, 
                           Reference::Addend *addend) = 0;

  /// Analyzes a pair of relocations from a .o file and returns the info
  /// (kind, target, addend) needed to instantiate a Reference.
  /// Two helper functions are passed as parameters to find the target atom
  /// given a symbol index or address.
  virtual std::error_code 
      getPairReferenceInfo(const normalized::Relocation &reloc1,
                           const normalized::Relocation &reloc2,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind, 
                           const lld::Atom **target, 
                           Reference::Addend *addend) = 0;
                           
  /// Fixup an atom when generating a final linked binary.
  virtual void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress, uint64_t inAtomAddress) = 0;

protected:
  KindHandler();
  
  // Handy way to pack mach-o r_type and other bit fields into one 16-bit value.
  typedef uint16_t RelocPattern;
  enum {
    rScattered = 0x8000,
    rPcRel     = 0x4000,
    rExtern    = 0x2000,
    rLength1   = 0x0000,
    rLength2   = 0x0100,
    rLength4   = 0x0200,
    rLength8   = 0x0300
  };
  static RelocPattern relocPattern(const normalized::Relocation &reloc);

  static int16_t  readS16(bool swap, const uint8_t *addr);
  static int32_t  readS32(bool swap, const uint8_t *addr);
  static uint32_t readU32(bool swap, const uint8_t *addr);
  static int64_t  readS64(bool swap, const uint8_t *addr);  
};



class KindHandler_x86_64 : public KindHandler {
public:
  static const Registry::KindStrings kindStrings[];

  virtual ~KindHandler_x86_64();
  bool isCallSite(const Reference &) override;
  bool isPointer(const Reference &) override;
  bool isLazyImmediate(const Reference &) override;
  bool isLazyTarget(const Reference &) override;
  bool isPairedReloc(const normalized::Relocation &) override;
  std::error_code getReferenceInfo(const normalized::Relocation &reloc,
                                   const DefinedAtom *inAtom,
                                   uint32_t offsetInAtom,
                                   uint64_t fixupAddress, bool swap,
                                   FindAtomBySectionAndAddress atomFromAddress,
                                   FindAtomBySymbolIndex atomFromSymbolIndex,
                                   Reference::KindValue *kind, 
                                   const lld::Atom **target, 
                                   Reference::Addend *addend) override;
  std::error_code 
      getPairReferenceInfo(const normalized::Relocation &reloc1,
                           const normalized::Relocation &reloc2,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind, 
                           const lld::Atom **target, 
                           Reference::Addend *addend) override;
                           
  virtual void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress, uint64_t inAtomAddress) 
                          override;

private:
  friend class X86_64LazyPointerAtom;
  friend class X86_64StubHelperAtom;
  friend class X86_64StubAtom;
  friend class X86_64StubHelperCommonAtom;
  friend class X86_64NonLazyPointerAtom;
 
  enum : Reference::KindValue {
    invalid,               /// for error condition
    
    // Kinds found in mach-o .o files:
    branch32,              /// ex: call _foo
    ripRel32,              /// ex: movq _foo(%rip), %rax
    ripRel32Minus1,        /// ex: movb $0x12, _foo(%rip)
    ripRel32Minus2,        /// ex: movw $0x1234, _foo(%rip)
    ripRel32Minus4,        /// ex: movl $0x12345678, _foo(%rip)
    ripRel32Anon,          /// ex: movq L1(%rip), %rax
    ripRel32GotLoad,       /// ex: movq  _foo@GOTPCREL(%rip), %rax
    ripRel32Got,           /// ex: pushq _foo@GOTPCREL(%rip)
    pointer64,             /// ex: .quad _foo
    pointer64Anon,         /// ex: .quad L1
    delta64,               /// ex: .quad _foo - .
    delta32,               /// ex: .long _foo - .
    delta64Anon,           /// ex: .quad L1 - .
    delta32Anon,           /// ex: .long L1 - .
    
    // Kinds introduced by Passes:
    ripRel32GotLoadNowLea, /// Target of GOT load is in linkage unit so 
                           ///  "movq  _foo@GOTPCREL(%rip), %rax" can be changed
                           /// to "leaq _foo(%rip), %rax
    lazyPointer,           /// Location contains a lazy pointer.
    lazyImmediateLocation, /// Location contains immediate value used in stub.
  };
  
  Reference::KindValue kindFromReloc(const normalized::Relocation &reloc);
  Reference::KindValue kindFromRelocPair(const normalized::Relocation &reloc1,
                                         const normalized::Relocation &reloc2);


};

class KindHandler_x86 : public KindHandler {
public:
  static const Registry::KindStrings kindStrings[];

  virtual ~KindHandler_x86();
  bool isCallSite(const Reference &) override;
  bool isPointer(const Reference &) override;
  bool isLazyImmediate(const Reference &) override;
  bool isLazyTarget(const Reference &) override;
  bool isPairedReloc(const normalized::Relocation &) override;
  std::error_code getReferenceInfo(const normalized::Relocation &reloc,
                                   const DefinedAtom *inAtom,
                                   uint32_t offsetInAtom,
                                   uint64_t fixupAddress, bool swap,
                                   FindAtomBySectionAndAddress atomFromAddress,
                                   FindAtomBySymbolIndex atomFromSymbolIndex,
                                   Reference::KindValue *kind,
                                   const lld::Atom **target,
                                   Reference::Addend *addend) override;
  std::error_code
      getPairReferenceInfo(const normalized::Relocation &reloc1,
                           const normalized::Relocation &reloc2,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind,
                           const lld::Atom **target,
                           Reference::Addend *addend) override;

  void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress, uint64_t inAtomAddress) 
                          override;

private:
  friend class X86LazyPointerAtom;
  friend class X86StubHelperAtom;
  friend class X86StubAtom;
  friend class X86StubHelperCommonAtom;
  friend class X86NonLazyPointerAtom;

  enum : Reference::KindValue {
    invalid,               /// for error condition

    // Kinds found in mach-o .o files:
    branch32,              /// ex: call _foo
    branch16,              /// ex: callw _foo
    abs32,                 /// ex: movl _foo, %eax
    funcRel32,             /// ex: movl _foo-L1(%eax), %eax
    pointer32,             /// ex: .long _foo
    delta32,               /// ex: .long _foo - .

    // Kinds introduced by Passes:
    lazyPointer,           /// Location contains a lazy pointer.
    lazyImmediateLocation, /// Location contains immediate value used in stub.
  };


};



class KindHandler_arm : public KindHandler {
public:
  static const Registry::KindStrings kindStrings[];

  virtual ~KindHandler_arm();
  bool isCallSite(const Reference &) override;
  bool isPointer(const Reference &) override;
  bool isLazyImmediate(const Reference &) override;
  bool isLazyTarget(const Reference &) override;
  bool isPairedReloc(const normalized::Relocation &) override;
  std::error_code getReferenceInfo(const normalized::Relocation &reloc,
                                   const DefinedAtom *inAtom,
                                   uint32_t offsetInAtom,
                                   uint64_t fixupAddress, bool swap,
                                   FindAtomBySectionAndAddress atomFromAddress,
                                   FindAtomBySymbolIndex atomFromSymbolIndex,
                                   Reference::KindValue *kind,
                                   const lld::Atom **target,
                                   Reference::Addend *addend) override;
  std::error_code
      getPairReferenceInfo(const normalized::Relocation &reloc1,
                           const normalized::Relocation &reloc2,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind,
                           const lld::Atom **target,
                           Reference::Addend *addend) override;

  void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                  Reference::KindValue kindValue, uint64_t addend,
                  uint8_t *location, uint64_t fixupAddress,
                  uint64_t targetAddress, uint64_t inAtomAddress) 
                  override;

private:
  enum : Reference::KindValue {
    invalid,               /// for error condition

    // Kinds found in mach-o .o files:
    thumb_b22,             /// ex: bl _foo
    thumb_movw,            /// ex: movw	r1, :lower16:_foo
    thumb_movt,            /// ex: movt	r1, :lower16:_foo
    thumb_movw_funcRel,    /// ex: movw	r1, :lower16:(_foo-(L1+4))
    thumb_movt_funcRel,    /// ex: movt r1, :upper16:(_foo-(L1+4))
    arm_b24,               /// ex: bl _foo
    arm_movw,              /// ex: movw	r1, :lower16:_foo
    arm_movt,              /// ex: movt	r1, :lower16:_foo
    arm_movw_funcRel,      /// ex: movw	r1, :lower16:(_foo-(L1+4))
    arm_movt_funcRel,      /// ex: movt r1, :upper16:(_foo-(L1+4))
    pointer32,             /// ex: .long _foo
    delta32,               /// ex: .long _foo - .

    // Kinds introduced by Passes:
    lazyPointer,           /// Location contains a lazy pointer.
    lazyImmediateLocation, /// Location contains immediate value used in stub.
  };

  int32_t getDisplacementFromThumbBranch(uint32_t instruction);
  int32_t getDisplacementFromArmBranch(uint32_t instruction);
  uint16_t getWordFromThumbMov(uint32_t instruction);
  uint16_t getWordFromArmMov(uint32_t instruction);
  uint32_t clearThumbBit(uint32_t value, const Atom* target);

};

} // namespace mach_o
} // namespace lld



#endif // LLD_READER_WRITER_MACHO_REFERENCE_KINDS_H

