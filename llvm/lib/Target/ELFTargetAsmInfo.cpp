//===-- ELFTargetAsmInfo.cpp - ELF asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on ELF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

ELFTargetAsmInfo::ELFTargetAsmInfo(const TargetMachine &TM)
  : TargetAsmInfo(TM) {

  BSSSection_  = getUnnamedSection("\t.bss",
                                   SectionFlags::Writable | SectionFlags::BSS);
  ReadOnlySection = getNamedSection("\t.rodata", SectionFlags::None);
  TLSDataSection = getNamedSection("\t.tdata",
                                   SectionFlags::Writable | SectionFlags::TLS);
  TLSBSSSection = getNamedSection("\t.tbss",
                SectionFlags::Writable | SectionFlags::TLS | SectionFlags::BSS);

  DataRelSection = getNamedSection("\t.data.rel", SectionFlags::Writable);
  DataRelLocalSection = getNamedSection("\t.data.rel.local",
                                        SectionFlags::Writable);
  DataRelROSection = getNamedSection("\t.data.rel.ro",
                                     SectionFlags::Writable);
  DataRelROLocalSection = getNamedSection("\t.data.rel.ro.local",
                                          SectionFlags::Writable);
}


const Section*
ELFTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV,
                                         SectionKind Kind) const {
  if (Kind.isText()) return TextSection;
  if (Kind.isMergableCString())
    return MergeableStringSection(cast<GlobalVariable>(GV));
  if (Kind.isMergableConst()) {
    const Type *Ty = cast<GlobalVariable>(GV)->getInitializer()->getType();
    const TargetData *TD = TM.getTargetData();
    return getSectionForMergableConstant(TD->getTypeAllocSize(Ty), 0);
  }
  if (Kind.isReadOnly())             return getReadOnlySection();
  
  
  if (Kind.isThreadData())           return TLSDataSection;
  if (Kind.isThreadBSS())            return TLSBSSSection;

  if (Kind.isBSS())                  return getBSSSection_();
  
  
  if (Kind.isDataNoRel())            return DataSection;
  if (Kind.isDataRelLocal())         return DataRelLocalSection;
  if (Kind.isDataRel())              return DataRelSection;
  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;
  
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

/// getSectionForMergableConstant - Given a mergable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const Section *
ELFTargetAsmInfo::getSectionForMergableConstant(uint64_t Size,
                                                unsigned ReloInfo) const {
  // FIXME: IF this global requires a relocation, can we really put it in
  // rodata???  This should check ReloInfo like darwin.
  
  const char *SecName = 0;
  switch (Size) {
  default: break;
  case 4:  SecName = ".rodata.cst4"; break;
  case 8:  SecName = ".rodata.cst8"; break;
  case 16: SecName = ".rodata.cst16"; break;
  }
  
  if (SecName)
    return getNamedSection(SecName,
                           SectionFlags::setEntitySize(SectionFlags::Mergeable,
                                                       Size));
  
  return getReadOnlySection();  // .rodata
}

/// getFlagsForNamedSection - If this target wants to be able to infer
/// section flags based on the name of the section specified for a global
/// variable, it can implement this.
unsigned ELFTargetAsmInfo::getFlagsForNamedSection(const char *Name) const {
  unsigned Flags = 0;
  if (Name[0] != '.') return 0;
  
  // Some lame default implementation based on some magic section names.
  if (strncmp(Name, ".gnu.linkonce.b.", 16) == 0 ||
      strncmp(Name, ".llvm.linkonce.b.", 17) == 0 ||
      strncmp(Name, ".gnu.linkonce.sb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.sb.", 18) == 0)
    Flags |= SectionFlags::BSS;
  else if (strcmp(Name, ".tdata") == 0 ||
           strncmp(Name, ".tdata.", 7) == 0 ||
           strncmp(Name, ".gnu.linkonce.td.", 17) == 0 ||
           strncmp(Name, ".llvm.linkonce.td.", 18) == 0)
    Flags |= SectionFlags::TLS;
  else if (strcmp(Name, ".tbss") == 0 ||
           strncmp(Name, ".tbss.", 6) == 0 ||
           strncmp(Name, ".gnu.linkonce.tb.", 17) == 0 ||
           strncmp(Name, ".llvm.linkonce.tb.", 18) == 0)
    Flags |= SectionFlags::BSS | SectionFlags::TLS;
  
  return Flags;
}



const char *
ELFTargetAsmInfo::getSectionPrefixForUniqueGlobal(SectionKind Kind) const{
  if (Kind.isText())                 return ".gnu.linkonce.t.";
  if (Kind.isReadOnly())             return ".gnu.linkonce.r.";
  
  if (Kind.isThreadData())           return ".gnu.linkonce.td.";
  if (Kind.isThreadBSS())            return ".gnu.linkonce.tb.";

  if (Kind.isBSS())                  return ".gnu.linkonce.b.";
  if (Kind.isDataNoRel())            return ".gnu.linkonce.d.";
  if (Kind.isDataRelLocal())         return ".gnu.linkonce.d.rel.local.";
  if (Kind.isDataRel())              return ".gnu.linkonce.d.rel.";
  if (Kind.isReadOnlyWithRelLocal()) return ".gnu.linkonce.d.rel.ro.local.";
  
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".gnu.linkonce.d.rel.ro.";
}



const Section*
ELFTargetAsmInfo::MergeableStringSection(const GlobalVariable *GV) const {
  const TargetData *TD = TM.getTargetData();
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const Type *Ty = cast<ArrayType>(C->getType())->getElementType();

  unsigned Size = TD->getTypeAllocSize(Ty);
  if (Size <= 16) {
    assert(getCStringSection() && "Should have string section prefix");

    // We also need alignment here
    unsigned Align = TD->getPrefTypeAlignment(Ty);
    if (Align < Size)
      Align = Size;

    std::string Name = getCStringSection() + utostr(Size) + '.' + utostr(Align);
    unsigned Flags = SectionFlags::setEntitySize(SectionFlags::Mergeable |
                                                 SectionFlags::Strings,
                                                 Size);
    return getNamedSection(Name.c_str(), Flags);
  }

  return getReadOnlySection();
}

std::string ELFTargetAsmInfo::printSectionFlags(unsigned flags) const {
  std::string Flags = ",\"";

  if (!(flags & SectionFlags::Debug))
    Flags += 'a';
  if (flags & SectionFlags::Code)
    Flags += 'x';
  if (flags & SectionFlags::Writable)
    Flags += 'w';
  if (flags & SectionFlags::Mergeable)
    Flags += 'M';
  if (flags & SectionFlags::Strings)
    Flags += 'S';
  if (flags & SectionFlags::TLS)
    Flags += 'T';

  Flags += "\",";

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (strcmp(CommentString, "@") == 0)
    Flags += '%';
  else
    Flags += '@';

  // FIXME: There can be exceptions here
  if (flags & SectionFlags::BSS)
    Flags += "nobits";
  else
    Flags += "progbits";

  if (unsigned entitySize = SectionFlags::getEntitySize(flags))
    Flags += "," + utostr(entitySize);

  return Flags;
}
