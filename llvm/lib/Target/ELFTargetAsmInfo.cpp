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

  ReadOnlySection = getNamedSection("\t.rodata", SectionKind::ReadOnly);
  TLSDataSection = getNamedSection("\t.tdata", SectionKind::ThreadData);
  TLSBSSSection = getNamedSection("\t.tbss", SectionKind::ThreadBSS);

  DataRelSection = getNamedSection("\t.data.rel", SectionKind::DataRel);
  DataRelLocalSection = getNamedSection("\t.data.rel.local",
                                        SectionKind::DataRelLocal);
  DataRelROSection = getNamedSection("\t.data.rel.ro",
                                     SectionKind::ReadOnlyWithRel);
  DataRelROLocalSection = getNamedSection("\t.data.rel.ro.local",
                                          SectionKind::ReadOnlyWithRelLocal);
    
  MergeableConst4Section = getNamedSection(".rodata.cst4",
                                           SectionKind::MergeableConst4);
  MergeableConst8Section = getNamedSection(".rodata.cst8",
                                           SectionKind::MergeableConst8);
  MergeableConst16Section = getNamedSection(".rodata.cst16",
                                            SectionKind::MergeableConst16);
}


const Section*
ELFTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV,
                                         SectionKind Kind) const {
  if (Kind.isText()) return TextSection;
  if (Kind.isMergeableCString())
    return MergeableStringSection(cast<GlobalVariable>(GV));
  
  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return MergeableConst4Section;
    if (Kind.isMergeableConst8())
      return MergeableConst8Section;
    if (Kind.isMergeableConst16())
      return MergeableConst16Section;
    return ReadOnlySection;  // .const
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

/// getSectionForMergeableConstant - Given a Mergeable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const Section *
ELFTargetAsmInfo::getSectionForMergeableConstant(SectionKind Kind) const {
  return SelectSectionForGlobal(0, Kind);
}

/// getFlagsForNamedSection - If this target wants to be able to infer
/// section flags based on the name of the section specified for a global
/// variable, it can implement this.
SectionKind::Kind ELFTargetAsmInfo::getKindForNamedSection(const char *Name,
                                                   SectionKind::Kind K) const {
  if (Name[0] != '.') return K;
  
  // Some lame default implementation based on some magic section names.
  if (strncmp(Name, ".gnu.linkonce.b.", 16) == 0 ||
      strncmp(Name, ".llvm.linkonce.b.", 17) == 0 ||
      strncmp(Name, ".gnu.linkonce.sb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.sb.", 18) == 0)
    return SectionKind::BSS;
  
  if (strcmp(Name, ".tdata") == 0 ||
      strncmp(Name, ".tdata.", 7) == 0 ||
      strncmp(Name, ".gnu.linkonce.td.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.td.", 18) == 0)
    return SectionKind::ThreadData;
  
  if (strcmp(Name, ".tbss") == 0 ||
      strncmp(Name, ".tbss.", 6) == 0 ||
      strncmp(Name, ".gnu.linkonce.tb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.tb.", 18) == 0)
    return SectionKind::ThreadBSS;
  
  return K;
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

    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the alignment
    // of the string!!
    unsigned Align = TD->getPrefTypeAlignment(Ty);
    if (Align < Size)
      Align = Size;

    std::string Name = getCStringSection() + utostr(Size) + '.' + utostr(Align);
    return getNamedSection(Name.c_str(), SectionKind::MergeableCString);
  }

  return getReadOnlySection();
}

void ELFTargetAsmInfo::getSectionFlagsAsString(SectionKind Kind,
                                             SmallVectorImpl<char> &Str) const {
  Str.push_back(',');
  Str.push_back('"');
  
  if (!Kind.isMetadata())
    Str.push_back('a');
  if (Kind.isText())
    Str.push_back('x');
  if (Kind.isWriteable())
    Str.push_back('w');
  if (Kind.isMergeableConst() || Kind.isMergeableCString())
    Str.push_back('M');
  if (Kind.isMergeableCString())
    Str.push_back('S');
  if (Kind.isThreadLocal())
    Str.push_back('T');

  Str.push_back('"');
  Str.push_back(',');

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (strcmp(CommentString, "@") == 0)
    Str.push_back('%');
  else
    Str.push_back('@');

  const char *KindStr;
  if (Kind.isBSS())
    KindStr = "nobits";
  else
    KindStr = "progbits";
  
  Str.append(KindStr, KindStr+strlen(KindStr));

  if (Kind.isMergeableCString()) {
    // TODO: Eventually handle multiple byte character strings.  For now, all
    // mergable C strings are single byte.
    Str.push_back(',');
    Str.push_back('1');
  } else if (Kind.isMergeableConst4()) {
    Str.push_back(',');
    Str.push_back('4');
  } else if (Kind.isMergeableConst8()) {
    Str.push_back(',');
    Str.push_back('8');
  } else if (Kind.isMergeableConst16()) {
    Str.push_back(',');
    Str.push_back('1');
    Str.push_back('6');
  }
}
