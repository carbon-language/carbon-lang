//===-- XCoreTargetObjectFile.cpp - XCore object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetObjectFile.h"
#include "XCoreSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;


void XCoreTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);

  BSSSection =
    Ctx.getELFSection(".dp.bss", ELF::SHT_NOBITS,
                      ELF::SHF_ALLOC | ELF::SHF_WRITE |
                      ELF::XCORE_SHF_DP_SECTION,
                      SectionKind::getBSS());
  BSSSectionLarge =
    Ctx.getELFSection(".dp.bss.large", ELF::SHT_NOBITS,
                      ELF::SHF_ALLOC | ELF::SHF_WRITE |
                      ELF::XCORE_SHF_DP_SECTION,
                      SectionKind::getBSS());
  DataSection =
    Ctx.getELFSection(".dp.data", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_WRITE |
                      ELF::XCORE_SHF_DP_SECTION,
                      SectionKind::getDataRel());
  DataSectionLarge =
    Ctx.getELFSection(".dp.data.large", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_WRITE |
                      ELF::XCORE_SHF_DP_SECTION,
                      SectionKind::getDataRel());
  // This is the wrong place to decide if const data should be placed
  // in the .cp or .dp section.
  // Ideally we should set up DataRelROSection to use the '.dp.'' and use this
  // for const data, unless the front end explicitly states a '.cp.'' section.
  ReadOnlySection =
    Ctx.getELFSection(".cp.rodata", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getReadOnlyWithRel());
  ReadOnlySectionLarge =
    Ctx.getELFSection(".cp.rodata.large", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getReadOnlyWithRel());
  MergeableConst4Section = 
    Ctx.getELFSection(".cp.rodata.cst4", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getMergeableConst4());
  MergeableConst8Section = 
    Ctx.getELFSection(".cp.rodata.cst8", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getMergeableConst8());
  MergeableConst16Section = 
    Ctx.getELFSection(".cp.rodata.cst16", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getMergeableConst16());
  CStringSection =
    Ctx.getELFSection(".cp.rodata.string", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getReadOnlyWithRel());
  // TextSection       - see MObjectFileInfo.cpp
  // StaticCtorSection - see MObjectFileInfo.cpp
  // StaticDtorSection - see MObjectFileInfo.cpp
 }

static SectionKind getXCoreKindForNamedSection(StringRef Name, SectionKind K) {
  if (Name.startswith(".cp."))
    return SectionKind::getReadOnly();
  return K;
}

static unsigned getXCoreSectionType(SectionKind K) {
  if (K.isBSS())
    return ELF::SHT_NOBITS;
  return ELF::SHT_PROGBITS;
}

static unsigned getXCoreSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (!K.isMetadata())
    Flags |= ELF::SHF_ALLOC;

  if (K.isText())
    Flags |= ELF::SHF_EXECINSTR;
  else if (K.isReadOnly())
    Flags |= ELF::XCORE_SHF_CP_SECTION;
  else
    Flags |= ELF::XCORE_SHF_DP_SECTION;

  if (K.isWriteable())
    Flags |= ELF::SHF_WRITE;

  if (K.isMergeableCString() || K.isMergeableConst4() ||
      K.isMergeableConst8() || K.isMergeableConst16())
    Flags |= ELF::SHF_MERGE;

  if (K.isMergeableCString())
    Flags |= ELF::SHF_STRINGS;

  return Flags;
}

const MCSection *XCoreTargetObjectFile::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler &Mang, const TargetMachine &TM) const {
  StringRef SectionName = GV->getSection();
  // Infer section flags from the section name if we can.
  Kind = getXCoreKindForNamedSection(SectionName, Kind);
  return getContext().getELFSection(SectionName, getXCoreSectionType(Kind),
                                    getXCoreSectionFlags(Kind), Kind);
}

const MCSection *XCoreTargetObjectFile::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
                       const TargetMachine &TM) const{
  if (Kind.isText())                      return TextSection;
  if (Kind.isMergeable1ByteCString())     return CStringSection;
  if (Kind.isMergeableConst4())           return MergeableConst4Section;
  if (Kind.isMergeableConst8())           return MergeableConst8Section;
  if (Kind.isMergeableConst16())          return MergeableConst16Section;

  Type *ObjType = GV->getType()->getPointerElementType();
  if (TM.getCodeModel() == CodeModel::Small ||
      !ObjType->isSized() ||
      TM.getDataLayout()->getTypeAllocSize(ObjType) < CodeModelLargeSize) {
    if (Kind.isReadOnly())                return ReadOnlySection;
    if (Kind.isBSS())                     return BSSSection;
    if (Kind.isDataRel())                 return DataSection;
    if (Kind.isReadOnlyWithRel())         return ReadOnlySection;
  } else {
    if (Kind.isReadOnly())                return ReadOnlySectionLarge;
    if (Kind.isBSS())                     return BSSSectionLarge;
    if (Kind.isDataRel())                 return DataSectionLarge;
    if (Kind.isReadOnlyWithRel())         return ReadOnlySectionLarge;
  }

  assert((Kind.isThreadLocal() || Kind.isCommon()) && "Unknown section kind");
  report_fatal_error("Target does not support TLS or Common sections");
}

const MCSection *XCoreTargetObjectFile::
getSectionForConstant(SectionKind Kind) const {
  if (Kind.isMergeableConst4())           return MergeableConst4Section;
  if (Kind.isMergeableConst8())           return MergeableConst8Section;
  if (Kind.isMergeableConst16())          return MergeableConst16Section;
  assert((Kind.isReadOnly() || Kind.isReadOnlyWithRel()) &&
         "Unknown section kind");
  // We assume the size of the object is never greater than CodeModelLargeSize.
  // To handle CodeModelLargeSize changes to AsmPrinter would be required.
  return ReadOnlySection;
}
