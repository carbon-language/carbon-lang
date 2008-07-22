//===-- MipsTargetAsmInfo.cpp - Mips asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"

using namespace llvm;

MipsTargetAsmInfo::MipsTargetAsmInfo(const MipsTargetMachine &TM):
  ELFTargetAsmInfo(TM) {

  MipsTM = &TM;

  AlignmentIsInBytes          = false;
  COMMDirectiveTakesAlignment = true;
  Data16bitsDirective         = "\t.half\t";
  Data32bitsDirective         = "\t.word\t";
  Data64bitsDirective         = NULL;
  PrivateGlobalPrefix         = "$";
  JumpTableDataSection        = "\t.rdata";
  CommentString               = "#";
  ReadOnlySection             = "\t.rdata";
  ZeroDirective               = "\t.space\t";
  BSSSection                  = "\t.section\t.bss";
  LCOMMDirective              = "\t.lcomm\t";
  CStringSection              = ".rodata.str";

  if (!TM.getSubtarget<MipsSubtarget>().hasABICall())
    JumpTableDirective = "\t.word\t";
  else
    JumpTableDirective = "\t.gpword\t";
  
  SmallDataSection = getNamedSection("\t.sdata", SectionFlags::Writeable);
  SmallBSSSection  = getNamedSection("\t.sbss",
                         SectionFlags::Writeable | SectionFlags::BSS);
}

SectionKind::Kind
MipsTargetAsmInfo::SectionKindForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind K = ELFTargetAsmInfo::SectionKindForGlobal(GV);

  if (K != SectionKind::Data && K != SectionKind::BSS && 
      K != SectionKind::RODataMergeConst)
    return K;

  if (isa<GlobalVariable>(GV)) {
    const TargetData *TD = ETM->getTargetData();
    unsigned Size = TD->getABITypeSize(GV->getType()->getElementType());
    unsigned Threshold = 
      MipsTM->getSubtarget<MipsSubtarget>().getSSectionThreshold();
     
    if (Size > 0 && Size <= Threshold) {
      if (K == SectionKind::BSS)
        return SectionKind::SmallBSS;
      else
        return SectionKind::SmallData;
    }
  }

  return K;
}

const Section*
MipsTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind K = SectionKindForGlobal(GV);
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);

  if (GVA && (!GVA->isWeakForLinker()))
    switch (K) {
      case SectionKind::SmallData:
        return getSmallDataSection();
      case SectionKind::SmallBSS:
        return getSmallBSSSection();
      default: break;
    }

  return ELFTargetAsmInfo::SelectSectionForGlobal(GV);
}
