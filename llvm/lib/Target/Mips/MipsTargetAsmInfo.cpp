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

static bool isSuitableForBSS(const GlobalVariable *GV) {
  if (!GV->hasInitializer())
    return true;

  // Leave constant zeros in readonly constant sections, so they can be shared
  Constant *C = GV->getInitializer();
  return (C->isNullValue() && !GV->isConstant() && !NoZerosInBSS);
}

SectionKind::Kind
MipsTargetAsmInfo::SectionKindForGlobal(const GlobalValue *GV) const {
  const TargetData *TD = ETM->getTargetData();
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);

  if (!GVA)
    return ELFTargetAsmInfo::SectionKindForGlobal(GV);
  
  // if this is a internal constant string, there is a special
  // section for it, but not in small data/bss.
  if (GVA->hasInitializer() && GV->hasInternalLinkage()) {
    Constant *C = GVA->getInitializer();
    const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
    if (CVA && CVA->isCString()) 
      return ELFTargetAsmInfo::SectionKindForGlobal(GV);
  }

  const Type *Ty = GV->getType()->getElementType();
  unsigned Size = TD->getABITypeSize(Ty);
  unsigned Threshold = 
    MipsTM->getSubtarget<MipsSubtarget>().getSSectionThreshold();

  if (Size > 0 && Size <= Threshold) {
    if (isSuitableForBSS(GVA))
      return SectionKind::SmallBSS;
    else
      return SectionKind::SmallData;
  }

  return ELFTargetAsmInfo::SectionKindForGlobal(GV);
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
