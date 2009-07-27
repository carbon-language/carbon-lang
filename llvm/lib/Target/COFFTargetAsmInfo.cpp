//===-- COFFTargetAsmInfo.cpp - COFF asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on COFF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/COFFTargetAsmInfo.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

COFFTargetAsmInfo::COFFTargetAsmInfo(const TargetMachine &TM)
  : TargetAsmInfo(TM) {

  TextSection = getOrCreateSection("_text", true, SectionKind::Text);
  DataSection = getOrCreateSection("_data", true, SectionKind::DataRel);

  GlobalPrefix = "_";
  LCOMMDirective = "\t.lcomm\t";
  COMMDirectiveTakesAlignment = false;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  StaticCtorsSection = "\t.section .ctors,\"aw\"";
  StaticDtorsSection = "\t.section .dtors,\"aw\"";
  HiddenDirective = NULL;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  AbsoluteDebugSectionOffsets = true;
  AbsoluteEHSectionOffsets = false;
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
  DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
  DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
  DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
  DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
  DwarfMacroInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
}

void COFFTargetAsmInfo::getSectionFlagsAsString(SectionKind Kind,
                                            SmallVectorImpl<char> &Str) const {
  // FIXME: Inefficient.
  std::string Res = ",\"";
  if (Kind.isText())
    Res += 'x';
  if (Kind.isWriteable())
    Res += 'w';
  Res += "\"";
  
  Str.append(Res.begin(), Res.end());
}

//===----------------------------------------------------------------------===//
// Move to AsmPrinter (mangler access).
//===----------------------------------------------------------------------===//

#include "llvm/GlobalVariable.h"

static const char *getSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text$linkonce";
  if (Kind.isWriteable())
    return ".data$linkonce";
  return ".rdata$linkonce";
}

const Section *
COFFTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV,
                                          SectionKind Kind) const {
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");
  
  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (Kind.isWeak()) {
    const char *Prefix = getSectionPrefixForUniqueGlobal(Kind);
    // FIXME: Use mangler interface (PR4584).
    std::string Name = Prefix+GV->getNameStr();
    return getOrCreateSection(Name.c_str(), false, Kind.getKind());
  }
  
  if (Kind.isText())
    return getTextSection();
  
  if (Kind.isBSS())
    if (const Section *S = getBSSSection_())
      return S;
  
  if (Kind.isReadOnly())
    if (const Section *S = getReadOnlySection())
      return S;
  
  return getDataSection();
}
