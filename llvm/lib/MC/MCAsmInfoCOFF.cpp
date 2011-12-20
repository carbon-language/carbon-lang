//===-- MCAsmInfoCOFF.cpp - COFF asm properties -----------------*- C++ -*-===//
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

#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

void MCAsmInfoCOFF::anchor() { }

MCAsmInfoCOFF::MCAsmInfoCOFF() {
  GlobalPrefix = "_";
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveType = LCOMM::ByteAlignment;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  LinkOnceDirective = "\t.linkonce discard\n";
  
  // Doesn't support visibility:
  HiddenVisibilityAttr = HiddenDeclarationVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
  HasMicrosoftFastStdCallMangling = true;

  SupportsDataRegions = false;
}

void MCAsmInfoMicrosoft::anchor() { }

MCAsmInfoMicrosoft::MCAsmInfoMicrosoft() {
  AllowQuotesInName = true;
}

void MCAsmInfoGNUCOFF::anchor() { }

MCAsmInfoGNUCOFF::MCAsmInfoGNUCOFF() {

}
