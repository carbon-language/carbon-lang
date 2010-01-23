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

MCAsmInfoCOFF::MCAsmInfoCOFF() {
  GlobalPrefix = "_";
  LCOMMDirective = "\t.lcomm\t";
  COMMDirectiveTakesAlignment = false;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  LinkOnceDirective = "\t.linkonce same_size\n";
  SetDirective = "\t.set\t";
  
  // Doesn't support visibility:
  HiddenVisibilityAttr = ProtectedVisibilityAttr = MCSA_Invalid;

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  AbsoluteDebugSectionOffsets = true;
  AbsoluteEHSectionOffsets = false;
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
}

