//===-- MCAsmInfoDarwin.cpp - Darwin asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoDarwin.h"
using namespace llvm;

MCAsmInfoDarwin::MCAsmInfoDarwin() {
  // Common settings for all Darwin targets.
  // Syntax:
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  LinkerPrivateGlobalPrefix = "l";
  AllowQuotesInName = true;
  HasSingleParameterDotFile = false;
  HasSubsectionsViaSymbols = true;

  AlignmentIsInBytes = false;
  COMMDirectiveAlignmentIsInBytes = false;
  InlineAsmStart = " InlineAsm Start";
  InlineAsmEnd = " InlineAsm End";

  // Directives:
  WeakDefDirective = "\t.weak_definition ";
  WeakRefDirective = "\t.weak_reference ";
  ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
  HasMachoZeroFillDirective = true;  // Uses .zerofill
  HasMachoTBSSDirective = true; // Uses .tbss
  HasStaticCtorDtorReferenceInStaticMode = true;
  
  HiddenVisibilityAttr = MCSA_PrivateExtern;
  // Doesn't support protected visibility.
  ProtectedVisibilityAttr = MCSA_Global;
  
  HasDotTypeDotSizeDirective = false;
  HasNoDeadStrip = true;

  DwarfUsesAbsoluteLabelForStmtList = false;
  DwarfUsesLabelOffsetForRanges = false;
}

