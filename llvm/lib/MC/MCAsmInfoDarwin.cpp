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
  HasStaticCtorDtorReferenceInStaticMode = true;
  SetDirective = "\t.set";
  
  HiddenVisibilityAttr = MCSA_PrivateExtern;
  // Doesn't support protected visibility.
  ProtectedVisibilityAttr = MCSA_Global;

  
  HasDotTypeDotSizeDirective = false;
  HasNoDeadStrip = true;
  // Note: Even though darwin has the .lcomm directive, it is just a synonym for
  // zerofill, so we prefer to use .zerofill.

  // _foo.eh symbols are currently always exported so that the linker knows
  // about them.  This is not necessary on 10.6 and later, but it
  // doesn't hurt anything.
  // FIXME: I need to get this from Triple.
  Is_EHSymbolPrivate = false;
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
}

