//===-- DarwinTargetAsmInfo.cpp - Darwin asm properties ---------*- C++ -*-===//
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

#include "llvm/Target/DarwinTargetAsmInfo.h"
using namespace llvm;

DarwinTargetAsmInfo::DarwinTargetAsmInfo() {
  // Common settings for all Darwin targets.
  // Syntax:
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  LinkerPrivateGlobalPrefix = "l";  // Marker for some ObjC metadata
  NeedsSet = true;
  NeedsIndirectEncoding = true;
  AllowQuotesInName = true;
  HasSingleParameterDotFile = false;

  // In non-PIC modes, emit a special label before jump tables so that the
  // linker can perform more accurate dead code stripping.  We do not check the
  // relocation model here since it can be overridden later.
  JumpTableSpecialLabelPrefix = "l";
    
  // Directives:
  WeakDefDirective = "\t.weak_definition ";
  WeakRefDirective = "\t.weak_reference ";
  HiddenDirective = "\t.private_extern ";
  LCOMMDirective = "\t.lcomm\t";

  // _foo.eh symbols are currently always exported so that the linker knows
  // about them.  This is not necessary on 10.6 and later, but it
  // doesn't hurt anything.
  // FIXME: I need to get this from Triple.
  Is_EHSymbolPrivate = false;
}

