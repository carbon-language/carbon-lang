//===-- XCoreMCAsmInfo.cpp - XCore asm properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreMCAsmInfo.h"
#include "llvm/ADT/StringRef.h"
using namespace llvm;

void XCoreMCAsmInfo::anchor() { }

XCoreMCAsmInfo::XCoreMCAsmInfo(StringRef TT) {
  SupportsDebugInformation = true;
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = 0;
  ZeroDirective = "\t.space\t";
  CommentString = "#";
    
  PrivateGlobalPrefix = ".L";
  AscizDirective = ".asciiz";

  HiddenVisibilityAttr = MCSA_Invalid;
  HiddenDeclarationVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  // Debug
  HasLEB128 = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  DwarfRegNumForCFI = true;
}

