//===-- PIC16TargetAsmInfo.cpp - PIC16 asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the PIC16TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PIC16TargetAsmInfo.h"

// FIXME: Layering violation to get enums and static function, should be moved
// to separate headers.
#include "PIC16.h"
#include "PIC16ISelLowering.h"
using namespace llvm;

PIC16TargetAsmInfo::PIC16TargetAsmInfo(const Target &T, const StringRef &TT) {
  CommentString = ";";
  GlobalPrefix = PAN::getTagName(PAN::PREFIX_SYMBOL);
  GlobalDirective = "\tglobal\t";
  ExternDirective = "\textern\t";

  Data8bitsDirective = " db ";
  Data16bitsDirective = " dw ";
  Data32bitsDirective = " dl ";
  Data64bitsDirective = NULL;
  ZeroDirective = NULL;
  AsciiDirective = " dt ";
  AscizDirective = NULL;
    
  RomData8bitsDirective = " dw ";
  RomData16bitsDirective = " rom_di ";
  RomData32bitsDirective = " rom_dl ";
    
    
  // Set it to false because we weed to generate c file name and not bc file
  // name.
  HasSingleParameterDotFile = false;
}

const char *PIC16TargetAsmInfo::
getDataASDirective(unsigned Size, unsigned AS) const {
  if (AS != PIC16ISD::ROM_SPACE)
    return 0;
  
  switch (Size) {
  case  8: return RomData8bitsDirective;
  case 16: return RomData16bitsDirective;
  case 32: return RomData32bitsDirective;
  default: return NULL;
  }
}

