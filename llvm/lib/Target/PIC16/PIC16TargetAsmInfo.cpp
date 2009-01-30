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
#include "PIC16TargetMachine.h"
#include "llvm/GlobalValue.h"

using namespace llvm;

PIC16TargetAsmInfo::
PIC16TargetAsmInfo(const PIC16TargetMachine &TM) 
  : TargetAsmInfo(TM) {
  CommentString = ";";
  Data8bitsDirective = " db ";
  Data16bitsDirective = " dw ";
  Data32bitsDirective = " dl ";
  RomData8bitsDirective = " dw ";
  RomData16bitsDirective = " rom_di ";
  RomData8bitsDirective = " rom_dl ";
  ZeroDirective = NULL;
  AsciiDirective = " dt ";
  AscizDirective = NULL;
  BSSSection_  = getNamedSection("udata.# UDATA",
                              SectionFlags::Writeable | SectionFlags::BSS);
  ReadOnlySection = getNamedSection("romdata.# ROMDATA", SectionFlags::None);
  DataSection = getNamedSection("idata.# IDATA", SectionFlags::Writeable);
  SwitchToSectionDirective = "";
}

const char *PIC16TargetAsmInfo::getData8bitsDirective(unsigned AddrSpace)
                                                       const {
      if (AddrSpace == PIC16ISD::ROM_SPACE)
        return RomData8bitsDirective;
      else 
        return Data8bitsDirective; 
  }

const char *PIC16TargetAsmInfo::getData16bitsDirective(unsigned AddrSpace)
                                                       const {
      if (AddrSpace == PIC16ISD::ROM_SPACE)
        return RomData16bitsDirective;
      else
        return Data16bitsDirective;
  }

const char *PIC16TargetAsmInfo::getData32bitsDirective(unsigned AddrSpace)
                                                       const {
      if (AddrSpace == PIC16ISD::ROM_SPACE)
        return RomData32bitsDirective;
      else
        return Data32bitsDirective;
  }

