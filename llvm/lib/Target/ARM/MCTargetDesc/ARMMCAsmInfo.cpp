//===-- ARMMCAsmInfo.cpp - ARM asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMMCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

void ARMMCAsmInfoDarwin::anchor() { }

ARMMCAsmInfoDarwin::ARMMCAsmInfoDarwin(StringRef TT) {
  Triple TheTriple(TT);
  if ((TheTriple.getArch() == Triple::armeb) ||
      (TheTriple.getArch() == Triple::thumbeb))
    IsLittleEndian = false;

  Data64bitsDirective = nullptr;
  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";
  UseDataRegionDirectives = true;

  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::SjLj;

  UseIntegratedAssembler = true;
}

void ARMELFMCAsmInfo::anchor() { }

ARMELFMCAsmInfo::ARMELFMCAsmInfo(StringRef TT) {
  Triple TheTriple(TT);
  if ((TheTriple.getArch() == Triple::armeb) ||
      (TheTriple.getArch() == Triple::thumbeb))
    IsLittleEndian = false;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  Data64bitsDirective = nullptr;
  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";

  HasLEB128 = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::ARM;

  // foo(plt) instead of foo@plt
  UseParensForSymbolVariant = true;

  UseIntegratedAssembler = true;
}

void ARMELFMCAsmInfo::setUseIntegratedAssembler(bool Value) {
  UseIntegratedAssembler = Value;
  if (!UseIntegratedAssembler) {
    // gas doesn't handle VFP register names in cfi directives,
    // so don't use register names with external assembler.
    // See https://sourceware.org/bugzilla/show_bug.cgi?id=16694
    DwarfRegNumForCFI = true;
  }
}

void ARMCOFFMCAsmInfoMicrosoft::anchor() { }

ARMCOFFMCAsmInfoMicrosoft::ARMCOFFMCAsmInfoMicrosoft() {
  AlignmentIsInBytes = false;

  PrivateGlobalPrefix = "$M";
}

void ARMCOFFMCAsmInfoGNU::anchor() { }

ARMCOFFMCAsmInfoGNU::ARMCOFFMCAsmInfoGNU() {
  AlignmentIsInBytes = false;
  HasSingleParameterDotFile = true;

  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";
  PrivateGlobalPrefix = ".L";

  HasLEB128 = true;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::None;
  UseParensForSymbolVariant = true;

  UseIntegratedAssembler = false;
  DwarfRegNumForCFI = true;
}

