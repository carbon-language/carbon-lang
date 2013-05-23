//===-- NVPTXMCAsmInfo.cpp - NVPTX asm properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the NVPTXMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "NVPTXMCAsmInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// -debug-compile - Command line option to inform opt and llc passes to
// compile for debugging
static cl::opt<bool> CompileForDebugging("debug-compile",
                                         cl::desc("Compile for debugging"),
                                         cl::Hidden, cl::init(false));

void NVPTXMCAsmInfo::anchor() {}

NVPTXMCAsmInfo::NVPTXMCAsmInfo(const StringRef &TT) {
  Triple TheTriple(TT);
  if (TheTriple.getArch() == Triple::nvptx64) {
    PointerSize = CalleeSaveStackSlotSize = 8;
  }

  CommentString = "//";

  PrivateGlobalPrefix = "$L__";

  AllowPeriodsInName = false;

  HasSetDirective = false;

  HasSingleParameterDotFile = false;

  InlineAsmStart = " inline asm";
  InlineAsmEnd = " inline asm";

  SupportsDebugInformation = CompileForDebugging;
  HasDotTypeDotSizeDirective = false;

  Data8bitsDirective = " .b8 ";
  Data16bitsDirective = " .b16 ";
  Data32bitsDirective = " .b32 ";
  Data64bitsDirective = " .b64 ";
  PrivateGlobalPrefix = "";
  ZeroDirective = " .b8";
  AsciiDirective = " .b8";
  AscizDirective = " .b8";

  // @TODO: Can we just disable this?
  GlobalDirective = "\t// .globl\t";
}
