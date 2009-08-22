//===-- X86MCAsmInfo.cpp - X86 asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86MCAsmInfo.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

enum AsmWriterFlavorTy {
  // Note: This numbering has to match the GCC assembler dialects for inline
  // asm alternatives to work right.
  ATT = 0, Intel = 1
};

static cl::opt<AsmWriterFlavorTy>
AsmWriterFlavor("x86-asm-syntax", cl::init(ATT),
  cl::desc("Choose style of code to emit from X86 backend:"),
  cl::values(clEnumValN(ATT,   "att",   "Emit AT&T-style assembly"),
             clEnumValN(Intel, "intel", "Emit Intel-style assembly"),
             clEnumValEnd));


static const char *const x86_asm_table[] = {
  "{si}", "S",
  "{di}", "D",
  "{ax}", "a",
  "{cx}", "c",
  "{memory}", "memory",
  "{flags}", "",
  "{dirflag}", "",
  "{fpsr}", "",
  "{cc}", "cc",
  0,0};

X86MCAsmInfoDarwin::X86MCAsmInfoDarwin(const Triple &Triple) {
  AsmTransCBE = x86_asm_table;
  AssemblerDialect = AsmWriterFlavor;
    
  bool is64Bit = Triple.getArch() == Triple::x86_64;

  TextAlignFillValue = 0x90;

  if (!is64Bit)
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit

  // Leopard and above support aligned common symbols.
  COMMDirectiveTakesAlignment = Triple.getDarwinMajorNumber() >= 9;

  if (is64Bit) {
    PersonalityPrefix = "";
    PersonalitySuffix = "+4@GOTPCREL";
  } else {
    PersonalityPrefix = "L";
    PersonalitySuffix = "$non_lazy_ptr";
  }

  CommentString = "##";
  PCSymbol = ".";

  SupportsDebugInformation = true;
  DwarfUsesInlineInfoSection = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::Dwarf;
  AbsoluteEHSectionOffsets = false;
}

X86ELFMCAsmInfo::X86ELFMCAsmInfo(const Triple &Triple) {
  AsmTransCBE = x86_asm_table;
  AssemblerDialect = AsmWriterFlavor;

  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Debug Information
  AbsoluteDebugSectionOffsets = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::Dwarf;
  AbsoluteEHSectionOffsets = false;

  // On Linux we must declare when we can use a non-executable stack.
  if (Triple.getOS() == Triple::Linux)
    NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";
}

X86MCAsmInfoCOFF::X86MCAsmInfoCOFF(const Triple &Triple) {
  AsmTransCBE = x86_asm_table;
  AssemblerDialect = AsmWriterFlavor;
}


X86WinMCAsmInfo::X86WinMCAsmInfo(const Triple &Triple) {
  AsmTransCBE = x86_asm_table;
  AssemblerDialect = AsmWriterFlavor;

  GlobalPrefix = "_";
  CommentString = ";";

  PrivateGlobalPrefix = "$";
  AlignDirective = "\tALIGN\t";
  ZeroDirective = "\tdb\t";
  ZeroDirectiveSuffix = " dup(0)";
  AsciiDirective = "\tdb\t";
  AscizDirective = 0;
  Data8bitsDirective = "\tdb\t";
  Data16bitsDirective = "\tdw\t";
  Data32bitsDirective = "\tdd\t";
  Data64bitsDirective = "\tdq\t";
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;

  AlignmentIsInBytes = true;
}
