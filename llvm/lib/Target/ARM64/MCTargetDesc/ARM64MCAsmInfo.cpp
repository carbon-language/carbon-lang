//===-- ARM64MCAsmInfo.cpp - ARM64 asm properties -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARM64MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARM64MCAsmInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

enum AsmWriterVariantTy {
  Default = -1,
  Generic = 0,
  Apple = 1
};

static cl::opt<AsmWriterVariantTy> AsmWriterVariant(
    "arm64-neon-syntax", cl::init(Default),
    cl::desc("Choose style of NEON code to emit from ARM64 backend:"),
    cl::values(clEnumValN(Generic, "generic", "Emit generic NEON assembly"),
               clEnumValN(Apple, "apple", "Emit Apple-style NEON assembly"),
               clEnumValEnd));

ARM64MCAsmInfoDarwin::ARM64MCAsmInfoDarwin() {
  // We prefer NEON instructions to be printed in the short form.
  AssemblerDialect = AsmWriterVariant == Default ? 1 : AsmWriterVariant;

  PrivateGlobalPrefix = "L";
  SeparatorString = "%%";
  CommentString = ";";
  PointerSize = CalleeSaveStackSlotSize = 8;

  AlignmentIsInBytes = false;
  UsesELFSectionDirectiveForBSS = true;
  SupportsDebugInformation = true;
  UseDataRegionDirectives = true;

  ExceptionsType = ExceptionHandling::DwarfCFI;
}

const MCExpr *ARM64MCAsmInfoDarwin::getExprForPersonalitySymbol(
    const MCSymbol *Sym, unsigned Encoding, MCStreamer &Streamer) const {
  // On Darwin, we can reference dwarf symbols with foo@GOT-., which
  // is an indirect pc-relative reference. The default implementation
  // won't reference using the GOT, so we need this target-specific
  // version.
  MCContext &Context = Streamer.getContext();
  const MCExpr *Res =
      MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_GOT, Context);
  MCSymbol *PCSym = Context.CreateTempSymbol();
  Streamer.EmitLabel(PCSym);
  const MCExpr *PC = MCSymbolRefExpr::Create(PCSym, Context);
  return MCBinaryExpr::CreateSub(Res, PC, Context);
}

ARM64MCAsmInfoELF::ARM64MCAsmInfoELF(StringRef TT) {
  Triple T(TT);
  if (T.getArch() == Triple::aarch64_be)
    IsLittleEndian = false;

  // We prefer NEON instructions to be printed in the short form.
  AssemblerDialect = AsmWriterVariant == Default ? 0 : AsmWriterVariant;

  PointerSize = 8;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "//";
  PrivateGlobalPrefix = ".L";
  Code32Directive = ".code\t32";

  Data16bitsDirective = "\t.hword\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.xword\t";

  UseDataRegionDirectives = false;

  WeakRefDirective = "\t.weak\t";

  HasLEB128 = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  UseIntegratedAssembler = true;
}
