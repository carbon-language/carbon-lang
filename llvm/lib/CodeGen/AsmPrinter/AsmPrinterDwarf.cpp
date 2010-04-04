//===-- AsmPrinterDwarf.cpp - AsmPrinter Dwarf Support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dwarf emissions parts of AsmPrinter.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

/// EmitSLEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitSLEB128(int Value, const char *Desc) const {
  if (isVerbose() && Desc)
    OutStreamer.AddComment(Desc);
    
  if (MAI->hasLEB128()) {
    // FIXME: MCize.
    OutStreamer.EmitRawText("\t.sleb128\t" + Twine(Value));
    return;
  }

  // If we don't have .sleb128, emit as .bytes.
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    OutStreamer.EmitIntValue(Byte, 1, /*addrspace*/0);
  } while (IsMore);
}

/// EmitULEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitULEB128(unsigned Value, const char *Desc,
                             unsigned PadTo) const {
  if (isVerbose() && Desc)
    OutStreamer.AddComment(Desc);
 
  if (MAI->hasLEB128() && PadTo == 0) {
    // FIXME: MCize.
    OutStreamer.EmitRawText("\t.uleb128\t" + Twine(Value));
    return;
  }
  
  // If we don't have .uleb128 or we want to emit padding, emit as .bytes.
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    if (Value || PadTo != 0) Byte |= 0x80;
    OutStreamer.EmitIntValue(Byte, 1, /*addrspace*/0);
  } while (Value);

  if (PadTo) {
    if (PadTo > 1)
      OutStreamer.EmitFill(PadTo - 1, 0x80/*fillval*/, 0/*addrspace*/);
    OutStreamer.EmitFill(1, 0/*fillval*/, 0/*addrspace*/);
  }
}

