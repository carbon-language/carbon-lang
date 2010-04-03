//===- lib/MC/MCStreamer.cpp - Streaming Machine Code Output --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include <cstdlib>
using namespace llvm;

MCStreamer::MCStreamer(MCContext &_Context) : Context(_Context), CurSection(0) {
}

MCStreamer::~MCStreamer() {
}

raw_ostream &MCStreamer::GetCommentOS() {
  // By default, discard comments.
  return nulls();
}


/// EmitIntValue - Special case of EmitValue that avoids the client having to
/// pass in a MCExpr for constant integers.
void MCStreamer::EmitIntValue(uint64_t Value, unsigned Size,
                              unsigned AddrSpace) {
  EmitValue(MCConstantExpr::Create(Value, getContext()), Size, AddrSpace);
}

void MCStreamer::EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                                 unsigned AddrSpace) {
  EmitValue(MCSymbolRefExpr::Create(Sym, getContext()), Size, AddrSpace);
}

/// EmitFill - Emit NumBytes bytes worth of the value specified by
/// FillValue.  This implements directives such as '.space'.
void MCStreamer::EmitFill(uint64_t NumBytes, uint8_t FillValue,
                          unsigned AddrSpace) {
  const MCExpr *E = MCConstantExpr::Create(FillValue, getContext());
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    EmitValue(E, 1, AddrSpace);
}

/// EmitRawText - If this file is backed by a assembly streamer, this dumps
/// the specified string in the output .s file.  This capability is
/// indicated by the hasRawTextSupport() predicate.
void MCStreamer::EmitRawText(StringRef String) {
  errs() << "EmitRawText called on an MCStreamer that doesn't support it, "
  " something must not be fully mc'ized\n";
  abort();
}

void MCStreamer::EmitRawText(const Twine &T) {
  SmallString<128> Str;
  T.toVector(Str);
  EmitRawText(Str.str());
}
