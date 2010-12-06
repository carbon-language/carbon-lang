//===- lib/MC/MCStreamer.cpp - Streaming Machine Code Output --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include <cstdlib>
using namespace llvm;

MCStreamer::MCStreamer(MCContext &Ctx) : Context(Ctx), CurSection(0),
                                         PrevSection(0) {
}

MCStreamer::~MCStreamer() {
}

raw_ostream &MCStreamer::GetCommentOS() {
  // By default, discard comments.
  return nulls();
}

void MCStreamer::EmitDwarfSetLineAddr(int64_t LineDelta,
                                      const MCSymbol *Label, int PointerSize) {
  // emit the sequence to set the address
  EmitIntValue(dwarf::DW_LNS_extended_op, 1);
  EmitULEB128IntValue(PointerSize + 1);
  EmitIntValue(dwarf::DW_LNE_set_address, 1);
  EmitSymbolValue(Label, PointerSize);

  // emit the sequence for the LineDelta (from 1) and a zero address delta.
  MCDwarfLineAddr::Emit(this, LineDelta, 0);
}

/// EmitIntValue - Special case of EmitValue that avoids the client having to
/// pass in a MCExpr for constant integers.
void MCStreamer::EmitIntValue(uint64_t Value, unsigned Size,
                              unsigned AddrSpace) {
  assert(Size <= 8);
  char buf[8];
  // FIXME: Endianness assumption.
  for (unsigned i = 0; i != Size; ++i)
    buf[i] = uint8_t(Value >> (i * 8));
  EmitBytes(StringRef(buf, Size), AddrSpace);
}

/// EmitULEB128Value - Special case of EmitULEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitULEB128IntValue(uint64_t Value, unsigned AddrSpace) {
  SmallString<32> Tmp;
  raw_svector_ostream OSE(Tmp);
  MCObjectWriter::EncodeULEB128(Value, OSE);
  EmitBytes(OSE.str(), AddrSpace);
}

/// EmitSLEB128Value - Special case of EmitSLEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitSLEB128IntValue(int64_t Value, unsigned AddrSpace) {
  SmallString<32> Tmp;
  raw_svector_ostream OSE(Tmp);
  MCObjectWriter::EncodeSLEB128(Value, OSE);
  EmitBytes(OSE.str(), AddrSpace);
}

void MCStreamer::EmitAbsValue(const MCExpr *Value, unsigned Size) {
  MCSymbol *ABS = getContext().CreateTempSymbol();
  EmitAssignment(ABS, Value);
  EmitSymbolValue(ABS, Size, 0);
}

void MCStreamer::EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                                 unsigned AddrSpace) {
  EmitValue(MCSymbolRefExpr::Create(Sym, getContext()), Size, AddrSpace);
}

void MCStreamer::EmitGPRel32Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

/// EmitFill - Emit NumBytes bytes worth of the value specified by
/// FillValue.  This implements directives such as '.space'.
void MCStreamer::EmitFill(uint64_t NumBytes, uint8_t FillValue,
                          unsigned AddrSpace) {
  const MCExpr *E = MCConstantExpr::Create(FillValue, getContext());
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    EmitValue(E, 1, AddrSpace);
}

bool MCStreamer::EmitDwarfFileDirective(unsigned FileNo,
                                        StringRef Filename) {
  return getContext().GetDwarfFile(Filename, FileNo) == 0;
}

void MCStreamer::EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa,
                                       unsigned Discriminator) {
  getContext().setCurrentDwarfLoc(FileNo, Line, Column, Flags, Isa,
                                  Discriminator);
}

bool MCStreamer::EmitCFIStartProc() {
  return false;
}

bool MCStreamer::EmitCFIEndProc() {
  return false;
}

bool MCStreamer::EmitCFIDefCfaOffset(int64_t Offset) {
  return false;
}

bool MCStreamer::EmitCFIDefCfaRegister(int64_t Register) {
  return false;
}

bool MCStreamer::EmitCFIOffset(int64_t Register, int64_t Offset) {
  return false;
}

bool MCStreamer::EmitCFIPersonality(const MCSymbol *Sym) {
  return false;
}

bool MCStreamer::EmitCFILsda(const MCSymbol *Sym) {
  return false;
}

/// EmitRawText - If this file is backed by an assembly streamer, this dumps
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
