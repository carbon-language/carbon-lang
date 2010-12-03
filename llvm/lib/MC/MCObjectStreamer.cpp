//===- lib/MC/MCObjectStreamer.cpp - Object File MCStreamer Interface -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectStreamer.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

MCObjectStreamer::MCObjectStreamer(MCContext &Context, TargetAsmBackend &TAB,
                                   raw_ostream &_OS, MCCodeEmitter *_Emitter,
                                   bool _PadSectionToAlignment)
  : MCStreamer(Context), Assembler(new MCAssembler(Context, TAB,
                                                   *_Emitter,
                                                   _PadSectionToAlignment,
                                                   _OS)),
    CurSectionData(0)
{
}

MCObjectStreamer::~MCObjectStreamer() {
  delete &Assembler->getBackend();
  delete &Assembler->getEmitter();
  delete Assembler;
}

MCFragment *MCObjectStreamer::getCurrentFragment() const {
  assert(getCurrentSectionData() && "No current section!");

  if (!getCurrentSectionData()->empty())
    return &getCurrentSectionData()->getFragmentList().back();

  return 0;
}

MCDataFragment *MCObjectStreamer::getOrCreateDataFragment() const {
  MCDataFragment *F = dyn_cast_or_null<MCDataFragment>(getCurrentFragment());
  if (!F)
    F = new MCDataFragment(getCurrentSectionData());
  return F;
}

const MCExpr *MCObjectStreamer::AddValueSymbols(const MCExpr *Value) {
  switch (Value->getKind()) {
  case MCExpr::Target: llvm_unreachable("Can't handle target exprs yet!");
  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
    AddValueSymbols(BE->getLHS());
    AddValueSymbols(BE->getRHS());
    break;
  }

  case MCExpr::SymbolRef:
    Assembler->getOrCreateSymbolData(cast<MCSymbolRefExpr>(Value)->getSymbol());
    break;

  case MCExpr::Unary:
    AddValueSymbols(cast<MCUnaryExpr>(Value)->getSubExpr());
    break;
  }

  return Value;
}

void MCObjectStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                                 unsigned AddrSpace, bool UseSet) {
  assert(AddrSpace == 0 && "Address space must be 0!");
  MCDataFragment *DF = getOrCreateDataFragment();

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (AddValueSymbols(Value)->EvaluateAsAbsolute(AbsValue)) {
    // FIXME: Endianness assumption.
    for (unsigned i = 0; i != Size; ++i)
      DF->getContents().push_back(uint8_t(AbsValue >> (i * 8)));
  } else {
    DF->addFixup(MCFixup::Create(DF->getContents().size(),
                                 AddValueSymbols(Value),
                                 MCFixup::getKindForSize(Size, false)));
    DF->getContents().resize(DF->getContents().size() + Size, 0);
  }
}

void MCObjectStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(CurSection && "Cannot emit before setting section!");

  Symbol->setSection(*CurSection);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // FIXME: This is wasteful, we don't necessarily need to create a data
  // fragment. Instead, we should mark the symbol as pointing into the data
  // fragment if it exists, otherwise we should just queue the label and set its
  // fragment pointer when we emit the next fragment.
  MCDataFragment *F = getOrCreateDataFragment();
  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(F);
  SD.setOffset(F->getContents().size());
}

void MCObjectStreamer::EmitULEB128Value(const MCExpr *Value,
                                        unsigned AddrSpace) {
  int64_t IntValue;
  if (Value->EvaluateAsAbsolute(IntValue, &getAssembler())) {
    EmitULEB128IntValue(IntValue, AddrSpace);
    return;
  }
  new MCLEBFragment(*Value, false, getCurrentSectionData());
}

void MCObjectStreamer::EmitSLEB128Value(const MCExpr *Value,
                                        unsigned AddrSpace) {
  int64_t IntValue;
  if (Value->EvaluateAsAbsolute(IntValue, &getAssembler())) {
    EmitSLEB128IntValue(IntValue, AddrSpace);
    return;
  }
  new MCLEBFragment(*Value, true, getCurrentSectionData());
}

void MCObjectStreamer::EmitWeakReference(MCSymbol *Alias,
                                         const MCSymbol *Symbol) {
  report_fatal_error("This file format doesn't support weak aliases.");
}

void MCObjectStreamer::SwitchSection(const MCSection *Section) {
  assert(Section && "Cannot switch to a null section!");

  // If already in this section, then this is a noop.
  if (Section == CurSection) return;

  PrevSection = CurSection;
  CurSection = Section;
  CurSectionData = &getAssembler().getOrCreateSectionData(*Section);
}

void MCObjectStreamer::EmitInstruction(const MCInst &Inst) {
  // Scan for values.
  for (unsigned i = Inst.getNumOperands(); i--; )
    if (Inst.getOperand(i).isExpr())
      AddValueSymbols(Inst.getOperand(i).getExpr());

  getCurrentSectionData()->setHasInstructions(true);

  // Now that a machine instruction has been assembled into this section, make
  // a line entry for any .loc directive that has been seen.
  MCLineEntry::Make(this, getCurrentSection());

  // If this instruction doesn't need relaxation, just emit it as data.
  if (!getAssembler().getBackend().MayNeedRelaxation(Inst)) {
    EmitInstToData(Inst);
    return;
  }

  // Otherwise, if we are relaxing everything, relax the instruction as much as
  // possible and emit it as data.
  if (getAssembler().getRelaxAll()) {
    MCInst Relaxed;
    getAssembler().getBackend().RelaxInstruction(Inst, Relaxed);
    while (getAssembler().getBackend().MayNeedRelaxation(Relaxed))
      getAssembler().getBackend().RelaxInstruction(Relaxed, Relaxed);
    EmitInstToData(Relaxed);
    return;
  }

  // Otherwise emit to a separate fragment.
  EmitInstToFragment(Inst);
}

void MCObjectStreamer::EmitInstToFragment(const MCInst &Inst) {
  MCInstFragment *IF = new MCInstFragment(Inst, getCurrentSectionData());

  raw_svector_ostream VecOS(IF->getCode());
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, IF->getFixups());
}

void MCObjectStreamer::EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                                const MCSymbol *LastLabel,
                                                const MCSymbol *Label) {
  if (!LastLabel) {
    int PointerSize = getAssembler().getBackend().getPointerSize();
    EmitDwarfSetLineAddr(LineDelta, Label, PointerSize);
    return;
  }
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *LabelRef =
    MCSymbolRefExpr::Create(Label, Variant, getContext());
  const MCExpr *LastLabelRef =
    MCSymbolRefExpr::Create(LastLabel, Variant, getContext());
  const MCExpr *AddrDelta =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, LabelRef, LastLabelRef,
                         getContext());
  int64_t Res;
  if (AddrDelta->EvaluateAsAbsolute(Res, &getAssembler())) {
    MCDwarfLineAddr::Emit(this, LineDelta, Res);
    return;
  }
  new MCDwarfLineAddrFragment(LineDelta, *AddrDelta, getCurrentSectionData());
}

void MCObjectStreamer::EmitValueToOffset(const MCExpr *Offset,
                                        unsigned char Value) {
  new MCOrgFragment(*Offset, Value, getCurrentSectionData());
}

void MCObjectStreamer::Finish() {
  getAssembler().Finish();
}
