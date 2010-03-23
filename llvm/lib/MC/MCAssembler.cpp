//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "assembler"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"

// FIXME: Gross.
#include "../Target/X86/X86FixupKinds.h"

#include <vector>
using namespace llvm;

STATISTIC(EmittedFragments, "Number of emitted assembler fragments");

// FIXME FIXME FIXME: There are number of places in this file where we convert
// what is a 64-bit assembler value used for computation into a value in the
// object file, which may truncate it. We should detect that truncation where
// invalid and report errors back.

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *_Parent)
  : Kind(_Kind),
    Parent(_Parent),
    FileSize(~UINT64_C(0))
{
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

MCFragment::~MCFragment() {
}

uint64_t MCFragment::getAddress() const {
  assert(getParent() && "Missing Section!");
  return getParent()->getAddress() + Offset;
}

/* *** */

MCSectionData::MCSectionData() : Section(0) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(&_Section),
    Alignment(1),
    Address(~UINT64_C(0)),
    Size(~UINT64_C(0)),
    FileSize(~UINT64_C(0)),
    HasInstructions(false)
{
  if (A)
    A->getSectionList().push_back(this);
}

/* *** */

MCSymbolData::MCSymbolData() : Symbol(0) {}

MCSymbolData::MCSymbolData(const MCSymbol &_Symbol, MCFragment *_Fragment,
                           uint64_t _Offset, MCAssembler *A)
  : Symbol(&_Symbol), Fragment(_Fragment), Offset(_Offset),
    IsExternal(false), IsPrivateExtern(false),
    CommonSize(0), CommonAlign(0), Flags(0), Index(0)
{
  if (A)
    A->getSymbolList().push_back(this);
}

/* *** */

MCAssembler::MCAssembler(MCContext &_Context, TargetAsmBackend &_Backend,
                         MCCodeEmitter &_Emitter, raw_ostream &_OS)
  : Context(_Context), Backend(_Backend), Emitter(_Emitter),
    OS(_OS), SubsectionsViaSymbols(false)
{
}

MCAssembler::~MCAssembler() {
}

static bool isScatteredFixupFullyResolvedSimple(const MCAssembler &Asm,
                                                const MCAsmFixup &Fixup,
                                                const MCValue Target,
                                                const MCSection *BaseSection) {
  // The effective fixup address is
  //     addr(atom(A)) + offset(A)
  //   - addr(atom(B)) - offset(B)
  //   - addr(<base symbol>) + <fixup offset from base symbol>
  // and the offsets are not relocatable, so the fixup is fully resolved when
  //  addr(atom(A)) - addr(atom(B)) - addr(<base symbol>)) == 0.
  //
  // The simple (Darwin, except on x86_64) way of dealing with this was to
  // assume that any reference to a temporary symbol *must* be a temporary
  // symbol in the same atom, unless the sections differ. Therefore, any PCrel
  // relocation to a temporary symbol (in the same section) is fully
  // resolved. This also works in conjunction with absolutized .set, which
  // requires the compiler to use .set to absolutize the differences between
  // symbols which the compiler knows to be assembly time constants, so we don't
  // need to worry about consider symbol differences fully resolved.

  // Non-relative fixups are only resolved if constant.
  if (!BaseSection)
    return Target.isAbsolute();

  // Otherwise, relative fixups are only resolved if not a difference and the
  // target is a temporary in the same section.
  if (Target.isAbsolute() || Target.getSymB())
    return false;

  const MCSymbol *A = &Target.getSymA()->getSymbol();
  if (!A->isTemporary() || !A->isInSection() ||
      &A->getSection() != BaseSection)
    return false;

  return true;
}

static bool isScatteredFixupFullyResolved(const MCAssembler &Asm,
                                          const MCAsmFixup &Fixup,
                                          const MCValue Target,
                                          const MCSymbolData *BaseSymbol) {
  // The effective fixup address is
  //     addr(atom(A)) + offset(A)
  //   - addr(atom(B)) - offset(B)
  //   - addr(BaseSymbol) + <fixup offset from base symbol>
  // and the offsets are not relocatable, so the fixup is fully resolved when
  //  addr(atom(A)) - addr(atom(B)) - addr(BaseSymbol) == 0.
  //
  // Note that "false" is almost always conservatively correct (it means we emit
  // a relocation which is unnecessary), except when it would force us to emit a
  // relocation which the target cannot encode.

  const MCSymbolData *A_Base = 0, *B_Base = 0;
  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    // Modified symbol references cannot be resolved.
    if (A->getKind() != MCSymbolRefExpr::VK_None)
      return false;

    A_Base = Asm.getAtom(&Asm.getSymbolData(A->getSymbol()));
    if (!A_Base)
      return false;
  }

  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    // Modified symbol references cannot be resolved.
    if (B->getKind() != MCSymbolRefExpr::VK_None)
      return false;

    B_Base = Asm.getAtom(&Asm.getSymbolData(B->getSymbol()));
    if (!B_Base)
      return false;
  }

  // If there is no base, A and B have to be the same atom for this fixup to be
  // fully resolved.
  if (!BaseSymbol)
    return A_Base == B_Base;

  // Otherwise, B must be missing and A must be the base.
  return !B_Base && BaseSymbol == A_Base;
}

bool MCAssembler::isSymbolLinkerVisible(const MCSymbolData *SD) const {
  // Non-temporary labels should always be visible to the linker.
  if (!SD->getSymbol().isTemporary())
    return true;

  // Absolute temporary labels are never visible.
  if (!SD->getFragment())
    return false;

  // Otherwise, check if the section requires symbols even for temporary labels.
  return getBackend().doesSectionRequireSymbols(
    SD->getFragment()->getParent()->getSection());
}

const MCSymbolData *MCAssembler::getAtomForAddress(const MCSectionData *Section,
                                                   uint64_t Address) const {
  const MCSymbolData *Best = 0;
  for (MCAssembler::const_symbol_iterator it = symbol_begin(),
         ie = symbol_end(); it != ie; ++it) {
    // Ignore non-linker visible symbols.
    if (!isSymbolLinkerVisible(it))
      continue;

    // Ignore symbols not in the same section.
    if (!it->getFragment() || it->getFragment()->getParent() != Section)
      continue;

    // Otherwise, find the closest symbol preceding this address (ties are
    // resolved in favor of the last defined symbol).
    if (it->getAddress() <= Address &&
        (!Best || it->getAddress() >= Best->getAddress()))
      Best = it;
  }

  return Best;
}

const MCSymbolData *MCAssembler::getAtom(const MCSymbolData *SD) const {
  // Linker visible symbols define atoms.
  if (isSymbolLinkerVisible(SD))
    return SD;

  // Absolute and undefined symbols have no defining atom.
  if (!SD->getFragment())
    return 0;

  // Otherwise, search by address.
  return getAtomForAddress(SD->getFragment()->getParent(), SD->getAddress());
}

bool MCAssembler::EvaluateFixup(const MCAsmLayout &Layout,
                                const MCAsmFixup &Fixup, const MCFragment *DF,
                                MCValue &Target, uint64_t &Value) const {
  if (!Fixup.Value->EvaluateAsRelocatable(Target, &Layout))
    llvm_report_error("expected relocatable expression");

  // FIXME: How do non-scattered symbols work in ELF? I presume the linker
  // doesn't support small relocations, but then under what criteria does the
  // assembler allow symbol differences?

  Value = Target.getConstant();

  bool IsPCRel =
    Emitter.getFixupKindInfo(Fixup.Kind).Flags & MCFixupKindInfo::FKF_IsPCRel;
  bool IsResolved = true;
  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    if (A->getSymbol().isDefined())
      Value += getSymbolData(A->getSymbol()).getAddress();
    else
      IsResolved = false;
  }
  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    if (B->getSymbol().isDefined())
      Value -= getSymbolData(B->getSymbol()).getAddress();
    else
      IsResolved = false;
  }

  // If we are using scattered symbols, determine whether this value is actually
  // resolved; scattering may cause atoms to move.
  if (IsResolved && getBackend().hasScatteredSymbols()) {
    if (getBackend().hasReliableSymbolDifference()) {
      // If this is a PCrel relocation, find the base atom (identified by its
      // symbol) that the fixup value is relative to.
      const MCSymbolData *BaseSymbol = 0;
      if (IsPCRel) {
        BaseSymbol = getAtomForAddress(
          DF->getParent(), DF->getAddress() + Fixup.Offset);
        if (!BaseSymbol)
          IsResolved = false;
      }

      if (IsResolved)
        IsResolved = isScatteredFixupFullyResolved(*this, Fixup, Target,
                                                   BaseSymbol);
    } else {
      const MCSection *BaseSection = 0;
      if (IsPCRel)
        BaseSection = &DF->getParent()->getSection();

      IsResolved = isScatteredFixupFullyResolvedSimple(*this, Fixup, Target,
                                                       BaseSection);
    }
  }

  if (IsPCRel)
    Value -= DF->getAddress() + Fixup.Offset;

  return IsResolved;
}

void MCAssembler::LayoutSection(MCSectionData &SD,
                                MCAsmLayout &Layout) {
  uint64_t Address = SD.getAddress();

  for (MCSectionData::iterator it = SD.begin(), ie = SD.end(); it != ie; ++it) {
    MCFragment &F = *it;

    F.setOffset(Address - SD.getAddress());

    // Evaluate fragment size.
    switch (F.getKind()) {
    case MCFragment::FT_Align: {
      MCAlignFragment &AF = cast<MCAlignFragment>(F);

      uint64_t Size = OffsetToAlignment(Address, AF.getAlignment());
      if (Size > AF.getMaxBytesToEmit())
        AF.setFileSize(0);
      else
        AF.setFileSize(Size);
      break;
    }

    case MCFragment::FT_Data:
      F.setFileSize(cast<MCDataFragment>(F).getContents().size());
      break;

    case MCFragment::FT_Fill: {
      MCFillFragment &FF = cast<MCFillFragment>(F);
      F.setFileSize(FF.getValueSize() * FF.getCount());
      break;
    }

    case MCFragment::FT_Inst:
      F.setFileSize(cast<MCInstFragment>(F).getInstSize());
      break;

    case MCFragment::FT_Org: {
      MCOrgFragment &OF = cast<MCOrgFragment>(F);

      int64_t TargetLocation;
      if (!OF.getOffset().EvaluateAsAbsolute(TargetLocation, &Layout))
        llvm_report_error("expected assembly-time absolute expression");

      // FIXME: We need a way to communicate this error.
      int64_t Offset = TargetLocation - F.getOffset();
      if (Offset < 0)
        llvm_report_error("invalid .org offset '" + Twine(TargetLocation) +
                          "' (at offset '" + Twine(F.getOffset()) + "'");

      F.setFileSize(Offset);
      break;
    }

    case MCFragment::FT_ZeroFill: {
      MCZeroFillFragment &ZFF = cast<MCZeroFillFragment>(F);

      // Align the fragment offset; it is safe to adjust the offset freely since
      // this is only in virtual sections.
      Address = RoundUpToAlignment(Address, ZFF.getAlignment());
      F.setOffset(Address - SD.getAddress());

      // FIXME: This is misnamed.
      F.setFileSize(ZFF.getSize());
      break;
    }
    }

    Address += F.getFileSize();
  }

  // Set the section sizes.
  SD.setSize(Address - SD.getAddress());
  if (getBackend().isVirtualSection(SD.getSection()))
    SD.setFileSize(0);
  else
    SD.setFileSize(Address - SD.getAddress());
}

/// WriteNopData - Write optimal nops to the output file for the \arg Count
/// bytes.  This returns the number of bytes written.  It may return 0 if
/// the \arg Count is more than the maximum optimal nops.
///
/// FIXME this is X86 32-bit specific and should move to a better place.
static uint64_t WriteNopData(uint64_t Count, MCObjectWriter *OW) {
  static const uint8_t Nops[16][16] = {
    // nop
    {0x90},
    // xchg %ax,%ax
    {0x66, 0x90},
    // nopl (%[re]ax)
    {0x0f, 0x1f, 0x00},
    // nopl 0(%[re]ax)
    {0x0f, 0x1f, 0x40, 0x00},
    // nopl 0(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopw 0L(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopw %cs:0L(%[re]ax,%[re]ax,1)
    {0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopl 0(%[re]ax,%[re]ax,1)
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x44, 0x00, 0x00,
     0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
     0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    // nopl 0L(%[re]ax) */
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
     0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    // nopl 0L(%[re]ax)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
     0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    // nopl 0L(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
     0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00}
  };

  if (Count > 15)
    return 0;

  for (uint64_t i = 0; i < Count; i++)
    OW->Write8(uint8_t(Nops[Count - 1][i]));

  return Count;
}

/// WriteFragmentData - Write the \arg F data to the output file.
static void WriteFragmentData(const MCFragment &F, MCObjectWriter *OW) {
  uint64_t Start = OW->getStream().tell();
  (void) Start;

  ++EmittedFragments;

  // FIXME: Embed in fragments instead?
  switch (F.getKind()) {
  case MCFragment::FT_Align: {
    MCAlignFragment &AF = cast<MCAlignFragment>(F);
    uint64_t Count = AF.getFileSize() / AF.getValueSize();

    // FIXME: This error shouldn't actually occur (the front end should emit
    // multiple .align directives to enforce the semantics it wants), but is
    // severe enough that we want to report it. How to handle this?
    if (Count * AF.getValueSize() != AF.getFileSize())
      llvm_report_error("undefined .align directive, value size '" +
                        Twine(AF.getValueSize()) +
                        "' is not a divisor of padding size '" +
                        Twine(AF.getFileSize()) + "'");

    // See if we are aligning with nops, and if so do that first to try to fill
    // the Count bytes.  Then if that did not fill any bytes or there are any
    // bytes left to fill use the the Value and ValueSize to fill the rest.
    if (AF.getEmitNops()) {
      uint64_t NopByteCount = WriteNopData(Count, OW);
      Count -= NopByteCount;
    }

    for (uint64_t i = 0; i != Count; ++i) {
      switch (AF.getValueSize()) {
      default:
        assert(0 && "Invalid size!");
      case 1: OW->Write8 (uint8_t (AF.getValue())); break;
      case 2: OW->Write16(uint16_t(AF.getValue())); break;
      case 4: OW->Write32(uint32_t(AF.getValue())); break;
      case 8: OW->Write64(uint64_t(AF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_Data: {
    MCDataFragment &DF = cast<MCDataFragment>(F);
    assert(DF.getFileSize() == DF.getContents().size() && "Invalid size!");
    OW->WriteBytes(DF.getContents().str());
    break;
  }

  case MCFragment::FT_Fill: {
    MCFillFragment &FF = cast<MCFillFragment>(F);
    for (uint64_t i = 0, e = FF.getCount(); i != e; ++i) {
      switch (FF.getValueSize()) {
      default:
        assert(0 && "Invalid size!");
      case 1: OW->Write8 (uint8_t (FF.getValue())); break;
      case 2: OW->Write16(uint16_t(FF.getValue())); break;
      case 4: OW->Write32(uint32_t(FF.getValue())); break;
      case 8: OW->Write64(uint64_t(FF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_Inst:
    llvm_unreachable("unexpected inst fragment after lowering");
    break;

  case MCFragment::FT_Org: {
    MCOrgFragment &OF = cast<MCOrgFragment>(F);

    for (uint64_t i = 0, e = OF.getFileSize(); i != e; ++i)
      OW->Write8(uint8_t(OF.getValue()));

    break;
  }

  case MCFragment::FT_ZeroFill: {
    assert(0 && "Invalid zero fill fragment in concrete section!");
    break;
  }
  }

  assert(OW->getStream().tell() - Start == F.getFileSize());
}

void MCAssembler::WriteSectionData(const MCSectionData *SD,
                                   MCObjectWriter *OW) const {
  // Ignore virtual sections.
  if (getBackend().isVirtualSection(SD->getSection())) {
    assert(SD->getFileSize() == 0);
    return;
  }

  uint64_t Start = OW->getStream().tell();
  (void) Start;

  for (MCSectionData::const_iterator it = SD->begin(),
         ie = SD->end(); it != ie; ++it)
    WriteFragmentData(*it, OW);

  // Add section padding.
  assert(SD->getFileSize() >= SD->getSize() && "Invalid section sizes!");
  OW->WriteZeros(SD->getFileSize() - SD->getSize());

  assert(OW->getStream().tell() - Start == SD->getFileSize());
}

void MCAssembler::Finish() {
  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - pre-layout\n--\n";
      dump(); });

  // Layout until everything fits.
  MCAsmLayout Layout(*this);
  while (LayoutOnce(Layout))
    continue;

  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - post-relaxation\n--\n";
      dump(); });

  // Finalize the layout, including fragment lowering.
  FinishLayout(Layout);

  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - final-layout\n--\n";
      dump(); });

  llvm::OwningPtr<MCObjectWriter> Writer(getBackend().createObjectWriter(OS));
  if (!Writer)
    llvm_report_error("unable to create object writer!");

  // Allow the object writer a chance to perform post-layout binding (for
  // example, to set the index fields in the symbol data).
  Writer->ExecutePostLayoutBinding(*this);

  // Evaluate and apply the fixups, generating relocation entries as necessary.
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    for (MCSectionData::iterator it2 = it->begin(),
           ie2 = it->end(); it2 != ie2; ++it2) {
      MCDataFragment *DF = dyn_cast<MCDataFragment>(it2);
      if (!DF)
        continue;

      for (MCDataFragment::fixup_iterator it3 = DF->fixup_begin(),
             ie3 = DF->fixup_end(); it3 != ie3; ++it3) {
        MCAsmFixup &Fixup = *it3;

        // Evaluate the fixup.
        MCValue Target;
        uint64_t FixedValue;
        if (!EvaluateFixup(Layout, Fixup, DF, Target, FixedValue)) {
          // The fixup was unresolved, we need a relocation. Inform the object
          // writer of the relocation, and give it an opportunity to adjust the
          // fixup value if need be.
          Writer->RecordRelocation(*this, DF, Fixup, Target, FixedValue);
        }

        getBackend().ApplyFixup(Fixup, *DF, FixedValue);
      }
    }
  }

  // Write the object file.
  Writer->WriteObject(*this);
  OS.flush();
}

bool MCAssembler::FixupNeedsRelaxation(const MCAsmFixup &Fixup,
                                       const MCFragment *DF,
                                       const MCAsmLayout &Layout) const {
  // Currently we only need to relax X86::reloc_pcrel_1byte.
  if (unsigned(Fixup.Kind) != X86::reloc_pcrel_1byte)
    return false;

  // If we cannot resolve the fixup value, it requires relaxation.
  MCValue Target;
  uint64_t Value;
  if (!EvaluateFixup(Layout, Fixup, DF, Target, Value))
    return true;

  // Otherwise, relax if the value is too big for a (signed) i8.
  return int64_t(Value) != int64_t(int8_t(Value));
}

bool MCAssembler::LayoutOnce(MCAsmLayout &Layout) {
  // Layout the concrete sections and fragments.
  uint64_t Address = 0;
  MCSectionData *Prev = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    // Skip virtual sections.
    if (getBackend().isVirtualSection(SD.getSection()))
      continue;

    // Align this section if necessary by adding padding bytes to the previous
    // section.
    if (uint64_t Pad = OffsetToAlignment(Address, it->getAlignment())) {
      assert(Prev && "Missing prev section!");
      Prev->setFileSize(Prev->getFileSize() + Pad);
      Address += Pad;
    }

    // Layout the section fragments and its size.
    SD.setAddress(Address);
    LayoutSection(SD, Layout);
    Address += SD.getFileSize();

    Prev = &SD;
  }

  // Layout the virtual sections.
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    if (!getBackend().isVirtualSection(SD.getSection()))
      continue;

    // Align this section if necessary by adding padding bytes to the previous
    // section.
    if (uint64_t Pad = OffsetToAlignment(Address, it->getAlignment()))
      Address += Pad;

    SD.setAddress(Address);
    LayoutSection(SD, Layout);
    Address += SD.getSize();
  }

  // Scan the fixups in order and relax any that don't fit.
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    for (MCSectionData::iterator it2 = SD.begin(),
           ie2 = SD.end(); it2 != ie2; ++it2) {
      MCDataFragment *DF = dyn_cast<MCDataFragment>(it2);
      if (!DF)
        continue;

      for (MCDataFragment::fixup_iterator it3 = DF->fixup_begin(),
             ie3 = DF->fixup_end(); it3 != ie3; ++it3) {
        MCAsmFixup &Fixup = *it3;

        // Check whether we need to relax this fixup.
        if (!FixupNeedsRelaxation(Fixup, DF, Layout))
          continue;

        // Relax the instruction.
        //
        // FIXME: This is a huge temporary hack which just looks for x86
        // branches; the only thing we need to relax on x86 is
        // 'X86::reloc_pcrel_1byte'. Once we have MCInst fragments, this will be
        // replaced by a TargetAsmBackend hook (most likely tblgen'd) to relax
        // an individual MCInst.
        SmallVectorImpl<char> &C = DF->getContents();
        uint64_t PrevOffset = Fixup.Offset;
        unsigned Amt = 0;

          // jcc instructions
        if (unsigned(C[Fixup.Offset-1]) >= 0x70 &&
            unsigned(C[Fixup.Offset-1]) <= 0x7f) {
          C[Fixup.Offset] = C[Fixup.Offset-1] + 0x10;
          C[Fixup.Offset-1] = char(0x0f);
          ++Fixup.Offset;
          Amt = 4;

          // jmp rel8
        } else if (C[Fixup.Offset-1] == char(0xeb)) {
          C[Fixup.Offset-1] = char(0xe9);
          Amt = 3;

        } else
          llvm_unreachable("unknown 1 byte pcrel instruction!");

        Fixup.Value = MCBinaryExpr::Create(
          MCBinaryExpr::Sub, Fixup.Value,
          MCConstantExpr::Create(3, getContext()),
          getContext());
        C.insert(C.begin() + Fixup.Offset, Amt, char(0));
        Fixup.Kind = MCFixupKind(X86::reloc_pcrel_4byte);

        // Update the remaining fixups, which have slid.
        //
        // FIXME: This is bad for performance, but will be eliminated by the
        // move to MCInst specific fragments.
        ++it3;
        for (; it3 != ie3; ++it3)
          it3->Offset += Amt;

        // Update all the symbols for this fragment, which may have slid.
        //
        // FIXME: This is really really bad for performance, but will be
        // eliminated by the move to MCInst specific fragments.
        for (MCAssembler::symbol_iterator it = symbol_begin(),
               ie = symbol_end(); it != ie; ++it) {
          MCSymbolData &SD = *it;

          if (it->getFragment() != DF)
            continue;

          if (SD.getOffset() > PrevOffset)
            SD.setOffset(SD.getOffset() + Amt);
        }

        // Restart layout.
        //
        // FIXME-PERF: This is O(N^2), but will be eliminated once we have a
        // smart MCAsmLayout object.
        return true;
      }
    }
  }

  return false;
}

void MCAssembler::FinishLayout(MCAsmLayout &Layout) {
  // Lower out any instruction fragments, to simplify the fixup application and
  // output.
  //
  // FIXME-PERF: We don't have to do this, but the assumption is that it is
  // cheap (we will mostly end up eliminating fragments and appending on to data
  // fragments), so the extra complexity downstream isn't worth it. Evaluate
  // this assumption.
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    for (MCSectionData::iterator it2 = SD.begin(),
           ie2 = SD.end(); it2 != ie2; ++it2) {
      MCInstFragment *IF = dyn_cast<MCInstFragment>(it2);
      if (!IF)
        continue;

      // Create a new data fragment for the instruction.
      //
      // FIXME: Reuse previous data fragment if possible.
      MCDataFragment *DF = new MCDataFragment();
      SD.getFragmentList().insert(it2, DF);

      // Update the data fragments layout data.
      DF->setParent(IF->getParent());
      DF->setOffset(IF->getOffset());
      DF->setFileSize(IF->getInstSize());

      // Copy in the data and the fixups.
      DF->getContents().append(IF->getCode().begin(), IF->getCode().end());
      for (unsigned i = 0, e = IF->getFixups().size(); i != e; ++i)
        DF->getFixups().push_back(IF->getFixups()[i]);

      // Delete the instruction fragment and update the iterator.
      SD.getFragmentList().erase(IF);
      it2 = DF;
    }
  }
}

// Debugging methods

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const MCAsmFixup &AF) {
  OS << "<MCAsmFixup" << " Offset:" << AF.Offset << " Value:" << *AF.Value
     << " Kind:" << AF.Kind << ">";
  return OS;
}

}

void MCFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCFragment " << (void*) this << " Offset:" << Offset
     << " FileSize:" << FileSize;

  OS << ">";
}

void MCAlignFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCAlignFragment ";
  this->MCFragment::dump();
  OS << "\n       ";
  OS << " Alignment:" << getAlignment()
     << " Value:" << getValue() << " ValueSize:" << getValueSize()
     << " MaxBytesToEmit:" << getMaxBytesToEmit() << ">";
}

void MCDataFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCDataFragment ";
  this->MCFragment::dump();
  OS << "\n       ";
  OS << " Contents:[";
  for (unsigned i = 0, e = getContents().size(); i != e; ++i) {
    if (i) OS << ",";
    OS << hexdigit((Contents[i] >> 4) & 0xF) << hexdigit(Contents[i] & 0xF);
  }
  OS << "] (" << getContents().size() << " bytes)";

  if (!getFixups().empty()) {
    OS << ",\n       ";
    OS << " Fixups:[";
    for (fixup_iterator it = fixup_begin(), ie = fixup_end(); it != ie; ++it) {
      if (it != fixup_begin()) OS << ",\n                ";
      OS << *it;
    }
    OS << "]";
  }

  OS << ">";
}

void MCFillFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCFillFragment ";
  this->MCFragment::dump();
  OS << "\n       ";
  OS << " Value:" << getValue() << " ValueSize:" << getValueSize()
     << " Count:" << getCount() << ">";
}

void MCInstFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCInstFragment ";
  this->MCFragment::dump();
  OS << "\n       ";
  OS << " Inst:";
  getInst().dump_pretty(OS);
  OS << ">";
}

void MCOrgFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCOrgFragment ";
  this->MCFragment::dump();
  OS << "\n       ";
  OS << " Offset:" << getOffset() << " Value:" << getValue() << ">";
}

void MCZeroFillFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCZeroFillFragment ";
  this->MCFragment::dump();
  OS << "\n       ";
  OS << " Size:" << getSize() << " Alignment:" << getAlignment() << ">";
}

void MCSectionData::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCSectionData";
  OS << " Alignment:" << getAlignment() << " Address:" << Address
     << " Size:" << Size << " FileSize:" << FileSize
     << " Fragments:[\n      ";
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin()) OS << ",\n      ";
    it->dump();
  }
  OS << "]>";
}

void MCSymbolData::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCSymbolData Symbol:" << getSymbol()
     << " Fragment:" << getFragment() << " Offset:" << getOffset()
     << " Flags:" << getFlags() << " Index:" << getIndex();
  if (isCommon())
    OS << " (common, size:" << getCommonSize()
       << " align: " << getCommonAlignment() << ")";
  if (isExternal())
    OS << " (external)";
  if (isPrivateExtern())
    OS << " (private extern)";
  OS << ">";
}

void MCAssembler::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCAssembler\n";
  OS << "  Sections:[\n    ";
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin()) OS << ",\n    ";
    it->dump();
  }
  OS << "],\n";
  OS << "  Symbols:[";

  for (symbol_iterator it = symbol_begin(), ie = symbol_end(); it != ie; ++it) {
    if (it != symbol_begin()) OS << ",\n           ";
    it->dump();
  }
  OS << "]>\n";
}
