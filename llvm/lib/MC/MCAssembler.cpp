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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"

#include <vector>
using namespace llvm;

namespace {
namespace stats {
STATISTIC(EmittedFragments, "Number of emitted assembler fragments");
STATISTIC(EvaluateFixup, "Number of evaluated fixups");
STATISTIC(FragmentLayouts, "Number of fragment layouts");
STATISTIC(ObjectBytes, "Number of emitted object file bytes");
STATISTIC(RelaxationSteps, "Number of assembler layout and relaxation steps");
STATISTIC(RelaxedInstructions, "Number of relaxed instructions");
STATISTIC(SectionLayouts, "Number of section layouts");
}
}

// FIXME FIXME FIXME: There are number of places in this file where we convert
// what is a 64-bit assembler value used for computation into a value in the
// object file, which may truncate it. We should detect that truncation where
// invalid and report errors back.

/* *** */

void MCAsmLayout::UpdateForSlide(MCFragment *F, int SlideAmount) {
  // We shouldn't have to do anything special to support negative slides, and it
  // is a perfectly valid thing to do as long as other parts of the system are
  // can guarantee convergence.
  assert(SlideAmount >= 0 && "Negative slides not yet supported");

  // Update the layout by simply recomputing the layout for the entire
  // file. This is trivially correct, but very slow.
  //
  // FIXME-PERF: This is O(N^2), but will be eliminated once we get smarter.

  // Layout the concrete sections and fragments.
  MCAssembler &Asm = getAssembler();
  uint64_t Address = 0;
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it) {
    // Skip virtual sections.
    if (Asm.getBackend().isVirtualSection(it->getSection()))
      continue;

    // Layout the section fragments and its size.
    Address = Asm.LayoutSection(*it, *this, Address);
  }

  // Layout the virtual sections.
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it) {
    if (!Asm.getBackend().isVirtualSection(it->getSection()))
      continue;

    // Layout the section fragments and its size.
    Address = Asm.LayoutSection(*it, *this, Address);
  }
}

uint64_t MCAsmLayout::getFragmentAddress(const MCFragment *F) const {
  assert(F->getParent() && "Missing section()!");
  return getSectionAddress(F->getParent()) + getFragmentOffset(F);
}

uint64_t MCAsmLayout::getFragmentEffectiveSize(const MCFragment *F) const {
  assert(F->EffectiveSize != ~UINT64_C(0) && "Address not set!");
  return F->EffectiveSize;
}

void MCAsmLayout::setFragmentEffectiveSize(MCFragment *F, uint64_t Value) {
  F->EffectiveSize = Value;
}

uint64_t MCAsmLayout::getFragmentOffset(const MCFragment *F) const {
  assert(F->Offset != ~UINT64_C(0) && "Address not set!");
  return F->Offset;
}

void MCAsmLayout::setFragmentOffset(MCFragment *F, uint64_t Value) {
  F->Offset = Value;
}

uint64_t MCAsmLayout::getSymbolAddress(const MCSymbolData *SD) const {
  assert(SD->getFragment() && "Invalid getAddress() on undefined symbol!");
  return getFragmentAddress(SD->getFragment()) + SD->getOffset();
}

uint64_t MCAsmLayout::getSectionAddress(const MCSectionData *SD) const {
  assert(SD->Address != ~UINT64_C(0) && "Address not set!");
  return SD->Address;
}

void MCAsmLayout::setSectionAddress(MCSectionData *SD, uint64_t Value) {
  SD->Address = Value;
}

uint64_t MCAsmLayout::getSectionSize(const MCSectionData *SD) const {
  assert(SD->Size != ~UINT64_C(0) && "File size not set!");
  return SD->Size;
}
void MCAsmLayout::setSectionSize(MCSectionData *SD, uint64_t Value) {
  SD->Size = Value;
}

uint64_t MCAsmLayout::getSectionFileSize(const MCSectionData *SD) const {
  assert(SD->FileSize != ~UINT64_C(0) && "File size not set!");
  return SD->FileSize;
}
void MCAsmLayout::setSectionFileSize(MCSectionData *SD, uint64_t Value) {
  SD->FileSize = Value;
}

  /// @}

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *_Parent)
  : Kind(_Kind),
    Parent(_Parent),
    EffectiveSize(~UINT64_C(0))
{
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

MCFragment::~MCFragment() {
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
    OS(_OS), RelaxAll(false), SubsectionsViaSymbols(false)
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
  // need to worry about considering symbol differences fully resolved.

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
                                          const MCAsmLayout &Layout,
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

    A_Base = Asm.getAtom(Layout, &Asm.getSymbolData(A->getSymbol()));
    if (!A_Base)
      return false;
  }

  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    // Modified symbol references cannot be resolved.
    if (B->getKind() != MCSymbolRefExpr::VK_None)
      return false;

    B_Base = Asm.getAtom(Layout, &Asm.getSymbolData(B->getSymbol()));
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

// FIXME-PERF: This routine is really slow.
const MCSymbolData *MCAssembler::getAtomForAddress(const MCAsmLayout &Layout,
                                                   const MCSectionData *Section,
                                                   uint64_t Address) const {
  const MCSymbolData *Best = 0;
  uint64_t BestAddress = 0;

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
    uint64_t SymbolAddress = Layout.getSymbolAddress(it);
    if (SymbolAddress <= Address && (!Best || SymbolAddress >= BestAddress)) {
      Best = it;
      BestAddress = SymbolAddress;
    }
  }

  return Best;
}

// FIXME-PERF: This routine is really slow.
const MCSymbolData *MCAssembler::getAtom(const MCAsmLayout &Layout,
                                         const MCSymbolData *SD) const {
  // Linker visible symbols define atoms.
  if (isSymbolLinkerVisible(SD))
    return SD;

  // Absolute and undefined symbols have no defining atom.
  if (!SD->getFragment())
    return 0;

  // Otherwise, search by address.
  return getAtomForAddress(Layout, SD->getFragment()->getParent(),
                           Layout.getSymbolAddress(SD));
}

bool MCAssembler::EvaluateFixup(const MCAsmLayout &Layout,
                                const MCAsmFixup &Fixup, const MCFragment *DF,
                                MCValue &Target, uint64_t &Value) const {
  ++stats::EvaluateFixup;

  if (!Fixup.Value->EvaluateAsRelocatable(Target, &Layout))
    report_fatal_error("expected relocatable expression");

  // FIXME: How do non-scattered symbols work in ELF? I presume the linker
  // doesn't support small relocations, but then under what criteria does the
  // assembler allow symbol differences?

  Value = Target.getConstant();

  bool IsPCRel =
    Emitter.getFixupKindInfo(Fixup.Kind).Flags & MCFixupKindInfo::FKF_IsPCRel;
  bool IsResolved = true;
  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    if (A->getSymbol().isDefined())
      Value += Layout.getSymbolAddress(&getSymbolData(A->getSymbol()));
    else
      IsResolved = false;
  }
  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    if (B->getSymbol().isDefined())
      Value -= Layout.getSymbolAddress(&getSymbolData(B->getSymbol()));
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
          Layout, DF->getParent(), Layout.getFragmentAddress(DF)+Fixup.Offset);
        if (!BaseSymbol)
          IsResolved = false;
      }

      if (IsResolved)
        IsResolved = isScatteredFixupFullyResolved(*this, Layout, Fixup, Target,
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
    Value -= Layout.getFragmentAddress(DF) + Fixup.Offset;

  return IsResolved;
}

uint64_t MCAssembler::LayoutSection(MCSectionData &SD,
                                    MCAsmLayout &Layout,
                                    uint64_t StartAddress) {
  bool IsVirtual = getBackend().isVirtualSection(SD.getSection());

  ++stats::SectionLayouts;

  // Align this section if necessary by adding padding bytes to the previous
  // section. It is safe to adjust this out-of-band, because no symbol or
  // fragment is allowed to point past the end of the section at any time.
  if (uint64_t Pad = OffsetToAlignment(StartAddress, SD.getAlignment())) {
    // Unless this section is virtual (where we are allowed to adjust the offset
    // freely), the padding goes in the previous section.
    if (!IsVirtual) {
      // Find the previous non-virtual section.
      iterator it = &SD;
      assert(it != begin() && "Invalid initial section address!");
      for (--it; getBackend().isVirtualSection(it->getSection()); --it) ;
      Layout.setSectionFileSize(&*it, Layout.getSectionFileSize(&*it) + Pad);
    }

    StartAddress += Pad;
  }

  // Set the aligned section address.
  Layout.setSectionAddress(&SD, StartAddress);

  uint64_t Address = StartAddress;
  for (MCSectionData::iterator it = SD.begin(), ie = SD.end(); it != ie; ++it) {
    MCFragment &F = *it;

    ++stats::FragmentLayouts;

    uint64_t FragmentOffset = Address - StartAddress;
    Layout.setFragmentOffset(&F, FragmentOffset);

    // Evaluate fragment size.
    uint64_t EffectiveSize = 0;
    switch (F.getKind()) {
    case MCFragment::FT_Align: {
      MCAlignFragment &AF = cast<MCAlignFragment>(F);

      EffectiveSize = OffsetToAlignment(Address, AF.getAlignment());
      if (EffectiveSize > AF.getMaxBytesToEmit())
        EffectiveSize = 0;
      break;
    }

    case MCFragment::FT_Data:
      EffectiveSize = cast<MCDataFragment>(F).getContents().size();
      break;

    case MCFragment::FT_Fill: {
      MCFillFragment &FF = cast<MCFillFragment>(F);
      EffectiveSize = FF.getValueSize() * FF.getCount();
      break;
    }

    case MCFragment::FT_Inst:
      EffectiveSize = cast<MCInstFragment>(F).getInstSize();
      break;

    case MCFragment::FT_Org: {
      MCOrgFragment &OF = cast<MCOrgFragment>(F);

      int64_t TargetLocation;
      if (!OF.getOffset().EvaluateAsAbsolute(TargetLocation, &Layout))
        report_fatal_error("expected assembly-time absolute expression");

      // FIXME: We need a way to communicate this error.
      int64_t Offset = TargetLocation - FragmentOffset;
      if (Offset < 0)
        report_fatal_error("invalid .org offset '" + Twine(TargetLocation) +
                          "' (at offset '" + Twine(FragmentOffset) + "'");

      EffectiveSize = Offset;
      break;
    }

    case MCFragment::FT_ZeroFill: {
      MCZeroFillFragment &ZFF = cast<MCZeroFillFragment>(F);

      // Align the fragment offset; it is safe to adjust the offset freely since
      // this is only in virtual sections.
      //
      // FIXME: We shouldn't be doing this here.
      Address = RoundUpToAlignment(Address, ZFF.getAlignment());
      Layout.setFragmentOffset(&F, Address - StartAddress);

      EffectiveSize = ZFF.getSize();
      break;
    }
    }

    Layout.setFragmentEffectiveSize(&F, EffectiveSize);
    Address += EffectiveSize;
  }

  // Set the section sizes.
  Layout.setSectionSize(&SD, Address - StartAddress);
  if (IsVirtual)
    Layout.setSectionFileSize(&SD, 0);
  else
    Layout.setSectionFileSize(&SD, Address - StartAddress);

  return Address;
}

/// WriteFragmentData - Write the \arg F data to the output file.
static void WriteFragmentData(const MCAssembler &Asm, const MCAsmLayout &Layout,
                              const MCFragment &F, MCObjectWriter *OW) {
  uint64_t Start = OW->getStream().tell();
  (void) Start;

  ++stats::EmittedFragments;

  // FIXME: Embed in fragments instead?
  uint64_t FragmentSize = Layout.getFragmentEffectiveSize(&F);
  switch (F.getKind()) {
  case MCFragment::FT_Align: {
    MCAlignFragment &AF = cast<MCAlignFragment>(F);
    uint64_t Count = FragmentSize / AF.getValueSize();

    // FIXME: This error shouldn't actually occur (the front end should emit
    // multiple .align directives to enforce the semantics it wants), but is
    // severe enough that we want to report it. How to handle this?
    if (Count * AF.getValueSize() != FragmentSize)
      report_fatal_error("undefined .align directive, value size '" +
                        Twine(AF.getValueSize()) +
                        "' is not a divisor of padding size '" +
                        Twine(FragmentSize) + "'");

    // See if we are aligning with nops, and if so do that first to try to fill
    // the Count bytes.  Then if that did not fill any bytes or there are any
    // bytes left to fill use the the Value and ValueSize to fill the rest.
    // If we are aligning with nops, ask that target to emit the right data.
    if (AF.getEmitNops()) {
      if (!Asm.getBackend().WriteNopData(Count, OW))
        report_fatal_error("unable to write nop sequence of " +
                          Twine(Count) + " bytes");
      break;
    }

    // Otherwise, write out in multiples of the value size.
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
    assert(FragmentSize == DF.getContents().size() && "Invalid size!");
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

    for (uint64_t i = 0, e = FragmentSize; i != e; ++i)
      OW->Write8(uint8_t(OF.getValue()));

    break;
  }

  case MCFragment::FT_ZeroFill: {
    assert(0 && "Invalid zero fill fragment in concrete section!");
    break;
  }
  }

  assert(OW->getStream().tell() - Start == FragmentSize);
}

void MCAssembler::WriteSectionData(const MCSectionData *SD,
                                   const MCAsmLayout &Layout,
                                   MCObjectWriter *OW) const {
  uint64_t SectionSize = Layout.getSectionSize(SD);
  uint64_t SectionFileSize = Layout.getSectionFileSize(SD);

  // Ignore virtual sections.
  if (getBackend().isVirtualSection(SD->getSection())) {
    assert(SectionFileSize == 0 && "Invalid size for section!");
    return;
  }

  uint64_t Start = OW->getStream().tell();
  (void) Start;

  for (MCSectionData::const_iterator it = SD->begin(),
         ie = SD->end(); it != ie; ++it)
    WriteFragmentData(*this, Layout, *it, OW);

  // Add section padding.
  assert(SectionFileSize >= SectionSize && "Invalid section sizes!");
  OW->WriteZeros(SectionFileSize - SectionSize);

  assert(OW->getStream().tell() - Start == SectionFileSize);
}

void MCAssembler::Finish() {
  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - pre-layout\n--\n";
      dump(); });

  // Assign section and fragment ordinals, all subsequent backend code is
  // responsible for updating these in place.
  unsigned SectionIndex = 0;
  unsigned FragmentIndex = 0;
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    it->setOrdinal(SectionIndex++);

    for (MCSectionData::iterator it2 = it->begin(),
           ie2 = it->end(); it2 != ie2; ++it2)
      it2->setOrdinal(FragmentIndex++);
  }

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

  uint64_t StartOffset = OS.tell();
  llvm::OwningPtr<MCObjectWriter> Writer(getBackend().createObjectWriter(OS));
  if (!Writer)
    report_fatal_error("unable to create object writer!");

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
          Writer->RecordRelocation(*this, Layout, DF, Fixup, Target,FixedValue);
        }

        getBackend().ApplyFixup(Fixup, *DF, FixedValue);
      }
    }
  }

  // Write the object file.
  Writer->WriteObject(*this, Layout);
  OS.flush();

  stats::ObjectBytes += OS.tell() - StartOffset;
}

bool MCAssembler::FixupNeedsRelaxation(const MCAsmFixup &Fixup,
                                       const MCFragment *DF,
                                       const MCAsmLayout &Layout) const {
  if (getRelaxAll())
    return true;

  // If we cannot resolve the fixup value, it requires relaxation.
  MCValue Target;
  uint64_t Value;
  if (!EvaluateFixup(Layout, Fixup, DF, Target, Value))
    return true;

  // Otherwise, relax if the value is too big for a (signed) i8.
  //
  // FIXME: This is target dependent!
  return int64_t(Value) != int64_t(int8_t(Value));
}

bool MCAssembler::FragmentNeedsRelaxation(const MCInstFragment *IF,
                                          const MCAsmLayout &Layout) const {
  // If this inst doesn't ever need relaxation, ignore it. This occurs when we
  // are intentionally pushing out inst fragments, or because we relaxed a
  // previous instruction to one that doesn't need relaxation.
  if (!getBackend().MayNeedRelaxation(IF->getInst(), IF->getFixups()))
    return false;

  for (MCInstFragment::const_fixup_iterator it = IF->fixup_begin(),
         ie = IF->fixup_end(); it != ie; ++it)
    if (FixupNeedsRelaxation(*it, IF, Layout))
      return true;

  return false;
}

bool MCAssembler::LayoutOnce(MCAsmLayout &Layout) {
  ++stats::RelaxationSteps;

  // Layout the concrete sections and fragments.
  uint64_t Address = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    // Skip virtual sections.
    if (getBackend().isVirtualSection(it->getSection()))
      continue;

    // Layout the section fragments and its size.
    Address = LayoutSection(*it, Layout, Address);
  }

  // Layout the virtual sections.
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (!getBackend().isVirtualSection(it->getSection()))
      continue;

    // Layout the section fragments and its size.
    Address = LayoutSection(*it, Layout, Address);
  }

  // Scan for fragments that need relaxation.
  bool WasRelaxed = false;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;

    for (MCSectionData::iterator it2 = SD.begin(),
           ie2 = SD.end(); it2 != ie2; ++it2) {
      // Check if this is an instruction fragment that needs relaxation.
      MCInstFragment *IF = dyn_cast<MCInstFragment>(it2);
      if (!IF || !FragmentNeedsRelaxation(IF, Layout))
        continue;

      ++stats::RelaxedInstructions;

      // FIXME-PERF: We could immediately lower out instructions if we can tell
      // they are fully resolved, to avoid retesting on later passes.

      // Relax the fragment.

      MCInst Relaxed;
      getBackend().RelaxInstruction(IF, Relaxed);

      // Encode the new instruction.
      //
      // FIXME-PERF: If it matters, we could let the target do this. It can
      // probably do so more efficiently in many cases.
      SmallVector<MCFixup, 4> Fixups;
      SmallString<256> Code;
      raw_svector_ostream VecOS(Code);
      getEmitter().EncodeInstruction(Relaxed, VecOS, Fixups);
      VecOS.flush();

      // Update the instruction fragment.
      int SlideAmount = Code.size() - IF->getInstSize();
      IF->setInst(Relaxed);
      IF->getCode() = Code;
      IF->getFixups().clear();
      for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
        MCFixup &F = Fixups[i];
        IF->getFixups().push_back(MCAsmFixup(F.getOffset(), *F.getValue(),
                                             F.getKind()));
      }

      // Update the layout, and remember that we relaxed. If we are relaxing
      // everything, we can skip this step since nothing will depend on updating
      // the values.
      if (!getRelaxAll())
        Layout.UpdateForSlide(IF, SlideAmount);
      WasRelaxed = true;
    }
  }

  return WasRelaxed;
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
      // FIXME-PERF: Reuse previous data fragment if possible.
      MCDataFragment *DF = new MCDataFragment();
      SD.getFragmentList().insert(it2, DF);

      // Update the data fragments layout data.
      //
      // FIXME: Add MCAsmLayout utility for this.
      DF->setParent(IF->getParent());
      DF->setOrdinal(IF->getOrdinal());
      Layout.setFragmentOffset(DF, Layout.getFragmentOffset(IF));
      Layout.setFragmentEffectiveSize(DF, Layout.getFragmentEffectiveSize(IF));

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
     << " EffectiveSize:" << EffectiveSize;

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
