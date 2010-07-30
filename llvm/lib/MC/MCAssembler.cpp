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

MCAsmLayout::MCAsmLayout(MCAssembler &Asm)
  : Assembler(Asm), LastValidFragment(0)
 {
  // Compute the section layout order. Virtual sections must go last.
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
    if (!Asm.getBackend().isVirtualSection(it->getSection()))
      SectionOrder.push_back(&*it);
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
    if (Asm.getBackend().isVirtualSection(it->getSection()))
      SectionOrder.push_back(&*it);
}

bool MCAsmLayout::isSectionUpToDate(const MCSectionData *SD) const {
  // The first section is always up-to-date.
  unsigned Index = SD->getLayoutOrder();
  if (!Index)
    return true;

  // Otherwise, sections are always implicitly computed when the preceeding
  // fragment is layed out.
  const MCSectionData *Prev = getSectionOrder()[Index - 1];
  return isFragmentUpToDate(&(Prev->getFragmentList().back()));
}

bool MCAsmLayout::isFragmentUpToDate(const MCFragment *F) const {
  return (LastValidFragment &&
          F->getLayoutOrder() <= LastValidFragment->getLayoutOrder());
}

void MCAsmLayout::UpdateForSlide(MCFragment *F, int SlideAmount) {
  // If this fragment wasn't already up-to-date, we don't need to do anything.
  if (!isFragmentUpToDate(F))
    return;

  // Otherwise, reset the last valid fragment to the predecessor of the
  // invalidated fragment.
  LastValidFragment = F->getPrevNode();
  if (!LastValidFragment) {
    unsigned Index = F->getParent()->getLayoutOrder();
    if (Index != 0) {
      MCSectionData *Prev = getSectionOrder()[Index - 1];
      LastValidFragment = &(Prev->getFragmentList().back());
    }
  }
}

void MCAsmLayout::EnsureValid(const MCFragment *F) const {
  // Advance the layout position until the fragment is up-to-date.
  while (!isFragmentUpToDate(F)) {
    // Advance to the next fragment.
    MCFragment *Cur = LastValidFragment;
    if (Cur)
      Cur = Cur->getNextNode();
    if (!Cur) {
      unsigned NextIndex = 0;
      if (LastValidFragment)
        NextIndex = LastValidFragment->getParent()->getLayoutOrder() + 1;
      Cur = SectionOrder[NextIndex]->begin();
    }

    const_cast<MCAsmLayout*>(this)->LayoutFragment(Cur);
  }
}

void MCAsmLayout::FragmentReplaced(MCFragment *Src, MCFragment *Dst) {
  if (LastValidFragment == Src)
    LastValidFragment = Dst;

  Dst->Offset = Src->Offset;
  Dst->EffectiveSize = Src->EffectiveSize;
}

uint64_t MCAsmLayout::getFragmentAddress(const MCFragment *F) const {
  assert(F->getParent() && "Missing section()!");
  return getSectionAddress(F->getParent()) + getFragmentOffset(F);
}

uint64_t MCAsmLayout::getFragmentEffectiveSize(const MCFragment *F) const {
  EnsureValid(F);
  assert(F->EffectiveSize != ~UINT64_C(0) && "Address not set!");
  return F->EffectiveSize;
}

uint64_t MCAsmLayout::getFragmentOffset(const MCFragment *F) const {
  EnsureValid(F);
  assert(F->Offset != ~UINT64_C(0) && "Address not set!");
  return F->Offset;
}

uint64_t MCAsmLayout::getSymbolAddress(const MCSymbolData *SD) const {
  assert(SD->getFragment() && "Invalid getAddress() on undefined symbol!");
  return getFragmentAddress(SD->getFragment()) + SD->getOffset();
}

uint64_t MCAsmLayout::getSectionAddress(const MCSectionData *SD) const {
  EnsureValid(SD->begin());
  assert(SD->Address != ~UINT64_C(0) && "Address not set!");
  return SD->Address;
}

uint64_t MCAsmLayout::getSectionAddressSize(const MCSectionData *SD) const {
  // The size is the last fragment's end offset.
  const MCFragment &F = SD->getFragmentList().back();
  return getFragmentOffset(&F) + getFragmentEffectiveSize(&F);
}

uint64_t MCAsmLayout::getSectionFileSize(const MCSectionData *SD) const {
  // Virtual sections have no file size.
  if (getAssembler().getBackend().isVirtualSection(SD->getSection()))
    return 0;

  // Otherwise, the file size is the same as the address space size.
  return getSectionAddressSize(SD);
}

uint64_t MCAsmLayout::getSectionSize(const MCSectionData *SD) const {
  // The logical size is the address space size minus any tail padding.
  uint64_t Size = getSectionAddressSize(SD);
  const MCAlignFragment *AF =
    dyn_cast<MCAlignFragment>(&(SD->getFragmentList().back()));
  if (AF && AF->hasOnlyAlignAddress())
    Size -= getFragmentEffectiveSize(AF);

  return Size;
}

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::~MCFragment() {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *_Parent)
  : Kind(_Kind), Parent(_Parent), Atom(0), Offset(~UINT64_C(0)),
    EffectiveSize(~UINT64_C(0))
{
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

/* *** */

MCSectionData::MCSectionData() : Section(0) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(&_Section),
    Alignment(1),
    Address(~UINT64_C(0)),
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
                                                const MCFixup &Fixup,
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
                                          const MCFixup &Fixup,
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

bool MCAssembler::isSymbolLinkerVisible(const MCSymbol &Symbol) const {
  // Non-temporary labels should always be visible to the linker.
  if (!Symbol.isTemporary())
    return true;

  // Absolute temporary labels are never visible.
  if (!Symbol.isInSection())
    return false;

  // Otherwise, check if the section requires symbols even for temporary labels.
  return getBackend().doesSectionRequireSymbols(Symbol.getSection());
}

const MCSymbolData *MCAssembler::getAtom(const MCAsmLayout &Layout,
                                         const MCSymbolData *SD) const {
  // Linker visible symbols define atoms.
  if (isSymbolLinkerVisible(SD->getSymbol()))
    return SD;

  // Absolute and undefined symbols have no defining atom.
  if (!SD->getFragment())
    return 0;

  // Non-linker visible symbols in sections which can't be atomized have no
  // defining atom.
  if (!getBackend().isSectionAtomizable(
        SD->getFragment()->getParent()->getSection()))
    return 0;

  // Otherwise, return the atom for the containing fragment.
  return SD->getFragment()->getAtom();
}

bool MCAssembler::EvaluateFixup(const MCAsmLayout &Layout,
                                const MCFixup &Fixup, const MCFragment *DF,
                                MCValue &Target, uint64_t &Value) const {
  ++stats::EvaluateFixup;

  if (!Fixup.getValue()->EvaluateAsRelocatable(Target, &Layout))
    report_fatal_error("expected relocatable expression");

  // FIXME: How do non-scattered symbols work in ELF? I presume the linker
  // doesn't support small relocations, but then under what criteria does the
  // assembler allow symbol differences?

  Value = Target.getConstant();

  bool IsPCRel = Emitter.getFixupKindInfo(
    Fixup.getKind()).Flags & MCFixupKindInfo::FKF_IsPCRel;
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
        BaseSymbol = DF->getAtom();
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
    Value -= Layout.getFragmentAddress(DF) + Fixup.getOffset();

  return IsResolved;
}

uint64_t MCAssembler::ComputeFragmentSize(MCAsmLayout &Layout,
                                          const MCFragment &F,
                                          uint64_t SectionAddress,
                                          uint64_t FragmentOffset) const {
  switch (F.getKind()) {
  case MCFragment::FT_Data:
    return cast<MCDataFragment>(F).getContents().size();
  case MCFragment::FT_Fill:
    return cast<MCFillFragment>(F).getSize();
  case MCFragment::FT_Inst:
    return cast<MCInstFragment>(F).getInstSize();

  case MCFragment::FT_Align: {
    const MCAlignFragment &AF = cast<MCAlignFragment>(F);

    assert((!AF.hasOnlyAlignAddress() || !AF.getNextNode()) &&
           "Invalid OnlyAlignAddress bit, not the last fragment!");

    uint64_t Size = OffsetToAlignment(SectionAddress + FragmentOffset,
                                      AF.getAlignment());

    // Honor MaxBytesToEmit.
    if (Size > AF.getMaxBytesToEmit())
      return 0;

    return Size;
  }

  case MCFragment::FT_Org: {
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);

    // FIXME: We should compute this sooner, we don't want to recurse here, and
    // we would like to be more functional.
    int64_t TargetLocation;
    if (!OF.getOffset().EvaluateAsAbsolute(TargetLocation, &Layout))
      report_fatal_error("expected assembly-time absolute expression");

    // FIXME: We need a way to communicate this error.
    int64_t Offset = TargetLocation - FragmentOffset;
    if (Offset < 0)
      report_fatal_error("invalid .org offset '" + Twine(TargetLocation) +
                         "' (at offset '" + Twine(FragmentOffset) + "'");

    return Offset;
  }
  }

  assert(0 && "invalid fragment kind");
  return 0;
}

void MCAsmLayout::LayoutFile() {
  // Initialize the first section and set the valid fragment layout point. All
  // actual layout computations are done lazily.
  LastValidFragment = 0;
  if (!getSectionOrder().empty())
    getSectionOrder().front()->Address = 0;
}

void MCAsmLayout::LayoutFragment(MCFragment *F) {
  MCFragment *Prev = F->getPrevNode();

  // We should never try to recompute something which is up-to-date.
  assert(!isFragmentUpToDate(F) && "Attempt to recompute up-to-date fragment!");
  // We should never try to compute the fragment layout if the section isn't
  // up-to-date.
  assert(isSectionUpToDate(F->getParent()) &&
         "Attempt to compute fragment before it's section!");
  // We should never try to compute the fragment layout if it's predecessor
  // isn't up-to-date.
  assert((!Prev || isFragmentUpToDate(Prev)) &&
         "Attempt to compute fragment before it's predecessor!");

  ++stats::FragmentLayouts;

  // Compute the fragment start address.
  uint64_t StartAddress = F->getParent()->Address;
  uint64_t Address = StartAddress;
  if (Prev)
    Address += Prev->Offset + Prev->EffectiveSize;

  // Compute fragment offset and size.
  F->Offset = Address - StartAddress;
  F->EffectiveSize = getAssembler().ComputeFragmentSize(*this, *F, StartAddress,
                                                        F->Offset);
  LastValidFragment = F;

  // If this is the last fragment in a section, update the next section address.
  if (!F->getNextNode()) {
    unsigned NextIndex = F->getParent()->getLayoutOrder() + 1;
    if (NextIndex != getSectionOrder().size())
      LayoutSection(getSectionOrder()[NextIndex]);
  }
}

void MCAsmLayout::LayoutSection(MCSectionData *SD) {
  unsigned SectionOrderIndex = SD->getLayoutOrder();

  ++stats::SectionLayouts;

  // Compute the section start address.
  uint64_t StartAddress = 0;
  if (SectionOrderIndex) {
    MCSectionData *Prev = getSectionOrder()[SectionOrderIndex - 1];
    StartAddress = getSectionAddress(Prev) + getSectionAddressSize(Prev);
  }

  // Honor the section alignment requirements.
  StartAddress = RoundUpToAlignment(StartAddress, SD->getAlignment());

  // Set the section address.
  SD->Address = StartAddress;
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

    assert(AF.getValueSize() && "Invalid virtual align in concrete fragment!");

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
    if (AF.hasEmitNops()) {
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

    assert(FF.getValueSize() && "Invalid virtual align in concrete fragment!");

    for (uint64_t i = 0, e = FF.getSize() / FF.getValueSize(); i != e; ++i) {
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
  }

  assert(OW->getStream().tell() - Start == FragmentSize);
}

void MCAssembler::WriteSectionData(const MCSectionData *SD,
                                   const MCAsmLayout &Layout,
                                   MCObjectWriter *OW) const {
  // Ignore virtual sections.
  if (getBackend().isVirtualSection(SD->getSection())) {
    assert(Layout.getSectionFileSize(SD) == 0 && "Invalid size for section!");

    // Check that contents are only things legal inside a virtual section.
    for (MCSectionData::const_iterator it = SD->begin(),
           ie = SD->end(); it != ie; ++it) {
      switch (it->getKind()) {
      default:
        assert(0 && "Invalid fragment in virtual section!");
      case MCFragment::FT_Align:
        assert(!cast<MCAlignFragment>(it)->getValueSize() &&
               "Invalid align in virtual section!");
        break;
      case MCFragment::FT_Fill:
        assert(!cast<MCFillFragment>(it)->getValueSize() &&
               "Invalid fill in virtual section!");
        break;
      }
    }

    return;
  }

  uint64_t Start = OW->getStream().tell();
  (void) Start;

  for (MCSectionData::const_iterator it = SD->begin(),
         ie = SD->end(); it != ie; ++it)
    WriteFragmentData(*this, Layout, *it, OW);

  assert(OW->getStream().tell() - Start == Layout.getSectionFileSize(SD));
}

void MCAssembler::Finish(MCObjectWriter *Writer) {
  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - pre-layout\n--\n";
      dump(); });

  // Create the layout object.
  MCAsmLayout Layout(*this);

  // Insert additional align fragments for concrete sections to explicitly pad
  // the previous section to match their alignment requirements. This is for
  // 'gas' compatibility, it shouldn't strictly be necessary.
  //
  // FIXME: This may be Mach-O specific.
  for (unsigned i = 1, e = Layout.getSectionOrder().size(); i < e; ++i) {
    MCSectionData *SD = Layout.getSectionOrder()[i];

    // Ignore sections without alignment requirements.
    unsigned Align = SD->getAlignment();
    if (Align <= 1)
      continue;

    // Ignore virtual sections, they don't cause file size modifications.
    if (getBackend().isVirtualSection(SD->getSection()))
      continue;

    // Otherwise, create a new align fragment at the end of the previous
    // section.
    MCAlignFragment *AF = new MCAlignFragment(Align, 0, 1, Align,
                                              Layout.getSectionOrder()[i - 1]);
    AF->setOnlyAlignAddress(true);
  }

  // Create dummy fragments and assign section ordinals.
  unsigned SectionIndex = 0;
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    // Create dummy fragments to eliminate any empty sections, this simplifies
    // layout.
    if (it->getFragmentList().empty())
      new MCFillFragment(0, 1, 0, it);

    it->setOrdinal(SectionIndex++);
  }

  // Assign layout order indices to sections and fragments.
  unsigned FragmentIndex = 0;
  for (unsigned i = 0, e = Layout.getSectionOrder().size(); i != e; ++i) {
    MCSectionData *SD = Layout.getSectionOrder()[i];
    SD->setLayoutOrder(i);

    for (MCSectionData::iterator it2 = SD->begin(),
           ie2 = SD->end(); it2 != ie2; ++it2)
      it2->setLayoutOrder(FragmentIndex++);
  }

  // Layout until everything fits.
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

  llvm::OwningPtr<MCObjectWriter> OwnWriter(0);
  if (Writer == 0) {
    //no custom Writer_ : create the default one life-managed by OwningPtr
    OwnWriter.reset(getBackend().createObjectWriter(OS));
    Writer = OwnWriter.get();
    if (!Writer)
      report_fatal_error("unable to create object writer!");
  }

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
        MCFixup &Fixup = *it3;

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

  stats::ObjectBytes += OS.tell() - StartOffset;
}

bool MCAssembler::FixupNeedsRelaxation(const MCFixup &Fixup,
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
  if (!getBackend().MayNeedRelaxation(IF->getInst()))
    return false;

  for (MCInstFragment::const_fixup_iterator it = IF->fixup_begin(),
         ie = IF->fixup_end(); it != ie; ++it)
    if (FixupNeedsRelaxation(*it, IF, Layout))
      return true;

  return false;
}

bool MCAssembler::LayoutOnce(MCAsmLayout &Layout) {
  ++stats::RelaxationSteps;

  // Layout the sections in order.
  Layout.LayoutFile();

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
      getBackend().RelaxInstruction(IF->getInst(), Relaxed);

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
      // FIXME: Eliminate copy.
      for (unsigned i = 0, e = Fixups.size(); i != e; ++i)
        IF->getFixups().push_back(Fixups[i]);

      // Update the layout, and remember that we relaxed.
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
      DF->setParent(IF->getParent());
      DF->setAtom(IF->getAtom());
      DF->setLayoutOrder(IF->getLayoutOrder());
      Layout.FragmentReplaced(IF, DF);

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

raw_ostream &operator<<(raw_ostream &OS, const MCFixup &AF) {
  OS << "<MCFixup" << " Offset:" << AF.getOffset()
     << " Value:" << *AF.getValue()
     << " Kind:" << AF.getKind() << ">";
  return OS;
}

}

void MCFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<";
  switch (getKind()) {
  case MCFragment::FT_Align: OS << "MCAlignFragment"; break;
  case MCFragment::FT_Data:  OS << "MCDataFragment"; break;
  case MCFragment::FT_Fill:  OS << "MCFillFragment"; break;
  case MCFragment::FT_Inst:  OS << "MCInstFragment"; break;
  case MCFragment::FT_Org:   OS << "MCOrgFragment"; break;
  }

  OS << "<MCFragment " << (void*) this << " LayoutOrder:" << LayoutOrder
     << " Offset:" << Offset << " EffectiveSize:" << EffectiveSize << ">";

  switch (getKind()) {
  case MCFragment::FT_Align: {
    const MCAlignFragment *AF = cast<MCAlignFragment>(this);
    if (AF->hasEmitNops())
      OS << " (emit nops)";
    if (AF->hasOnlyAlignAddress())
      OS << " (only align section)";
    OS << "\n       ";
    OS << " Alignment:" << AF->getAlignment()
       << " Value:" << AF->getValue() << " ValueSize:" << AF->getValueSize()
       << " MaxBytesToEmit:" << AF->getMaxBytesToEmit() << ">";
    break;
  }
  case MCFragment::FT_Data:  {
    const MCDataFragment *DF = cast<MCDataFragment>(this);
    OS << "\n       ";
    OS << " Contents:[";
    const SmallVectorImpl<char> &Contents = DF->getContents();
    for (unsigned i = 0, e = Contents.size(); i != e; ++i) {
      if (i) OS << ",";
      OS << hexdigit((Contents[i] >> 4) & 0xF) << hexdigit(Contents[i] & 0xF);
    }
    OS << "] (" << Contents.size() << " bytes)";

    if (!DF->getFixups().empty()) {
      OS << ",\n       ";
      OS << " Fixups:[";
      for (MCDataFragment::const_fixup_iterator it = DF->fixup_begin(),
             ie = DF->fixup_end(); it != ie; ++it) {
        if (it != DF->fixup_begin()) OS << ",\n                ";
        OS << *it;
      }
      OS << "]";
    }
    break;
  }
  case MCFragment::FT_Fill:  {
    const MCFillFragment *FF = cast<MCFillFragment>(this);
    OS << " Value:" << FF->getValue() << " ValueSize:" << FF->getValueSize()
       << " Size:" << FF->getSize();
    break;
  }
  case MCFragment::FT_Inst:  {
    const MCInstFragment *IF = cast<MCInstFragment>(this);
    OS << "\n       ";
    OS << " Inst:";
    IF->getInst().dump_pretty(OS);
    break;
  }
  case MCFragment::FT_Org:  {
    const MCOrgFragment *OF = cast<MCOrgFragment>(this);
    OS << "\n       ";
    OS << " Offset:" << OF->getOffset() << " Value:" << OF->getValue();
    break;
  }
  }
  OS << ">";
}

void MCSectionData::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCSectionData";
  OS << " Alignment:" << getAlignment() << " Address:" << Address
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
