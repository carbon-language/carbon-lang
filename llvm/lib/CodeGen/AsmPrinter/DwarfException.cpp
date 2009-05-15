//===-- CodeGen/AsmPrinter/DwarfException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

static TimerGroup &getDwarfTimerGroup() {
  static TimerGroup DwarfTimerGroup("Dwarf Exception");
  return DwarfTimerGroup;
}

void DwarfException::EmitCommonEHFrame(const Function *Personality,
                                       unsigned Index) {
  // Size and sign of stack growth.
  int stackGrowth =
    Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
    TargetFrameInfo::StackGrowsUp ?
    TD->getPointerSize() : -TD->getPointerSize();

  // Begin eh frame section.
  Asm->SwitchToTextSection(TAI->getDwarfEHFrameSection());

  if (!TAI->doesRequireNonLocalEHFrameLabel())
    O << TAI->getEHGlobalPrefix();

  O << "EH_frame" << Index << ":\n";
  EmitLabel("section_eh_frame", Index);

  // Define base labels.
  EmitLabel("eh_frame_common", Index);

  // Define the eh frame length.
  EmitDifference("eh_frame_common_end", Index,
                 "eh_frame_common_begin", Index, true);
  Asm->EOL("Length of Common Information Entry");

  // EH frame header.
  EmitLabel("eh_frame_common_begin", Index);
  Asm->EmitInt32((int)0);
  Asm->EOL("CIE Identifier Tag");
  Asm->EmitInt8(dwarf::DW_CIE_VERSION);
  Asm->EOL("CIE Version");

  // The personality presence indicates that language specific information will
  // show up in the eh frame.
  Asm->EmitString(Personality ? "zPLR" : "zR");
  Asm->EOL("CIE Augmentation");

  // Round out reader.
  Asm->EmitULEB128Bytes(1);
  Asm->EOL("CIE Code Alignment Factor");
  Asm->EmitSLEB128Bytes(stackGrowth);
  Asm->EOL("CIE Data Alignment Factor");
  Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), true));
  Asm->EOL("CIE Return Address Column");

  // If there is a personality, we need to indicate the functions location.
  if (Personality) {
    Asm->EmitULEB128Bytes(7);
    Asm->EOL("Augmentation Size");

    if (TAI->getNeedsIndirectEncoding()) {
      Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4 |
                    dwarf::DW_EH_PE_indirect);
      Asm->EOL("Personality (pcrel sdata4 indirect)");
    } else {
      Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
      Asm->EOL("Personality (pcrel sdata4)");
    }

    PrintRelDirective(true);
    O << TAI->getPersonalityPrefix();
    Asm->EmitExternalGlobal((const GlobalVariable *)(Personality));
    O << TAI->getPersonalitySuffix();
    if (strcmp(TAI->getPersonalitySuffix(), "+4@GOTPCREL"))
      O << "-" << TAI->getPCSymbol();
    Asm->EOL("Personality");

    Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    Asm->EOL("LSDA Encoding (pcrel sdata4)");

    Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    Asm->EOL("FDE Encoding (pcrel sdata4)");
  } else {
    Asm->EmitULEB128Bytes(1);
    Asm->EOL("Augmentation Size");

    Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    Asm->EOL("FDE Encoding (pcrel sdata4)");
  }

  // Indicate locations of general callee saved registers in frame.
  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);
  EmitFrameMoves(NULL, 0, Moves, true);

  // On Darwin the linker honors the alignment of eh_frame, which means it must
  // be 8-byte on 64-bit targets to match what gcc does.  Otherwise you get
  // holes which confuse readers of eh_frame.
  Asm->EmitAlignment(TD->getPointerSize() == sizeof(int32_t) ? 2 : 3,
                     0, 0, false);
  EmitLabel("eh_frame_common_end", Index);

  Asm->EOL();
}

/// EmitEHFrame - Emit function exception frame information.
///
void DwarfException::EmitEHFrame(const FunctionEHFrameInfo &EHFrameInfo) {
  assert(!EHFrameInfo.function->hasAvailableExternallyLinkage() && 
         "Should not emit 'available externally' functions at all");

  Function::LinkageTypes linkage = EHFrameInfo.function->getLinkage();
  Asm->SwitchToTextSection(TAI->getDwarfEHFrameSection());

  // Externally visible entry into the functions eh frame info. If the
  // corresponding function is static, this should not be externally visible.
  if (linkage != Function::InternalLinkage &&
      linkage != Function::PrivateLinkage) {
    if (const char *GlobalEHDirective = TAI->getGlobalEHDirective())
      O << GlobalEHDirective << EHFrameInfo.FnName << "\n";
  }

  // If corresponding function is weak definition, this should be too.
  if ((linkage == Function::WeakAnyLinkage ||
       linkage == Function::WeakODRLinkage ||
       linkage == Function::LinkOnceAnyLinkage ||
       linkage == Function::LinkOnceODRLinkage) &&
      TAI->getWeakDefDirective())
    O << TAI->getWeakDefDirective() << EHFrameInfo.FnName << "\n";

  // If there are no calls then you can't unwind.  This may mean we can omit the
  // EH Frame, but some environments do not handle weak absolute symbols. If
  // UnwindTablesMandatory is set we cannot do this optimization; the unwind
  // info is to be available for non-EH uses.
  if (!EHFrameInfo.hasCalls &&
      !UnwindTablesMandatory &&
      ((linkage != Function::WeakAnyLinkage &&
        linkage != Function::WeakODRLinkage &&
        linkage != Function::LinkOnceAnyLinkage &&
        linkage != Function::LinkOnceODRLinkage) ||
       !TAI->getWeakDefDirective() ||
       TAI->getSupportsWeakOmittedEHFrame())) {
    O << EHFrameInfo.FnName << " = 0\n";
    // This name has no connection to the function, so it might get
    // dead-stripped when the function is not, erroneously.  Prohibit
    // dead-stripping unconditionally.
    if (const char *UsedDirective = TAI->getUsedDirective())
      O << UsedDirective << EHFrameInfo.FnName << "\n\n";
  } else {
    O << EHFrameInfo.FnName << ":\n";

    // EH frame header.
    EmitDifference("eh_frame_end", EHFrameInfo.Number,
                   "eh_frame_begin", EHFrameInfo.Number, true);
    Asm->EOL("Length of Frame Information Entry");

    EmitLabel("eh_frame_begin", EHFrameInfo.Number);

    if (TAI->doesRequireNonLocalEHFrameLabel()) {
      PrintRelDirective(true, true);
      PrintLabelName("eh_frame_begin", EHFrameInfo.Number);

      if (!TAI->isAbsoluteEHSectionOffsets())
        O << "-EH_frame" << EHFrameInfo.PersonalityIndex;
    } else {
      EmitSectionOffset("eh_frame_begin", "eh_frame_common",
                        EHFrameInfo.Number, EHFrameInfo.PersonalityIndex,
                        true, true, false);
    }

    Asm->EOL("FDE CIE offset");

    EmitReference("eh_func_begin", EHFrameInfo.Number, true, true);
    Asm->EOL("FDE initial location");
    EmitDifference("eh_func_end", EHFrameInfo.Number,
                   "eh_func_begin", EHFrameInfo.Number, true);
    Asm->EOL("FDE address range");

    // If there is a personality and landing pads then point to the language
    // specific data area in the exception table.
    if (EHFrameInfo.PersonalityIndex) {
      Asm->EmitULEB128Bytes(4);
      Asm->EOL("Augmentation size");

      if (EHFrameInfo.hasLandingPads)
        EmitReference("exception", EHFrameInfo.Number, true, true);
      else
        Asm->EmitInt32((int)0);
      Asm->EOL("Language Specific Data Area");
    } else {
      Asm->EmitULEB128Bytes(0);
      Asm->EOL("Augmentation size");
    }

    // Indicate locations of function specific callee saved registers in frame.
    EmitFrameMoves("eh_func_begin", EHFrameInfo.Number, EHFrameInfo.Moves, 
                   true);

    // On Darwin the linker honors the alignment of eh_frame, which means it
    // must be 8-byte on 64-bit targets to match what gcc does.  Otherwise you
    // get holes which confuse readers of eh_frame.
    Asm->EmitAlignment(TD->getPointerSize() == sizeof(int32_t) ? 2 : 3,
                       0, 0, false);
    EmitLabel("eh_frame_end", EHFrameInfo.Number);

    // If the function is marked used, this table should be also.  We cannot
    // make the mark unconditional in this case, since retaining the table also
    // retains the function in this case, and there is code around that depends
    // on unused functions (calling undefined externals) being dead-stripped to
    // link correctly.  Yes, there really is.
    if (MMI->getUsedFunctions().count(EHFrameInfo.function))
      if (const char *UsedDirective = TAI->getUsedDirective())
        O << UsedDirective << EHFrameInfo.FnName << "\n\n";
  }
}

/// EmitExceptionTable - Emit landing pads and actions.
///
/// The general organization of the table is complex, but the basic concepts are
/// easy.  First there is a header which describes the location and organization
/// of the three components that follow.
/// 
///  1. The landing pad site information describes the range of code covered by
///     the try.  In our case it's an accumulation of the ranges covered by the
///     invokes in the try.  There is also a reference to the landing pad that
///     handles the exception once processed.  Finally an index into the actions
///     table.
///  2. The action table, in our case, is composed of pairs of type ids and next
///     action offset.  Starting with the action index from the landing pad
///     site, each type Id is checked for a match to the current exception.  If
///     it matches then the exception and type id are passed on to the landing
///     pad.  Otherwise the next action is looked up.  This chain is terminated
///     with a next action of zero.  If no type id is found the the frame is
///     unwound and handling continues.
///  3. Type id table contains references to all the C++ typeinfo for all
///     catches in the function.  This tables is reversed indexed base 1.

/// SharedTypeIds - How many leading type ids two landing pads have in common.
unsigned DwarfException::SharedTypeIds(const LandingPadInfo *L,
                                       const LandingPadInfo *R) {
  const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
  unsigned LSize = LIds.size(), RSize = RIds.size();
  unsigned MinSize = LSize < RSize ? LSize : RSize;
  unsigned Count = 0;

  for (; Count != MinSize; ++Count)
    if (LIds[Count] != RIds[Count])
      return Count;

  return Count;
}

/// PadLT - Order landing pads lexicographically by type id.
bool DwarfException::PadLT(const LandingPadInfo *L, const LandingPadInfo *R) {
  const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
  unsigned LSize = LIds.size(), RSize = RIds.size();
  unsigned MinSize = LSize < RSize ? LSize : RSize;

  for (unsigned i = 0; i != MinSize; ++i)
    if (LIds[i] != RIds[i])
      return LIds[i] < RIds[i];

  return LSize < RSize;
}

void DwarfException::EmitExceptionTable() {
  const std::vector<GlobalVariable *> &TypeInfos = MMI->getTypeInfos();
  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  if (PadInfos.empty()) return;

  // Sort the landing pads in order of their type ids.  This is used to fold
  // duplicate actions.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());
  for (unsigned i = 0, N = PadInfos.size(); i != N; ++i)
    LandingPads.push_back(&PadInfos[i]);
  std::sort(LandingPads.begin(), LandingPads.end(), PadLT);

  // Negative type ids index into FilterIds, positive type ids index into
  // TypeInfos.  The value written for a positive type id is just the type id
  // itself.  For a negative type id, however, the value written is the
  // (negative) byte offset of the corresponding FilterIds entry.  The byte
  // offset is usually equal to the type id, because the FilterIds entries are
  // written using a variable width encoding which outputs one byte per entry as
  // long as the value written is not too large, but can differ.  This kind of
  // complication does not occur for positive type ids because type infos are
  // output using a fixed width encoding.  FilterOffsets[i] holds the byte
  // offset corresponding to FilterIds[i].
  SmallVector<int, 16> FilterOffsets;
  FilterOffsets.reserve(FilterIds.size());
  int Offset = -1;
  for(std::vector<unsigned>::const_iterator I = FilterIds.begin(),
        E = FilterIds.end(); I != E; ++I) {
    FilterOffsets.push_back(Offset);
    Offset -= TargetAsmInfo::getULEB128Size(*I);
  }

  // Compute the actions table and gather the first action index for each
  // landing pad site.
  SmallVector<ActionEntry, 32> Actions;
  SmallVector<unsigned, 64> FirstActions;
  FirstActions.reserve(LandingPads.size());

  int FirstAction = 0;
  unsigned SizeActions = 0;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LP = LandingPads[i];
    const std::vector<int> &TypeIds = LP->TypeIds;
    const unsigned NumShared = i ? SharedTypeIds(LP, LandingPads[i-1]) : 0;
    unsigned SizeSiteActions = 0;

    if (NumShared < TypeIds.size()) {
      unsigned SizeAction = 0;
      ActionEntry *PrevAction = 0;

      if (NumShared) {
        const unsigned SizePrevIds = LandingPads[i-1]->TypeIds.size();
        assert(Actions.size());
        PrevAction = &Actions.back();
        SizeAction = TargetAsmInfo::getSLEB128Size(PrevAction->NextAction) +
          TargetAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);

        for (unsigned j = NumShared; j != SizePrevIds; ++j) {
          SizeAction -=
            TargetAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
          SizeAction += -PrevAction->NextAction;
          PrevAction = PrevAction->Previous;
        }
      }

      // Compute the actions.
      for (unsigned I = NumShared, M = TypeIds.size(); I != M; ++I) {
        int TypeID = TypeIds[I];
        assert(-1-TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
        int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
        unsigned SizeTypeID = TargetAsmInfo::getSLEB128Size(ValueForTypeID);

        int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
        SizeAction = SizeTypeID + TargetAsmInfo::getSLEB128Size(NextAction);
        SizeSiteActions += SizeAction;

        ActionEntry Action = {ValueForTypeID, NextAction, PrevAction};
        Actions.push_back(Action);

        PrevAction = &Actions.back();
      }

      // Record the first action of the landing pad site.
      FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
    } // else identical - re-use previous FirstAction

    FirstActions.push_back(FirstAction);

    // Compute this sites contribution to size.
    SizeActions += SizeSiteActions;
  }

  // Compute the call-site table.  The entry for an invoke has a try-range
  // containing the call, a non-zero landing pad and an appropriate action.  The
  // entry for an ordinary call has a try-range containing the call and zero for
  // the landing pad and the action.  Calls marked 'nounwind' have no entry and
  // must not be contained in the try-range of any entry - they form gaps in the
  // table.  Entries must be ordered by try-range address.
  SmallVector<CallSiteEntry, 64> CallSites;

  RangeMapType PadMap;

  // Invokes and nounwind calls have entries in PadMap (due to being bracketed
  // by try-range labels when lowered).  Ordinary calls do not, so appropriate
  // try-ranges for them need be deduced.
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LandingPad = LandingPads[i];
    for (unsigned j = 0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
      unsigned BeginLabel = LandingPad->BeginLabels[j];
      assert(!PadMap.count(BeginLabel) && "Duplicate landing pad labels!");
      PadRange P = { i, j };
      PadMap[BeginLabel] = P;
    }
  }

  // The end label of the previous invoke or nounwind try-range.
  unsigned LastLabel = 0;

  // Whether there is a potentially throwing instruction (currently this means
  // an ordinary call) between the end of the previous try-range and now.
  bool SawPotentiallyThrowing = false;

  // Whether the last callsite entry was for an invoke.
  bool PreviousIsInvoke = false;

  // Visit all instructions in order of address.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
         MI != E; ++MI) {
      if (!MI->isLabel()) {
        SawPotentiallyThrowing |= MI->getDesc().isCall();
        continue;
      }

      unsigned BeginLabel = MI->getOperand(0).getImm();
      assert(BeginLabel && "Invalid label!");

      // End of the previous try-range?
      if (BeginLabel == LastLabel)
        SawPotentiallyThrowing = false;

      // Beginning of a new try-range?
      RangeMapType::iterator L = PadMap.find(BeginLabel);
      if (L == PadMap.end())
        // Nope, it was just some random label.
        continue;

      PadRange P = L->second;
      const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];

      assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
             "Inconsistent landing pad map!");

      // If some instruction between the previous try-range and this one may
      // throw, create a call-site entry with no landing pad for the region
      // between the try-ranges.
      if (SawPotentiallyThrowing) {
        CallSiteEntry Site = {LastLabel, BeginLabel, 0, 0};
        CallSites.push_back(Site);
        PreviousIsInvoke = false;
      }

      LastLabel = LandingPad->EndLabels[P.RangeIndex];
      assert(BeginLabel && LastLabel && "Invalid landing pad!");

      if (LandingPad->LandingPadLabel) {
        // This try-range is for an invoke.
        CallSiteEntry Site = {BeginLabel, LastLabel,
                              LandingPad->LandingPadLabel,
                              FirstActions[P.PadIndex]};

        // Try to merge with the previous call-site.
        if (PreviousIsInvoke) {
          CallSiteEntry &Prev = CallSites.back();
          if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
            // Extend the range of the previous entry.
            Prev.EndLabel = Site.EndLabel;
            continue;
          }
        }

        // Otherwise, create a new call-site.
        CallSites.push_back(Site);
        PreviousIsInvoke = true;
      } else {
        // Create a gap.
        PreviousIsInvoke = false;
      }
    }
  }

  // If some instruction between the previous try-range and the end of the
  // function may throw, create a call-site entry with no landing pad for the
  // region following the try-range.
  if (SawPotentiallyThrowing) {
    CallSiteEntry Site = {LastLabel, 0, 0, 0};
    CallSites.push_back(Site);
  }

  // Final tallies.

  // Call sites.
  const unsigned SiteStartSize  = sizeof(int32_t); // DW_EH_PE_udata4
  const unsigned SiteLengthSize = sizeof(int32_t); // DW_EH_PE_udata4
  const unsigned LandingPadSize = sizeof(int32_t); // DW_EH_PE_udata4
  unsigned SizeSites = CallSites.size() * (SiteStartSize +
                                           SiteLengthSize +
                                           LandingPadSize);
  for (unsigned i = 0, e = CallSites.size(); i < e; ++i)
    SizeSites += TargetAsmInfo::getULEB128Size(CallSites[i].Action);

  // Type infos.
  const unsigned TypeInfoSize = TD->getPointerSize(); // DW_EH_PE_absptr
  unsigned SizeTypes = TypeInfos.size() * TypeInfoSize;

  unsigned TypeOffset = sizeof(int8_t) + // Call site format
    TargetAsmInfo::getULEB128Size(SizeSites) + // Call-site table length
    SizeSites + SizeActions + SizeTypes;

  unsigned TotalSize = sizeof(int8_t) + // LPStart format
                       sizeof(int8_t) + // TType format
           TargetAsmInfo::getULEB128Size(TypeOffset) + // TType base offset
                       TypeOffset;

  unsigned SizeAlign = (4 - TotalSize) & 3;

  // Begin the exception table.
  Asm->SwitchToDataSection(TAI->getDwarfExceptionSection());
  Asm->EmitAlignment(2, 0, 0, false);
  O << "GCC_except_table" << SubprogramCount << ":\n";

  for (unsigned i = 0; i != SizeAlign; ++i) {
    Asm->EmitInt8(0);
    Asm->EOL("Padding");
    }

  EmitLabel("exception", SubprogramCount);

  // Emit the header.
  Asm->EmitInt8(dwarf::DW_EH_PE_omit);
  Asm->EOL("LPStart format (DW_EH_PE_omit)");
  Asm->EmitInt8(dwarf::DW_EH_PE_absptr);
  Asm->EOL("TType format (DW_EH_PE_absptr)");
  Asm->EmitULEB128Bytes(TypeOffset);
  Asm->EOL("TType base offset");
  Asm->EmitInt8(dwarf::DW_EH_PE_udata4);
  Asm->EOL("Call site format (DW_EH_PE_udata4)");
  Asm->EmitULEB128Bytes(SizeSites);
  Asm->EOL("Call-site table length");

  // Emit the landing pad site information.
  for (unsigned i = 0; i < CallSites.size(); ++i) {
    CallSiteEntry &S = CallSites[i];
    const char *BeginTag;
    unsigned BeginNumber;

    if (!S.BeginLabel) {
      BeginTag = "eh_func_begin";
      BeginNumber = SubprogramCount;
    } else {
      BeginTag = "label";
      BeginNumber = S.BeginLabel;
    }

    EmitSectionOffset(BeginTag, "eh_func_begin", BeginNumber, SubprogramCount,
                      true, true);
    Asm->EOL("Region start");

    if (!S.EndLabel)
      EmitDifference("eh_func_end", SubprogramCount, BeginTag, BeginNumber,
                     true);
    else
      EmitDifference("label", S.EndLabel, BeginTag, BeginNumber, true);

    Asm->EOL("Region length");

    if (!S.PadLabel)
      Asm->EmitInt32(0);
    else
      EmitSectionOffset("label", "eh_func_begin", S.PadLabel, SubprogramCount,
                        true, true);

    Asm->EOL("Landing pad");

    Asm->EmitULEB128Bytes(S.Action);
    Asm->EOL("Action");
  }

  // Emit the actions.
  for (unsigned I = 0, N = Actions.size(); I != N; ++I) {
    ActionEntry &Action = Actions[I];

    Asm->EmitSLEB128Bytes(Action.ValueForTypeID);
    Asm->EOL("TypeInfo index");
    Asm->EmitSLEB128Bytes(Action.NextAction);
    Asm->EOL("Next action");
  }

  // Emit the type ids.
  for (unsigned M = TypeInfos.size(); M; --M) {
    GlobalVariable *GV = TypeInfos[M - 1];
    PrintRelDirective();

    if (GV) {
      std::string GLN;
      O << Asm->getGlobalLinkName(GV, GLN);
    } else {
      O << "0";
    }

    Asm->EOL("TypeInfo");
  }

  // Emit the filter typeids.
  for (unsigned j = 0, M = FilterIds.size(); j < M; ++j) {
    unsigned TypeID = FilterIds[j];
    Asm->EmitULEB128Bytes(TypeID);
    Asm->EOL("Filter TypeInfo index");
  }

  Asm->EmitAlignment(2, 0, 0, false);
}

  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
DwarfException::DwarfException(raw_ostream &OS, AsmPrinter *A,
                               const TargetAsmInfo *T)
  : Dwarf(OS, A, T, "eh"), shouldEmitTable(false), shouldEmitMoves(false),
    shouldEmitTableModule(false), shouldEmitMovesModule(false),
    ExceptionTimer(0) {
  if (TimePassesIsEnabled) 
    ExceptionTimer = new Timer("Dwarf Exception Writer",
                               getDwarfTimerGroup());
}

DwarfException::~DwarfException() {
  delete ExceptionTimer;
}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfException::EndModule() {
  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  if (shouldEmitMovesModule || shouldEmitTableModule) {
    const std::vector<Function *> Personalities = MMI->getPersonalities();
    for (unsigned i = 0; i < Personalities.size(); ++i)
      EmitCommonEHFrame(Personalities[i], i);

    for (std::vector<FunctionEHFrameInfo>::iterator I = EHFrames.begin(),
           E = EHFrames.end(); I != E; ++I)
      EmitEHFrame(*I);
  }

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}

/// BeginFunction - Gather pre-function exception information.  Assumes being
/// emitted immediately after the function entry point.
void DwarfException::BeginFunction(MachineFunction *MF) {
  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  this->MF = MF;
  shouldEmitTable = shouldEmitMoves = false;

  if (MMI && TAI->doesSupportExceptionHandling()) {
    // Map all labels and get rid of any dead landing pads.
    MMI->TidyLandingPads();

    // If any landing pads survive, we need an EH table.
    if (MMI->getLandingPads().size())
      shouldEmitTable = true;

    // See if we need frame move info.
    if (!MF->getFunction()->doesNotThrow() || UnwindTablesMandatory)
      shouldEmitMoves = true;

    if (shouldEmitMoves || shouldEmitTable)
      // Assumes in correct section after the entry point.
      EmitLabel("eh_func_begin", ++SubprogramCount);
  }

  shouldEmitTableModule |= shouldEmitTable;
  shouldEmitMovesModule |= shouldEmitMoves;

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfException::EndFunction() {
  if (TimePassesIsEnabled) 
    ExceptionTimer->startTimer();

  if (shouldEmitMoves || shouldEmitTable) {
    EmitLabel("eh_func_end", SubprogramCount);
    EmitExceptionTable();

    // Save EH frame information
    std::string Name;
    EHFrames.push_back(
        FunctionEHFrameInfo(getAsm()->getCurrentFunctionEHName(MF, Name),
                            SubprogramCount,
                            MMI->getPersonalityIndex(),
                            MF->getFrameInfo()->hasCalls(),
                            !MMI->getLandingPads().empty(),
                            MMI->getFrameMoves(),
                            MF->getFunction()));
  }

  if (TimePassesIsEnabled) 
    ExceptionTimer->stopTimer();
}
