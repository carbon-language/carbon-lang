//===-- CodeGen/AsmPrinter/DwarfException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing DWARF exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

DwarfException::DwarfException(raw_ostream &OS, AsmPrinter *A,
                               const MCAsmInfo *T)
  : Dwarf(OS, A, T, "eh"), shouldEmitTable(false), shouldEmitMoves(false),
    shouldEmitTableModule(false), shouldEmitMovesModule(false),
    ExceptionTimer(0) {
  if (TimePassesIsEnabled)
    ExceptionTimer = new Timer("DWARF Exception Writer");
}

DwarfException::~DwarfException() {
  delete ExceptionTimer;
}

/// SizeOfEncodedValue - Return the size of the encoding in bytes.
unsigned DwarfException::SizeOfEncodedValue(unsigned Encoding) {
  if (Encoding == dwarf::DW_EH_PE_omit)
    return 0;

  switch (Encoding & 0x07) {
  case dwarf::DW_EH_PE_absptr:
    return TD->getPointerSize();
  case dwarf::DW_EH_PE_udata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
    return 8;
  }

  assert(0 && "Invalid encoded value.");
  return 0;
}

/// CreateLabelDiff - Emit a label and subtract it from the expression we
/// already have.  This is equivalent to emitting "foo - .", but we have to emit
/// the label for "." directly.
const MCExpr *DwarfException::CreateLabelDiff(const MCExpr *ExprRef,
                                              const char *LabelName,
                                              unsigned Index) {
  SmallString<64> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix()
                            << LabelName << Asm->getFunctionNumber()
                            << "_" << Index;
  MCSymbol *DotSym = Asm->OutContext.GetOrCreateSymbol(Name.str());
  Asm->OutStreamer.EmitLabel(DotSym);

  return MCBinaryExpr::CreateSub(ExprRef,
                                 MCSymbolRefExpr::Create(DotSym,
                                                         Asm->OutContext),
                                 Asm->OutContext);
}

/// EmitCIE - Emit a Common Information Entry (CIE). This holds information that
/// is shared among many Frame Description Entries.  There is at least one CIE
/// in every non-empty .debug_frame section.
void DwarfException::EmitCIE(const Function *PersonalityFn, unsigned Index) {
  // Size and sign of stack growth.
  int stackGrowth =
    Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
    TargetFrameInfo::StackGrowsUp ?
    TD->getPointerSize() : -TD->getPointerSize();

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  
  // Begin eh frame section.
  Asm->OutStreamer.SwitchSection(TLOF.getEHFrameSection());

  if (MAI->is_EHSymbolPrivate())
    O << MAI->getPrivateGlobalPrefix();
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
  Asm->OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
  Asm->EOL("CIE Identifier Tag");
  Asm->EmitInt8(dwarf::DW_CIE_VERSION);
  Asm->EOL("CIE Version");

  // The personality presence indicates that language specific information will
  // show up in the eh frame.  Find out how we are supposed to lower the
  // personality function reference:
  const MCExpr *PersonalityRef = 0;
  bool IsPersonalityIndirect = false, IsPersonalityPCRel = false;
  if (PersonalityFn) {
    // FIXME: HANDLE STATIC CODEGEN MODEL HERE.
    
    // In non-static mode, ask the object file how to represent this reference.
    PersonalityRef =
      TLOF.getSymbolForDwarfGlobalReference(PersonalityFn, Asm->Mang,
                                            Asm->MMI,
                                            IsPersonalityIndirect,
                                            IsPersonalityPCRel);
  }
  
  unsigned PerEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
  if (IsPersonalityIndirect)
    PerEncoding |= dwarf::DW_EH_PE_indirect;
  unsigned LSDAEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
  unsigned FDEEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;

  char Augmentation[5] = { 0 };
  unsigned AugmentationSize = 0;
  char *APtr = Augmentation + 1;

  if (PersonalityRef) {
    // There is a personality function.
    *APtr++ = 'P';
    AugmentationSize += 1 + SizeOfEncodedValue(PerEncoding);
  }

  if (UsesLSDA[Index]) {
    // An LSDA pointer is in the FDE augmentation.
    *APtr++ = 'L';
    ++AugmentationSize;
  }

  if (FDEEncoding != dwarf::DW_EH_PE_absptr) {
    // A non-default pointer encoding for the FDE.
    *APtr++ = 'R';
    ++AugmentationSize;
  }

  if (APtr != Augmentation + 1)
    Augmentation[0] = 'z';

  Asm->EmitString(Augmentation);
  Asm->EOL("CIE Augmentation");

  // Round out reader.
  Asm->EmitULEB128Bytes(1);
  Asm->EOL("CIE Code Alignment Factor");
  Asm->EmitSLEB128Bytes(stackGrowth);
  Asm->EOL("CIE Data Alignment Factor");
  Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), true));
  Asm->EOL("CIE Return Address Column");

  Asm->EmitULEB128Bytes(AugmentationSize);
  Asm->EOL("Augmentation Size");

  Asm->EmitInt8(PerEncoding);
  Asm->EOL("Personality", PerEncoding);

  // If there is a personality, we need to indicate the function's location.
  if (PersonalityRef) {
    if (!IsPersonalityPCRel)
      PersonalityRef = CreateLabelDiff(PersonalityRef, "personalityref_addr",
                                       Index);

    O << MAI->getData32bitsDirective() << *PersonalityRef;
    Asm->EOL("Personality");

    Asm->EmitInt8(LSDAEncoding);
    Asm->EOL("LSDA Encoding", LSDAEncoding);

    Asm->EmitInt8(FDEEncoding);
    Asm->EOL("FDE Encoding", FDEEncoding);
  }

  // Indicate locations of general callee saved registers in frame.
  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);
  EmitFrameMoves(NULL, 0, Moves, true);

  // On Darwin the linker honors the alignment of eh_frame, which means it must
  // be 8-byte on 64-bit targets to match what gcc does.  Otherwise you get
  // holes which confuse readers of eh_frame.
  Asm->EmitAlignment(TD->getPointerSize() == 4 ? 2 : 3, 0, 0, false);
  EmitLabel("eh_frame_common_end", Index);

  Asm->EOL();
}

/// EmitFDE - Emit the Frame Description Entry (FDE) for the function.
void DwarfException::EmitFDE(const FunctionEHFrameInfo &EHFrameInfo) {
  assert(!EHFrameInfo.function->hasAvailableExternallyLinkage() &&
         "Should not emit 'available externally' functions at all");

  const Function *TheFunc = EHFrameInfo.function;

  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getEHFrameSection());

  // Externally visible entry into the functions eh frame info. If the
  // corresponding function is static, this should not be externally visible.
  if (!TheFunc->hasLocalLinkage())
    if (const char *GlobalEHDirective = MAI->getGlobalEHDirective())
      O << GlobalEHDirective << *EHFrameInfo.FunctionEHSym << '\n';

  // If corresponding function is weak definition, this should be too.
  if (TheFunc->isWeakForLinker() && MAI->getWeakDefDirective())
    O << MAI->getWeakDefDirective() << *EHFrameInfo.FunctionEHSym << '\n';

  // If corresponding function is hidden, this should be too.
  if (TheFunc->hasHiddenVisibility())
    if (const char *HiddenDirective = MAI->getHiddenDirective())
      O << HiddenDirective << *EHFrameInfo.FunctionEHSym << '\n';

  // If there are no calls then you can't unwind.  This may mean we can omit the
  // EH Frame, but some environments do not handle weak absolute symbols. If
  // UnwindTablesMandatory is set we cannot do this optimization; the unwind
  // info is to be available for non-EH uses.
  if (!EHFrameInfo.hasCalls && !UnwindTablesMandatory &&
      (!TheFunc->isWeakForLinker() ||
       !MAI->getWeakDefDirective() ||
       MAI->getSupportsWeakOmittedEHFrame())) {
    O << *EHFrameInfo.FunctionEHSym << " = 0\n";
    // This name has no connection to the function, so it might get
    // dead-stripped when the function is not, erroneously.  Prohibit
    // dead-stripping unconditionally.
    if (const char *UsedDirective = MAI->getUsedDirective())
      O << UsedDirective << *EHFrameInfo.FunctionEHSym << "\n\n";
  } else {
    O << *EHFrameInfo.FunctionEHSym << ":\n";

    // EH frame header.
    EmitDifference("eh_frame_end", EHFrameInfo.Number,
                   "eh_frame_begin", EHFrameInfo.Number, true);
    Asm->EOL("Length of Frame Information Entry");

    EmitLabel("eh_frame_begin", EHFrameInfo.Number);

    EmitSectionOffset("eh_frame_begin", "eh_frame_common",
                      EHFrameInfo.Number, EHFrameInfo.PersonalityIndex,
                      true, true, false);

    Asm->EOL("FDE CIE offset");

    EmitReference("eh_func_begin", EHFrameInfo.Number, true, true);
    Asm->EOL("FDE initial location");
    EmitDifference("eh_func_end", EHFrameInfo.Number,
                   "eh_func_begin", EHFrameInfo.Number, true);
    Asm->EOL("FDE address range");

    // If there is a personality and landing pads then point to the language
    // specific data area in the exception table.
    if (MMI->getPersonalities()[0] != NULL) {

      if (Asm->TM.getLSDAEncoding() != DwarfLSDAEncoding::EightByte) {
        Asm->EmitULEB128Bytes(4);
        Asm->EOL("Augmentation size");

        if (EHFrameInfo.hasLandingPads)
          EmitReference("exception", EHFrameInfo.Number, true, true);
        else
          Asm->OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
      } else {
        Asm->EmitULEB128Bytes(TD->getPointerSize());
        Asm->EOL("Augmentation size");

        if (EHFrameInfo.hasLandingPads) {
          EmitReference("exception", EHFrameInfo.Number, true, false);
        } else {
          Asm->OutStreamer.EmitIntValue(0, TD->getPointerSize(),
                                        0/*addrspace*/);
        }
      }

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
    if (MMI->isUsedFunction(EHFrameInfo.function))
      if (const char *UsedDirective = MAI->getUsedDirective()) {
        O << UsedDirective << *EHFrameInfo.FunctionEHSym << "\n\n";
      }
  }

  Asm->EOL();
}

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

/// ComputeActionsTable - Compute the actions table and gather the first action
/// index for each landing pad site.
unsigned DwarfException::
ComputeActionsTable(const SmallVectorImpl<const LandingPadInfo*> &LandingPads,
                    SmallVectorImpl<ActionEntry> &Actions,
                    SmallVectorImpl<unsigned> &FirstActions) {

  // The action table follows the call-site table in the LSDA. The individual
  // records are of two types:
  //
  //   * Catch clause
  //   * Exception specification
  //
  // The two record kinds have the same format, with only small differences.
  // They are distinguished by the "switch value" field: Catch clauses
  // (TypeInfos) have strictly positive switch values, and exception
  // specifications (FilterIds) have strictly negative switch values. Value 0
  // indicates a catch-all clause.
  //
  // Negative type IDs index into FilterIds. Positive type IDs index into
  // TypeInfos.  The value written for a positive type ID is just the type ID
  // itself.  For a negative type ID, however, the value written is the
  // (negative) byte offset of the corresponding FilterIds entry.  The byte
  // offset is usually equal to the type ID (because the FilterIds entries are
  // written using a variable width encoding, which outputs one byte per entry
  // as long as the value written is not too large) but can differ.  This kind
  // of complication does not occur for positive type IDs because type infos are
  // output using a fixed width encoding.  FilterOffsets[i] holds the byte
  // offset corresponding to FilterIds[i].

  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  SmallVector<int, 16> FilterOffsets;
  FilterOffsets.reserve(FilterIds.size());
  int Offset = -1;

  for (std::vector<unsigned>::const_iterator
         I = FilterIds.begin(), E = FilterIds.end(); I != E; ++I) {
    FilterOffsets.push_back(Offset);
    Offset -= MCAsmInfo::getULEB128Size(*I);
  }

  FirstActions.reserve(LandingPads.size());

  int FirstAction = 0;
  unsigned SizeActions = 0;
  const LandingPadInfo *PrevLPI = 0;

  for (SmallVectorImpl<const LandingPadInfo *>::const_iterator
         I = LandingPads.begin(), E = LandingPads.end(); I != E; ++I) {
    const LandingPadInfo *LPI = *I;
    const std::vector<int> &TypeIds = LPI->TypeIds;
    const unsigned NumShared = PrevLPI ? SharedTypeIds(LPI, PrevLPI) : 0;
    unsigned SizeSiteActions = 0;

    if (NumShared < TypeIds.size()) {
      unsigned SizeAction = 0;
      ActionEntry *PrevAction = 0;

      if (NumShared) {
        const unsigned SizePrevIds = PrevLPI->TypeIds.size();
        assert(Actions.size());
        PrevAction = &Actions.back();
        SizeAction = MCAsmInfo::getSLEB128Size(PrevAction->NextAction) +
          MCAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);

        for (unsigned j = NumShared; j != SizePrevIds; ++j) {
          SizeAction -=
            MCAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
          SizeAction += -PrevAction->NextAction;
          PrevAction = PrevAction->Previous;
        }
      }

      // Compute the actions.
      for (unsigned J = NumShared, M = TypeIds.size(); J != M; ++J) {
        int TypeID = TypeIds[J];
        assert(-1 - TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
        int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
        unsigned SizeTypeID = MCAsmInfo::getSLEB128Size(ValueForTypeID);

        int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
        SizeAction = SizeTypeID + MCAsmInfo::getSLEB128Size(NextAction);
        SizeSiteActions += SizeAction;

        ActionEntry Action = { ValueForTypeID, NextAction, PrevAction };
        Actions.push_back(Action);
        PrevAction = &Actions.back();
      }

      // Record the first action of the landing pad site.
      FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
    } // else identical - re-use previous FirstAction

    // Information used when created the call-site table. The action record
    // field of the call site record is the offset of the first associated
    // action record, relative to the start of the actions table. This value is
    // biased by 1 (1 in dicating the start of the actions table), and 0
    // indicates that there are no actions.
    FirstActions.push_back(FirstAction);

    // Compute this sites contribution to size.
    SizeActions += SizeSiteActions;

    PrevLPI = LPI;
  }

  return SizeActions;
}

/// CallToNoUnwindFunction - Return `true' if this is a call to a function
/// marked `nounwind'. Return `false' otherwise.
bool DwarfException::CallToNoUnwindFunction(const MachineInstr *MI) {
  assert(MI->getDesc().isCall() && "This should be a call instruction!");

  bool MarkedNoUnwind = false;
  bool SawFunc = false;

  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);

    if (MO.isGlobal()) {
      if (Function *F = dyn_cast<Function>(MO.getGlobal())) {
        if (SawFunc) {
          // Be conservative. If we have more than one function operand for this
          // call, then we can't make the assumption that it's the callee and
          // not a parameter to the call.
          // 
          // FIXME: Determine if there's a way to say that `F' is the callee or
          // parameter.
          MarkedNoUnwind = false;
          break;
        }

        MarkedNoUnwind = F->doesNotThrow();
        SawFunc = true;
      }
    }
  }

  return MarkedNoUnwind;
}

/// ComputeCallSiteTable - Compute the call-site table.  The entry for an invoke
/// has a try-range containing the call, a non-zero landing pad, and an
/// appropriate action.  The entry for an ordinary call has a try-range
/// containing the call and zero for the landing pad and the action.  Calls
/// marked 'nounwind' have no entry and must not be contained in the try-range
/// of any entry - they form gaps in the table.  Entries must be ordered by
/// try-range address.
void DwarfException::
ComputeCallSiteTable(SmallVectorImpl<CallSiteEntry> &CallSites,
                     const RangeMapType &PadMap,
                     const SmallVectorImpl<const LandingPadInfo *> &LandingPads,
                     const SmallVectorImpl<unsigned> &FirstActions) {
  // The end label of the previous invoke or nounwind try-range.
  unsigned LastLabel = 0;

  // Whether there is a potentially throwing instruction (currently this means
  // an ordinary call) between the end of the previous try-range and now.
  bool SawPotentiallyThrowing = false;

  // Whether the last CallSite entry was for an invoke.
  bool PreviousIsInvoke = false;

  // Visit all instructions in order of address.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
         MI != E; ++MI) {
      if (!MI->isLabel()) {
        if (MI->getDesc().isCall())
          SawPotentiallyThrowing |= !CallToNoUnwindFunction(MI);

        continue;
      }

      unsigned BeginLabel = MI->getOperand(0).getImm();
      assert(BeginLabel && "Invalid label!");

      // End of the previous try-range?
      if (BeginLabel == LastLabel)
        SawPotentiallyThrowing = false;

      // Beginning of a new try-range?
      RangeMapType::const_iterator L = PadMap.find(BeginLabel);
      if (L == PadMap.end())
        // Nope, it was just some random label.
        continue;

      const PadRange &P = L->second;
      const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];
      assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
             "Inconsistent landing pad map!");

      // For Dwarf exception handling (SjLj handling doesn't use this). If some
      // instruction between the previous try-range and this one may throw,
      // create a call-site entry with no landing pad for the region between the
      // try-ranges.
      if (SawPotentiallyThrowing &&
          MAI->getExceptionHandlingType() == ExceptionHandling::Dwarf) {
        CallSiteEntry Site = { LastLabel, BeginLabel, 0, 0 };
        CallSites.push_back(Site);
        PreviousIsInvoke = false;
      }

      LastLabel = LandingPad->EndLabels[P.RangeIndex];
      assert(BeginLabel && LastLabel && "Invalid landing pad!");

      if (LandingPad->LandingPadLabel) {
        // This try-range is for an invoke.
        CallSiteEntry Site = {
          BeginLabel,
          LastLabel,
          LandingPad->LandingPadLabel,
          FirstActions[P.PadIndex]
        };

        // Try to merge with the previous call-site. SJLJ doesn't do this
        if (PreviousIsInvoke &&
          MAI->getExceptionHandlingType() == ExceptionHandling::Dwarf) {
          CallSiteEntry &Prev = CallSites.back();
          if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
            // Extend the range of the previous entry.
            Prev.EndLabel = Site.EndLabel;
            continue;
          }
        }

        // Otherwise, create a new call-site.
        if (MAI->getExceptionHandlingType() == ExceptionHandling::Dwarf)
          CallSites.push_back(Site);
        else {
          // SjLj EH must maintain the call sites in the order assigned
          // to them by the SjLjPrepare pass.
          unsigned SiteNo = MMI->getCallSiteBeginLabel(BeginLabel);
          if (CallSites.size() < SiteNo)
            CallSites.resize(SiteNo);
          CallSites[SiteNo - 1] = Site;
        }
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
  if (SawPotentiallyThrowing &&
      MAI->getExceptionHandlingType() == ExceptionHandling::Dwarf) {
    CallSiteEntry Site = { LastLabel, 0, 0, 0 };
    CallSites.push_back(Site);
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
///  2. The action table, in our case, is composed of pairs of type IDs and next
///     action offset.  Starting with the action index from the landing pad
///     site, each type ID is checked for a match to the current exception.  If
///     it matches then the exception and type id are passed on to the landing
///     pad.  Otherwise the next action is looked up.  This chain is terminated
///     with a next action of zero.  If no type id is found then the frame is
///     unwound and handling continues.
///  3. Type ID table contains references to all the C++ typeinfo for all
///     catches in the function.  This tables is reverse indexed base 1.
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

  // Compute the actions table and gather the first action index for each
  // landing pad site.
  SmallVector<ActionEntry, 32> Actions;
  SmallVector<unsigned, 64> FirstActions;
  unsigned SizeActions = ComputeActionsTable(LandingPads, Actions,
                                             FirstActions);

  // Invokes and nounwind calls have entries in PadMap (due to being bracketed
  // by try-range labels when lowered).  Ordinary calls do not, so appropriate
  // try-ranges for them need be deduced when using DWARF exception handling.
  RangeMapType PadMap;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LandingPad = LandingPads[i];
    for (unsigned j = 0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
      unsigned BeginLabel = LandingPad->BeginLabels[j];
      assert(!PadMap.count(BeginLabel) && "Duplicate landing pad labels!");
      PadRange P = { i, j };
      PadMap[BeginLabel] = P;
    }
  }

  // Compute the call-site table.
  SmallVector<CallSiteEntry, 64> CallSites;
  ComputeCallSiteTable(CallSites, PadMap, LandingPads, FirstActions);

  // Final tallies.

  // Call sites.
  const unsigned SiteStartSize  = SizeOfEncodedValue(dwarf::DW_EH_PE_udata4);
  const unsigned SiteLengthSize = SizeOfEncodedValue(dwarf::DW_EH_PE_udata4);
  const unsigned LandingPadSize = SizeOfEncodedValue(dwarf::DW_EH_PE_udata4);
  bool IsSJLJ = MAI->getExceptionHandlingType() == ExceptionHandling::SjLj;
  bool HaveTTData = IsSJLJ ? (!TypeInfos.empty() || !FilterIds.empty()) : true;
  unsigned SizeSites;

  if (IsSJLJ)
    SizeSites = 0;
  else
    SizeSites = CallSites.size() *
      (SiteStartSize + SiteLengthSize + LandingPadSize);

  for (unsigned i = 0, e = CallSites.size(); i < e; ++i) {
    SizeSites += MCAsmInfo::getULEB128Size(CallSites[i].Action);
    if (IsSJLJ)
      SizeSites += MCAsmInfo::getULEB128Size(i);
  }

  // Type infos.
  const MCSection *LSDASection = Asm->getObjFileLowering().getLSDASection();
  unsigned TTypeFormat;
  unsigned TypeFormatSize;

  if (!HaveTTData) {
    // For SjLj exceptions, if there is no TypeInfo, then we just explicitly say
    // that we're omitting that bit.
    TTypeFormat = dwarf::DW_EH_PE_omit;
    TypeFormatSize = SizeOfEncodedValue(dwarf::DW_EH_PE_absptr);
  } else {
    // Okay, we have actual filters or typeinfos to emit.  As such, we need to
    // pick a type encoding for them.  We're about to emit a list of pointers to
    // typeinfo objects at the end of the LSDA.  However, unless we're in static
    // mode, this reference will require a relocation by the dynamic linker.
    //
    // Because of this, we have a couple of options:
    // 
    //   1) If we are in -static mode, we can always use an absolute reference
    //      from the LSDA, because the static linker will resolve it.
    //      
    //   2) Otherwise, if the LSDA section is writable, we can output the direct
    //      reference to the typeinfo and allow the dynamic linker to relocate
    //      it.  Since it is in a writable section, the dynamic linker won't
    //      have a problem.
    //      
    //   3) Finally, if we're in PIC mode and the LDSA section isn't writable,
    //      we need to use some form of indirection.  For example, on Darwin,
    //      we can output a statically-relocatable reference to a dyld stub. The
    //      offset to the stub is constant, but the contents are in a section
    //      that is updated by the dynamic linker.  This is easy enough, but we
    //      need to tell the personality function of the unwinder to indirect
    //      through the dyld stub.
    //
    // FIXME: When (3) is actually implemented, we'll have to emit the stubs
    // somewhere.  This predicate should be moved to a shared location that is
    // in target-independent code.
    //
    if (LSDASection->getKind().isWriteable() ||
        Asm->TM.getRelocationModel() == Reloc::Static)
      TTypeFormat = dwarf::DW_EH_PE_absptr;
    else
      TTypeFormat = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;

    TypeFormatSize = SizeOfEncodedValue(TTypeFormat);
  }

  // Begin the exception table.
  Asm->OutStreamer.SwitchSection(LSDASection);
  Asm->EmitAlignment(2, 0, 0, false);

  O << "GCC_except_table" << SubprogramCount << ":\n";

  // The type infos need to be aligned. GCC does this by inserting padding just
  // before the type infos. However, this changes the size of the exception
  // table, so you need to take this into account when you output the exception
  // table size. However, the size is output using a variable length encoding.
  // So by increasing the size by inserting padding, you may increase the number
  // of bytes used for writing the size. If it increases, say by one byte, then
  // you now need to output one less byte of padding to get the type infos
  // aligned.  However this decreases the size of the exception table. This
  // changes the value you have to output for the exception table size. Due to
  // the variable length encoding, the number of bytes used for writing the
  // length may decrease. If so, you then have to increase the amount of
  // padding. And so on. If you look carefully at the GCC code you will see that
  // it indeed does this in a loop, going on and on until the values stabilize.
  // We chose another solution: don't output padding inside the table like GCC
  // does, instead output it before the table.
  unsigned SizeTypes = TypeInfos.size() * TypeFormatSize;
  unsigned TyOffset = sizeof(int8_t) +          // Call site format
    MCAsmInfo::getULEB128Size(SizeSites) +      // Call-site table length
    SizeSites + SizeActions + SizeTypes;
  unsigned TotalSize = sizeof(int8_t) +         // LPStart format
                       sizeof(int8_t) +         // TType format
    (HaveTTData ?
     MCAsmInfo::getULEB128Size(TyOffset) : 0) + // TType base offset
    TyOffset;
  unsigned SizeAlign = (4 - TotalSize) & 3;

  for (unsigned i = 0; i != SizeAlign; ++i) {
    Asm->EmitInt8(0);
    Asm->EOL("Padding");
  }

  EmitLabel("exception", SubprogramCount);

  if (IsSJLJ) {
    SmallString<16> LSDAName;
    raw_svector_ostream(LSDAName) << MAI->getPrivateGlobalPrefix() <<
      "_LSDA_" << Asm->getFunctionNumber();
    O << LSDAName.str() << ":\n";
  }

  // Emit the header.
  Asm->EmitInt8(dwarf::DW_EH_PE_omit);
  Asm->EOL("@LPStart format", dwarf::DW_EH_PE_omit);

  Asm->EmitInt8(TTypeFormat);
  Asm->EOL("@TType format", TTypeFormat);

  if (HaveTTData) {
    Asm->EmitULEB128Bytes(TyOffset);
    Asm->EOL("@TType base offset");
  }

  // SjLj Exception handling
  if (IsSJLJ) {
    Asm->EmitInt8(dwarf::DW_EH_PE_udata4);
    Asm->EOL("Call site format", dwarf::DW_EH_PE_udata4);
    Asm->EmitULEB128Bytes(SizeSites);
    Asm->EOL("Call site table length");

    // Emit the landing pad site information.
    unsigned idx = 0;
    for (SmallVectorImpl<CallSiteEntry>::const_iterator
         I = CallSites.begin(), E = CallSites.end(); I != E; ++I, ++idx) {
      const CallSiteEntry &S = *I;

      // Offset of the landing pad, counted in 16-byte bundles relative to the
      // @LPStart address.
      Asm->EmitULEB128Bytes(idx);
      Asm->EOL("Landing pad");

      // Offset of the first associated action record, relative to the start of
      // the action table. This value is biased by 1 (1 indicates the start of
      // the action table), and 0 indicates that there are no actions.
      Asm->EmitULEB128Bytes(S.Action);
      Asm->EOL("Action");
    }
  } else {
    // DWARF Exception handling
    assert(MAI->getExceptionHandlingType() == ExceptionHandling::Dwarf);

    // The call-site table is a list of all call sites that may throw an
    // exception (including C++ 'throw' statements) in the procedure
    // fragment. It immediately follows the LSDA header. Each entry indicates,
    // for a given call, the first corresponding action record and corresponding
    // landing pad.
    //
    // The table begins with the number of bytes, stored as an LEB128
    // compressed, unsigned integer. The records immediately follow the record
    // count. They are sorted in increasing call-site address. Each record
    // indicates:
    //
    //   * The position of the call-site.
    //   * The position of the landing pad.
    //   * The first action record for that call site.
    //
    // A missing entry in the call-site table indicates that a call is not
    // supposed to throw.

    // Emit the landing pad call site table.
    Asm->EmitInt8(dwarf::DW_EH_PE_udata4);
    Asm->EOL("Call site format", dwarf::DW_EH_PE_udata4);
    Asm->EmitULEB128Bytes(SizeSites);
    Asm->EOL("Call site table size");

    for (SmallVectorImpl<CallSiteEntry>::const_iterator
         I = CallSites.begin(), E = CallSites.end(); I != E; ++I) {
      const CallSiteEntry &S = *I;
      const char *BeginTag;
      unsigned BeginNumber;

      if (!S.BeginLabel) {
        BeginTag = "eh_func_begin";
        BeginNumber = SubprogramCount;
      } else {
        BeginTag = "label";
        BeginNumber = S.BeginLabel;
      }

      // Offset of the call site relative to the previous call site, counted in
      // number of 16-byte bundles. The first call site is counted relative to
      // the start of the procedure fragment.
      EmitSectionOffset(BeginTag, "eh_func_begin", BeginNumber, SubprogramCount,
                        true, true);
      Asm->EOL("Region start");

      if (!S.EndLabel)
        EmitDifference("eh_func_end", SubprogramCount, BeginTag, BeginNumber,
                       true);
      else
        EmitDifference("label", S.EndLabel, BeginTag, BeginNumber, true);

      Asm->EOL("Region length");

      // Offset of the landing pad, counted in 16-byte bundles relative to the
      // @LPStart address.
      if (!S.PadLabel)
        Asm->OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
      else
        EmitSectionOffset("label", "eh_func_begin", S.PadLabel, SubprogramCount,
                          true, true);

      Asm->EOL("Landing pad");

      // Offset of the first associated action record, relative to the start of
      // the action table. This value is biased by 1 (1 indicates the start of
      // the action table), and 0 indicates that there are no actions.
      Asm->EmitULEB128Bytes(S.Action);
      Asm->EOL("Action");
    }
  }

  // Emit the Action Table.
  for (SmallVectorImpl<ActionEntry>::const_iterator
         I = Actions.begin(), E = Actions.end(); I != E; ++I) {
    const ActionEntry &Action = *I;

    // Type Filter
    //
    //   Used by the runtime to match the type of the thrown exception to the
    //   type of the catch clauses or the types in the exception specification.

    Asm->EmitSLEB128Bytes(Action.ValueForTypeID);
    Asm->EOL("TypeInfo index");

    // Action Record
    //
    //   Self-relative signed displacement in bytes of the next action record,
    //   or 0 if there is no next action record.

    Asm->EmitSLEB128Bytes(Action.NextAction);
    Asm->EOL("Next action");
  }

  // Emit the Catch TypeInfos.
  for (std::vector<GlobalVariable *>::const_reverse_iterator
         I = TypeInfos.rbegin(), E = TypeInfos.rend(); I != E; ++I) {
    const GlobalVariable *GV = *I;
    PrintRelDirective();

    if (GV) {
      O << *Asm->GetGlobalValueSymbol(GV);
    } else {
      O << "0x0";
    }

    Asm->EOL("TypeInfo");
  }

  // Emit the Exception Specifications.
  for (std::vector<unsigned>::const_iterator
         I = FilterIds.begin(), E = FilterIds.end(); I < E; ++I) {
    unsigned TypeID = *I;
    Asm->EmitULEB128Bytes(TypeID);
    if (TypeID != 0)
      Asm->EOL("Exception specification");
    else
      Asm->EOL();
  }

  Asm->EmitAlignment(2, 0, 0, false);
}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfException::EndModule() {
  if (MAI->getExceptionHandlingType() != ExceptionHandling::Dwarf)
    return;

  if (!shouldEmitMovesModule && !shouldEmitTableModule)
    return;

  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  const std::vector<Function *> Personalities = MMI->getPersonalities();

  for (unsigned I = 0, E = Personalities.size(); I < E; ++I)
    EmitCIE(Personalities[I], I);

  for (std::vector<FunctionEHFrameInfo>::iterator
         I = EHFrames.begin(), E = EHFrames.end(); I != E; ++I)
    EmitFDE(*I);

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void DwarfException::BeginFunction(MachineFunction *MF) {
  if (!MMI || !MAI->doesSupportExceptionHandling()) return;

  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  this->MF = MF;
  shouldEmitTable = shouldEmitMoves = false;

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  // If any landing pads survive, we need an EH table.
  if (!MMI->getLandingPads().empty())
    shouldEmitTable = true;

  // See if we need frame move info.
  if (!MF->getFunction()->doesNotThrow() || UnwindTablesMandatory)
    shouldEmitMoves = true;

  if (shouldEmitMoves || shouldEmitTable)
    // Assumes in correct section after the entry point.
    EmitLabel("eh_func_begin", ++SubprogramCount);

  shouldEmitTableModule |= shouldEmitTable;
  shouldEmitMovesModule |= shouldEmitMoves;

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfException::EndFunction() {
  if (!shouldEmitMoves && !shouldEmitTable) return;

  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  EmitLabel("eh_func_end", SubprogramCount);
  EmitExceptionTable();

  const MCSymbol *FunctionEHSym =
    Asm->GetSymbolWithGlobalValueBase(MF->getFunction(), ".eh",
                                      Asm->MAI->is_EHSymbolPrivate());
  
  // Save EH frame information
  EHFrames.push_back(FunctionEHFrameInfo(FunctionEHSym, SubprogramCount,
                                         MMI->getPersonalityIndex(),
                                         MF->getFrameInfo()->hasCalls(),
                                         !MMI->getLandingPads().empty(),
                                         MMI->getFrameMoves(),
                                         MF->getFunction()));

  // Record if this personality index uses a landing pad.
  UsesLSDA[MMI->getPersonalityIndex()] |= !MMI->getLandingPads().empty();

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}
