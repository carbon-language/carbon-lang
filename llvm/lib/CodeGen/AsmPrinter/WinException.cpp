//===-- CodeGen/AsmPrinter/WinException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Win64 exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "WinException.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

WinException::WinException(AsmPrinter *A) : EHStreamer(A) {
  // MSVC's EH tables are always composed of 32-bit words.  All known 64-bit
  // platforms use an imagerel32 relocation to refer to symbols.
  useImageRel32 = (A->getDataLayout().getPointerSizeInBits() == 64);
}

WinException::~WinException() {}

/// endModule - Emit all exception information that should come after the
/// content.
void WinException::endModule() {
  auto &OS = *Asm->OutStreamer;
  const Module *M = MMI->getModule();
  for (const Function &F : *M)
    if (F.hasFnAttribute("safeseh"))
      OS.EmitCOFFSafeSEH(Asm->getSymbol(&F));
}

void WinException::beginFunction(const MachineFunction *MF) {
  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();
  bool hasEHFunclets = MMI->hasEHFunclets();

  const Function *F = MF->getFunction();
  const Function *ParentF = MMI->getWinEHParent(F);

  shouldEmitMoves = Asm->needsSEHMoves();

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  const Function *Per = nullptr;
  if (F->hasPersonalityFn())
    Per = dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());

  bool forceEmitPersonality =
    F->hasPersonalityFn() && !isNoOpWithoutInvoke(classifyEHPersonality(Per)) &&
    F->needsUnwindTableEntry();

  shouldEmitPersonality =
      forceEmitPersonality || ((hasLandingPads || hasEHFunclets) &&
                               PerEncoding != dwarf::DW_EH_PE_omit && Per);

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  shouldEmitLSDA = shouldEmitPersonality &&
    LSDAEncoding != dwarf::DW_EH_PE_omit;

  // If we're not using CFI, we don't want the CFI or the personality, but we
  // might want EH tables if we had EH pads.
  // FIXME: If WinEHPrepare outlined something, we should emit the LSDA. Remove
  // this once WinEHPrepare stops doing that.
  if (!Asm->MAI->usesWindowsCFI()) {
    shouldEmitLSDA =
        hasEHFunclets || (F->hasFnAttribute("wineh-parent") && F == ParentF);
    shouldEmitPersonality = false;
    return;
  }

  beginFunclet(MF->front(), Asm->CurrentFnSym);
}

/// endFunction - Gather and emit post-function exception information.
///
void WinException::endFunction(const MachineFunction *MF) {
  if (!shouldEmitPersonality && !shouldEmitMoves && !shouldEmitLSDA)
    return;

  const Function *F = MF->getFunction();
  EHPersonality Per = EHPersonality::Unknown;
  if (F->hasPersonalityFn())
    Per = classifyEHPersonality(F->getPersonalityFn());

  // Get rid of any dead landing pads if we're not using a Windows EH scheme. In
  // Windows EH schemes, the landing pad is not actually reachable. It only
  // exists so that we can emit the right table data.
  if (!isMSVCEHPersonality(Per))
    MMI->TidyLandingPads();

  endFunclet();

  // endFunclet will emit the necessary .xdata tables for x64 SEH.
  if (Per == EHPersonality::MSVC_Win64SEH && MMI->hasEHFunclets())
    return;

  if (shouldEmitPersonality || shouldEmitLSDA) {
    Asm->OutStreamer->PushSection();

    // Just switch sections to the right xdata section. This use of CurrentFnSym
    // assumes that we only emit the LSDA when ending the parent function.
    MCSection *XData = WinEH::UnwindEmitter::getXDataSection(Asm->CurrentFnSym,
                                                             Asm->OutContext);
    Asm->OutStreamer->SwitchSection(XData);

    // Emit the tables appropriate to the personality function in use. If we
    // don't recognize the personality, assume it uses an Itanium-style LSDA.
    if (Per == EHPersonality::MSVC_Win64SEH)
      emitCSpecificHandlerTable(MF);
    else if (Per == EHPersonality::MSVC_X86SEH)
      emitExceptHandlerTable(MF);
    else if (Per == EHPersonality::MSVC_CXX)
      emitCXXFrameHandler3Table(MF);
    else
      emitExceptionTable();

    Asm->OutStreamer->PopSection();
  }
}

/// Retreive the MCSymbol for a GlobalValue or MachineBasicBlock. GlobalValues
/// are used in the old WinEH scheme, and they will be removed eventually.
static MCSymbol *getMCSymbolForMBBOrGV(AsmPrinter *Asm, ValueOrMBB Handler) {
  if (!Handler)
    return nullptr;
  if (Handler.is<const MachineBasicBlock *>()) {
    auto *MBB = Handler.get<const MachineBasicBlock *>();
    assert(MBB->isEHFuncletEntry());

    // Give catches and cleanups a name based off of their parent function and
    // their funclet entry block's number.
    const MachineFunction *MF = MBB->getParent();
    const Function *F = MF->getFunction();
    StringRef FuncLinkageName = GlobalValue::getRealLinkageName(F->getName());
    MCContext &Ctx = MF->getContext();
    StringRef HandlerPrefix = MBB->isCleanupFuncletEntry() ? "dtor" : "catch";
    return Ctx.getOrCreateSymbol("?" + HandlerPrefix + "$" +
                                 Twine(MBB->getNumber()) + "@?0?" +
                                 FuncLinkageName + "@4HA");
  }
  return Asm->getSymbol(cast<GlobalValue>(Handler.get<const Value *>()));
}

void WinException::beginFunclet(const MachineBasicBlock &MBB,
                                MCSymbol *Sym) {
  CurrentFuncletEntry = &MBB;

  const Function *F = Asm->MF->getFunction();
  // If a symbol was not provided for the funclet, invent one.
  if (!Sym) {
    Sym = getMCSymbolForMBBOrGV(Asm, &MBB);

    // Describe our funclet symbol as a function with internal linkage.
    Asm->OutStreamer->BeginCOFFSymbolDef(Sym);
    Asm->OutStreamer->EmitCOFFSymbolStorageClass(COFF::IMAGE_SYM_CLASS_STATIC);
    Asm->OutStreamer->EmitCOFFSymbolType(COFF::IMAGE_SYM_DTYPE_FUNCTION
                                         << COFF::SCT_COMPLEX_TYPE_SHIFT);
    Asm->OutStreamer->EndCOFFSymbolDef();

    // We want our funclet's entry point to be aligned such that no nops will be
    // present after the label.
    Asm->EmitAlignment(std::max(Asm->MF->getAlignment(), MBB.getAlignment()),
                       F);

    // Now that we've emitted the alignment directive, point at our funclet.
    Asm->OutStreamer->EmitLabel(Sym);
  }

  // Mark 'Sym' as starting our funclet.
  if (shouldEmitMoves || shouldEmitPersonality)
    Asm->OutStreamer->EmitWinCFIStartProc(Sym);

  if (shouldEmitPersonality) {
    const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
    const Function *PerFn = nullptr;

    // Determine which personality routine we are using for this funclet.
    if (F->hasPersonalityFn())
      PerFn = dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());
    const MCSymbol *PersHandlerSym =
        TLOF.getCFIPersonalitySymbol(PerFn, *Asm->Mang, Asm->TM, MMI);

    // Classify the personality routine so that we may reason about it.
    EHPersonality Per = EHPersonality::Unknown;
    if (F->hasPersonalityFn())
      Per = classifyEHPersonality(F->getPersonalityFn());

    // Do not emit a .seh_handler directive if it is a C++ cleanup funclet.
    if (Per != EHPersonality::MSVC_CXX ||
        !CurrentFuncletEntry->isCleanupFuncletEntry())
      Asm->OutStreamer->EmitWinEHHandler(PersHandlerSym, true, true);
  }
}

void WinException::endFunclet() {
  // No funclet to process?  Great, we have nothing to do.
  if (!CurrentFuncletEntry)
    return;

  if (shouldEmitMoves || shouldEmitPersonality) {
    const Function *F = Asm->MF->getFunction();
    EHPersonality Per = EHPersonality::Unknown;
    if (F->hasPersonalityFn())
      Per = classifyEHPersonality(F->getPersonalityFn());

    // The .seh_handlerdata directive implicitly switches section, push the
    // current section so that we may return to it.
    Asm->OutStreamer->PushSection();

    // Emit an UNWIND_INFO struct describing the prologue.
    Asm->OutStreamer->EmitWinEHHandlerData();

    if (Per == EHPersonality::MSVC_CXX && shouldEmitPersonality &&
        !CurrentFuncletEntry->isCleanupFuncletEntry()) {
      // If this is a C++ catch funclet (or the parent function),
      // emit a reference to the LSDA for the parent function.
      StringRef FuncLinkageName = GlobalValue::getRealLinkageName(F->getName());
      MCSymbol *FuncInfoXData = Asm->OutContext.getOrCreateSymbol(
          Twine("$cppxdata$", FuncLinkageName));
      Asm->OutStreamer->EmitValue(create32bitRef(FuncInfoXData), 4);
    } else if (Per == EHPersonality::MSVC_Win64SEH && MMI->hasEHFunclets() &&
               !CurrentFuncletEntry->isEHFuncletEntry()) {
      // If this is the parent function in Win64 SEH, emit the LSDA immediately
      // following .seh_handlerdata.
      emitCSpecificHandlerTable(Asm->MF);
    }

    // Switch back to the previous section now that we are done writing to
    // .xdata.
    Asm->OutStreamer->PopSection();

    // Emit a .seh_endproc directive to mark the end of the function.
    Asm->OutStreamer->EmitWinCFIEndProc();
  }

  // Let's make sure we don't try to end the same funclet twice.
  CurrentFuncletEntry = nullptr;
}

const MCExpr *WinException::create32bitRef(const MCSymbol *Value) {
  if (!Value)
    return MCConstantExpr::create(0, Asm->OutContext);
  return MCSymbolRefExpr::create(Value, useImageRel32
                                            ? MCSymbolRefExpr::VK_COFF_IMGREL32
                                            : MCSymbolRefExpr::VK_None,
                                 Asm->OutContext);
}

const MCExpr *WinException::create32bitRef(const Value *V) {
  if (!V)
    return MCConstantExpr::create(0, Asm->OutContext);
  // FIXME: Delete the GlobalValue case once the new IR is fully functional.
  if (const auto *GV = dyn_cast<GlobalValue>(V))
    return create32bitRef(Asm->getSymbol(GV));
  return create32bitRef(MMI->getAddrLabelSymbol(cast<BasicBlock>(V)));
}

const MCExpr *WinException::getLabelPlusOne(MCSymbol *Label) {
  return MCBinaryExpr::createAdd(create32bitRef(Label),
                                 MCConstantExpr::create(1, Asm->OutContext),
                                 Asm->OutContext);
}

/// Information describing an invoke range.
struct InvokeRange {
  MCSymbol *BeginLabel = nullptr;
  MCSymbol *EndLabel = nullptr;
  int State = -1;

  /// If we saw a potentially throwing call between this range and the last
  /// range.
  bool SawPotentiallyThrowing = false;
};

/// Iterator over the begin/end label pairs of invokes within a basic block.
class InvokeLabelIterator {
public:
  InvokeLabelIterator(WinEHFuncInfo &EHInfo,
                      MachineBasicBlock::const_iterator MBBI,
                      MachineBasicBlock::const_iterator MBBIEnd)
      : EHInfo(EHInfo), MBBI(MBBI), MBBIEnd(MBBIEnd) {
    scan();
  }

  // Iterator methods.
  bool operator==(const InvokeLabelIterator &o) const { return MBBI == o.MBBI; }
  bool operator!=(const InvokeLabelIterator &o) const { return MBBI != o.MBBI; }
  InvokeRange &operator*() { return CurRange; }
  InvokeRange *operator->() { return &CurRange; }
  InvokeLabelIterator &operator++() { return scan(); }

private:
  // Scan forward to find the next invoke range, or hit the end iterator.
  InvokeLabelIterator &scan();

  WinEHFuncInfo &EHInfo;
  MachineBasicBlock::const_iterator MBBI;
  MachineBasicBlock::const_iterator MBBIEnd;
  InvokeRange CurRange;
};

/// Invoke label range iteration logic. Increment MBBI until we find the next
/// EH_LABEL pair, and then update MBBI to point after the end label.
InvokeLabelIterator &InvokeLabelIterator::scan() {
  // Reset our state.
  CurRange = InvokeRange{};

  for (const MachineInstr &MI : make_range(MBBI, MBBIEnd)) {
    // Remember if we had to cross a potentially throwing call instruction that
    // must unwind to caller.
    if (MI.isCall()) {
      CurRange.SawPotentiallyThrowing |=
          !EHStreamer::callToNoUnwindFunction(&MI);
      continue;
    }
    // Find the next EH_LABEL instruction.
    if (!MI.isEHLabel())
      continue;

    // If this is a begin label, break out with the state and end label.
    // Otherwise this is probably a CFI EH_LABEL that we should continue past.
    MCSymbol *Label = MI.getOperand(0).getMCSymbol();
    auto StateAndEnd = EHInfo.InvokeToStateMap.find(Label);
    if (StateAndEnd == EHInfo.InvokeToStateMap.end())
      continue;
    MBBI = MachineBasicBlock::const_iterator(&MI);
    CurRange.BeginLabel = Label;
    CurRange.EndLabel = StateAndEnd->second.second;
    CurRange.State = StateAndEnd->second.first;
    break;
  }

  // If we didn't find a begin label, we are done, return the end iterator.
  if (!CurRange.BeginLabel) {
    MBBI = MBBIEnd;
    return *this;
  }

  // If this is a begin label, update MBBI to point past the end label.
  for (; MBBI != MBBIEnd; ++MBBI)
    if (MBBI->isEHLabel() &&
        MBBI->getOperand(0).getMCSymbol() == CurRange.EndLabel)
      break;
  return *this;
}

/// Utility for making a range for all the invoke ranges.
static iterator_range<InvokeLabelIterator>
invoke_ranges(WinEHFuncInfo &EHInfo, const MachineBasicBlock &MBB) {
  return make_range(InvokeLabelIterator(EHInfo, MBB.begin(), MBB.end()),
                    InvokeLabelIterator(EHInfo, MBB.end(), MBB.end()));
}

/// Emit the language-specific data that __C_specific_handler expects.  This
/// handler lives in the x64 Microsoft C runtime and allows catching or cleaning
/// up after faults with __try, __except, and __finally.  The typeinfo values
/// are not really RTTI data, but pointers to filter functions that return an
/// integer (1, 0, or -1) indicating how to handle the exception. For __finally
/// blocks and other cleanups, the landing pad label is zero, and the filter
/// function is actually a cleanup handler with the same prototype.  A catch-all
/// entry is modeled with a null filter function field and a non-zero landing
/// pad label.
///
/// Possible filter function return values:
///   EXCEPTION_EXECUTE_HANDLER (1):
///     Jump to the landing pad label after cleanups.
///   EXCEPTION_CONTINUE_SEARCH (0):
///     Continue searching this table or continue unwinding.
///   EXCEPTION_CONTINUE_EXECUTION (-1):
///     Resume execution at the trapping PC.
///
/// Inferred table structure:
///   struct Table {
///     int NumEntries;
///     struct Entry {
///       imagerel32 LabelStart;
///       imagerel32 LabelEnd;
///       imagerel32 FilterOrFinally;  // One means catch-all.
///       imagerel32 LabelLPad;        // Zero means __finally.
///     } Entries[NumEntries];
///   };
void WinException::emitCSpecificHandlerTable(const MachineFunction *MF) {
  auto &OS = *Asm->OutStreamer;
  MCContext &Ctx = Asm->OutContext;

  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(MF->getFunction());
  if (!FuncInfo.SEHUnwindMap.empty()) {
    // Remember what state we were in the last time we found a begin try label.
    // This allows us to coalesce many nearby invokes with the same state into
    // one entry.
    int LastEHState = -1;
    MCSymbol *LastBeginLabel = nullptr;
    MCSymbol *LastEndLabel = nullptr;

    // Use the assembler to compute the number of table entries through label
    // difference and division.
    MCSymbol *TableBegin =
        Ctx.createTempSymbol("lsda_begin", /*AlwaysAddSuffix=*/true);
    MCSymbol *TableEnd =
        Ctx.createTempSymbol("lsda_end", /*AlwaysAddSuffix=*/true);
    const MCExpr *LabelDiff =
        MCBinaryExpr::createSub(MCSymbolRefExpr::create(TableEnd, Ctx),
                                MCSymbolRefExpr::create(TableBegin, Ctx), Ctx);
    const MCExpr *EntrySize = MCConstantExpr::create(16, Ctx);
    const MCExpr *EntryCount =
        MCBinaryExpr::createDiv(LabelDiff, EntrySize, Ctx);
    OS.EmitValue(EntryCount, 4);

    OS.EmitLabel(TableBegin);

    // Iterate over all the invoke try ranges. Unlike MSVC, LLVM currently only
    // models exceptions from invokes. LLVM also allows arbitrary reordering of
    // the code, so our tables end up looking a bit different. Rather than
    // trying to match MSVC's tables exactly, we emit a denormalized table.  For
    // each range of invokes in the same state, we emit table entries for all
    // the actions that would be taken in that state. This means our tables are
    // slightly bigger, which is OK.
    for (const auto &MBB : *MF) {
      for (InvokeRange &I : invoke_ranges(FuncInfo, MBB)) {
        // If this invoke is in the same state as the last invoke and there were
        // no non-throwing calls between it, extend the range to include both
        // and continue.
        if (!I.SawPotentiallyThrowing && I.State == LastEHState) {
          LastEndLabel = I.EndLabel;
          continue;
        }

        // If this invoke ends a previous one, emit all the actions for this
        // state.
        if (LastEHState != -1) {
          assert(LastBeginLabel && LastEndLabel);
          for (int State = LastEHState; State != -1;) {
            SEHUnwindMapEntry &UME = FuncInfo.SEHUnwindMap[State];
            const MCExpr *FilterOrFinally;
            const MCExpr *ExceptOrNull;
            auto *Handler = UME.Handler.get<MachineBasicBlock *>();
            if (UME.IsFinally) {
              FilterOrFinally = create32bitRef(Handler->getSymbol());
              ExceptOrNull = MCConstantExpr::create(0, Ctx);
            } else {
              // For an except, the filter can be 1 (catch-all) or a function
              // label.
              FilterOrFinally = UME.Filter ? create32bitRef(UME.Filter)
                                           : MCConstantExpr::create(1, Ctx);
              ExceptOrNull = create32bitRef(Handler->getSymbol());
            }

            OS.EmitValue(getLabelPlusOne(LastBeginLabel), 4);
            OS.EmitValue(getLabelPlusOne(LastEndLabel), 4);
            OS.EmitValue(FilterOrFinally, 4);
            OS.EmitValue(ExceptOrNull, 4);

            State = UME.ToState;
          }
        }

        LastBeginLabel = I.BeginLabel;
        LastEndLabel = I.EndLabel;
        LastEHState = I.State;
      }
    }
    OS.EmitLabel(TableEnd);
    return;
  }

  // Simplifying assumptions for first implementation:
  // - Cleanups are not implemented.
  // - Filters are not implemented.

  // The Itanium LSDA table sorts similar landing pads together to simplify the
  // actions table, but we don't need that.
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());
  for (const auto &LP : PadInfos)
    LandingPads.push_back(&LP);

  // Compute label ranges for call sites as we would for the Itanium LSDA, but
  // use an all zero action table because we aren't using these actions.
  SmallVector<unsigned, 64> FirstActions;
  FirstActions.resize(LandingPads.size());
  SmallVector<CallSiteEntry, 64> CallSites;
  computeCallSiteTable(CallSites, LandingPads, FirstActions);

  MCSymbol *EHFuncBeginSym = Asm->getFunctionBegin();
  MCSymbol *EHFuncEndSym = Asm->getFunctionEnd();

  // Emit the number of table entries.
  unsigned NumEntries = 0;
  for (const CallSiteEntry &CSE : CallSites) {
    if (!CSE.LPad)
      continue; // Ignore gaps.
    NumEntries += CSE.LPad->SEHHandlers.size();
  }
  OS.EmitIntValue(NumEntries, 4);

  // If there are no actions, we don't need to iterate again.
  if (NumEntries == 0)
    return;

  // Emit the four-label records for each call site entry. The table has to be
  // sorted in layout order, and the call sites should already be sorted.
  for (const CallSiteEntry &CSE : CallSites) {
    // Ignore gaps. Unlike the Itanium model, unwinding through a frame without
    // an EH table entry will propagate the exception rather than terminating
    // the program.
    if (!CSE.LPad)
      continue;
    const LandingPadInfo *LPad = CSE.LPad;

    // Compute the label range. We may reuse the function begin and end labels
    // rather than forming new ones.
    const MCExpr *Begin =
        create32bitRef(CSE.BeginLabel ? CSE.BeginLabel : EHFuncBeginSym);
    const MCExpr *End;
    if (CSE.EndLabel) {
      // The interval is half-open, so we have to add one to include the return
      // address of the last invoke in the range.
      End = getLabelPlusOne(CSE.EndLabel);
    } else {
      End = create32bitRef(EHFuncEndSym);
    }

    // Emit an entry for each action.
    for (SEHHandler Handler : LPad->SEHHandlers) {
      OS.EmitValue(Begin, 4);
      OS.EmitValue(End, 4);

      // Emit the filter or finally function pointer, if present. Otherwise,
      // emit '1' to indicate a catch-all.
      const Function *F = Handler.FilterOrFinally;
      if (F)
        OS.EmitValue(create32bitRef(Asm->getSymbol(F)), 4);
      else
        OS.EmitIntValue(1, 4);

      // Emit the recovery address, if present. Otherwise, this must be a
      // finally.
      const BlockAddress *BA = Handler.RecoverBA;
      if (BA)
        OS.EmitValue(
            create32bitRef(Asm->GetBlockAddressSymbol(BA)), 4);
      else
        OS.EmitIntValue(0, 4);
    }
  }
}

void WinException::emitCXXFrameHandler3Table(const MachineFunction *MF) {
  const Function *F = MF->getFunction();
  auto &OS = *Asm->OutStreamer;
  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(F);

  StringRef FuncLinkageName = GlobalValue::getRealLinkageName(F->getName());

  SmallVector<std::pair<const MCExpr *, int>, 4> IPToStateTable;
  MCSymbol *FuncInfoXData = nullptr;
  if (shouldEmitPersonality) {
    // If we're 64-bit, emit a pointer to the C++ EH data, and build a map from
    // IPs to state numbers.
    FuncInfoXData =
        Asm->OutContext.getOrCreateSymbol(Twine("$cppxdata$", FuncLinkageName));
    computeIP2StateTable(MF, FuncInfo, IPToStateTable);
  } else {
    FuncInfoXData = Asm->OutContext.getOrCreateLSDASymbol(FuncLinkageName);
    emitEHRegistrationOffsetLabel(FuncInfo, FuncLinkageName);
  }

  MCSymbol *UnwindMapXData = nullptr;
  MCSymbol *TryBlockMapXData = nullptr;
  MCSymbol *IPToStateXData = nullptr;
  if (!FuncInfo.UnwindMap.empty())
    UnwindMapXData = Asm->OutContext.getOrCreateSymbol(
        Twine("$stateUnwindMap$", FuncLinkageName));
  if (!FuncInfo.TryBlockMap.empty())
    TryBlockMapXData =
        Asm->OutContext.getOrCreateSymbol(Twine("$tryMap$", FuncLinkageName));
  if (!IPToStateTable.empty())
    IPToStateXData =
        Asm->OutContext.getOrCreateSymbol(Twine("$ip2state$", FuncLinkageName));

  // FuncInfo {
  //   uint32_t           MagicNumber
  //   int32_t            MaxState;
  //   UnwindMapEntry    *UnwindMap;
  //   uint32_t           NumTryBlocks;
  //   TryBlockMapEntry  *TryBlockMap;
  //   uint32_t           IPMapEntries; // always 0 for x86
  //   IPToStateMapEntry *IPToStateMap; // always 0 for x86
  //   uint32_t           UnwindHelp;   // non-x86 only
  //   ESTypeList        *ESTypeList;
  //   int32_t            EHFlags;
  // }
  // EHFlags & 1 -> Synchronous exceptions only, no async exceptions.
  // EHFlags & 2 -> ???
  // EHFlags & 4 -> The function is noexcept(true), unwinding can't continue.
  OS.EmitValueToAlignment(4);
  OS.EmitLabel(FuncInfoXData);
  OS.EmitIntValue(0x19930522, 4);                      // MagicNumber
  OS.EmitIntValue(FuncInfo.UnwindMap.size(), 4);       // MaxState
  OS.EmitValue(create32bitRef(UnwindMapXData), 4);     // UnwindMap
  OS.EmitIntValue(FuncInfo.TryBlockMap.size(), 4);     // NumTryBlocks
  OS.EmitValue(create32bitRef(TryBlockMapXData), 4);   // TryBlockMap
  OS.EmitIntValue(IPToStateTable.size(), 4);           // IPMapEntries
  OS.EmitValue(create32bitRef(IPToStateXData), 4);     // IPToStateMap
  if (Asm->MAI->usesWindowsCFI())
    OS.EmitIntValue(FuncInfo.UnwindHelpFrameOffset, 4); // UnwindHelp
  OS.EmitIntValue(0, 4);                               // ESTypeList
  OS.EmitIntValue(1, 4);                               // EHFlags

  // UnwindMapEntry {
  //   int32_t ToState;
  //   void  (*Action)();
  // };
  if (UnwindMapXData) {
    OS.EmitLabel(UnwindMapXData);
    for (const WinEHUnwindMapEntry &UME : FuncInfo.UnwindMap) {
      MCSymbol *CleanupSym = getMCSymbolForMBBOrGV(Asm, UME.Cleanup);
      OS.EmitIntValue(UME.ToState, 4);             // ToState
      OS.EmitValue(create32bitRef(CleanupSym), 4); // Action
    }
  }

  // TryBlockMap {
  //   int32_t      TryLow;
  //   int32_t      TryHigh;
  //   int32_t      CatchHigh;
  //   int32_t      NumCatches;
  //   HandlerType *HandlerArray;
  // };
  if (TryBlockMapXData) {
    OS.EmitLabel(TryBlockMapXData);
    SmallVector<MCSymbol *, 1> HandlerMaps;
    for (size_t I = 0, E = FuncInfo.TryBlockMap.size(); I != E; ++I) {
      WinEHTryBlockMapEntry &TBME = FuncInfo.TryBlockMap[I];

      MCSymbol *HandlerMapXData = nullptr;
      if (!TBME.HandlerArray.empty())
        HandlerMapXData =
            Asm->OutContext.getOrCreateSymbol(Twine("$handlerMap$")
                                                  .concat(Twine(I))
                                                  .concat("$")
                                                  .concat(FuncLinkageName));
      HandlerMaps.push_back(HandlerMapXData);

      // TBMEs should form intervals.
      assert(0 <= TBME.TryLow && "bad trymap interval");
      assert(TBME.TryLow <= TBME.TryHigh && "bad trymap interval");
      assert(TBME.TryHigh < TBME.CatchHigh && "bad trymap interval");
      assert(TBME.CatchHigh < int(FuncInfo.UnwindMap.size()) &&
             "bad trymap interval");

      OS.EmitIntValue(TBME.TryLow, 4);                    // TryLow
      OS.EmitIntValue(TBME.TryHigh, 4);                   // TryHigh
      OS.EmitIntValue(TBME.CatchHigh, 4);                 // CatchHigh
      OS.EmitIntValue(TBME.HandlerArray.size(), 4);       // NumCatches
      OS.EmitValue(create32bitRef(HandlerMapXData), 4);   // HandlerArray
    }

    for (size_t I = 0, E = FuncInfo.TryBlockMap.size(); I != E; ++I) {
      WinEHTryBlockMapEntry &TBME = FuncInfo.TryBlockMap[I];
      MCSymbol *HandlerMapXData = HandlerMaps[I];
      if (!HandlerMapXData)
        continue;
      // HandlerType {
      //   int32_t         Adjectives;
      //   TypeDescriptor *Type;
      //   int32_t         CatchObjOffset;
      //   void          (*Handler)();
      //   int32_t         ParentFrameOffset; // x64 only
      // };
      OS.EmitLabel(HandlerMapXData);
      for (const WinEHHandlerType &HT : TBME.HandlerArray) {
        // Get the frame escape label with the offset of the catch object. If
        // the index is -1, then there is no catch object, and we should emit an
        // offset of zero, indicating that no copy will occur.
        const MCExpr *FrameAllocOffsetRef = nullptr;
        if (HT.CatchObjRecoverIdx >= 0) {
          MCSymbol *FrameAllocOffset =
              Asm->OutContext.getOrCreateFrameAllocSymbol(
                  FuncLinkageName, HT.CatchObjRecoverIdx);
          FrameAllocOffsetRef = MCSymbolRefExpr::create(
              FrameAllocOffset, MCSymbolRefExpr::VK_None, Asm->OutContext);
        } else if (HT.CatchObj.FrameOffset != INT_MAX) {
          int Offset = HT.CatchObj.FrameOffset;
          // For 32-bit, the catch object offset is relative to the end of the
          // EH registration node. For 64-bit, it's relative to SP at the end of
          // the prologue.
          if (!shouldEmitPersonality) {
            assert(FuncInfo.EHRegNodeEndOffset != INT_MAX);
            Offset += FuncInfo.EHRegNodeEndOffset;
          }
          FrameAllocOffsetRef = MCConstantExpr::create(Offset, Asm->OutContext);
        } else {
          FrameAllocOffsetRef = MCConstantExpr::create(0, Asm->OutContext);
        }

        MCSymbol *HandlerSym = getMCSymbolForMBBOrGV(Asm, HT.Handler);

        OS.EmitIntValue(HT.Adjectives, 4);                  // Adjectives
        OS.EmitValue(create32bitRef(HT.TypeDescriptor), 4); // Type
        OS.EmitValue(FrameAllocOffsetRef, 4);               // CatchObjOffset
        OS.EmitValue(create32bitRef(HandlerSym), 4);        // Handler

        if (shouldEmitPersonality) {
          // With the new IR, this is always 16 + 8 + getMaxCallFrameSize().
          // Keep this in sync with X86FrameLowering::emitPrologue.
          int ParentFrameOffset =
              16 + 8 + MF->getFrameInfo()->getMaxCallFrameSize();
          OS.EmitIntValue(ParentFrameOffset, 4); // ParentFrameOffset
        }
      }
    }
  }

  // IPToStateMapEntry {
  //   void   *IP;
  //   int32_t State;
  // };
  if (IPToStateXData) {
    OS.EmitLabel(IPToStateXData);
    for (auto &IPStatePair : IPToStateTable) {
      OS.EmitValue(IPStatePair.first, 4);     // IP
      OS.EmitIntValue(IPStatePair.second, 4); // State
    }
  }
}

void WinException::computeIP2StateTable(
    const MachineFunction *MF, WinEHFuncInfo &FuncInfo,
    SmallVectorImpl<std::pair<const MCExpr *, int>> &IPToStateTable) {
  // Remember what state we were in the last time we found a begin try label.
  // This allows us to coalesce many nearby invokes with the same state into one
  // entry.
  int LastEHState = -1;
  MCSymbol *LastEndLabel = Asm->getFunctionBegin();
  assert(LastEndLabel && "need local function start label");

  // Indicate that all calls from the prologue to the first invoke unwind to
  // caller. We handle this as a special case since other ranges starting at end
  // labels need to use LtmpN+1.
  IPToStateTable.push_back(std::make_pair(create32bitRef(LastEndLabel), -1));

  for (const auto &MBB : *MF) {
    // FIXME: Do we need to emit entries for funclet base states?

    for (InvokeRange &I : invoke_ranges(FuncInfo, MBB)) {
      assert(I.BeginLabel && I.EndLabel);
      // If there was a potentially throwing call between this begin label and
      // the last end label, we need an extra base state entry to indicate that
      // those calls unwind directly to the caller.
      if (I.SawPotentiallyThrowing && LastEHState != -1) {
        IPToStateTable.push_back(
            std::make_pair(getLabelPlusOne(LastEndLabel), -1));
        LastEHState = -1;
      }

      // Emit an entry indicating that PCs after 'Label' have this EH state.
      if (I.State != LastEHState)
        IPToStateTable.push_back(
            std::make_pair(create32bitRef(I.BeginLabel), I.State));
      LastEHState = I.State;
      LastEndLabel = I.EndLabel;
    }
  }

  if (LastEndLabel != Asm->getFunctionBegin()) {
    // Indicate that all calls from the last invoke until the epilogue unwind to
    // caller. This also ensures that we have at least one ip2state entry, if
    // somehow all invokes were deleted during CodeGen.
    IPToStateTable.push_back(std::make_pair(getLabelPlusOne(LastEndLabel), -1));
  }
}

void WinException::emitEHRegistrationOffsetLabel(const WinEHFuncInfo &FuncInfo,
                                                 StringRef FLinkageName) {
  // Outlined helpers called by the EH runtime need to know the offset of the EH
  // registration in order to recover the parent frame pointer. Now that we know
  // we've code generated the parent, we can emit the label assignment that
  // those helpers use to get the offset of the registration node.
  assert(FuncInfo.EHRegNodeEscapeIndex != INT_MAX &&
         "no EH reg node localescape index");
  MCSymbol *ParentFrameOffset =
      Asm->OutContext.getOrCreateParentFrameOffsetSymbol(FLinkageName);
  MCSymbol *RegistrationOffsetSym = Asm->OutContext.getOrCreateFrameAllocSymbol(
      FLinkageName, FuncInfo.EHRegNodeEscapeIndex);
  const MCExpr *RegistrationOffsetSymRef =
      MCSymbolRefExpr::create(RegistrationOffsetSym, Asm->OutContext);
  Asm->OutStreamer->EmitAssignment(ParentFrameOffset, RegistrationOffsetSymRef);
}

/// Emit the language-specific data that _except_handler3 and 4 expect. This is
/// functionally equivalent to the __C_specific_handler table, except it is
/// indexed by state number instead of IP.
void WinException::emitExceptHandlerTable(const MachineFunction *MF) {
  MCStreamer &OS = *Asm->OutStreamer;
  const Function *F = MF->getFunction();
  StringRef FLinkageName = GlobalValue::getRealLinkageName(F->getName());

  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(F);
  emitEHRegistrationOffsetLabel(FuncInfo, FLinkageName);

  // Emit the __ehtable label that we use for llvm.x86.seh.lsda.
  MCSymbol *LSDALabel = Asm->OutContext.getOrCreateLSDASymbol(FLinkageName);
  OS.EmitValueToAlignment(4);
  OS.EmitLabel(LSDALabel);

  const Function *Per =
      dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());
  StringRef PerName = Per->getName();
  int BaseState = -1;
  if (PerName == "_except_handler4") {
    // The LSDA for _except_handler4 starts with this struct, followed by the
    // scope table:
    //
    // struct EH4ScopeTable {
    //   int32_t GSCookieOffset;
    //   int32_t GSCookieXOROffset;
    //   int32_t EHCookieOffset;
    //   int32_t EHCookieXOROffset;
    //   ScopeTableEntry ScopeRecord[];
    // };
    //
    // Only the EHCookieOffset field appears to vary, and it appears to be the
    // offset from the final saved SP value to the retaddr.
    OS.EmitIntValue(-2, 4);
    OS.EmitIntValue(0, 4);
    // FIXME: Calculate.
    OS.EmitIntValue(9999, 4);
    OS.EmitIntValue(0, 4);
    BaseState = -2;
  }

  if (!FuncInfo.SEHUnwindMap.empty()) {
    for (SEHUnwindMapEntry &UME : FuncInfo.SEHUnwindMap) {
      MCSymbol *ExceptOrFinally =
          UME.Handler.get<MachineBasicBlock *>()->getSymbol();
      OS.EmitIntValue(UME.ToState, 4);                  // ToState
      OS.EmitValue(create32bitRef(UME.Filter), 4);      // Filter
      OS.EmitValue(create32bitRef(ExceptOrFinally), 4); // Except/Finally
    }
    return;
  }
  // FIXME: The following code is for the old landingpad-based SEH
  // implementation. Remove it when possible.

  // Build a list of pointers to LandingPadInfos and then sort by WinEHState.
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  SmallVector<const LandingPadInfo *, 4> LPads;
  LPads.reserve((PadInfos.size()));
  for (const LandingPadInfo &LPInfo : PadInfos)
    LPads.push_back(&LPInfo);
  std::sort(LPads.begin(), LPads.end(),
            [](const LandingPadInfo *L, const LandingPadInfo *R) {
              return L->WinEHState < R->WinEHState;
            });

  // For each action in each lpad, emit one of these:
  // struct ScopeTableEntry {
  //   int32_t EnclosingLevel;
  //   int32_t (__cdecl *Filter)();
  //   void *HandlerOrFinally;
  // };
  //
  // The "outermost" action will use BaseState as its enclosing level. Each
  // other action will refer to the previous state as its enclosing level.
  int CurState = 0;
  for (const LandingPadInfo *LPInfo : LPads) {
    int EnclosingLevel = BaseState;
    assert(CurState + int(LPInfo->SEHHandlers.size()) - 1 ==
               LPInfo->WinEHState &&
           "gaps in the SEH scope table");
    for (auto I = LPInfo->SEHHandlers.rbegin(), E = LPInfo->SEHHandlers.rend();
         I != E; ++I) {
      const SEHHandler &Handler = *I;
      const BlockAddress *BA = Handler.RecoverBA;
      const Function *F = Handler.FilterOrFinally;
      assert(F && "cannot catch all in 32-bit SEH without filter function");
      const MCExpr *FilterOrNull =
          create32bitRef(BA ? Asm->getSymbol(F) : nullptr);
      const MCExpr *ExceptOrFinally = create32bitRef(
          BA ? Asm->GetBlockAddressSymbol(BA) : Asm->getSymbol(F));

      OS.EmitIntValue(EnclosingLevel, 4);
      OS.EmitValue(FilterOrNull, 4);
      OS.EmitValue(ExceptOrFinally, 4);

      // The next state unwinds to this state.
      EnclosingLevel = CurState;
      CurState++;
    }
  }
}
