//===-- CodeGen/AsmPrinter/DwarfTableException.cpp - Dwarf Exception Impl --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing DWARF exception info into asm files.
// The implementation emits all the necessary tables "by hands".
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
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
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

DwarfTableException::DwarfTableException(AsmPrinter *A)
  :  DwarfException(A),
     shouldEmitTable(false), shouldEmitMoves(false),
     shouldEmitTableModule(false), shouldEmitMovesModule(false) {}

DwarfTableException::~DwarfTableException() {}

/// EmitCIE - Emit a Common Information Entry (CIE). This holds information that
/// is shared among many Frame Description Entries.  There is at least one CIE
/// in every non-empty .debug_frame section.
void DwarfTableException::EmitCIE(const Function *PersonalityFn, unsigned Index) {
  // Size and sign of stack growth.
  int stackGrowth = Asm->getTargetData().getPointerSize();
  if (Asm->TM.getFrameLowering()->getStackGrowthDirection() ==
      TargetFrameLowering::StackGrowsDown)
    stackGrowth *= -1;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  // Begin eh frame section.
  Asm->OutStreamer.SwitchSection(TLOF.getEHFrameSection());

  MCSymbol *EHFrameSym;
  if (TLOF.isFunctionEHFrameSymbolPrivate())
    EHFrameSym = Asm->GetTempSymbol("EH_frame", Index);
  else
    EHFrameSym = Asm->OutContext.GetOrCreateSymbol(Twine("EH_frame") +
                                                   Twine(Index));
  Asm->OutStreamer.EmitLabel(EHFrameSym);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("section_eh_frame", Index));

  // Define base labels.
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_frame_common", Index));

  // Define the eh frame length.
  Asm->OutStreamer.AddComment("Length of Common Information Entry");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("eh_frame_common_end", Index),
                           Asm->GetTempSymbol("eh_frame_common_begin", Index),
                           4);

  // EH frame header.
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_frame_common_begin",Index));
  Asm->OutStreamer.AddComment("CIE Identifier Tag");
  Asm->OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
  Asm->OutStreamer.AddComment("DW_CIE_VERSION");
  Asm->OutStreamer.EmitIntValue(dwarf::DW_CIE_VERSION, 1/*size*/, 0/*addr*/);

  // The personality presence indicates that language specific information will
  // show up in the eh frame.  Find out how we are supposed to lower the
  // personality function reference:

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  unsigned FDEEncoding = TLOF.getFDEEncoding(false);
  unsigned PerEncoding = TLOF.getPersonalityEncoding();

  char Augmentation[6] = { 0 };
  unsigned AugmentationSize = 0;
  char *APtr = Augmentation + 1;

  if (PersonalityFn) {
    // There is a personality function.
    *APtr++ = 'P';
    AugmentationSize += 1 + Asm->GetSizeOfEncodedValue(PerEncoding);
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

  Asm->OutStreamer.AddComment("CIE Augmentation");
  Asm->OutStreamer.EmitBytes(StringRef(Augmentation, strlen(Augmentation)+1),0);

  // Round out reader.
  Asm->EmitULEB128(1, "CIE Code Alignment Factor");
  Asm->EmitSLEB128(stackGrowth, "CIE Data Alignment Factor");
  Asm->OutStreamer.AddComment("CIE Return Address Column");

  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  const TargetFrameLowering *TFI = Asm->TM.getFrameLowering();
  Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), true));

  if (Augmentation[0]) {
    Asm->EmitULEB128(AugmentationSize, "Augmentation Size");

    // If there is a personality, we need to indicate the function's location.
    if (PersonalityFn) {
      Asm->EmitEncodingByte(PerEncoding, "Personality");
      Asm->OutStreamer.AddComment("Personality");
      Asm->EmitReference(PersonalityFn, PerEncoding);
    }
    if (UsesLSDA[Index])
      Asm->EmitEncodingByte(LSDAEncoding, "LSDA");
    if (FDEEncoding != dwarf::DW_EH_PE_absptr)
      Asm->EmitEncodingByte(FDEEncoding, "FDE");
  }

  // Indicate locations of general callee saved registers in frame.
  std::vector<MachineMove> Moves;
  TFI->getInitialFrameState(Moves);
  Asm->EmitFrameMoves(Moves, 0, true);

  // On Darwin the linker honors the alignment of eh_frame, which means it must
  // be 8-byte on 64-bit targets to match what gcc does.  Otherwise you get
  // holes which confuse readers of eh_frame.
  Asm->EmitAlignment(Asm->getTargetData().getPointerSize() == 4 ? 2 : 3);
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_frame_common_end", Index));
}

/// EmitFDE - Emit the Frame Description Entry (FDE) for the function.
void DwarfTableException::EmitFDE(const FunctionEHFrameInfo &EHFrameInfo) {
  assert(!EHFrameInfo.function->hasAvailableExternallyLinkage() &&
         "Should not emit 'available externally' functions at all");

  const Function *TheFunc = EHFrameInfo.function;
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  unsigned FDEEncoding = TLOF.getFDEEncoding(false);

  Asm->OutStreamer.SwitchSection(TLOF.getEHFrameSection());

  // Externally visible entry into the functions eh frame info. If the
  // corresponding function is static, this should not be externally visible.
  if (!TheFunc->hasLocalLinkage() && TLOF.isFunctionEHSymbolGlobal())
    Asm->OutStreamer.EmitSymbolAttribute(EHFrameInfo.FunctionEHSym,MCSA_Global);

  // If corresponding function is weak definition, this should be too.
  if (TheFunc->isWeakForLinker() && Asm->MAI->getWeakDefDirective())
    Asm->OutStreamer.EmitSymbolAttribute(EHFrameInfo.FunctionEHSym,
                                         MCSA_WeakDefinition);

  // If corresponding function is hidden, this should be too.
  if (TheFunc->hasHiddenVisibility())
    if (MCSymbolAttr HiddenAttr = Asm->MAI->getHiddenVisibilityAttr())
      Asm->OutStreamer.EmitSymbolAttribute(EHFrameInfo.FunctionEHSym,
                                           HiddenAttr);

  // If there are no calls then you can't unwind.  This may mean we can omit the
  // EH Frame, but some environments do not handle weak absolute symbols. If
  // UnwindTablesMandatory is set we cannot do this optimization; the unwind
  // info is to be available for non-EH uses.
  if (!EHFrameInfo.adjustsStack && !UnwindTablesMandatory &&
      (!TheFunc->isWeakForLinker() ||
       !Asm->MAI->getWeakDefDirective() ||
       TLOF.getSupportsWeakOmittedEHFrame())) {
    Asm->OutStreamer.EmitAssignment(EHFrameInfo.FunctionEHSym,
                                    MCConstantExpr::Create(0, Asm->OutContext));
    // This name has no connection to the function, so it might get
    // dead-stripped when the function is not, erroneously.  Prohibit
    // dead-stripping unconditionally.
    if (Asm->MAI->hasNoDeadStrip())
      Asm->OutStreamer.EmitSymbolAttribute(EHFrameInfo.FunctionEHSym,
                                           MCSA_NoDeadStrip);
  } else {
    Asm->OutStreamer.EmitLabel(EHFrameInfo.FunctionEHSym);

    // EH frame header.
    Asm->OutStreamer.AddComment("Length of Frame Information Entry");
    Asm->EmitLabelDifference(
                Asm->GetTempSymbol("eh_frame_end", EHFrameInfo.Number),
                Asm->GetTempSymbol("eh_frame_begin", EHFrameInfo.Number), 4);

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_frame_begin",
                                                  EHFrameInfo.Number));

    Asm->OutStreamer.AddComment("FDE CIE offset");
    Asm->EmitLabelDifference(
                       Asm->GetTempSymbol("eh_frame_begin", EHFrameInfo.Number),
                       Asm->GetTempSymbol("eh_frame_common",
                                          EHFrameInfo.PersonalityIndex), 4);

    MCSymbol *EHFuncBeginSym =
      Asm->GetTempSymbol("eh_func_begin", EHFrameInfo.Number);

    Asm->OutStreamer.AddComment("FDE initial location");
    Asm->EmitReference(EHFuncBeginSym, FDEEncoding);

    Asm->OutStreamer.AddComment("FDE address range");
    Asm->EmitLabelDifference(Asm->GetTempSymbol("eh_func_end",
                                                EHFrameInfo.Number),
                             EHFuncBeginSym,
                             Asm->GetSizeOfEncodedValue(FDEEncoding));

    // If there is a personality and landing pads then point to the language
    // specific data area in the exception table.
    if (MMI->getPersonalities()[0] != NULL) {
      unsigned Size = Asm->GetSizeOfEncodedValue(LSDAEncoding);

      Asm->EmitULEB128(Size, "Augmentation size");
      Asm->OutStreamer.AddComment("Language Specific Data Area");
      if (EHFrameInfo.hasLandingPads)
        Asm->EmitReference(Asm->GetTempSymbol("exception", EHFrameInfo.Number),
                           LSDAEncoding);
      else
        Asm->OutStreamer.EmitIntValue(0, Size/*size*/, 0/*addrspace*/);

    } else {
      Asm->EmitULEB128(0, "Augmentation size");
    }

    // Indicate locations of function specific callee saved registers in frame.
    Asm->EmitFrameMoves(EHFrameInfo.Moves, EHFuncBeginSym, true);

    // On Darwin the linker honors the alignment of eh_frame, which means it
    // must be 8-byte on 64-bit targets to match what gcc does.  Otherwise you
    // get holes which confuse readers of eh_frame.
    Asm->EmitAlignment(Asm->getTargetData().getPointerSize() == 4 ? 2 : 3);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_frame_end",
                                                  EHFrameInfo.Number));

    // If the function is marked used, this table should be also.  We cannot
    // make the mark unconditional in this case, since retaining the table also
    // retains the function in this case, and there is code around that depends
    // on unused functions (calling undefined externals) being dead-stripped to
    // link correctly.  Yes, there really is.
    if (MMI->isUsedFunction(EHFrameInfo.function))
      if (Asm->MAI->hasNoDeadStrip())
        Asm->OutStreamer.EmitSymbolAttribute(EHFrameInfo.FunctionEHSym,
                                             MCSA_NoDeadStrip);
  }
  Asm->OutStreamer.AddBlankLine();
}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfTableException::EndModule() {
  if (!Asm->MAI->isExceptionHandlingDwarf())
    return;

  if (!shouldEmitMovesModule && !shouldEmitTableModule)
    return;

  const std::vector<const Function*> &Personalities = MMI->getPersonalities();

  for (unsigned I = 0, E = Personalities.size(); I < E; ++I)
    EmitCIE(Personalities[I], I);

  for (std::vector<FunctionEHFrameInfo>::iterator
         I = EHFrames.begin(), E = EHFrames.end(); I != E; ++I)
    EmitFDE(*I);
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void DwarfTableException::BeginFunction(const MachineFunction *MF) {
  shouldEmitTable = shouldEmitMoves = false;

  // If any landing pads survive, we need an EH table.
  shouldEmitTable = !MMI->getLandingPads().empty();

  // See if we need frame move info.
  shouldEmitMoves =
    !Asm->MF->getFunction()->doesNotThrow() || UnwindTablesMandatory;

  if (shouldEmitMoves || shouldEmitTable)
    // Assumes in correct section after the entry point.
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_begin",
                                                  Asm->getFunctionNumber()));

  shouldEmitTableModule |= shouldEmitTable;
  shouldEmitMovesModule |= shouldEmitMoves;
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfTableException::EndFunction() {
  if (!shouldEmitMoves && !shouldEmitTable) return;

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_end",
                                                Asm->getFunctionNumber()));

  // Record if this personality index uses a landing pad.
  bool HasLandingPad = !MMI->getLandingPads().empty();
  UsesLSDA[MMI->getPersonalityIndex()] |= HasLandingPad;

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  if (HasLandingPad)
    EmitExceptionTable();

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  MCSymbol *FunctionEHSym =
    Asm->GetSymbolWithGlobalValueBase(Asm->MF->getFunction(), ".eh",
                                      TLOF.isFunctionEHFrameSymbolPrivate());

  // Save EH frame information
  EHFrames.
    push_back(FunctionEHFrameInfo(FunctionEHSym,
                                  Asm->getFunctionNumber(),
                                  MMI->getPersonalityIndex(),
                                  Asm->MF->getFrameInfo()->adjustsStack(),
                                  !MMI->getLandingPads().empty(),
                                  MMI->getFrameMoves(),
                                  Asm->MF->getFunction()));
}
