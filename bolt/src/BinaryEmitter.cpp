//===--- BinaryEmitter.cpp - collection of functions to emit code and data ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryContext.h"
#include "BinaryEmitter.h"
#include "BinaryFunction.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/SMLoc.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

extern cl::opt<uint32_t> X86AlignBranchBoundary;

namespace opts {

extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;
extern cl::OptionCategory BoltRelocCategory;

extern cl::opt<unsigned> AlignText;
extern cl::opt<bool> HotText;
extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::opt<bool> PreserveBlocksAlignment;
extern cl::opt<bool> PrintCacheMetrics;
extern cl::opt<bool> UpdateDebugSections;
extern cl::opt<bool> UpdateEnd;
extern cl::opt<unsigned> Verbosity;

cl::opt<bool>
AlignBlocks("align-blocks",
  cl::desc("align basic blocks"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<MacroFusionType>
AlignMacroOpFusion("align-macro-fusion",
  cl::desc("fix instruction alignment for macro-fusion (x86 relocation mode)"),
  cl::init(MFT_HOT),
  cl::values(clEnumValN(MFT_NONE, "none",
               "do not insert alignment no-ops for macro-fusion"),
             clEnumValN(MFT_HOT, "hot",
               "only insert alignment no-ops on hot execution paths (default)"),
             clEnumValN(MFT_ALL, "all",
               "always align instructions to allow macro-fusion")),
  cl::ZeroOrMore,
  cl::cat(BoltRelocCategory));

static cl::list<std::string>
BreakFunctionNames("break-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to core dump on (debugging)"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string>
FunctionPadSpec("pad-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to pad with amount of bytes"),
  cl::value_desc("func1:pad1,func2:pad2,func3:pad3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
MarkFuncs("mark-funcs",
  cl::desc("mark function boundaries with break instruction to make "
           "sure we accidentally don't cross them"),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

static cl::opt<bool>
PrintJumpTables("print-jump-tables",
  cl::desc("print jump tables"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
X86AlignBranchBoundaryHotOnly("x86-align-branch-boundary-hot-only",
  cl::desc("only apply branch boundary alignment in hot code"),
  cl::init(true),
  cl::cat(BoltOptCategory));

size_t padFunction(const BinaryFunction &Function) {
  static std::map<std::string, size_t> FunctionPadding;

  if (FunctionPadding.empty() && !FunctionPadSpec.empty()) {
    for (auto &Spec : FunctionPadSpec) {
      auto N = Spec.find(':');
      if (N == std::string::npos)
        continue;
      auto Name = Spec.substr(0, N);
      auto Padding = std::stoull(Spec.substr(N+1));
      FunctionPadding[Name] = Padding;
    }
  }

  for (auto &FPI : FunctionPadding) {
    auto Name = FPI.first;
    auto Padding = FPI.second;
    if (Function.hasNameRegex(Name)) {
      return Padding;
    }
  }

  return 0;
}

} // namespace opts

namespace {

class BinaryEmitter {
private:
  BinaryEmitter(const BinaryEmitter &) = delete;
  BinaryEmitter &operator=(const BinaryEmitter &) = delete;

  MCStreamer &Streamer;
  BinaryContext &BC;

public:
  BinaryEmitter(MCStreamer &Streamer, BinaryContext &BC)
    : Streamer(Streamer),
      BC(BC) {}

  /// Emit all code and data.
  void emitAll(StringRef OrgSecPrefix);

  /// Emit function code. The caller is responsible for emitting function
  /// symbol(s) and setting the section to emit the code to.
  void emitFunctionBody(BinaryFunction &BF, bool EmitColdPart,
                        bool EmitCodeOnly = false);

private:
  /// Emit function code.
  void emitFunctions();

  /// Emit a single function.
  bool emitFunction(BinaryFunction &BF, bool EmitColdPart);

  /// Helper for emitFunctionBody to write data inside a function
  /// (used for AArch64)
  void emitConstantIslands(BinaryFunction &BF, bool EmitColdPart,
                           BinaryFunction *OnBehalfOf = nullptr);

  /// Emit jump tables for the function.
  void emitJumpTables(const BinaryFunction &BF);

  /// Emit jump table data. Callee supplies sections for the data.
  void emitJumpTable(const JumpTable &JT, MCSection *HotSection,
                     MCSection *ColdSection);

  /// Emit exception handling ranges for the function.
  void emitLSDA(BinaryFunction &BF, bool EmitColdPart);

  /// Emit line number information corresponding to \p NewLoc. \p PrevLoc
  /// provides a context for de-duplication of line number info.
  /// \p FirstInstr indicates if \p NewLoc represents the first instruction
  /// in a sequence, such as a function fragment.
  ///
  /// Return new current location which is either \p NewLoc or \p PrevLoc.
  SMLoc emitLineInfo(const BinaryFunction &BF, SMLoc NewLoc, SMLoc PrevLoc,
                     bool FirstInstr);

  /// Emit debug line information for functions that were not emitted.
  void emitDebugLineInfoForNonSimpleFunctions();

  /// Emit function as a blob with relocations and labels for relocations.
  void emitFunctionBodyRaw(BinaryFunction &BF) LLVM_ATTRIBUTE_UNUSED;

  /// Emit data sections that have code references in them.
  void emitDataSections(StringRef OrgSecPrefix);
};

} // anonymous namespace

void BinaryEmitter::emitAll(StringRef OrgSecPrefix) {
  Streamer.InitSections(false);

  if (auto *RtLibrary = BC.getRuntimeLibrary()) {
    RtLibrary->emitBinary(BC, Streamer);
  }

  BC.getTextSection()->setAlignment(opts::AlignText);

  emitFunctions();

  if (!BC.HasRelocations && opts::UpdateDebugSections)
    emitDebugLineInfoForNonSimpleFunctions();

  emitDataSections(OrgSecPrefix);

  // Update _end if needed.
  if (opts::UpdateEnd) {
    Streamer.EmitLabel(BC.Ctx->getOrCreateSymbol("_end"));
  }
}

void BinaryEmitter::emitFunctions() {
  auto emit = [&](const std::vector<BinaryFunction *> &Functions) {
    const auto HasProfile = BC.NumProfiledFuncs > 0;
    const uint32_t OriginalBranchBoundaryAlign = X86AlignBranchBoundary;
    for (auto *Function : Functions) {
      if (!BC.shouldEmit(*Function)) {
        continue;
      }

      DEBUG(dbgs() << "BOLT: generating code for function \""
                   << *Function << "\" : "
                   << Function->getFunctionNumber() << '\n');

      // Was any part of the function emitted.
      bool Emitted{false};

      // Turn off Intel JCC Erratum mitigation for cold code if requested
      if (HasProfile && opts::X86AlignBranchBoundaryHotOnly &&
          !Function->hasValidProfile())
        X86AlignBranchBoundary = 0;

      Emitted |= emitFunction(*Function, /*EmitColdPart=*/false);

      if (Function->isSplit()) {
        if (opts::X86AlignBranchBoundaryHotOnly)
          X86AlignBranchBoundary = 0;
        Emitted |= emitFunction(*Function, /*EmitColdPart=*/true);
      }
      X86AlignBranchBoundary = OriginalBranchBoundaryAlign;

      if (Emitted)
        Function->setEmitted(/*KeepCFG=*/opts::PrintCacheMetrics);
    }
  };

  // Mark the start of hot text.
  if (opts::HotText) {
    Streamer.SwitchSection(BC.getTextSection());
    Streamer.EmitLabel(BC.getHotTextStartSymbol());
  }

  // Emit functions in sorted order.
  std::vector<BinaryFunction *> SortedFunctions = BC.getSortedFunctions();
  emit(SortedFunctions);

  // Emit functions added by BOLT.
  emit(BC.getInjectedBinaryFunctions());

  // Mark the end of hot text.
  if (opts::HotText) {
    Streamer.SwitchSection(BC.getTextSection());
    Streamer.EmitLabel(BC.getHotTextEndSymbol());
  }
}

bool BinaryEmitter::emitFunction(BinaryFunction &Function, bool EmitColdPart) {
  if (Function.size() == 0)
    return false;

  if (Function.getState() == BinaryFunction::State::Empty)
    return false;

  auto *Section =
      BC.getCodeSection(EmitColdPart ? Function.getColdCodeSectionName()
                                     : Function.getCodeSectionName());
  Streamer.SwitchSection(Section);
  Section->setHasInstructions(true);
  BC.Ctx->addGenDwarfSection(Section);

  if (BC.HasRelocations) {
    Streamer.EmitCodeAlignment(BinaryFunction::MinAlign);
    auto MaxAlignBytes = EmitColdPart
      ? Function.getMaxColdAlignmentBytes()
      : Function.getMaxAlignmentBytes();
    if (MaxAlignBytes > 0)
      Streamer.EmitCodeAlignment(Function.getAlignment(), MaxAlignBytes);
  } else {
    Streamer.EmitCodeAlignment(Function.getAlignment());
  }

  MCContext &Context = Streamer.getContext();
  const MCAsmInfo *MAI = Context.getAsmInfo();

  // Emit all symbols associated with the main function entry.
  if (!EmitColdPart) {
    for (auto *Symbol : Function.getSymbols()) {
      Streamer.EmitSymbolAttribute(Symbol, MCSA_ELF_TypeFunction);
      Streamer.EmitLabel(Symbol);
    }
  } else {
    auto *Symbol = Function.getColdSymbol();
    Streamer.EmitSymbolAttribute(Symbol, MCSA_ELF_TypeFunction);
    Streamer.EmitLabel(Symbol);
  }

  // Emit CFI start
  if (Function.hasCFI()) {
    Streamer.EmitCFIStartProc(/*IsSimple=*/false);
    if (Function.getPersonalityFunction() != nullptr) {
      Streamer.EmitCFIPersonality(Function.getPersonalityFunction(),
                                  Function.getPersonalityEncoding());
    }
    auto *LSDASymbol = EmitColdPart ? Function.getColdLSDASymbol()
                                    : Function.getLSDASymbol();
    if (LSDASymbol) {
      Streamer.EmitCFILsda(LSDASymbol, BC.MOFI->getLSDAEncoding());
    } else {
      Streamer.EmitCFILsda(0, dwarf::DW_EH_PE_omit);
    }
    // Emit CFI instructions relative to the CIE
    for (const auto &CFIInstr : Function.cie()) {
      // Only write CIE CFI insns that LLVM will not already emit
      const std::vector<MCCFIInstruction> &FrameInstrs =
          MAI->getInitialFrameState();
      if (std::find(FrameInstrs.begin(), FrameInstrs.end(), CFIInstr) ==
          FrameInstrs.end())
        Streamer.EmitCFIInstruction(CFIInstr);
    }
  }

  assert((Function.empty() || !(*Function.begin()).isCold()) &&
         "first basic block should never be cold");

  // Emit UD2 at the beginning if requested by user.
  if (!opts::BreakFunctionNames.empty()) {
    for (auto &Name : opts::BreakFunctionNames) {
      if (Function.hasNameRegex(Name)) {
        Streamer.EmitIntValue(0x0B0F, 2); // UD2: 0F 0B
        break;
      }
    }
  }

  // Emit code.
  emitFunctionBody(Function, EmitColdPart, /*EmitCodeOnly=*/false);

  // Emit padding if requested.
  if (auto Padding = opts::padFunction(Function)) {
    DEBUG(dbgs() << "BOLT-DEBUG: padding function " << Function << " with "
                 << Padding << " bytes\n");
    Streamer.emitFill(Padding, MAI->getTextAlignFillValue());
  }

  if (opts::MarkFuncs) {
    Streamer.EmitIntValue(MAI->getTrapFillValue(), 1);
  }

  // Emit CFI end
  if (Function.hasCFI())
    Streamer.EmitCFIEndProc();

  Streamer.EmitLabel(EmitColdPart ? Function.getFunctionColdEndLabel()
                                  : Function.getFunctionEndLabel());

  // Exception handling info for the function.
  emitLSDA(Function, EmitColdPart);

  if (!EmitColdPart && opts::JumpTables > JTS_NONE)
    emitJumpTables(Function);

  return true;
}

void BinaryEmitter::emitFunctionBody(BinaryFunction &BF, bool EmitColdPart,
                                     bool EmitCodeOnly) {
  if (!EmitCodeOnly && EmitColdPart && BF.hasConstantIsland())
    BF.duplicateConstantIslands();

  // Track the first emitted instruction with debug info.
  bool FirstInstr = true;
  for (auto BB : BF.layout()) {
    if (EmitColdPart != BB->isCold())
      continue;

    if ((opts::AlignBlocks || opts::PreserveBlocksAlignment)
        && BB->getAlignment() > 1) {
      Streamer.EmitCodeAlignment(BB->getAlignment(),
                                 BB->getAlignmentMaxBytes());
    }
    Streamer.EmitLabel(BB->getLabel());
    if (auto *EntrySymbol = BF.getSecondaryEntryPointSymbol(*BB)) {
      Streamer.EmitLabel(EntrySymbol);
    }

    // Check if special alignment for macro-fusion is needed.
    bool MayNeedMacroFusionAlignment =
      (opts::AlignMacroOpFusion == MFT_ALL) ||
      (opts::AlignMacroOpFusion == MFT_HOT &&
       BB->getKnownExecutionCount());
    BinaryBasicBlock::const_iterator MacroFusionPair;
    if (MayNeedMacroFusionAlignment) {
      MacroFusionPair = BB->getMacroOpFusionPair();
      if (MacroFusionPair == BB->end())
        MayNeedMacroFusionAlignment = false;
    }

    SMLoc LastLocSeen;
    // Remember if the last instruction emitted was a prefix.
    bool LastIsPrefix = false;
    for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
      auto &Instr = *I;

      if (EmitCodeOnly && BC.MII->get(Instr.getOpcode()).isPseudo())
        continue;

      // Handle pseudo instructions.
      if (BC.MIB->isEHLabel(Instr)) {
        const auto *Label = BC.MIB->getTargetSymbol(Instr);
        assert(Instr.getNumOperands() >= 1 && Label &&
               "bad EH_LABEL instruction");
        Streamer.EmitLabel(const_cast<MCSymbol *>(Label));
        continue;
      }
      if (BC.MIB->isCFI(Instr)) {
        Streamer.EmitCFIInstruction(*BF.getCFIFor(Instr));
        continue;
      }

      // Handle macro-fusion alignment. If we emitted a prefix as
      // the last instruction, we should've already emitted the associated
      // alignment hint, so don't emit it twice.
      if (MayNeedMacroFusionAlignment && !LastIsPrefix && I == MacroFusionPair){
        // This assumes the second instruction in the macro-op pair will get
        // assigned to its own MCRelaxableFragment. Since all JCC instructions
        // are relaxable, we should be safe.
        Streamer.EmitNeverAlignCodeAtEnd(/*Alignment to avoid=*/64);
      }

      if (!EmitCodeOnly && opts::UpdateDebugSections &&
          BF.getDWARFUnitLineTable().first) {
        LastLocSeen = emitLineInfo(BF, Instr.getLoc(), LastLocSeen, FirstInstr);
        FirstInstr = false;
      }

      // Prepare to tag this location with a label if we need to keep track of
      // the location of calls/returns for BOLT address translation maps
      if (!EmitCodeOnly && BF.requiresAddressTranslation() &&
          BC.MIB->hasAnnotation(Instr, "Offset")) {
        const auto Offset = BC.MIB->getAnnotationAs<uint32_t>(Instr, "Offset");
        MCSymbol *LocSym = BC.Ctx->createTempSymbol(/*CanBeUnnamed=*/true);
        Streamer.EmitLabel(LocSym);
        BB->getLocSyms().emplace_back(std::make_pair(Offset, LocSym));
      }

      Streamer.EmitInstruction(Instr, *BC.STI);
      LastIsPrefix = BC.MIB->isPrefix(Instr);
    }
  }

  if (!EmitCodeOnly)
    emitConstantIslands(BF, EmitColdPart);
}

void BinaryEmitter::emitConstantIslands(BinaryFunction &BF, bool EmitColdPart,
                                        BinaryFunction *OnBehalfOf) {
  BinaryFunction::IslandInfo &Islands = BF.getIslandInfo();
  if (Islands.DataOffsets.empty() && Islands.Dependency.empty())
    return;

  if (!OnBehalfOf) {
    if (!EmitColdPart)
      Streamer.EmitLabel(BF.getFunctionConstantIslandLabel());
    else
      Streamer.EmitLabel(BF.getFunctionColdConstantIslandLabel());
  }

  assert((!OnBehalfOf || Islands.Proxies[OnBehalfOf].size() > 0) &&
         "spurious OnBehalfOf constant island emission");

  assert(!BF.isInjected() &&
         "injected functions should not have constant islands");
  // Raw contents of the function.
  StringRef SectionContents = BF.getSection().getContents();

  // Raw contents of the function.
  StringRef FunctionContents =
      SectionContents.substr(
          BF.getAddress() - BF.getSection().getAddress(),
          BF.getMaxSize());

  if (opts::Verbosity && !OnBehalfOf)
    outs() << "BOLT-INFO: emitting constant island for function " << BF << "\n";

  // We split the island into smaller blocks and output labels between them.
  auto IS = Islands.Offsets.begin();
  for (auto DataIter = Islands.DataOffsets.begin();
       DataIter != Islands.DataOffsets.end();
       ++DataIter) {
    uint64_t FunctionOffset = *DataIter;
    uint64_t EndOffset = 0ULL;

    // Determine size of this data chunk
    auto NextData = std::next(DataIter);
    auto CodeIter = Islands.CodeOffsets.lower_bound(*DataIter);
    if (CodeIter == Islands.CodeOffsets.end() &&
        NextData == Islands.DataOffsets.end()) {
      EndOffset = BF.getMaxSize();
    } else if (CodeIter == Islands.CodeOffsets.end()) {
      EndOffset = *NextData;
    } else if (NextData == Islands.DataOffsets.end()) {
      EndOffset = *CodeIter;
    } else {
      EndOffset = (*CodeIter > *NextData) ? *NextData : *CodeIter;
    }

    if (FunctionOffset == EndOffset)
      continue;    // Size is zero, nothing to emit

    // Emit labels, relocs and data
    auto RI = BF.getMoveRelocations().lower_bound(FunctionOffset);
    while ((IS != Islands.Offsets.end() && IS->first < EndOffset) ||
           (RI != BF.getMoveRelocations().end() && RI->first < EndOffset)) {
      auto NextLabelOffset =
        IS == Islands.Offsets.end() ? EndOffset : IS->first;
      auto NextRelOffset =
        RI == BF.getMoveRelocations().end() ? EndOffset : RI->first;
      auto NextStop = std::min(NextLabelOffset, NextRelOffset);
      assert(NextStop <= EndOffset && "internal overflow error");
      if (FunctionOffset < NextStop) {
        Streamer.EmitBytes(FunctionContents.slice(FunctionOffset, NextStop));
        FunctionOffset = NextStop;
      }
      if (IS != Islands.Offsets.end() && FunctionOffset == IS->first) {
        // This is a slightly complex code to decide which label to emit. We
        // have 4 cases to handle: regular symbol, cold symbol, regular or cold
        // symbol being emitted on behalf of an external function.
        if (!OnBehalfOf) {
          if (!EmitColdPart) {
            DEBUG(dbgs() << "BOLT-DEBUG: emitted label "
                         << IS->second->getName() << " at offset 0x"
                         << Twine::utohexstr(IS->first) << '\n');
            if (IS->second->isUndefined())
              Streamer.EmitLabel(IS->second);
            else
              assert(BF.hasName(IS->second->getName()));
          } else if (Islands.ColdSymbols.count(IS->second) != 0) {
            DEBUG(dbgs() << "BOLT-DEBUG: emitted label "
                         << Islands.ColdSymbols[IS->second]->getName()
                         << '\n');
            if (Islands.ColdSymbols[IS->second]->isUndefined())
              Streamer.EmitLabel(Islands.ColdSymbols[IS->second]);
          }
        } else {
          if (!EmitColdPart) {
            if (MCSymbol *Sym = Islands.Proxies[OnBehalfOf][IS->second]) {
              DEBUG(dbgs() << "BOLT-DEBUG: emitted label " << Sym->getName()
                           << '\n');
              Streamer.EmitLabel(Sym);
            }
          } else if (MCSymbol *Sym =
                         Islands.ColdProxies[OnBehalfOf][IS->second]) {
            DEBUG(dbgs() << "BOLT-DEBUG: emitted label " << Sym->getName()
                         << '\n');
            Streamer.EmitLabel(Sym);
          }
        }
        ++IS;
      }
      if (RI != BF.getMoveRelocations().end() && FunctionOffset == RI->first) {
        auto RelocationSize = RI->second.emit(&Streamer);
        DEBUG(dbgs() << "BOLT-DEBUG: emitted relocation for symbol "
                     << RI->second.Symbol->getName() << " at offset 0x"
                     << Twine::utohexstr(RI->first)
                     << " with size " << RelocationSize << '\n');
        FunctionOffset += RelocationSize;
        ++RI;
      }
    }
    assert(FunctionOffset <= EndOffset && "overflow error");
    if (FunctionOffset < EndOffset) {
      Streamer.EmitBytes(FunctionContents.slice(FunctionOffset, EndOffset));
    }
  }
  assert(IS == Islands.Offsets.end() && "some symbols were not emitted!");

  if (OnBehalfOf)
    return;
  // Now emit constant islands from other functions that we may have used in
  // this function.
  for (auto *ExternalFunc : Islands.Dependency) {
    emitConstantIslands(*ExternalFunc, EmitColdPart, &BF);
  }
}

SMLoc BinaryEmitter::emitLineInfo(const BinaryFunction &BF, SMLoc NewLoc,
                                  SMLoc PrevLoc, bool FirstInstr) {
  auto *FunctionCU = BF.getDWARFUnitLineTable().first;
  const auto *FunctionLineTable = BF.getDWARFUnitLineTable().second;
  assert(FunctionCU && "cannot emit line info for function without CU");

  auto RowReference = DebugLineTableRowRef::fromSMLoc(NewLoc);

  // Check if no new line info needs to be emitted.
  if (RowReference == DebugLineTableRowRef::NULL_ROW ||
      NewLoc.getPointer() == PrevLoc.getPointer())
    return PrevLoc;

  unsigned CurrentFilenum = 0;
  const auto *CurrentLineTable = FunctionLineTable;

  // If the CU id from the current instruction location does not
  // match the CU id from the current function, it means that we
  // have come across some inlined code.  We must look up the CU
  // for the instruction's original function and get the line table
  // from that.
  const auto FunctionUnitIndex = FunctionCU->getOffset();
  const auto CurrentUnitIndex = RowReference.DwCompileUnitIndex;
  if (CurrentUnitIndex != FunctionUnitIndex) {
    CurrentLineTable = BC.DwCtx->getLineTableForUnit(
        BC.DwCtx->getCompileUnitForOffset(CurrentUnitIndex));
    // Add filename from the inlined function to the current CU.
    CurrentFilenum =
      BC.addDebugFilenameToUnit(FunctionUnitIndex, CurrentUnitIndex,
        CurrentLineTable->Rows[RowReference.RowIndex - 1].File);
  }

  const auto &CurrentRow = CurrentLineTable->Rows[RowReference.RowIndex - 1];
  if (!CurrentFilenum)
    CurrentFilenum = CurrentRow.File;

  unsigned Flags = (DWARF2_FLAG_IS_STMT * CurrentRow.IsStmt) |
                   (DWARF2_FLAG_BASIC_BLOCK * CurrentRow.BasicBlock) |
                   (DWARF2_FLAG_PROLOGUE_END * CurrentRow.PrologueEnd) |
                   (DWARF2_FLAG_EPILOGUE_BEGIN * CurrentRow.EpilogueBegin);

  // Always emit is_stmt at the beginning of function fragment.
  if (FirstInstr)
    Flags |= DWARF2_FLAG_IS_STMT;

  BC.Ctx->setCurrentDwarfLoc(
    CurrentFilenum,
    CurrentRow.Line,
    CurrentRow.Column,
    Flags,
    CurrentRow.Isa,
    CurrentRow.Discriminator);
  BC.Ctx->setDwarfCompileUnitID(FunctionUnitIndex);

  return NewLoc;
}

void BinaryEmitter::emitJumpTables(const BinaryFunction &BF) {
  if (!BF.hasJumpTables())
    return;

  if (opts::PrintJumpTables) {
    outs() << "BOLT-INFO: jump tables for function " << BF << ":\n";
  }

  for (auto &JTI : BF.jumpTables()) {
    auto &JT = *JTI.second;
    if (opts::PrintJumpTables)
      JT.print(outs());
    if ((opts::JumpTables == JTS_BASIC || !BF.isSimple()) &&
        BC.HasRelocations) {
      JT.updateOriginal();
    } else {
      MCSection *HotSection, *ColdSection;
      if (opts::JumpTables == JTS_BASIC) {
        std::string Name = ".local." + JT.Labels[0]->getName().str();
        std::replace(Name.begin(), Name.end(), '/', '.');
        auto &Section = BC.registerOrUpdateSection(Name,
                                                   ELF::SHT_PROGBITS,
                                                   ELF::SHF_ALLOC);
        Section.setAnonymous(true);
        JT.setOutputSection(Section);
        HotSection = BC.getDataSection(Name);
        ColdSection = HotSection;
      } else {
        if (BF.isSimple()) {
          HotSection = BC.MOFI->getReadOnlySection();
          ColdSection = BC.MOFI->getReadOnlyColdSection();
        } else {
          HotSection = BF.hasProfile() ? BC.MOFI->getReadOnlySection()
                                       : BC.MOFI->getReadOnlyColdSection();
          ColdSection = HotSection;
        }
      }
      emitJumpTable(JT, HotSection, ColdSection);
    }
  }
}

void BinaryEmitter::emitJumpTable(const JumpTable &JT, MCSection *HotSection,
                                  MCSection *ColdSection) {
  // Pre-process entries for aggressive splitting.
  // Each label represents a separate switch table and gets its own count
  // determining its destination.
  std::map<MCSymbol *, uint64_t> LabelCounts;
  if (opts::JumpTables > JTS_SPLIT && !JT.Counts.empty()) {
    MCSymbol *CurrentLabel = JT.Labels.at(0);
    uint64_t CurrentLabelCount = 0;
    for (unsigned Index = 0; Index < JT.Entries.size(); ++Index) {
      auto LI = JT.Labels.find(Index * JT.EntrySize);
      if (LI != JT.Labels.end()) {
        LabelCounts[CurrentLabel] = CurrentLabelCount;
        CurrentLabel = LI->second;
        CurrentLabelCount = 0;
      }
      CurrentLabelCount += JT.Counts[Index].Count;
    }
    LabelCounts[CurrentLabel] = CurrentLabelCount;
  } else {
    Streamer.SwitchSection(JT.Count > 0 ? HotSection : ColdSection);
    Streamer.EmitValueToAlignment(JT.EntrySize);
  }
  MCSymbol *LastLabel = nullptr;
  uint64_t Offset = 0;
  for (auto *Entry : JT.Entries) {
    auto LI = JT.Labels.find(Offset);
    if (LI != JT.Labels.end()) {
      DEBUG(dbgs() << "BOLT-DEBUG: emitting jump table "
                   << LI->second->getName() << " (originally was at address 0x"
                   << Twine::utohexstr(JT.getAddress() + Offset)
                   << (Offset ? "as part of larger jump table\n" : "\n"));
      if (!LabelCounts.empty()) {
        DEBUG(dbgs() << "BOLT-DEBUG: jump table count: "
                     << LabelCounts[LI->second] << '\n');
        if (LabelCounts[LI->second] > 0) {
          Streamer.SwitchSection(HotSection);
        } else {
          Streamer.SwitchSection(ColdSection);
        }
        Streamer.EmitValueToAlignment(JT.EntrySize);
      }
      Streamer.EmitLabel(LI->second);
      LastLabel = LI->second;
    }
    if (JT.Type == JumpTable::JTT_NORMAL) {
      Streamer.EmitSymbolValue(Entry, JT.OutputEntrySize);
    } else { // JTT_PIC
      auto JTExpr = MCSymbolRefExpr::create(LastLabel, Streamer.getContext());
      auto E = MCSymbolRefExpr::create(Entry, Streamer.getContext());
      auto Value = MCBinaryExpr::createSub(E, JTExpr, Streamer.getContext());
      Streamer.EmitValue(Value, JT.EntrySize);
    }
    Offset += JT.EntrySize;
  }
}

// The code is based on EHStreamer::emitExceptionTable().
void BinaryEmitter::emitLSDA(BinaryFunction &BF, bool EmitColdPart) {
  const auto *Sites =
    EmitColdPart ? &BF.getColdCallSites() : &BF.getCallSites();
  if (Sites->empty()) {
    return;
  }

  // Calculate callsite table size. Size of each callsite entry is:
  //
  //  sizeof(start) + sizeof(length) + sizeof(LP) + sizeof(uleb128(action))
  //
  // or
  //
  //  sizeof(dwarf::DW_EH_PE_data4) * 3 + sizeof(uleb128(action))
  uint64_t CallSiteTableLength = Sites->size() * 4 * 3;
  for (const auto &CallSite : *Sites) {
    CallSiteTableLength += getULEB128Size(CallSite.Action);
  }

  Streamer.SwitchSection(BC.MOFI->getLSDASection());

  const auto TTypeEncoding = BC.MOFI->getTTypeEncoding();
  const auto TTypeEncodingSize = BC.getDWARFEncodingSize(TTypeEncoding);
  const auto TTypeAlignment = 4;

  // Type tables have to be aligned at 4 bytes.
  Streamer.EmitValueToAlignment(TTypeAlignment);

  // Emit the LSDA label.
  auto *LSDASymbol = EmitColdPart ? BF.getColdLSDASymbol() : BF.getLSDASymbol();
  assert(LSDASymbol && "no LSDA symbol set");
  Streamer.EmitLabel(LSDASymbol);

  // Corresponding FDE start.
  const auto *StartSymbol = EmitColdPart ? BF.getColdSymbol() : BF.getSymbol();

  // Emit the LSDA header.

  // If LPStart is omitted, then the start of the FDE is used as a base for
  // landing pad displacements. Then if a cold fragment starts with
  // a landing pad, this means that the first landing pad offset will be 0.
  // As a result, an exception handling runtime will ignore this landing pad,
  // because zero offset denotes the absence of a landing pad.
  // For this reason, we emit LPStart value of 0 and output an absolute value
  // of the landing pad in the table.
  //
  // FIXME: this may break PIEs and DSOs where the base address is not 0.
  Streamer.EmitIntValue(dwarf::DW_EH_PE_udata4, 1); // LPStart format
  Streamer.EmitIntValue(0, 4);
  auto emitLandingPad = [&](const MCSymbol *LPSymbol) {
    if (!LPSymbol) {
      Streamer.EmitIntValue(0, 4);
      return;
    }
    Streamer.EmitSymbolValue(LPSymbol, 4);
  };

  Streamer.EmitIntValue(TTypeEncoding, 1);        // TType format

  // See the comment in EHStreamer::emitExceptionTable() on to use
  // uleb128 encoding (which can use variable number of bytes to encode the same
  // value) to ensure type info table is properly aligned at 4 bytes without
  // iteratively fixing sizes of the tables.
  unsigned CallSiteTableLengthSize = getULEB128Size(CallSiteTableLength);
  unsigned TTypeBaseOffset =
    sizeof(int8_t) +                            // Call site format
    CallSiteTableLengthSize +                   // Call site table length size
    CallSiteTableLength +                       // Call site table length
    BF.getLSDAActionTable().size() +            // Actions table size
    BF.getLSDATypeTable().size() * TTypeEncodingSize; // Types table size
  unsigned TTypeBaseOffsetSize = getULEB128Size(TTypeBaseOffset);
  unsigned TotalSize =
    sizeof(int8_t) +                            // LPStart format
    sizeof(int8_t) +                            // TType format
    TTypeBaseOffsetSize +                       // TType base offset size
    TTypeBaseOffset;                            // TType base offset
  unsigned SizeAlign = (4 - TotalSize) & 3;

  // Account for any extra padding that will be added to the call site table
  // length.
  Streamer.EmitPaddedULEB128IntValue(TTypeBaseOffset,
                                     TTypeBaseOffsetSize + SizeAlign);

  // Emit the landing pad call site table. We use signed data4 since we can emit
  // a landing pad in a different part of the split function that could appear
  // earlier in the address space than LPStart.
  Streamer.EmitIntValue(dwarf::DW_EH_PE_sdata4, 1);
  Streamer.EmitULEB128IntValue(CallSiteTableLength);

  for (const auto &CallSite : *Sites) {
    const auto *BeginLabel = CallSite.Start;
    const auto *EndLabel = CallSite.End;

    assert(BeginLabel && "start EH label expected");
    assert(EndLabel && "end EH label expected");

    // Start of the range is emitted relative to the start of current
    // function split part.
    Streamer.emitAbsoluteSymbolDiff(BeginLabel, StartSymbol, 4);
    Streamer.emitAbsoluteSymbolDiff(EndLabel, BeginLabel, 4);
    emitLandingPad(CallSite.LP);
    Streamer.EmitULEB128IntValue(CallSite.Action);
  }

  // Write out action, type, and type index tables at the end.
  //
  // For action and type index tables there's no need to change the original
  // table format unless we are doing function splitting, in which case we can
  // split and optimize the tables.
  //
  // For type table we (re-)encode the table using TTypeEncoding matching
  // the current assembler mode.
  for (auto const &Byte : BF.getLSDAActionTable()) {
    Streamer.EmitIntValue(Byte, 1);
  }
  assert(!(TTypeEncoding & dwarf::DW_EH_PE_indirect) &&
         "indirect type info encoding is not supported yet");
  for (int Index = BF.getLSDATypeTable().size() - 1; Index >= 0; --Index) {
    // Note: the address could be an indirect one.
    const auto TypeAddress = BF.getLSDATypeTable()[Index];
    switch (TTypeEncoding & 0x70) {
    default:
      llvm_unreachable("unsupported TTypeEncoding");
    case 0:
      Streamer.EmitIntValue(TypeAddress, TTypeEncodingSize);
      break;
    case dwarf::DW_EH_PE_pcrel: {
      if (TypeAddress) {
        const auto *TypeSymbol =
          BC.getOrCreateGlobalSymbol(TypeAddress,
                                     "TI",
                                     TTypeEncodingSize,
                                     TTypeAlignment);
        auto *DotSymbol = BC.Ctx->createTempSymbol();
        Streamer.EmitLabel(DotSymbol);
        const auto *SubDotExpr = MCBinaryExpr::createSub(
            MCSymbolRefExpr::create(TypeSymbol, *BC.Ctx),
            MCSymbolRefExpr::create(DotSymbol, *BC.Ctx),
            *BC.Ctx);
        Streamer.EmitValue(SubDotExpr, TTypeEncodingSize);
      } else {
        Streamer.EmitIntValue(0, TTypeEncodingSize);
      }
      break;
    }
    }
  }
  for (auto const &Byte : BF.getLSDATypeIndexTable()) {
    Streamer.EmitIntValue(Byte, 1);
  }
}

void BinaryEmitter::emitDebugLineInfoForNonSimpleFunctions() {
  for (auto &It : BC.getBinaryFunctions()) {
    const auto &Function = It.second;

    if (Function.isSimple())
      continue;

    auto ULT = Function.getDWARFUnitLineTable();
    auto Unit = ULT.first;
    auto LineTable = ULT.second;

    if (!LineTable)
      continue; // nothing to update for this function

    std::vector<uint32_t> Results;
    MCSection *FunctionSection =
        BC.getCodeSection(Function.getCodeSectionName());

    uint64_t Address = It.first;
    if (LineTable->lookupAddressRange(Address, Function.getMaxSize(),
                                      Results)) {
      auto &OutputLineTable =
          BC.Ctx->getMCDwarfLineTable(Unit->getOffset()).getMCLineSections();
      for (auto RowIndex : Results) {
        const auto &Row = LineTable->Rows[RowIndex];
        BC.Ctx->setCurrentDwarfLoc(
            Row.File,
            Row.Line,
            Row.Column,
            (DWARF2_FLAG_IS_STMT * Row.IsStmt) |
            (DWARF2_FLAG_BASIC_BLOCK * Row.BasicBlock) |
            (DWARF2_FLAG_PROLOGUE_END * Row.PrologueEnd) |
            (DWARF2_FLAG_EPILOGUE_BEGIN * Row.EpilogueBegin),
            Row.Isa,
            Row.Discriminator,
            Row.Address);
        auto Loc = BC.Ctx->getCurrentDwarfLoc();
        BC.Ctx->clearDwarfLocSeen();
        OutputLineTable.addLineEntry(MCDwarfLineEntry{nullptr, Loc},
                                     FunctionSection);
      }
      // Add an empty entry past the end of the function
      // for end_sequence mark.
      BC.Ctx->setCurrentDwarfLoc(0, 0, 0, 0, 0, 0,
                                 Address + Function.getMaxSize());
      auto Loc = BC.Ctx->getCurrentDwarfLoc();
      BC.Ctx->clearDwarfLocSeen();
      OutputLineTable.addLineEntry(MCDwarfLineEntry{nullptr, Loc},
                                   FunctionSection);
    } else {
      DEBUG(dbgs() << "BOLT-DEBUG: function " << Function
                   << " has no associated line number information\n");
    }
  }
}

void BinaryEmitter::emitFunctionBodyRaw(BinaryFunction &BF) {
  // #14998851: Fix gold linker's '--emit-relocs'.
  llvm_unreachable(
      "cannot emit raw body unless relocation accuracy is guaranteed");

  assert(!BF.isInjected() && "cannot emit raw body of injected function");

  // Raw contents of the function.
  StringRef SectionContents = BF.getSection().getContents();

  // Raw contents of the function.
  StringRef FunctionContents = SectionContents.substr(
      BF.getAddress() - BF.getSection().getAddress(), BF.getSize());

  if (opts::Verbosity)
    outs() << "BOLT-INFO: emitting function " << BF << " in raw ("
           << BF.getSize() << " bytes)\n";

  // We split the function blob into smaller blocks and output relocations
  // and/or labels between them.
  uint64_t FunctionOffset = 0;
  auto LI = BF.getLabels().begin();
  auto RI = BF.getMoveRelocations().begin();
  while (LI != BF.getLabels().end() ||
         RI != BF.getMoveRelocations().end()) {
    uint64_t NextLabelOffset =
      (LI == BF.getLabels().end() ? BF.getSize() : LI->first);
    uint64_t NextRelocationOffset =
      (RI == BF.getMoveRelocations().end() ? BF.getSize() : RI->first);
    auto NextStop = std::min(NextLabelOffset, NextRelocationOffset);
    assert(NextStop <= BF.getSize() && "internal overflow error");
    if (FunctionOffset < NextStop) {
      Streamer.EmitBytes(FunctionContents.slice(FunctionOffset, NextStop));
      FunctionOffset = NextStop;
    }
    if (LI != BF.getLabels().end() && FunctionOffset == LI->first) {
      Streamer.EmitLabel(LI->second);
      DEBUG(dbgs() << "BOLT-DEBUG: emitted label " << LI->second->getName()
                   << " at offset 0x" << Twine::utohexstr(LI->first) << '\n');
      ++LI;
    }
    if (RI != BF.getMoveRelocations().end() && FunctionOffset == RI->first) {
      auto RelocationSize = RI->second.emit(&Streamer);
      DEBUG(dbgs() << "BOLT-DEBUG: emitted relocation for symbol "
                   << RI->second.Symbol->getName() << " at offset 0x"
                   << Twine::utohexstr(RI->first)
                   << " with size " << RelocationSize << '\n');
      FunctionOffset += RelocationSize;
      ++RI;
    }
  }
  assert(FunctionOffset <= BF.getSize() && "overflow error");
  if (FunctionOffset < BF.getSize()) {
    Streamer.EmitBytes(FunctionContents.substr(FunctionOffset));
  }
}

void BinaryEmitter::emitDataSections(StringRef OrgSecPrefix) {
  for (auto &Section : BC.sections()) {
    if (!Section.hasRelocations() || !Section.hasSectionRef())
      continue;

    StringRef SectionName = Section.getName();
    std::string EmitName = Section.isReordered()
      ? std::string(Section.getOutputName())
      : OrgSecPrefix.str() + std::string(SectionName);
    Section.emitAsData(Streamer, EmitName);
    Section.clearRelocations();
  }
}

namespace llvm {
namespace bolt {

void emitBinaryContext(MCStreamer &Streamer, BinaryContext &BC,
                       StringRef OrgSecPrefix) {
  BinaryEmitter(Streamer, BC).emitAll(OrgSecPrefix);
}

void emitFunctionBody(MCStreamer &Streamer, BinaryFunction &BF,
                      bool EmitColdPart, bool EmitCodeOnly) {
  BinaryEmitter(Streamer, BF.getBinaryContext()).
    emitFunctionBody(BF, EmitColdPart, EmitCodeOnly);
}

} // namespace bolt
} // namespace llvm
