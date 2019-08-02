//===--- Passes/Instrumentation.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Instrumentation.h"
#include "Passes/DataflowInfoManager.h"
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "bolt-instrumentation"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;

extern bool shouldProcess(const llvm::bolt::BinaryFunction &Function);

cl::opt<std::string> InstrumentationFilename(
    "instrumentation-file",
    cl::desc("file name where instrumented profile will be saved"),
    cl::init("/tmp/prof.fdata"),
    cl::Optional,
    cl::cat(BoltCategory));

cl::opt<bool> InstrumentHotOnly(
    "instrument-hot-only",
    cl::desc("only insert instrumentation on hot functions (need profile)"),
    cl::init(false),
    cl::Optional,
    cl::cat(BoltCategory));
}

namespace llvm {
namespace bolt {

uint32_t Instrumentation::getFunctionNameIndex(const BinaryFunction &Function) {
  auto Iter = FuncToStringIdx.find(&Function);
  if (Iter != FuncToStringIdx.end())
    return Iter->second;
  auto Idx = StringTable.size();
  FuncToStringIdx.emplace(std::make_pair(&Function, Idx));
  StringTable.append(Function.getNames()[0]);
  StringTable.append(1, '\0');
  return Idx;
}

Instrumentation::CounterDescription Instrumentation::createDescription(
    const BinaryFunction &FromFunction, uint32_t From,
    const BinaryFunction &ToFunction, uint32_t To) {
  CounterDescription Res;
  Res.FromFuncStringIdx = getFunctionNameIndex(FromFunction);
  Res.FromOffset = From;
  Res.ToFuncStringIdx = getFunctionNameIndex(ToFunction);
  Res.ToOffset = To;
  return Res;
}

std::vector<MCInst> Instrumentation::createInstrumentationSnippet(
    BinaryFunction &FromFunction, uint32_t FromOffset, BinaryFunction &ToFunc,
    uint32_t ToOffset) {
  Descriptions.emplace_back(
      createDescription(FromFunction, FromOffset, ToFunc, ToOffset));

  BinaryContext &BC = FromFunction.getBinaryContext();
  MCSymbol *Label =
      BC.Ctx->createTempSymbol("InstrEntry", true);
  Labels.emplace_back(Label);
  std::vector<MCInst> CounterInstrs(5);
  // Don't clobber application red zone (ABI dependent)
  BC.MIB->createStackPointerIncrement(CounterInstrs[0], 128,
                                      /*NoFlagsClobber=*/true);
  BC.MIB->createPushFlags(CounterInstrs[1], 2);
  BC.MIB->createIncMemory(CounterInstrs[2], Label, &*BC.Ctx);
  BC.MIB->createPopFlags(CounterInstrs[3], 2);
  BC.MIB->createStackPointerDecrement(CounterInstrs[4], 128,
                                      /*NoFlagsClobber=*/true);
  return CounterInstrs;
}

bool Instrumentation::instrumentOneTarget(BinaryBasicBlock::iterator &Iter,
                                          BinaryFunction &FromFunction,
                                          BinaryBasicBlock &FromBB,
                                          uint32_t From, BinaryFunction &ToFunc,
                                          BinaryBasicBlock *TargetBB,
                                          uint32_t ToOffset) {
  std::vector<MCInst> CounterInstrs =
      createInstrumentationSnippet(FromFunction, From, ToFunc, ToOffset);

  BinaryContext &BC = FromFunction.getBinaryContext();
  const MCInst &Inst = *Iter;
  if (BC.MIB->isCall(Inst) && !TargetBB) {
    for (auto &NewInst : CounterInstrs) {
      Iter = FromBB.insertInstruction(Iter, NewInst);
      ++Iter;
    }
    return true;
  }

  if (!TargetBB)
    return false;

  // Indirect branch, conditional branches or fall-throughs
  // Regular cond branch, put counter at start of target block
  if (TargetBB->pred_size() == 1 && &FromBB != TargetBB &&
      !TargetBB->isEntryPoint()) {
    auto RemoteIter = TargetBB->begin();
    for (auto &NewInst : CounterInstrs) {
      RemoteIter = TargetBB->insertInstruction(RemoteIter, NewInst);
      ++RemoteIter;
    }
    return true;
  }
  if (FromBB.succ_size() == 1 && &FromBB != TargetBB) {
    for (auto &NewInst : CounterInstrs) {
      Iter = FromBB.insertInstruction(Iter, NewInst);
      ++Iter;
    }
    return true;
  }
  // Critical edge, create BB and put counter there
  SplitWorklist.emplace_back(std::make_pair(&FromBB, TargetBB));
  SplitInstrs.emplace_back(std::move(CounterInstrs));
  return true;
}

void Instrumentation::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return;

  const auto Flags = BinarySection::getFlags(/*IsReadOnly=*/false,
                                             /*IsText=*/false,
                                             /*IsAllocatable=*/true);
  BC.registerOrUpdateSection(".bolt.instr.counters", ELF::SHT_PROGBITS, Flags,
                             nullptr, 0, 1,
                             /*local=*/true);

  BC.registerOrUpdateNoteSection(".bolt.instr.tables", nullptr,
                                  0,
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);

  uint64_t InstrumentationSites{0ULL};
  uint64_t InstrumentationSitesSavingFlags{0ULL};
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isSimple() || !opts::shouldProcess(Function)
        || (opts::InstrumentHotOnly && !Function.getKnownExecutionCount()))
      continue;
    Function.disambiguateJumpTables();
    SplitWorklist.clear();
    SplitInstrs.clear();

    for (auto BBI = Function.begin(); BBI != Function.end(); ++BBI) {
      auto &BB{*BBI};
      bool HasUnconditionalBranch{false};
      bool HasJumpTable{false};

      for (auto I = BB.begin(); I != BB.end(); ++I) {
        const auto &Inst = *I;
        if (!BC.MIB->hasAnnotation(Inst, "Offset"))
          continue;

        const bool IsJumpTable = Function.getJumpTable(Inst);
        if (IsJumpTable)
          HasJumpTable = true;
        else if (BC.MIB->isUnconditionalBranch(Inst))
          HasUnconditionalBranch = true;
        else if ((!BC.MIB->isCall(Inst) &&
                  !BC.MIB->isConditionalBranch(Inst)) ||
                 BC.MIB->isUnsupportedBranch(Inst.getOpcode()))
          continue;

        uint32_t FromOffset = BC.MIB->getAnnotationAs<uint32_t>(Inst, "Offset");
        const MCSymbol *Target = BC.MIB->getTargetSymbol(Inst);
        BinaryBasicBlock *TargetBB = Function.getBasicBlockForLabel(Target);
        uint32_t ToOffset = TargetBB ? TargetBB->getInputOffset() : 0;
        BinaryFunction *TargetFunc =
            TargetBB ? &Function : BC.getFunctionForSymbol(Target);
        // Should be null for indirect branches/calls
        if (TargetFunc) {
          if (instrumentOneTarget(I, Function, BB, FromOffset, *TargetFunc,
                                  TargetBB, ToOffset))
            ++InstrumentationSites;
          continue;
        }

        if (IsJumpTable) {
          for (auto &Succ : BB.successors()) {
            if (instrumentOneTarget(I, Function, BB, FromOffset, Function,
                                    &*Succ, Succ->getInputOffset()))
              ++InstrumentationSites;
          }
          continue;
        }

        // FIXME: handle indirect calls
      } // End of instructions loop

      // Instrument fallthroughs (when the direct jump instruction is missing)
      if (!HasUnconditionalBranch && !HasJumpTable && BB.succ_size() > 0 &&
          BB.size() > 0) {
        auto *FTBB = BB.getFallthrough();
        assert(FTBB && "expected valid fall-through basic block");
        auto I = BB.begin();
        auto LastInstr = BB.end();
        --LastInstr;
        while (LastInstr != I && BC.MIB->isPseudo(*LastInstr))
          --LastInstr;
        uint32_t FromOffset = 0;
        // The last instruction in the BB should have an annotation, except
        // if it was branching to the end of the function as a result of
        // __builtin_unreachable(), in which case it was deleted by fixBranches.
        // Ignore this case. FIXME: force fixBranches() to preserve the offset.
        if (!BC.MIB->hasAnnotation(*LastInstr, "Offset"))
          continue;

        FromOffset = BC.MIB->getAnnotationAs<uint32_t>(*LastInstr, "Offset");
        if (instrumentOneTarget(I, Function, BB, FromOffset, Function, FTBB,
                                FTBB->getInputOffset()))
          ++InstrumentationSites;
      }
    } // End of BBs loop

    // Consume list of critical edges: split them and add instrumentation to the
    // newly created BBs
    auto Iter = SplitInstrs.begin();
    for (auto &BBPair : SplitWorklist) {
      auto *NewBB = Function.splitEdge(BBPair.first, BBPair.second);
      NewBB->addInstructions(Iter->begin(), Iter->end());
      ++Iter;
    }
  }

  outs() << "BOLT-INSTRUMENTER: Instrumented " << InstrumentationSites
         << " sites, " << InstrumentationSitesSavingFlags << " saving flags.\n";
}

void Instrumentation::emitTablesAsELFNote(BinaryContext &BC) {
  std::string TablesStr;
  raw_string_ostream OS(TablesStr);

  // Start of the vector with descriptions (one CounterDescription for each
  // counter), vector size is Labels.size() CounterDescription-sized elmts
  for (const auto &Desc : Descriptions) {
    OS.write(reinterpret_cast<const char *>(&Desc.FromFuncStringIdx), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.FromOffset), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.ToFuncStringIdx), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.ToOffset), 4);
  }
  // Our string table lives immediately after descriptions vector
  OS << StringTable;
  OS.flush();
  const auto BoltInfo = BinarySection::encodeELFNote(
      "BOLT", TablesStr, BinarySection::NT_BOLT_INSTRUMENTATION_TABLES);
  BC.registerOrUpdateNoteSection(".bolt.instr.tables", copyByteArray(BoltInfo),
                                 BoltInfo.size(),
                                 /*Alignment=*/1,
                                 /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

void Instrumentation::emit(BinaryContext &BC, MCStreamer &Streamer) {
  emitTablesAsELFNote(BC);

  const auto Flags = BinarySection::getFlags(/*IsReadOnly=*/false,
                                             /*IsText=*/false,
                                             /*IsAllocatable=*/true);
  auto *Section = BC.Ctx->getELFSection(".bolt.instr.counters",
                                        ELF::SHT_PROGBITS,
                                        Flags);

  // All of the following symbols will be exported as globals to be used by the
  // instrumentation runtime library to dump the instrumentation data to disk.
  // Label marking start of the memory region containing instrumentation
  // counters, total vector size is Labels.size() 8-byte counters
  MCSymbol *Locs = BC.Ctx->getOrCreateSymbol("__bolt_instr_locations");
  MCSymbol *NumLocs = BC.Ctx->getOrCreateSymbol("__bolt_instr_num_locs");
  /// File name where profile is going to written to after target binary
  /// finishes a run
  MCSymbol *FilenameSym = BC.Ctx->getOrCreateSymbol("__bolt_instr_filename");

  Streamer.SwitchSection(Section);
  Streamer.EmitLabel(Locs);
  Streamer.EmitSymbolAttribute(Locs,
                               MCSymbolAttr::MCSA_Global);
  for (const auto &Label : Labels) {
    Streamer.EmitLabel(Label);
    Streamer.emitFill(8, 0);
  }
  Streamer.EmitLabel(NumLocs);
  Streamer.EmitSymbolAttribute(NumLocs,
                               MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(Labels.size(), /*Size=*/4);
  Streamer.EmitLabel(FilenameSym);
  Streamer.EmitBytes(opts::InstrumentationFilename);
  Streamer.emitFill(1, 0);
  outs() << "BOLT-INSTRUMENTER: Total size of counters: "
         << (Labels.size() * 8) << " bytes (static alloc memory)\n";
  outs() << "BOLT-INSTRUMENTER: Total size of string table emitted: "
         << StringTable.size() << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Total size of descriptors: "
         << (Labels.size() * 16) << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Profile will be saved to file "
         << opts::InstrumentationFilename << "\n";
}

}
}
