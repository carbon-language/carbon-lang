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
#include "ParallelUtilities.h"
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

cl::opt<bool> InstrumentCalls(
    "instrument-calls",
    cl::desc("record profile for inter-function control flow activity"),
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

void Instrumentation::createCallDescription(
    const BinaryFunction &FromFunction, uint32_t From,
    const BinaryFunction &ToFunction, uint32_t To) {
  CallDescription CD;
  CD.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  CD.FromLoc.Offset = From;
  CD.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  CD.ToLoc.Offset = To;
  CD.Counter = Counters.size();
  CallDescriptions.emplace_back(CD);
}

bool Instrumentation::createEdgeDescription(
    FunctionDescription &FuncDesc,
    const BinaryFunction &FromFunction, uint32_t From,
    uint32_t FromNodeID,
    const BinaryFunction &ToFunction, uint32_t To,
    uint32_t ToNodeID, bool Instrumented) {
  EdgeDescription ED;
  auto Result = FuncDesc.EdgesSet.insert(std::make_pair(FromNodeID, ToNodeID));
  // Avoid creating duplicated edge descriptions. This happens in CFGs where a
  // block jumps to its fall-through.
  if (Result.second == false)
    return false;
  ED.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  ED.FromLoc.Offset = From;
  ED.FromNode = FromNodeID;
  ED.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  ED.ToLoc.Offset = To;
  ED.ToNode = ToNodeID;
  ED.Counter = Instrumented ? Counters.size() : 0xffffffff;
  FuncDesc.Edges.emplace_back(ED);
  return true;
}

void Instrumentation::createExitNodeDescription(FunctionDescription &FuncDesc,
                                                uint32_t Node) {
  InstrumentedNode IN;
  IN.Node = Node;
  IN.Counter = Counters.size();
  FuncDesc.ExitNodes.emplace_back(IN);
}

std::vector<MCInst>
Instrumentation::createInstrumentationSnippet(BinaryContext &BC, bool IsLeaf) {
  auto L = BC.scopeLock();
  MCSymbol *Label;
  Label = BC.Ctx->createTempSymbol("InstrEntry", true);
  Counters.emplace_back(Label);
  std::vector<MCInst> CounterInstrs(5);
  // Don't clobber application red zone (ABI dependent)
  if (IsLeaf)
    BC.MIB->createStackPointerIncrement(CounterInstrs[0], 128,
                                        /*NoFlagsClobber=*/true);
  BC.MIB->createPushFlags(CounterInstrs[1], 2);
  BC.MIB->createIncMemory(CounterInstrs[2], Label, &*BC.Ctx);
  BC.MIB->createPopFlags(CounterInstrs[3], 2);
  if (IsLeaf)
    BC.MIB->createStackPointerDecrement(CounterInstrs[4], 128,
                                        /*NoFlagsClobber=*/true);
  return CounterInstrs;
}

void Instrumentation::instrumentExitNode(BinaryContext &BC,
                                         BinaryBasicBlock &BB,
                                         BinaryBasicBlock::iterator Iter,
                                         bool IsLeaf,
                                         FunctionDescription &FuncDesc,
                                         uint32_t Node) {
  createExitNodeDescription(FuncDesc, Node);
  std::vector<MCInst> CounterInstrs = createInstrumentationSnippet(BC, IsLeaf);

  for (auto &NewInst : CounterInstrs) {
    Iter = BB.insertInstruction(Iter, NewInst);
    ++Iter;
  }
}

bool Instrumentation::instrumentOneTarget(
    SplitWorklistTy &SplitWorklist, SplitInstrsTy &SplitInstrs,
    BinaryBasicBlock::iterator &Iter, BinaryFunction &FromFunction,
    BinaryBasicBlock &FromBB, uint32_t From, BinaryFunction &ToFunc,
    BinaryBasicBlock *TargetBB, uint32_t ToOffset, bool IsLeaf,
    FunctionDescription *FuncDesc, uint32_t FromNodeID, uint32_t ToNodeID) {
  {
    auto L = FromFunction.getBinaryContext().scopeLock();
    bool Created{true};
    if (!TargetBB)
      createCallDescription(FromFunction, From, ToFunc, ToOffset);
    else
      Created = createEdgeDescription(*FuncDesc, FromFunction, From, FromNodeID,
                                      ToFunc, ToOffset, ToNodeID,
                                      /*Instrumented=*/true);
    if (!Created)
      return false;
  }

  std::vector<MCInst> CounterInstrs =
    createInstrumentationSnippet(FromFunction.getBinaryContext(), IsLeaf);

  BinaryContext &BC = FromFunction.getBinaryContext();
  const MCInst &Inst = *Iter;
  if (BC.MIB->isCall(Inst) && !TargetBB) {
    for (auto &NewInst : CounterInstrs) {
      Iter = FromBB.insertInstruction(Iter, NewInst);
      ++Iter;
    }
    return true;
  }

  if (!TargetBB || !FuncDesc)
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

void Instrumentation::instrumentFunction(BinaryContext &BC,
                                         BinaryFunction &Function,
                                         MCPlusBuilder::AllocatorIdTy AllocId) {
  SplitWorklistTy SplitWorklist;
  SplitInstrsTy SplitInstrs;

  FunctionDescription *FuncDesc = nullptr;
  {
    std::unique_lock<std::shared_timed_mutex> L(FDMutex);
    FunctionDescriptions.emplace_back();
    FuncDesc = &FunctionDescriptions.back();
  }

  Function.disambiguateJumpTables(AllocId);

  std::unordered_map<const BinaryBasicBlock *, uint32_t> BBToID;
  uint32_t Id = 0;
  for (auto BBI = Function.begin(); BBI != Function.end(); ++BBI) {
    BBToID[&*BBI] = Id++;
  }
  std::unordered_set<const BinaryBasicBlock *> VisitedSet;
  // DFS to establish edges we will use for a spanning tree. Edges in the
  // spanning tree can be instrumentation-free since their count can be
  // inferred by solving flow equations on a bottom-up traversal of the tree.
  // Exit basic blocks are always instrumented so we start the traversal with
  // a minimum number of defined variables to make the equation solvable.
  std::stack<std::pair<const BinaryBasicBlock *, BinaryBasicBlock *>> Stack;
  std::unordered_map<const BinaryBasicBlock *,
                     std::set<const BinaryBasicBlock *>>
      STOutSet;
  for (auto BBI = Function.layout_rbegin(); BBI != Function.layout_rend();
       ++BBI) {
    if ((*BBI)->isEntryPoint())
      Stack.push(std::make_pair(nullptr, *BBI));
  }

  // Modified version of BinaryFunction::dfs() to build a spanning tree
  while (!Stack.empty()) {
    BinaryBasicBlock *BB;
    const BinaryBasicBlock *Pred;
    std::tie(Pred, BB) = Stack.top();
    Stack.pop();
    if (VisitedSet.find(BB) != VisitedSet.end())
      continue;

    VisitedSet.insert(BB);
    if (Pred)
      STOutSet[Pred].insert(BB);

    for (auto *SuccBB : BB->landing_pads())
      Stack.push(std::make_pair(BB, SuccBB));

    for (auto *SuccBB : BB->successors())
      Stack.push(std::make_pair(BB, SuccBB));
  }

  // Determine whether this is a leaf function, which needs special
  // instructions to protect the red zone
  bool IsLeafFunction{true};
  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
    for (auto I = BBI->begin(), E = BBI->end(); I != E; ++I) {
      if (BC.MIB->isCall(*I)) {
        IsLeafFunction = false;
        break;
      }
    }
    if (!IsLeafFunction)
      break;
  }

  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
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
      else if ((!BC.MIB->isCall(Inst) && !BC.MIB->isConditionalBranch(Inst)) ||
               BC.MIB->isUnsupportedBranch(Inst.getOpcode()))
        continue;

      uint32_t FromOffset = BC.MIB->getAnnotationAs<uint32_t>(Inst, "Offset");
      const MCSymbol *Target = BC.MIB->getTargetSymbol(Inst);
      BinaryBasicBlock *TargetBB = Function.getBasicBlockForLabel(Target);
      uint32_t ToOffset = TargetBB ? TargetBB->getInputOffset() : 0;
      BinaryFunction *TargetFunc =
          TargetBB ? &Function : BC.getFunctionForSymbol(Target);
      // Should be null for indirect branches/calls
      if (TargetFunc && !TargetBB) {
        if (opts::InstrumentCalls)
          instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                              FromOffset, *TargetFunc, TargetBB, ToOffset,
                              IsLeafFunction);
        continue;
      }
      if (TargetFunc) {
        // Do not instrument edges in the spanning tree
        if (STOutSet[&BB].find(TargetBB) != STOutSet[&BB].end()) {
          auto L = BC.scopeLock();
          createEdgeDescription(*FuncDesc, Function, FromOffset, BBToID[&BB],
                                Function, ToOffset, BBToID[TargetBB],
                                /*Instrumented=*/false);
          continue;
        }
        instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                            FromOffset, *TargetFunc, TargetBB, ToOffset,
                            IsLeafFunction, FuncDesc, BBToID[&BB],
                            BBToID[TargetBB]);
        continue;
      }

      if (IsJumpTable) {
        for (auto &Succ : BB.successors()) {
          // Do not instrument edges in the spanning tree
          if (STOutSet[&BB].find(&*Succ) != STOutSet[&BB].end()) {
            auto L = BC.scopeLock();
            createEdgeDescription(*FuncDesc, Function, FromOffset, BBToID[&BB],
                                  Function, Succ->getInputOffset(),
                                  BBToID[&*Succ], /*Instrumented=*/false);
            continue;
          }
          instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                              FromOffset, Function, &*Succ,
                              Succ->getInputOffset(), IsLeafFunction, FuncDesc,
                              BBToID[&BB], BBToID[&*Succ]);
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

      // Do not instrument edges in the spanning tree
      if (STOutSet[&BB].find(FTBB) != STOutSet[&BB].end()) {
        auto L = BC.scopeLock();
        createEdgeDescription(*FuncDesc, Function, FromOffset, BBToID[&BB],
                              Function, FTBB->getInputOffset(), BBToID[FTBB],
                              /*Instrumented=*/false);
        continue;
      }
      instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                          FromOffset, Function, FTBB, FTBB->getInputOffset(),
                          IsLeafFunction, FuncDesc, BBToID[&BB], BBToID[FTBB]);
    }
  } // End of BBs loop

  // Instrument spanning tree leaves
  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
    auto &BB{*BBI};
    if (STOutSet[&BB].size() == 0 && BB.size() > 0)
      instrumentExitNode(BC, BB, BB.begin(), IsLeafFunction, *FuncDesc,
                         BBToID[&BB]);
  }

  // Consume list of critical edges: split them and add instrumentation to the
  // newly created BBs
  auto Iter = SplitInstrs.begin();
  for (auto &BBPair : SplitWorklist) {
    auto *NewBB = Function.splitEdge(BBPair.first, BBPair.second);
    NewBB->addInstructions(Iter->begin(), Iter->end());
    ++Iter;
  }

  // Unused now
  FuncDesc->EdgesSet.clear();
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

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return (!BF.isSimple() || !opts::shouldProcess(BF) ||
            (opts::InstrumentHotOnly && !BF.getKnownExecutionCount()));
  };

  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocatorId) {
        instrumentFunction(BC, BF, AllocatorId);
      };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_QUADRATIC, WorkFun,
      SkipPredicate, "instrumentation", /* ForceSequential=*/true);
}

uint32_t Instrumentation::getFDSize() const {
  uint32_t FuncDescSize = 0;
  for (const auto &Func : FunctionDescriptions) {
    FuncDescSize += 8 + Func.Edges.size() * sizeof(EdgeDescription) +
                    Func.ExitNodes.size() * sizeof(InstrumentedNode);
  }
  return FuncDescSize;
}

void Instrumentation::emitTablesAsELFNote(BinaryContext &BC) {
  std::string TablesStr;
  raw_string_ostream OS(TablesStr);

  // Start of the vector with descriptions (one CounterDescription for each
  // counter), vector size is Counters.size() CounterDescription-sized elmts
  const auto CDSize = CallDescriptions.size() * sizeof(CallDescription);
  OS.write(reinterpret_cast<const char *>(&CDSize), 4);
  for (const auto &Desc : CallDescriptions) {
    OS.write(reinterpret_cast<const char *>(&Desc.FromLoc.FuncString), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.FromLoc.Offset), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.ToLoc.FuncString), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.ToLoc.Offset), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.Counter), 4);
  }
  const auto FDSize = getFDSize();
  OS.write(reinterpret_cast<const char *>(&FDSize), 4);
  for (const auto &Desc : FunctionDescriptions) {
    const auto ExitsNum = Desc.ExitNodes.size();
    OS.write(reinterpret_cast<const char *>(&ExitsNum), 4);
    for (const auto &ExitNode : Desc.ExitNodes) {
      OS.write(reinterpret_cast<const char *>(&ExitNode.Node), 4);
      OS.write(reinterpret_cast<const char *>(&ExitNode.Counter), 4);
    }
    const auto EdgesNum = Desc.Edges.size();
    OS.write(reinterpret_cast<const char *>(&EdgesNum), 4);
    for (const auto &Edge : Desc.Edges) {
      OS.write(reinterpret_cast<const char *>(&Edge.FromLoc.FuncString), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.FromLoc.Offset), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.FromNode), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.ToLoc.FuncString), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.ToLoc.Offset), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.ToNode), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.Counter), 4);
    }
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
  // counters, total vector size is Counters.size() 8-byte counters
  MCSymbol *Locs = BC.Ctx->getOrCreateSymbol("__bolt_instr_locations");
  MCSymbol *NumCalls = BC.Ctx->getOrCreateSymbol("__bolt_instr_num_calls");
  MCSymbol *NumFuncs = BC.Ctx->getOrCreateSymbol("__bolt_instr_num_funcs");
  /// File name where profile is going to written to after target binary
  /// finishes a run
  MCSymbol *FilenameSym = BC.Ctx->getOrCreateSymbol("__bolt_instr_filename");

  Streamer.SwitchSection(Section);
  Streamer.EmitLabel(Locs);
  Streamer.EmitSymbolAttribute(Locs,
                               MCSymbolAttr::MCSA_Global);
  for (const auto &Label : Counters) {
    Streamer.EmitLabel(Label);
    Streamer.emitFill(8, 0);
  }
  Streamer.EmitLabel(NumCalls);
  Streamer.EmitSymbolAttribute(NumCalls,
                               MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(CallDescriptions.size(), /*Size=*/4);
  Streamer.EmitLabel(NumFuncs);
  Streamer.EmitSymbolAttribute(NumFuncs,
                               MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(FunctionDescriptions.size(), /*Size=*/4);
  Streamer.EmitLabel(FilenameSym);
  Streamer.EmitBytes(opts::InstrumentationFilename);
  Streamer.emitFill(1, 0);

  uint32_t FuncDescSize = getFDSize();
  outs() << "BOLT-INSTRUMENTER: Number of call descriptors: "
         << CallDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of function descriptors: "
         << FunctionDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of counters: " << Counters.size()
         << "\n";
  outs() << "BOLT-INSTRUMENTER: Total size of counters: "
         << (Counters.size() * 8) << " bytes (static alloc memory)\n";
  outs() << "BOLT-INSTRUMENTER: Total size of string table emitted: "
         << StringTable.size() << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Total size of descriptors: "
         << (FuncDescSize + CallDescriptions.size() * sizeof(CallDescription))
         << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Profile will be saved to file "
         << opts::InstrumentationFilename << "\n";
}

}
}
