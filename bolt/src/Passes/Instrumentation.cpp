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

cl::opt<std::string> InstrumentationFilename(
    "instrumentation-file",
    cl::desc("file name where instrumented profile will be saved"),
    cl::init("/tmp/prof.fdata"),
    cl::Optional,
    cl::cat(BoltCategory));

cl::opt<bool> InstrumentationFileAppendPID(
    "instrumentation-file-append-pid",
    cl::desc("append PID to saved profile file name (default: false)"),
    cl::init(false),
    cl::Optional,
    cl::cat(BoltCategory));

cl::opt<bool> ConservativeInstrumentation(
    "conservative-instrumentation",
    cl::desc(
        "don't trust our CFG and disable spanning trees and any counter "
        "inference, put a counter everywhere (for debugging, default: false)"),
    cl::init(false), cl::Optional, cl::cat(BoltCategory));

cl::opt<uint32_t>
    InstrumentationSleepTime("instrumentation-sleep-time",
                             cl::desc("interval between profile writes, "
                                      "default: 0 = write only at program end"),
                             cl::init(0), cl::Optional,
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
    cl::init(true),
    cl::Optional,
    cl::cat(BoltCategory));
}

namespace llvm {
namespace bolt {

uint32_t Instrumentation::getFunctionNameIndex(const BinaryFunction &Function) {
  auto Iter = FuncToStringIdx.find(&Function);
  if (Iter != FuncToStringIdx.end())
    return Iter->second;
  auto Idx = Summary->StringTable.size();
  FuncToStringIdx.emplace(std::make_pair(&Function, Idx));
  Summary->StringTable.append(Function.getOneName());
  Summary->StringTable.append(1, '\0');
  return Idx;
}

bool Instrumentation::createCallDescription(FunctionDescription &FuncDesc,
                                            const BinaryFunction &FromFunction,
                                            uint32_t From, uint32_t FromNodeID,
                                            const BinaryFunction &ToFunction,
                                            uint32_t To, bool IsInvoke) {
  CallDescription CD;
  // Ordinarily, we don't augment direct calls with an explicit counter, except
  // when forced to do so or when we know this callee could be throwing
  // exceptions, in which case there is no other way to accurately record its
  // frequency.
  bool ForceInstrumentation = opts::ConservativeInstrumentation || IsInvoke;
  CD.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  CD.FromLoc.Offset = From;
  CD.FromNode = FromNodeID;
  CD.Target = &ToFunction;
  CD.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  CD.ToLoc.Offset = To;
  CD.Counter = ForceInstrumentation ? Summary->Counters.size() : 0xffffffff;
  if (ForceInstrumentation)
    ++DirectCallCounters;
  FuncDesc.Calls.emplace_back(CD);
  return ForceInstrumentation;
}

void Instrumentation::createIndCallDescription(
    const BinaryFunction &FromFunction, uint32_t From) {
  IndCallDescription ICD;
  ICD.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  ICD.FromLoc.Offset = From;
  Summary->IndCallDescriptions.emplace_back(ICD);
}

void Instrumentation::createIndCallTargetDescription(
    const BinaryFunction &ToFunction, uint32_t To) {
  IndCallTargetDescription ICD;
  ICD.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  ICD.ToLoc.Offset = To;
  ICD.Target = &ToFunction;
  Summary->IndCallTargetDescriptions.emplace_back(ICD);
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
  ED.Counter = Instrumented ? Summary->Counters.size() : 0xffffffff;
  if (Instrumented)
    ++BranchCounters;
  FuncDesc.Edges.emplace_back(ED);
  return Instrumented;
}

void Instrumentation::createLeafNodeDescription(FunctionDescription &FuncDesc,
                                                uint32_t Node) {
  InstrumentedNode IN;
  IN.Node = Node;
  IN.Counter = Summary->Counters.size();
  ++LeafNodeCounters;
  FuncDesc.LeafNodes.emplace_back(IN);
}

std::vector<MCInst>
Instrumentation::createInstrumentationSnippet(BinaryContext &BC, bool IsLeaf) {
  auto L = BC.scopeLock();
  MCSymbol *Label;
  Label = BC.Ctx->createTempSymbol("InstrEntry", true);
  Summary->Counters.emplace_back(Label);
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

namespace {

// Helper instruction sequence insertion function
BinaryBasicBlock::iterator
insertInstructions(std::vector<MCInst>& Instrs,
                   BinaryBasicBlock &BB,
                   BinaryBasicBlock::iterator Iter) {
  for (auto &NewInst : Instrs) {
    Iter = BB.insertInstruction(Iter, NewInst);
    ++Iter;
  }
  return Iter;
}

}

void Instrumentation::instrumentLeafNode(BinaryContext &BC,
                                         BinaryBasicBlock &BB,
                                         BinaryBasicBlock::iterator Iter,
                                         bool IsLeaf,
                                         FunctionDescription &FuncDesc,
                                         uint32_t Node) {
  createLeafNodeDescription(FuncDesc, Node);
  std::vector<MCInst> CounterInstrs = createInstrumentationSnippet(BC, IsLeaf);
  insertInstructions(CounterInstrs, BB, Iter);
}

void Instrumentation::instrumentIndirectTarget(BinaryBasicBlock &BB,
                                               BinaryBasicBlock::iterator &Iter,
                                               BinaryFunction &FromFunction,
                                               uint32_t From) {
  auto L = FromFunction.getBinaryContext().scopeLock();
  const auto IndCallSiteID = Summary->IndCallDescriptions.size();
  createIndCallDescription(FromFunction, From);

  BinaryContext &BC = FromFunction.getBinaryContext();
  bool IsTailCall = BC.MIB->isTailCall(*Iter);
  std::vector<MCInst> CounterInstrs = BC.MIB->createInstrumentedIndirectCall(
      *Iter, IsTailCall,
      IsTailCall ? Summary->IndTailCallHandlerFunc
                 : Summary->IndCallHandlerFunc,
      IndCallSiteID, &*BC.Ctx);

  Iter = BB.eraseInstruction(Iter);
  Iter = insertInstructions(CounterInstrs, BB, Iter);
  --Iter;
}

bool Instrumentation::instrumentOneTarget(
    SplitWorklistTy &SplitWorklist, SplitInstrsTy &SplitInstrs,
    BinaryBasicBlock::iterator &Iter, BinaryFunction &FromFunction,
    BinaryBasicBlock &FromBB, uint32_t From, BinaryFunction &ToFunc,
    BinaryBasicBlock *TargetBB, uint32_t ToOffset, bool IsLeaf, bool IsInvoke,
    FunctionDescription *FuncDesc, uint32_t FromNodeID, uint32_t ToNodeID) {
  {
    auto L = FromFunction.getBinaryContext().scopeLock();
    bool Created{true};
    if (!TargetBB)
      Created = createCallDescription(*FuncDesc, FromFunction, From, FromNodeID,
                                      ToFunc, ToOffset, IsInvoke);
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
  if (BC.MIB->isCall(Inst)) {
    // This code handles both
    // - (regular) inter-function calls (cross-function control transfer),
    // - (rare) intra-function calls (function-local control transfer)
    Iter = insertInstructions(CounterInstrs, FromBB, Iter);
    return true;
  }

  if (!TargetBB || !FuncDesc)
    return false;

  // Indirect branch, conditional branches or fall-throughs
  // Regular cond branch, put counter at start of target block
  //
  // N.B.: (FromBB != TargetBBs) checks below handle conditional jumps where
  // we can't put the instrumentation counter in this block because not all
  // paths that reach it at this point will be taken and going to the target.
  if (TargetBB->pred_size() == 1 && &FromBB != TargetBB &&
      !TargetBB->isEntryPoint()) {
    insertInstructions(CounterInstrs, *TargetBB, TargetBB->begin());
    return true;
  }
  if (FromBB.succ_size() == 1 && &FromBB != TargetBB) {
    Iter = insertInstructions(CounterInstrs, FromBB, Iter);
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
  if (Function.hasUnknownControlFlow())
    return;

  SplitWorklistTy SplitWorklist;
  SplitInstrsTy SplitInstrs;

  FunctionDescription *FuncDesc = nullptr;
  {
    std::unique_lock<std::shared_timed_mutex> L(FDMutex);
    Summary->FunctionDescriptions.emplace_back();
    FuncDesc = &Summary->FunctionDescriptions.back();
  }

  FuncDesc->Function = &Function;
  Function.disambiguateJumpTables(AllocId);
  Function.deleteConservativeEdges();

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
    if ((*BBI)->isEntryPoint() || (*BBI)->isLandingPad()) {
      Stack.push(std::make_pair(nullptr, *BBI));
      if (opts::InstrumentCalls && (*BBI)->isEntryPoint()) {
        EntryNode E;
        E.Node = BBToID[&**BBI];
        E.Address = (*BBI)->getInputOffset();
        FuncDesc->EntryNodes.emplace_back(E);
        createIndCallTargetDescription(Function, (*BBI)->getInputOffset());
      }
    }
  }

  // Modified version of BinaryFunction::dfs() to build a spanning tree
  if (!opts::ConservativeInstrumentation) {
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

      for (auto *SuccBB : BB->successors())
        Stack.push(std::make_pair(BB, SuccBB));
    }
  }

  // Determine whether this is a leaf function, which needs special
  // instructions to protect the red zone
  bool IsLeafFunction{true};
  DenseSet<const BinaryBasicBlock *> InvokeBlocks;
  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
    for (auto I = BBI->begin(), E = BBI->end(); I != E; ++I) {
      if (BC.MIB->isCall(*I)) {
        if (BC.MIB->isInvoke(*I)) {
          InvokeBlocks.insert(&*BBI);
        }
        IsLeafFunction = false;
      }
    }
  }

  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
    auto &BB{*BBI};
    bool HasUnconditionalBranch{false};
    bool HasJumpTable{false};
    bool IsInvokeBlock = InvokeBlocks.count(&BB) > 0;

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
      if (TargetFunc && BC.MIB->isCall(Inst)) {
        if (opts::InstrumentCalls) {
          const auto *ForeignBB = TargetFunc->getBasicBlockForLabel(Target);
          if (ForeignBB)
            ToOffset = ForeignBB->getInputOffset();
          instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                              FromOffset, *TargetFunc, TargetBB, ToOffset,
                              IsLeafFunction, IsInvokeBlock, FuncDesc,
                              BBToID[&BB]);
        }
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
                            IsLeafFunction, IsInvokeBlock, FuncDesc,
                            BBToID[&BB], BBToID[TargetBB]);
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
          instrumentOneTarget(
              SplitWorklist, SplitInstrs, I, Function, BB, FromOffset, Function,
              &*Succ, Succ->getInputOffset(), IsLeafFunction, IsInvokeBlock,
              FuncDesc, BBToID[&BB], BBToID[&*Succ]);
        }
        continue;
      }

      // Handle indirect calls -- could be direct calls with unknown targets
      // or secondary entry points of known functions, so check it is indirect
      // to be sure.
      if (opts::InstrumentCalls && BC.MIB->isIndirectCall(*I))
        instrumentIndirectTarget(BB, I, Function, FromOffset);

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
                          IsLeafFunction, IsInvokeBlock, FuncDesc, BBToID[&BB],
                          BBToID[FTBB]);
    }
  } // End of BBs loop

  // Instrument spanning tree leaves
  if (!opts::ConservativeInstrumentation) {
    for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
      auto &BB{*BBI};
      if (STOutSet[&BB].size() == 0)
        instrumentLeafNode(BC, BB, BB.begin(), IsLeafFunction, *FuncDesc,
                           BBToID[&BB]);
    }
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
                             nullptr, 0, 1);

  BC.registerOrUpdateNoteSection(".bolt.instr.tables", nullptr,
                                  0,
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);

  Summary->IndCallHandlerFunc =
      BC.Ctx->getOrCreateSymbol("__bolt_trampoline_ind_call");
  Summary->IndTailCallHandlerFunc =
      BC.Ctx->getOrCreateSymbol("__bolt_trampoline_ind_tailcall");

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return (!BF.isSimple() || BF.isIgnored() ||
            (opts::InstrumentHotOnly && !BF.getKnownExecutionCount()));
  };

  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocatorId) {
        instrumentFunction(BC, BF, AllocatorId);
      };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_QUADRATIC, WorkFun,
      SkipPredicate, "instrumentation", /* ForceSequential=*/true);

  createAuxiliaryFunctions(BC);

  if (BC.isMachO() && BC.StartFunctionAddress) {
    BinaryFunction *Main =
        BC.getBinaryFunctionAtAddress(*BC.StartFunctionAddress);
    assert(Main && "Entry point function not found");
    BinaryBasicBlock &BB = Main->front();

    ErrorOr<BinarySection &> SetupSection =
        BC.getUniqueSectionByName("I__setup");
    if (!SetupSection) {
      llvm::errs() << "Cannot find I__setup section\n";
      exit(1);
    }
    MCSymbol *Target = BC.registerNameAtAddress(
        "__bolt_instr_setup", SetupSection->getAddress(), 0, 0);
    MCInst NewInst;
    BC.MIB->createCall(NewInst, Target, BC.Ctx.get());
    BB.insertInstruction(BB.begin(), std::move(NewInst));
  }

  setupRuntimeLibrary(BC);
}

void Instrumentation::createAuxiliaryFunctions(BinaryContext &BC) {
  auto createSimpleFunction =
      [&](StringRef Title, std::vector<MCInst> Instrs) -> BinaryFunction * {
    BinaryFunction *Func = BC.createInjectedBinaryFunction(Title);

    std::vector<std::unique_ptr<BinaryBasicBlock>> BBs;
    BBs.emplace_back(
        Func->createBasicBlock(BinaryBasicBlock::INVALID_OFFSET, nullptr));
    BBs.back()->addInstructions(Instrs.begin(), Instrs.end());
    BBs.back()->setCFIState(0);
    Func->insertBasicBlocks(nullptr, std::move(BBs),
                            /*UpdateLayout=*/true,
                            /*UpdateCFIState=*/false);
    Func->updateState(BinaryFunction::State::CFG_Finalized);
    return Func;
  };

  Summary->InitialIndCallHandlerFunction =
      createSimpleFunction("__bolt_instr_default_ind_call_handler",
                           BC.MIB->createInstrumentedNoopIndCallHandler());

  Summary->InitialIndTailCallHandlerFunction =
      createSimpleFunction("__bolt_instr_default_ind_tailcall_handler",
                           BC.MIB->createInstrumentedNoopIndTailCallHandler());

  // TODO: Remove this code once we start loading the runtime library for OSX.
  if (BC.isMachO()) {
    std::vector<MCInst> Instrs(8);
    for (MCInst &Instruction : Instrs)
      BC.MIB->createNoop(Instruction);
    BC.MIB->createReturn(Instrs.back());
    BinaryFunction *Placeholder = createSimpleFunction(
        "__bolt_instr_setup_placeholder", std::move(Instrs));
    ErrorOr<BinarySection &> SetupSection =
        BC.getUniqueSectionByName("I__setup");
    if (!SetupSection) {
      llvm::errs() << "Cannot find I__setup section\n";
      exit(1);
    }
    Placeholder->setOutputAddress(SetupSection->getAddress());
    Placeholder->setFileOffset(SetupSection->getInputFileOffset());
    Placeholder->setOriginSection(&*SetupSection);
  }
}

void Instrumentation::setupRuntimeLibrary(BinaryContext &BC) {
  auto FuncDescSize = Summary->getFDSize();

  outs() << "BOLT-INSTRUMENTER: Number of indirect call site descriptors: "
         << Summary->IndCallDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of indirect call target descriptors: "
         << Summary->IndCallTargetDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of function descriptors: "
         << Summary->FunctionDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of branch counters: " << BranchCounters
         << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of ST leaf node counters: "
         << LeafNodeCounters << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of direct call counters: "
         << DirectCallCounters << "\n";
  outs() << "BOLT-INSTRUMENTER: Total number of counters: "
         << Summary->Counters.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Total size of counters: "
         << (Summary->Counters.size() * 8) << " bytes (static alloc memory)\n";
  outs() << "BOLT-INSTRUMENTER: Total size of string table emitted: "
         << Summary->StringTable.size() << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Total size of descriptors: "
         << (FuncDescSize +
             Summary->IndCallDescriptions.size() * sizeof(IndCallDescription) +
             Summary->IndCallTargetDescriptions.size() *
                 sizeof(IndCallTargetDescription))
         << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Profile will be saved to file "
         << opts::InstrumentationFilename << "\n";

  BC.setRuntimeLibrary(
      llvm::make_unique<InstrumentationRuntimeLibrary>(std::move(Summary)));
}
}
}
