//===--- ReorderFunctions.cpp - Function reordering pass ------------ -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "ReorderFunctions.h"
#include "HFSort.h"
#include "llvm/Support/Options.h"
#include <fstream>

#define DEBUG_TYPE "hfsort"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;
extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> Relocs;
extern cl::opt<uint32_t> RandomSeed;

extern bool shouldProcess(const bolt::BinaryFunction &Function);
extern size_t padFunction(const bolt::BinaryFunction &Function);

cl::opt<bolt::BinaryFunction::ReorderType>
ReorderFunctions("reorder-functions",
  cl::desc("reorder and cluster functions (works only with relocations)"),
  cl::init(bolt::BinaryFunction::RT_NONE),
  cl::values(clEnumValN(bolt::BinaryFunction::RT_NONE,
      "none",
      "do not reorder functions"),
    clEnumValN(bolt::BinaryFunction::RT_EXEC_COUNT,
      "exec-count",
      "order by execution count"),
    clEnumValN(bolt::BinaryFunction::RT_HFSORT,
      "hfsort",
      "use hfsort algorithm"),
    clEnumValN(bolt::BinaryFunction::RT_HFSORT_PLUS,
      "hfsort+",
      "use hfsort+ algorithm"),
    clEnumValN(bolt::BinaryFunction::RT_PETTIS_HANSEN,
      "pettis-hansen",
      "use Pettis-Hansen algorithm"),
    clEnumValN(bolt::BinaryFunction::RT_RANDOM,
      "random",
      "reorder functions randomly"),
    clEnumValN(bolt::BinaryFunction::RT_USER,
      "user",
      "use function order specified by -function-order"),
    clEnumValEnd),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ReorderFunctionsUseHotSize("reorder-functions-use-hot-size",
  cl::desc("use a function's hot size when doing clustering"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<std::string>
FunctionOrderFile("function-order",
  cl::desc("file containing an ordered list of functions to use for function "
           "reordering"),
  cl::cat(BoltOptCategory));

static cl::opt<std::string>
GenerateFunctionOrderFile("generate-function-order",
  cl::desc("file to dump the ordered list of functions to use for function "
           "reordering"),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
UseEdgeCounts("use-edge-counts",
  cl::desc("use edge count data when doing clustering"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
CgFromPerfData("cg-from-perf-data",
  cl::desc("use perf data directly when constructing the call graph"
           " for stale functions"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
CgIgnoreRecursiveCalls("cg-ignore-recursive-calls",
  cl::desc("ignore recursive calls when constructing the call graph"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
CgUseSplitHotSize("cg-use-split-hot-size",
  cl::desc("use hot/cold data on basic blocks to determine hot sizes for "
           "call graph functions"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
UseGainCache("hfsort+-use-cache",
  llvm::cl::desc("Use a cache for mergeGain results when computing hfsort+."),
  llvm::cl::ZeroOrMore,
  llvm::cl::init(true),
  llvm::cl::Hidden,
  llvm::cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
UseShortCallCache("hfsort+-use-short-call-cache",
  llvm::cl::desc("Use a cache for shortCall results when computing hfsort+."),
  llvm::cl::ZeroOrMore,
  llvm::cl::init(true),
  llvm::cl::Hidden,
  llvm::cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;  

void ReorderFunctions::reorder(std::vector<Cluster> &&Clusters,
                               std::map<uint64_t, BinaryFunction> &BFs) {
  std::vector<uint64_t> FuncAddr(Cg.numNodes());  // Just for computing stats
  uint64_t TotalSize = 0;
  uint32_t Index = 0;

  // Set order of hot functions based on clusters.
  for (const auto& Cluster : Clusters) {
    for (const auto FuncId : Cluster.targets()) {
      assert(Cg.samples(FuncId) > 0);
      Cg.nodeIdToFunc(FuncId)->setIndex(Index++);
      FuncAddr[FuncId] = TotalSize;
      TotalSize += Cg.size(FuncId);
    }
  }

  if (opts::ReorderFunctions == BinaryFunction::RT_NONE)
    return;

  if (opts::Verbosity == 0) {
#ifndef NDEBUG
    if (!DebugFlag || !isCurrentDebugType("hfsort"))
      return;
#else
    return;
#endif
  }

  bool PrintDetailed = opts::Verbosity > 1;
#ifndef NDEBUG
  PrintDetailed |=
    (DebugFlag && isCurrentDebugType("hfsort") && opts::Verbosity > 0);
#endif
  TotalSize   = 0;
  uint64_t CurPage     = 0;
  uint64_t Hotfuncs    = 0;
  double TotalDistance = 0;
  double TotalCalls    = 0;
  double TotalCalls64B = 0;
  double TotalCalls4KB = 0;
  double TotalCalls2MB = 0;
  if (PrintDetailed) {
    outs() << "BOLT-INFO: Function reordering page layout\n"
           << "BOLT-INFO: ============== page 0 ==============\n";
  }
  for (auto& Cluster : Clusters) {
    if (PrintDetailed) {
      outs() <<
        format("BOLT-INFO: -------- density = %.3lf (%u / %u) --------\n",
               Cluster.density(), Cluster.samples(), Cluster.size());
    }

    for (auto FuncId : Cluster.targets()) {
      if (Cg.samples(FuncId) > 0) {
        Hotfuncs++;

        if (PrintDetailed) {
          outs() << "BOLT-INFO: hot func " << *Cg.nodeIdToFunc(FuncId)
                 << " (" << Cg.size(FuncId) << ")\n";
        }

        uint64_t Dist = 0;
        uint64_t Calls = 0;
        for (auto Dst : Cg.successors(FuncId)) {
          if (FuncId == Dst) // ignore recursive calls in stats
            continue;
          const auto& Arc = *Cg.findArc(FuncId, Dst);
          const auto D = std::abs(FuncAddr[Arc.dst()] -
                                  (FuncAddr[FuncId] + Arc.avgCallOffset()));
          const auto W = Arc.weight();
          if (D < 64 && PrintDetailed && opts::Verbosity > 2) {
            outs() << "BOLT-INFO: short (" << D << "B) call:\n"
                   << "BOLT-INFO:   Src: " << *Cg.nodeIdToFunc(FuncId) << "\n"
                   << "BOLT-INFO:   Dst: " << *Cg.nodeIdToFunc(Dst) << "\n"
                   << "BOLT-INFO:   Weight = " << W << "\n"
                   << "BOLT-INFO:   AvgOffset = " << Arc.avgCallOffset() << "\n";
          }
          Calls += W;
          if (D < 64)        TotalCalls64B += W;
          if (D < 4096)      TotalCalls4KB += W;
          if (D < (2 << 20)) TotalCalls2MB += W;
          Dist += Arc.weight() * D;
          if (PrintDetailed) {
            outs() << format("BOLT-INFO: arc: %u [@%lu+%.1lf] -> %u [@%lu]: "
                             "weight = %.0lf, callDist = %f\n",
                             Arc.src(),
                             FuncAddr[Arc.src()],
                             Arc.avgCallOffset(),
                             Arc.dst(),
                             FuncAddr[Arc.dst()],
                             Arc.weight(), D);
          }
        }
        TotalCalls += Calls;
        TotalDistance += Dist;
        TotalSize += Cg.size(FuncId);

        if (PrintDetailed) {
          outs() << format("BOLT-INFO: start = %6u : avgCallDist = %lu : %s\n",
                           TotalSize,
                           Calls ? Dist / Calls : 0,
                           Cg.nodeIdToFunc(FuncId)->getPrintName().c_str());
          const auto NewPage = TotalSize / HugePageSize;
          if (NewPage != CurPage) {
            CurPage = NewPage;
            outs() <<
              format("BOLT-INFO: ============== page %u ==============\n",
                     CurPage);
          }
        }
      }
    }
  }
  outs() << "BOLT-INFO: Function reordering stats\n"
         << format("BOLT-INFO:  Number of hot functions: %u\n"
                   "BOLT-INFO:  Number of clusters: %lu\n",
                   Hotfuncs, Clusters.size())
         << format("BOLT-INFO:  Final average call distance = %.1lf "
                   "(%.0lf / %.0lf)\n",
                   TotalCalls ? TotalDistance / TotalCalls : 0,
                   TotalDistance, TotalCalls)
         << format("BOLT-INFO:  Total Calls = %.0lf\n", TotalCalls);
  if (TotalCalls) {
    outs() << format("BOLT-INFO:  Total Calls within 64B = %.0lf (%.2lf%%)\n",
                     TotalCalls64B, 100 * TotalCalls64B / TotalCalls)
           << format("BOLT-INFO:  Total Calls within 4KB = %.0lf (%.2lf%%)\n",
                     TotalCalls4KB, 100 * TotalCalls4KB / TotalCalls)
           << format("BOLT-INFO:  Total Calls within 2MB = %.0lf (%.2lf%%)\n",
                     TotalCalls2MB, 100 * TotalCalls2MB / TotalCalls);
  }
}

namespace {

std::vector<std::string> readFunctionOrderFile() {
  std::vector<std::string> FunctionNames;
  std::ifstream FuncsFile(opts::FunctionOrderFile, std::ios::in);
  if (!FuncsFile) {
    errs() << "Ordered functions file \"" << opts::FunctionOrderFile
           << "\" can't be opened.\n";
    exit(1);
  }
  std::string FuncName;
  while (std::getline(FuncsFile, FuncName)) {
    FunctionNames.push_back(FuncName);
  }
  return FunctionNames;
}

}

void ReorderFunctions::runOnFunctions(BinaryContext &BC,
                                      std::map<uint64_t, BinaryFunction> &BFs,
                                      std::set<uint64_t> &LargeFunctions) {
  if (!opts::Relocs && opts::ReorderFunctions != BinaryFunction::RT_NONE) {
    errs() << "BOLT-ERROR: Function reordering only works when "
           << "relocs are enabled.\n";
    exit(1);
  }

  if (opts::ReorderFunctions != BinaryFunction::RT_NONE &&
      opts::ReorderFunctions != BinaryFunction::RT_EXEC_COUNT &&
      opts::ReorderFunctions != BinaryFunction::RT_USER) {
    Cg = buildCallGraph(BC,
                        BFs,
                        [this](const BinaryFunction &BF) {
                          return !shouldOptimize(BF) || !BF.hasProfile();
                        },
                        opts::CgFromPerfData,
                        false, // IncludeColdCalls
                        opts::ReorderFunctionsUseHotSize,
                        opts::CgUseSplitHotSize,
                        opts::UseEdgeCounts,
                        opts::CgIgnoreRecursiveCalls);
    Cg.normalizeArcWeights(opts::UseEdgeCounts);
  }

  std::vector<Cluster> Clusters;

  switch(opts::ReorderFunctions) {
  case BinaryFunction::RT_NONE:
    break;
  case BinaryFunction::RT_EXEC_COUNT:
    {
      std::vector<BinaryFunction *> SortedFunctions(BFs.size());
      uint32_t Index = 0;
      std::transform(BFs.begin(),
                     BFs.end(),
                     SortedFunctions.begin(),
                     [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                       return &BFI.second;
                     });
      std::stable_sort(SortedFunctions.begin(), SortedFunctions.end(),
                       [&](const BinaryFunction *A, const BinaryFunction *B) {
                         if (!opts::shouldProcess(*A))
                           return false;
                         const auto PadA = opts::padFunction(*A);
                         const auto PadB = opts::padFunction(*B);
                         if (!PadA || !PadB) {
                           if (PadA)
                             return true;
                           if (PadB)
                             return false;
                         }
                         return !A->hasProfile() &&
                           (B->hasProfile() ||
                            (A->getExecutionCount() > B->getExecutionCount()));
                       });
      for (auto *BF : SortedFunctions) {
        if (BF->hasProfile())
          BF->setIndex(Index++);
      }
    }
    break;
  case BinaryFunction::RT_HFSORT:
    Clusters = clusterize(Cg);
    break;
  case BinaryFunction::RT_HFSORT_PLUS:
    Clusters = hfsortPlus(Cg, opts::UseGainCache, opts::UseShortCallCache);
    break;
  case BinaryFunction::RT_PETTIS_HANSEN:
    Clusters = pettisAndHansen(Cg);
    break;
  case BinaryFunction::RT_RANDOM:
    std::srand(opts::RandomSeed);
    Clusters = randomClusters(Cg);
    break;
  case BinaryFunction::RT_USER:
    {
      uint32_t Index = 0;
      for (const auto &Function : readFunctionOrderFile()) {
        std::vector<uint64_t> FuncAddrs;

        auto Itr = BC.GlobalSymbols.find(Function);
        if (Itr == BC.GlobalSymbols.end()) {
          uint32_t LocalID = 1;
          while(1) {
            // If we can't find the main symbol name, look for alternates.
            Itr = BC.GlobalSymbols.find(Function + "/" + std::to_string(LocalID));
            if (Itr != BC.GlobalSymbols.end())
              FuncAddrs.push_back(Itr->second);
            else
              break;
            LocalID++;
          }
        } else {
          FuncAddrs.push_back(Itr->second);
        }

        if (FuncAddrs.empty()) {
          errs() << "BOLT-WARNING: Reorder functions: can't find function for "
                 << Function << ".\n";
          continue;
        }

        for (const auto FuncAddr : FuncAddrs) {
          const auto *FuncSym = BC.getOrCreateGlobalSymbol(FuncAddr, "FUNCat");
          assert(FuncSym);

          auto *BF = BC.getFunctionForSymbol(FuncSym);
          if (!BF) {
            errs() << "BOLT-WARNING: Reorder functions: can't find function for "
                   << Function << ".\n";
            break;
          }
          if (!BF->hasValidIndex()) {
            BF->setIndex(Index++);
          } else if (opts::Verbosity > 0) {
            errs() << "BOLT-WARNING: Duplicate reorder entry for " << Function << ".\n";
          }
        }
      }
    }
    break;
  }

  reorder(std::move(Clusters), BFs);

  if (!opts::GenerateFunctionOrderFile.empty()) {
    std::ofstream FuncsFile(opts::GenerateFunctionOrderFile, std::ios::out);
    if (!FuncsFile) {
      errs() << "Ordered functions file \"" << opts::GenerateFunctionOrderFile
             << "\" can't be opened.\n";
      exit(1);
    }

    std::vector<BinaryFunction *> SortedFunctions(BFs.size());

    std::transform(BFs.begin(),
                   BFs.end(),
                   SortedFunctions.begin(),
                   [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                     return &BFI.second;
                   });

    // Sort functions by index.
    std::stable_sort(
      SortedFunctions.begin(),
      SortedFunctions.end(),
      [](const BinaryFunction *A, const BinaryFunction *B) {
        if (A->hasValidIndex() && B->hasValidIndex()) {
          return A->getIndex() < B->getIndex();
        } else if (A->hasValidIndex() && !B->hasValidIndex()) {
          return true;
        } else if (!A->hasValidIndex() && B->hasValidIndex()) {
          return false;
        } else {
          return A->getAddress() < B->getAddress();
        }
      });

    for (const auto *Func : SortedFunctions) {
      if (!Func->hasValidIndex())
        break;
      if (Func->isPLTFunction())
        continue;
      const char *Indent = "";
      for (auto Name : Func->getNames()) {
        const auto SlashPos = Name.find('/');
        if (SlashPos != std::string::npos) {
          // Avoid duplicates for local functions.
          if (Name.find('/', SlashPos + 1) != std::string::npos)
            continue;
          Name = Name.substr(0, SlashPos);
        }
        FuncsFile << Indent << ".text." << Name << "\n";
        Indent = " ";
      }
    }
    FuncsFile.close();

    outs() << "BOLT-INFO: dumped function order to \""
           << opts::GenerateFunctionOrderFile << "\"\n";

    exit(0);
  }
}

} // namespace bolt
} // namespace llvm
