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

} // namespace opts

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;  

void ReorderFunctions::normalizeArcWeights() {
  // Normalize arc weights.
  if (!opts::UseEdgeCounts) {
    for (NodeId FuncId = 0; FuncId < Cg.Nodes.size(); ++FuncId) {
      auto& Func = Cg.Nodes[FuncId];
      for (auto Caller : Func.Preds) {
        auto& A = *Cg.Arcs.find(Arc(Caller, FuncId));
        A.NormalizedWeight = A.Weight / Func.Samples;
        A.AvgCallOffset /= A.Weight;
        assert(A.AvgCallOffset < Cg.Nodes[Caller].Size);
      }
    }
  } else {
    for (NodeId FuncId = 0; FuncId < Cg.Nodes.size(); ++FuncId) {
      auto &Func = Cg.Nodes[FuncId];
      for (auto Caller : Func.Preds) {
        auto& A = *Cg.Arcs.find(Arc(Caller, FuncId));
        A.NormalizedWeight = A.Weight / Func.Samples;
      }
    }
  }
}

void ReorderFunctions::reorder(std::vector<Cluster> &&Clusters,
                               std::map<uint64_t, BinaryFunction> &BFs) {
  std::vector<uint64_t> FuncAddr(Cg.Nodes.size());  // Just for computing stats
  uint64_t TotalSize = 0;
  uint32_t Index = 0;

  // Set order of hot functions based on clusters.
  for (const auto& Cluster : Clusters) {
    for (const auto FuncId : Cluster.Targets) {
      assert(Cg.Nodes[FuncId].Samples > 0);
      Cg.Funcs[FuncId]->setIndex(Index++);
      FuncAddr[FuncId] = TotalSize;
      TotalSize += Cg.Nodes[FuncId].Size;
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

  TotalSize   = 0;
  uint64_t CurPage     = 0;
  uint64_t Hotfuncs    = 0;
  double TotalDistance = 0;
  double TotalCalls    = 0;
  double TotalCalls64B = 0;
  double TotalCalls4KB = 0;
  double TotalCalls2MB = 0;
  dbgs() << "============== page 0 ==============\n";
  for (auto& Cluster : Clusters) {
    dbgs() <<
      format("-------- density = %.3lf (%u / %u) --------\n",
             (double) Cluster.Samples / Cluster.Size,
             Cluster.Samples, Cluster.Size);

    for (auto FuncId : Cluster.Targets) {
      if (Cg.Nodes[FuncId].Samples > 0) {
        Hotfuncs++;

        dbgs() << "BOLT-INFO: hot func " << *Cg.Funcs[FuncId]
               << " (" << Cg.Nodes[FuncId].Size << ")\n";

        uint64_t Dist = 0;
        uint64_t Calls = 0;
        for (auto Dst : Cg.Nodes[FuncId].Succs) {
          auto& A = *Cg.Arcs.find(Arc(FuncId, Dst));
          auto D =
            std::abs(FuncAddr[A.Dst] - (FuncAddr[FuncId] + A.AvgCallOffset));
          auto W = A.Weight;
          Calls += W;
          if (D < 64)        TotalCalls64B += W;
          if (D < 4096)      TotalCalls4KB += W;
          if (D < (2 << 20)) TotalCalls2MB += W;
          Dist += A.Weight * D;
          dbgs() << format("arc: %u [@%lu+%.1lf] -> %u [@%lu]: "
                           "weight = %.0lf, callDist = %f\n",
                           A.Src, FuncAddr[A.Src], A.AvgCallOffset,
                           A.Dst, FuncAddr[A.Dst], A.Weight, D);
        }
        TotalCalls += Calls;
        TotalDistance += Dist;
        dbgs() << format("start = %6u : avgCallDist = %lu : %s\n",
                         TotalSize,
                         Calls ? Dist / Calls : 0,
                         Cg.Funcs[FuncId]->getPrintName().c_str());
        TotalSize += Cg.Nodes[FuncId].Size;
        auto NewPage = TotalSize / HugePageSize;
        if (NewPage != CurPage) {
          CurPage = NewPage;
          dbgs() << format("============== page %u ==============\n", CurPage);
        }
      }
    }
  }
  dbgs() << format("  Number of hot functions: %u\n"
                   "  Number of clusters: %lu\n",
                   Hotfuncs, Clusters.size())
         << format("  Final average call distance = %.1lf (%.0lf / %.0lf)\n",
                   TotalCalls ? TotalDistance / TotalCalls : 0,
                   TotalDistance, TotalCalls)
         << format("  Total Calls = %.0lf\n", TotalCalls);
  if (TotalCalls) {
    dbgs() << format("  Total Calls within 64B = %.0lf (%.2lf%%)\n",
                     TotalCalls64B, 100 * TotalCalls64B / TotalCalls)
           << format("  Total Calls within 4KB = %.0lf (%.2lf%%)\n",
                     TotalCalls4KB, 100 * TotalCalls4KB / TotalCalls)
           << format("  Total Calls within 2MB = %.0lf (%.2lf%%)\n",
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
                        false, // IncludeColdCalls
                        opts::ReorderFunctionsUseHotSize,
                        opts::UseEdgeCounts);
    normalizeArcWeights();
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
    Clusters = hfsortPlus(Cg);
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
      FuncsFile << Func->getSymbol()->getName().data() << "\n";
    }
    FuncsFile.close();

    outs() << "BOLT-INFO: dumped function order to \""
           << opts::GenerateFunctionOrderFile << "\"\n";

    exit(0);
  }
}

} // namespace bolt
} // namespace llvm
