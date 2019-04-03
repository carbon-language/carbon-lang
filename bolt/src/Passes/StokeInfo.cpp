#include "StokeInfo.h"
#include "llvm/Support/Options.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "stoke"

using namespace llvm;
using namespace bolt;

namespace opts {
cl::OptionCategory StokeOptCategory("STOKE pass options");

static cl::opt<std::string>
StokeOutputDataFilename("stoke-out",
  cl::desc("output data (.csv) for Stoke's use"),
  cl::Optional,
  cl::cat(StokeOptCategory));
}

namespace llvm {
namespace bolt {


void getRegNameFromBitVec(const BinaryContext &BC, const BitVector &RegV,
    std::set<std::string> *NameVec = nullptr) {
  int RegIdx = RegV.find_first();
  while (RegIdx != -1) {
    DEBUG(dbgs() << BC.MRI->getName(RegIdx) << " ");
    if (NameVec) {
      NameVec->insert(std::string(BC.MRI->getName(RegIdx)));
    }
    RegIdx = RegV.find_next(RegIdx);
  }
  DEBUG(dbgs() << "\n");
}

void StokeInfo::checkInstr(const BinaryContext &BC, const BinaryFunction &BF,
    StokeFuncInfo &FuncInfo) {

  BitVector RegV(NumRegs, false);
  for (auto *BB : BF.layout()) {
    if (BB->empty()) {
      continue;
    }
    for (auto &It : *BB) {
      auto &InstDesc = BC.MII->get(It.getOpcode());
      if (InstDesc.isPseudo()) {
        continue;
      }
      // skip function with exception handling yet
      if (BC.MIB->isEHLabel(It) || BC.MIB->isInvoke(It)) {
        FuncInfo.Omitted = true;
        return;
      }
      // check if this function contains call instruction
      if (BC.MIB->isCall(It)) {
        FuncInfo.HasCall = true;
        const auto *TargetSymbol = BC.MIB->getTargetSymbol(It);
        // if it is an indirect call, skip
        if (TargetSymbol == nullptr) {
          FuncInfo.Omitted = true;
          return;
        }
      }
      // check if this function modify stack or heap
      // TODO: more accurate analysis
      auto IsPush = BC.MIB->isPush(It);
      auto IsRipAddr = BC.MIB->hasPCRelOperand(It);
      if (IsPush) {
        FuncInfo.StackOut = true;
      }
      if (BC.MIB->isStore(It) && !IsPush && !IsRipAddr) {
        FuncInfo.HeapOut = true;
      }
      if (IsRipAddr) {
        FuncInfo.HasRipAddr = true;
      }

    } // end of for (auto &It : ...)
  } // end of for (auto *BB : ...)
}

bool StokeInfo::checkFunction(const BinaryContext &BC, BinaryFunction &BF,
    DataflowInfoManager &DInfo, RegAnalysis &RA,
    StokeFuncInfo &FuncInfo) {

  std::string Name = BF.getSymbol()->getName().str();

  if (!BF.isSimple() || BF.isMultiEntry() || BF.empty()) {
    return false;
  }
  outs() << " STOKE-INFO: analyzing function " << Name << "\n";

  FuncInfo.FuncName = Name;
  FuncInfo.Offset = BF.getFileOffset();
  FuncInfo.Size = BF.getMaxSize();
  FuncInfo.NumInstrs = BF.getNumNonPseudos();
  FuncInfo.NumBlocks = BF.size();
  // early stop for large functions
  if (FuncInfo.NumInstrs > 500) {
    return false;
  }

  FuncInfo.IsLoopFree = BF.isLoopFree();
  if (!FuncInfo.IsLoopFree) {
    auto &BLI = BF.getLoopInfo();
    FuncInfo.NumLoops = BLI.OuterLoops;
    FuncInfo.MaxLoopDepth = BLI.MaximumDepth;
  }

  FuncInfo.HotSize = BF.estimateHotSize();
  FuncInfo.TotalSize = BF.estimateSize();
  FuncInfo.Score = BF.getFunctionScore();

  checkInstr(BC, BF, FuncInfo);

  // register analysis
  BinaryBasicBlock &EntryBB = BF.front();
  assert(EntryBB.isEntryPoint() && "Weird, this should be the entry block!");

  auto *FirstNonPseudo = EntryBB.getFirstNonPseudoInstr();
  if (!FirstNonPseudo) {
    return false;
  }

  DEBUG(dbgs() << "\t [DefIn]\n\t ");
  auto LiveInBV = *(DInfo.getLivenessAnalysis().getStateAt(FirstNonPseudo));
  LiveInBV &= DefaultDefInMask;
  getRegNameFromBitVec(BC, LiveInBV, &FuncInfo.DefIn);

  DEBUG(dbgs() << "\t [LiveOut]\n\t ");
  auto LiveOutBV = RA.getFunctionClobberList(&BF);
  LiveOutBV &= DefaultLiveOutMask;
  getRegNameFromBitVec(BC, LiveOutBV, &FuncInfo.LiveOut);

  outs() << " STOKE-INFO: end function \n";
  return true;
}

void StokeInfo::runOnFunctions(BinaryContext &BC) {
  outs() << "STOKE-INFO: begin of stoke pass\n";

  std::ofstream Outfile;
  if (!opts::StokeOutputDataFilename.empty()) {
    Outfile.open(opts::StokeOutputDataFilename);
  } else {
    errs() << "STOKE-INFO: output file is required\n";
    return;
  }

  // check some context meta data
  DEBUG(dbgs() << "\tTarget: " << BC.TheTarget->getName() << "\n");
  DEBUG(dbgs() << "\tTripleName " << BC.TripleName << "\n");
  DEBUG(dbgs() << "\tgetNumRegs " << BC.MRI->getNumRegs() << "\n");

  auto CG = buildCallGraph(BC);
  RegAnalysis RA(BC, &BC.getBinaryFunctions(), &CG);

  NumRegs = BC.MRI->getNumRegs();
  assert(NumRegs > 0 && "STOKE-INFO: the target register number is incorrect!");

  DefaultDefInMask.resize(NumRegs, false);
  DefaultLiveOutMask.resize(NumRegs, false);

  BC.MIB->getDefaultDefIn(DefaultDefInMask);
  BC.MIB->getDefaultLiveOut(DefaultLiveOutMask);

  getRegNameFromBitVec(BC, DefaultDefInMask);
  getRegNameFromBitVec(BC, DefaultLiveOutMask);

  StokeFuncInfo FuncInfo;
  // analyze all functions
  FuncInfo.printCsvHeader(Outfile);
  for (auto &BF : BC.getBinaryFunctions()) {
    DataflowInfoManager DInfo(BC, BF.second, &RA/*RA.get()*/, nullptr);
    FuncInfo.reset();
    if (checkFunction(BC, BF.second, DInfo, RA, FuncInfo)) {
      FuncInfo.printData(Outfile);
    }
  }

  outs() << "STOKE-INFO: end of stoke pass\n";
}

} // namespace bolt
} // namespace llvm
