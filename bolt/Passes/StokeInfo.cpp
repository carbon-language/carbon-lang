#include "StokeInfo.h"
#include "llvm/Support/Options.h"

using namespace llvm;
using namespace bolt;

namespace opts {

cl::OptionCategory StokeOptCategory("STOKE generic options");

static cl::opt<std::string>
StokeOutputDataFilename("stoke-data",
  cl::desc("<info data for stoke>"),
  cl::Optional,
  cl::cat(StokeOptCategory));
}

namespace llvm {
namespace bolt {

void dumpRegNameFromBitVec(const BitVector &RegV, const BinaryContext &BC) {
  dbgs() << "\t ";
  int RegIdx = RegV.find_first();
  while (RegIdx != -1) {
    dbgs() << RegIdx << ":" << BC.MRI->getName(RegIdx) << " ";
    RegIdx = RegV.find_next(RegIdx);
  }
  dbgs() << "\n";
}

void getRegNameFromBitVec(const BitVector &RegV, std::set<std::string> &NameVec,
    const BinaryContext &BC) {
  int RegIdx = RegV.find_first();
  while (RegIdx != -1) {
    dbgs() << RegIdx << BC.MRI->getName(RegIdx) << "<>";
    NameVec.insert(std::string(BC.MRI->getName(RegIdx)));
    RegIdx = RegV.find_next(RegIdx);
  }
  dbgs() << "\n";
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
      if (BC.MIA->isEHLabel(It) || BC.MIA->isInvoke(It) || BC.MIA->hasEHInfo(It)) {
        outs() << "\t exception\n";
        FuncInfo.Omitted = true;
        return;
      }
      if (BC.MIA->hasRIPOperand(It)) {
        outs() << "\t rip operand\n";
      }
      // check if this function contains call instruction
      if (BC.MIA->isCall(It)) {
        FuncInfo.HasCall = true;
        const auto *TargetSymbol = BC.MIA->getTargetSymbol(It);
        // if it is an indirect call, skip
        if (TargetSymbol == nullptr) {
          FuncInfo.Omitted = true;
          return;
        } else {
          outs() << "\t calling " << TargetSymbol->getName() << "\n";
        }
      }
      // check if this function modify stack or heap
      // TODO: more accurate analysis
      auto IsPush = BC.MIA->isPush(It);
      if (IsPush) {
        FuncInfo.StackOut = true;
      }
      if (BC.MIA->isStore(It) && !IsPush && !BC.MIA->hasRIPOperand(It)) {
        FuncInfo.HeapOut = true;
      }

    } // end of for (auto &It : ...)
  } // end of for (auto *BB : ...)
}

bool StokeInfo::analyze(const BinaryContext &BC, BinaryFunction &BF,
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
  FuncInfo.IsLoopFree = BF.isLoopFree();
  FuncInfo.HotSize = BF.estimateHotSize();
  FuncInfo.TotalSize = BF.estimateSize();

  if (!FuncInfo.IsLoopFree) {
    auto &BLI = BF.getLoopInfo();
    FuncInfo.NumLoops = BLI.OuterLoops;
    FuncInfo.MaxLoopDepth = BLI.MaximumDepth;
  }
  // early stop for large functions
  if (FuncInfo.NumInstrs > 500) {
    return false;
  }

  BinaryBasicBlock &EntryBB = BF.front();
  assert(EntryBB.isEntryPoint() && "Weird, this block should be the entry block!");

  dbgs() << "\t EntryBB offset: " << EntryBB.getInputOffset() << "\n";
  auto *FirstNonPseudo = EntryBB.getFirstNonPseudoInstr();
  if (!FirstNonPseudo) {
    return false;
  }
  dbgs() << "\t " << BC.InstPrinter->getOpcodeName(FirstNonPseudo->getOpcode()) << "\n";

  dbgs() << "\t [DefIn at entry point]\n\t ";
  auto LiveInBV = *(DInfo.getLivenessAnalysis().getStateAt(FirstNonPseudo));
  LiveInBV &= DefaultDefInMask;
  getRegNameFromBitVec(LiveInBV, FuncInfo.DefIn, BC);

  outs() << "\t [LiveOut at return point]\n\t ";
  auto LiveOutBV = RA.getFunctionClobberList(&BF);
  LiveOutBV &= DefaultLiveOutMask;
  getRegNameFromBitVec(LiveOutBV, FuncInfo.LiveOut, BC);

  checkInstr(BC, BF, FuncInfo);

  outs() << " STOKE-INFO: end function \n";
  return true;
}

void StokeInfo::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &) {
  outs() << "STOKE-INFO: begin of stoke pass\n";

  std::ofstream Outfile;
  if (!opts::StokeOutputDataFilename.empty()) {
    Outfile.open(opts::StokeOutputDataFilename);
  } else {
    outs() << "STOKE-INFO: output file is required\n";
    return;
  }

  // check some context meta data
  outs() << "\tTarget: " << BC.TheTarget->getName() << "\n";
  outs() << "\tTripleName " << BC.TripleName << "\n";
  outs() << "\tgetNumRegs " << BC.MRI->getNumRegs() << "\n";

  auto CG = buildCallGraph(BC, BFs);
  RegAnalysis RA(BC, BFs, CG);

  NumRegs = BC.MRI->getNumRegs();
  assert(NumRegs > 0 && "STOKE-INFO: the target register number is incorrect!");

  DefaultDefInMask.resize(NumRegs, false);
  DefaultLiveOutMask.resize(NumRegs, false);

  BC.MIA->getDefaultDefIn(DefaultDefInMask, *BC.MRI);
  BC.MIA->getDefaultLiveOut(DefaultLiveOutMask, *BC.MRI);

  dumpRegNameFromBitVec(DefaultDefInMask, BC);
  dumpRegNameFromBitVec(DefaultLiveOutMask, BC);

  StokeFuncInfo FuncInfo;
  // analyze all functions
  FuncInfo.printCsvHeader(Outfile);
  for (auto &BF : BFs) {
    DataflowInfoManager DInfo(BC, BF.second, &RA/*RA.get()*/, nullptr);
    FuncInfo.reset();
    if (analyze(BC, BF.second, DInfo, RA, FuncInfo)) {
      FuncInfo.printData(Outfile);
    }
  }

  outs() << "STOKE-INFO: end of stoke pass\n";
}

} // namespace bolt
} // namespace llvm
