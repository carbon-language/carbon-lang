//===--- BinaryFunction.cpp - Interface for machine-level function --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "DataReader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <queue>
#include <string>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "flo"

namespace llvm {
namespace flo {

BinaryBasicBlock *
BinaryFunction::getBasicBlockContainingOffset(uint64_t Offset) {
  if (Offset > Size)
    return nullptr;

  if (BasicBlocks.empty())
    return nullptr;

  auto I = std::upper_bound(BasicBlocks.begin(),
                            BasicBlocks.end(),
                            BinaryBasicBlock(Offset));
  assert(I != BasicBlocks.begin() && "first basic block not at offset 0");

  return &(*--I);
}

unsigned BinaryFunction::eraseDeadBBs(
    std::map<BinaryBasicBlock *, bool> &ToPreserve) {
  BasicBlockOrderType NewLayout;
  unsigned Count = 0;
  for (auto I = BasicBlocksLayout.begin(), E = BasicBlocksLayout.end(); I != E;
       ++I) {
    if (ToPreserve[*I])
      NewLayout.push_back(*I);
    else
      ++Count;
  }
  BasicBlocksLayout = std::move(NewLayout);
  return Count;
}

void BinaryFunction::print(raw_ostream &OS, std::string Annotation,
                           bool PrintInstructions) const {
  StringRef SectionName;
  Section.getName(SectionName);
  OS << "Binary Function \"" << getName() << "\" " << Annotation << " {"
     << "\n  State       : "   << CurrentState
     << "\n  Address     : 0x" << Twine::utohexstr(Address)
     << "\n  Size        : 0x" << Twine::utohexstr(Size)
     << "\n  MaxSize     : 0x" << Twine::utohexstr(MaxSize)
     << "\n  Offset      : 0x" << Twine::utohexstr(FileOffset)
     << "\n  Section     : "   << SectionName
     << "\n  Orc Section : "   << getCodeSectionName()
     << "\n  IsSimple    : "   << IsSimple
     << "\n  BB Count    : "   << BasicBlocksLayout.size();
  if (BasicBlocksLayout.size()) {
    OS << "\n  BB Layout   : ";
    auto Sep = "";
    for (auto BB : BasicBlocksLayout) {
      OS << Sep << BB->getName();
      Sep = ", ";
    }
  }
  if (ImageAddress)
    OS << "\n  Image       : 0x" << Twine::utohexstr(ImageAddress);
  if (ExecutionCount != COUNT_NO_PROFILE)
    OS << "\n  Exec Count  : " << ExecutionCount;

  OS << "\n}\n";

  if (!PrintInstructions || !BC.InstPrinter)
    return;

  // Offset of the instruction in function.
  uint64_t Offset{0};

  if (BasicBlocks.empty() && !Instructions.empty()) {
    // Print before CFG was built.
    for (const auto &II : Instructions) {
      auto Offset = II.first;

      // Print label if exists at this offset.
      auto LI = Labels.find(Offset);
      if (LI != Labels.end())
        OS << LI->second->getName() << ":\n";

      auto &Instruction = II.second;
      OS << format("    %08" PRIx64 ": ", Offset);
      BC.InstPrinter->printInst(&Instruction, OS, "", *BC.STI);
      OS << "\n";
    }
  }

  for (auto BB : BasicBlocksLayout) {
    OS << BB->getName() << " ("
       << BB->Instructions.size() << " instructions, align : "
       << BB->getAlignment() << ")\n";

    uint64_t BBExecCount = BB->getExecutionCount();
    if (BBExecCount != BinaryBasicBlock::COUNT_NO_PROFILE) {
      OS << "  Exec Count : " << BBExecCount << "\n";
    }
    if (!BB->Predecessors.empty()) {
      OS << "  Predecessors: ";
      auto Sep = "";
      for (auto Pred : BB->Predecessors) {
        OS << Sep << Pred->getName();
        Sep = ", ";
      }
      OS << '\n';
    }

    Offset = RoundUpToAlignment(Offset, BB->getAlignment());

    for (auto &Instr : *BB) {
      OS << format("    %08" PRIx64 ": ", Offset);
      BC.InstPrinter->printInst(&Instr, OS, "", *BC.STI);
      OS << "\n";

      // In case we need MCInst printer:
      // Instr.dump_pretty(OS, InstructionPrinter.get());

      // Calculate the size of the instruction.
      // Note: this is imprecise since happening prior to relaxation.
      SmallString<256> Code;
      SmallVector<MCFixup, 4> Fixups;
      raw_svector_ostream VecOS(Code);
      BC.MCE->encodeInstruction(Instr, VecOS, Fixups, *BC.STI);
      Offset += Code.size();
    }

    if (!BB->Successors.empty()) {
      OS << "  Successors: ";
      auto BI = BB->BranchInfo.begin();
      auto Sep = "";
      for (auto Succ : BB->Successors) {
        assert(BI != BB->BranchInfo.end() && "missing BranchInfo entry");
        OS << Sep << Succ->getName();
        if (ExecutionCount != COUNT_NO_PROFILE &&
            BI->MispredictedCount != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
          OS << " (mispreds: " << BI->MispredictedCount
             << ", count: " << BI->Count << ")";
        } else if (ExecutionCount != COUNT_NO_PROFILE &&
                   BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
          OS << " (inferred count: " << BI->Count << ")";
        }
        Sep = ", ";
        ++BI;
      }
      OS << '\n';
    }

    OS << '\n';
  }

  OS << "End of Function \"" << getName() << "\"\n";
}

bool BinaryFunction::disassemble(ArrayRef<uint8_t> FunctionData) {
  assert(FunctionData.size() == getSize() &&
         "function size does not match raw data size");

  auto &Ctx = BC.Ctx;
  auto &MIA = BC.MIA;

  // Insert a label at the beginning of the function. This will be our first
  // basic block.
  Labels[0] = Ctx->createTempSymbol("BB0", false);

  bool IsSimple = true;
  for (uint64_t Offset = 0; IsSimple && (Offset < getSize()); ) {
    MCInst Instruction;
    uint64_t Size;
    if (!BC.DisAsm->getInstruction(Instruction,
                                   Size,
                                   FunctionData.slice(Offset),
                                   getAddress() + Offset,
                                   nulls(),
                                   nulls())) {
      // Ignore this function. Skip to the next one.
      IsSimple = false;
      break;
    }

    if (MIA->isUnsupported(Instruction)) {
      DEBUG(dbgs() << "FLO: unsupported instruction seen. Skipping function "
                   << getName() << ".\n");
      IsSimple = false;
      break;
    }

    if (MIA->isIndirectBranch(Instruction)) {
      DEBUG(dbgs() << "FLO: indirect branch seen. Skipping function "
                   << getName() << ".\n");
      IsSimple = false;
      break;
    }

    uint64_t AbsoluteInstrAddr = getAddress() + Offset;
    if (MIA->isBranch(Instruction) || MIA->isCall(Instruction)) {
      uint64_t InstructionTarget = 0;
      if (MIA->evaluateBranch(Instruction,
                              AbsoluteInstrAddr,
                              Size,
                              InstructionTarget)) {
        // Check if the target is within the same function. Otherwise it's
        // a call, possibly a tail call.
        //
        // If the target *is* the function address it could be either a branch
        // or a recursive call.
        bool IsCall = MIA->isCall(Instruction);
        MCSymbol *TargetSymbol{nullptr};
        uint64_t TargetOffset{0};

        if (IsCall && containsAddress(InstructionTarget)) {
          if (InstructionTarget == getAddress()) {
            // Recursive call.
            TargetSymbol = Ctx->getOrCreateSymbol(getName());
          } else {
            // Possibly an old-style PIC code
            DEBUG(dbgs() << "FLO: internal call detected at 0x"
                         << Twine::utohexstr(AbsoluteInstrAddr)
                         << " in function " << getName() << "\n");
            IsSimple = false;
            break;
          }
        }

        if (!TargetSymbol) {
          // Create either local label or external symbol.
          if (containsAddress(InstructionTarget)) {
            // Check if there's already a registered label.
            TargetOffset = InstructionTarget - getAddress();
            auto LI = Labels.find(TargetOffset);
            if (LI == Labels.end()) {
              TargetSymbol = Ctx->createTempSymbol();
              Labels[TargetOffset] = TargetSymbol;
            } else {
              TargetSymbol = LI->second;
            }
          } else {
            if (!IsCall && Size == 2) {
              errs() << "FLO-WARNING: relaxed tail call detected at 0x"
                     << Twine::utohexstr(AbsoluteInstrAddr)
                     << ". Code size will be increased.\n";
            }

            // This is a call regardless of the opcode (e.g. tail call).
            IsCall = true;
            TargetSymbol = BC.getOrCreateGlobalSymbol(InstructionTarget,
                                                      "FUNCat");
          }
        }

        Instruction.clear();
        Instruction.addOperand(
            MCOperand::createExpr(
              MCSymbolRefExpr::create(TargetSymbol,
                                      MCSymbolRefExpr::VK_None,
                                      *Ctx)));
        if (!IsCall) {
          // Add local branch info.
          LocalBranches.push_back({Offset, TargetOffset});
        }

      } else {
        // Indirect call
        DEBUG(dbgs() << "FLO: indirect call detected (not yet supported)\n");
        IsSimple = false;
        break;
      }
    } else {
      if (MIA->hasRIPOperand(Instruction)) {
        uint64_t TargetAddress{0};
        MCSymbol *TargetSymbol{nullptr};
        if (!MIA->evaluateRIPOperand(Instruction, AbsoluteInstrAddr,
                                     Size, TargetAddress)) {
          DEBUG(
            dbgs() << "FLO: rip-relative operand could not be evaluated:\n";
            BC.InstPrinter->printInst(&Instruction, dbgs(), "", *BC.STI);
            dbgs() << '\n';
            Instruction.dump_pretty(dbgs(), BC.InstPrinter.get());
            dbgs() << '\n';
          );
          IsSimple = false;
          break;
        }
        // FIXME: check that the address is in data, not in code.
        TargetSymbol = BC.getOrCreateGlobalSymbol(TargetAddress, "DATAat");
        MIA->replaceRIPOperandDisp(
            Instruction,
            MCOperand::createExpr(
              MCSymbolRefExpr::create(TargetSymbol,
                                      MCSymbolRefExpr::VK_None,
                                      *Ctx)));
      }
    }

    addInstruction(Offset, std::move(Instruction));

    Offset += Size;
  }

  setSimple(IsSimple);

  // TODO: clear memory if not simple function?

  // Update state.
  updateState(State::Disassembled);

  return true;
}

bool BinaryFunction::buildCFG() {

  auto &MIA = BC.MIA;

  auto BranchDataOrErr = BC.DR.getFuncBranchData(getName());
  if (std::error_code EC = BranchDataOrErr.getError()) {
    DEBUG(dbgs() << "no branch data found for \"" << getName() << "\"\n");
  } else {
    ExecutionCount = BC.DR.countBranchesTo(getName());
  }

  if (!isSimple())
    return false;

  if (!(CurrentState == State::Disassembled))
    return false;

  assert(BasicBlocks.empty() && "basic block list should be empty");
  assert((Labels.find(0) != Labels.end()) &&
         "first instruction should always have a label");

  // Create basic blocks in the original layout order:
  //
  //  * Every instruction with associated label marks
  //    the beginning of a basic block.
  //  * Conditional instruction marks the end of a basic block,
  //    except when the following instruction is an
  //    unconditional branch, and the unconditional branch is not
  //    a destination of another branch. In the latter case, the
  //    basic block will consist of a single unconditional branch
  //    (missed optimization opportunity?).
  //
  // Created basic blocks are sorted in layout order since they are
  // created in the same order as instructions, and instructions are
  // sorted by offsets.
  BinaryBasicBlock *InsertBB{nullptr};
  BinaryBasicBlock *PrevBB{nullptr};
  bool IsLastInstrNop = false;
  MCInst *PrevInstr{nullptr};
  for (auto &InstrInfo : Instructions) {
    auto LI = Labels.find(InstrInfo.first);
    if (LI != Labels.end()) {
      // Always create new BB at branch destination.
      PrevBB = InsertBB;
      InsertBB = addBasicBlock(LI->first, LI->second,
                               /* DeriveAlignment = */ IsLastInstrNop);
    }
    if (!InsertBB) {
      // It must be a fallthrough or unreachable code. Create a new block unless
      // we see an unconditional branch following a conditional one.
      assert(PrevBB && "no previous basic block for a fall through");
      assert(PrevInstr && "no previous instruction for a fall through");
      if (MIA->isUnconditionalBranch(InstrInfo.second) &&
          !MIA->isUnconditionalBranch(*PrevInstr)) {
        // Temporarily restore inserter basic block.
        InsertBB = PrevBB;
      } else {
        InsertBB = addBasicBlock(InstrInfo.first,
                                 BC.Ctx->createTempSymbol("FT", true),
                                 /* DeriveAlignment = */ IsLastInstrNop);
      }
    }

    // Ignore nops. We use nops to derive alignment of the next basic block.
    // It will not always work, as some blocks are naturally aligned, but
    // it's just part of heuristic for block alignment.
    if (MIA->isNoop(InstrInfo.second)) {
      IsLastInstrNop = true;
      continue;
    }

    IsLastInstrNop = false;
    InsertBB->addInstruction(InstrInfo.second);
    PrevInstr = &InstrInfo.second;

    // How well do we detect tail calls here?
    if (MIA->isTerminator(InstrInfo.second)) {
      PrevBB = InsertBB;
      InsertBB = nullptr;
    }
  }

  // Set the basic block layout to the original order
  for (auto &BB : BasicBlocks) {
    BasicBlocksLayout.emplace_back(&BB);
  }

  // Intermediate dump.
  DEBUG(print(dbgs(), "after creating basic blocks"));

  // TODO: handle properly calls to no-return functions,
  // e.g. exit(3), etc. Otherwise we'll see a false fall-through
  // blocks.

  for (auto &Branch : LocalBranches) {

    DEBUG(dbgs() << "registering branch [0x" << Twine::utohexstr(Branch.first)
                 << "] -> [0x" << Twine::utohexstr(Branch.second) << "]\n");
    BinaryBasicBlock *FromBB = getBasicBlockContainingOffset(Branch.first);
    assert(FromBB && "cannot find BB containing FROM branch");
    BinaryBasicBlock *ToBB = getBasicBlockAtOffset(Branch.second);
    assert(ToBB && "cannot find BB containing TO branch");

    if (std::error_code EC = BranchDataOrErr.getError()) {
      FromBB->addSuccessor(ToBB);
    } else {
      const FuncBranchData &BranchData = BranchDataOrErr.get();
      auto BranchInfoOrErr = BranchData.getBranch(Branch.first, Branch.second);
      if (std::error_code EC = BranchInfoOrErr.getError()) {
        FromBB->addSuccessor(ToBB);
      } else {
        const BranchInfo &BInfo = BranchInfoOrErr.get();
        FromBB->addSuccessor(ToBB, BInfo.Branches, BInfo.Mispreds);
      }
    }
  }

  // Add fall-through branches.
  PrevBB = nullptr;
  bool IsPrevFT = false; // Is previous block a fall-through.
  for (auto &BB : BasicBlocks) {
    if (IsPrevFT) {
      PrevBB->addSuccessor(&BB, BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE,
                           BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE);
    }

    MCInst &LastInst = BB.back();
    if (BB.empty()) {
      IsPrevFT = true;
    } else if (BB.succ_size() == 0) {
      IsPrevFT = MIA->isTerminator(LastInst) ? false : true;
    } else if (BB.succ_size() == 1) {
      IsPrevFT =  MIA->isConditionalBranch(LastInst) ? true : false;
    } else {
      // Either ends with 2 branches, or with an indirect jump.
      IsPrevFT = false;
    }

    PrevBB = &BB;
  }

  if (!IsPrevFT) {
    // Possibly a call that does not return.
    DEBUG(dbgs() << "last block was marked as a fall-through\n");
  }

  // Infer frequency for non-taken branches
  if (ExecutionCount != COUNT_NO_PROFILE && !BranchDataOrErr.getError()) {
    inferFallThroughCounts();
  }

  // Clean-up memory taken by instructions and labels.
  clearInstructions();
  clearLabels();
  clearLocalBranches();

  // Update the state.
  CurrentState = State::CFG;

  return true;
}

void BinaryFunction::inferFallThroughCounts() {
  assert(!BasicBlocks.empty() && "basic block list should not be empty");

  // Compute preliminary execution time for each basic block
  for (auto &CurBB : BasicBlocks) {
    if (&CurBB == &*BasicBlocks.begin()) {
      CurBB.ExecutionCount = ExecutionCount;
      continue;
    }
    CurBB.ExecutionCount = 0;
  }

  for (auto &CurBB : BasicBlocks) {
    auto SuccCount = CurBB.BranchInfo.begin();
    for (auto Succ : CurBB.successors()) {
      // Do not update execution count of the entry block (when we have tail
      // calls). We already accounted for those when computing the func count.
      if (Succ == &*BasicBlocks.begin())
        continue;
      if (SuccCount->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Succ->ExecutionCount += SuccCount->Count;
      ++SuccCount;
    }
  }

  // Work on a basic block at a time, propagating frequency information forwards
  // It is important to walk in the layour order
  for (auto &CurBB : BasicBlocks) {
    uint64_t BBExecCount = CurBB.getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (CurBB.succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data
    uint64_t ReportedBranches = 0;
    for (auto &SuccCount : CurBB.BranchInfo) {
      if (SuccCount.Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        ReportedBranches += SuccCount.Count;
    }

    // Infer the frequency of the fall-through edge, representing not taking the
    // branch
    uint64_t Inferred = 0;
    if (BBExecCount > ReportedBranches)
      Inferred = BBExecCount - ReportedBranches;
    if (BBExecCount < ReportedBranches)
      errs() << "FLO-WARNING: Fall-through inference is slightly inconsistent. "
                "exec frequency is less than the outgoing edges frequency ("
             << BBExecCount << " < " << ReportedBranches
             << ") for  BB at offset 0x"
             << Twine::utohexstr(getAddress() + CurBB.getOffset()) << '\n';

    // Put this information into the fall-through edge
    if (CurBB.succ_size() == 0)
      continue;
    // If there is a FT, the last successor will be it.
    auto &SuccCount = CurBB.BranchInfo.back();
    auto &Succ = CurBB.Successors.back();
    if (SuccCount.Count == BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
      SuccCount.Count = Inferred;
      Succ->ExecutionCount += Inferred;
    }

  } // end for (CurBB : BasicBlocks)

  return;
}

void BinaryFunction::optimizeLayout() {
  // Bail if no profiling information or if empty
  if (getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE ||
      BasicBlocksLayout.empty()) {
    return;
  }

  // Work on optimal solution if problem is small enough
  if (BasicBlocksLayout.size() <= FUNC_SIZE_THRESHOLD)
    return solveOptimalLayout();

  DEBUG(dbgs() << "running block layout heuristics on " << getName() << "\n");

  // Greedy heuristic implementation for the TSP, applied to BB layout. Try to
  // maximize weight during a path traversing all BBs. In this way, we will
  // convert the hottest branches into fall-throughs.

  // Encode an edge between two basic blocks, source and destination
  typedef std::pair<BinaryBasicBlock *, BinaryBasicBlock *> EdgeTy;
  std::map<EdgeTy, uint64_t> Weight;

  // Define a comparison function to establish SWO between edges
  auto Comp = [&Weight](EdgeTy A, EdgeTy B) { return Weight[A] < Weight[B]; };
  std::priority_queue<EdgeTy, std::vector<EdgeTy>, decltype(Comp)> Queue(Comp);

  typedef std::vector<BinaryBasicBlock *> ClusterTy;
  typedef std::map<BinaryBasicBlock *, int> BBToClusterMapTy;
  std::vector<ClusterTy> Clusters;
  BBToClusterMapTy BBToClusterMap;

  // Populating priority queue with all edges
  for (auto BB : BasicBlocksLayout) {
    BBToClusterMap[BB] = -1; // Mark as unmapped
    auto BI = BB->BranchInfo.begin();
    for (auto &I : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Weight[std::make_pair(BB, I)] = BI->Count;
      Queue.push(std::make_pair(BB, I));
      ++BI;
    }
  }

  // Start a cluster with the entry point
  BinaryBasicBlock *Entry = *BasicBlocksLayout.begin();
  Clusters.emplace_back();
  auto &EntryCluster = Clusters.back();
  EntryCluster.push_back(Entry);
  BBToClusterMap[Entry] = 0;

  // Grow clusters in a greedy fashion
  while (!Queue.empty()) {
    auto elmt = Queue.top();
    Queue.pop();

    BinaryBasicBlock *BBSrc = elmt.first;
    BinaryBasicBlock *BBDst = elmt.second;
    int I = 0, J = 0;

    // Case 1: BBSrc and BBDst are the same. Ignore this edge
    if (BBSrc == BBDst || BBDst == Entry)
      continue;

    // Case 2: Both BBSrc and BBDst are already allocated
    if ((I = BBToClusterMap[BBSrc]) != -1 &&
        (J = BBToClusterMap[BBDst]) != -1) {
      // Case 2a: If they are already allocated at the same cluster, ignore
      if (I == J)
        continue;
      auto &ClusterA = Clusters[I];
      auto &ClusterB = Clusters[J];
      if (ClusterA.back() == BBSrc && ClusterB.front() == BBDst) {
        // Case 2b: BBSrc is at the end of a cluster and BBDst is at the start,
        // allowing us to merge two clusters
        for (auto BB : ClusterB)
          BBToClusterMap[BB] = I;
        ClusterA.insert(ClusterA.end(), ClusterB.begin(), ClusterB.end());
        ClusterB.clear();
      } else {
        // Case 2c: Both BBSrc and BBDst are allocated in positions we cannot
        // merge them, so we ignore this edge.
      }
      continue;
    }

    // Case 3: BBSrc is already allocated in a cluster
    if ((I = BBToClusterMap[BBSrc]) != -1) {
      auto &Cluster = Clusters[I];
      if (Cluster.back() == BBSrc) {
        // Case 3a: BBSrc is allocated at the end of this cluster. We put
        // BBSrc and BBDst together.
        Cluster.push_back(BBDst);
        BBToClusterMap[BBDst] = I;
      } else {
        // Case 3b: We cannot put BBSrc and BBDst in consecutive positions,
        // so we ignore this edge.
      }
      continue;
    }

    // Case 4: BBSrc is not in a cluster, but BBDst is
    if ((I = BBToClusterMap[BBDst]) != -1) {
      auto &Cluster = Clusters[I];
      if (Cluster.front() == BBDst) {
        // Case 4a: BBDst is allocated at the start of this cluster. We put
        // BBSrc and BBDst together.
        Cluster.insert(Cluster.begin(), BBSrc);
        BBToClusterMap[BBSrc] = I;
      } else {
        // Case 4b: We cannot put BBSrc and BBDst in consecutive positions,
        // so we ignore this edge.
      }
      continue;
    }

    // Case 5: Both BBSrc and BBDst are unallocated, so we create a new cluster
    // with them
    I = Clusters.size();
    Clusters.emplace_back();
    auto &Cluster = Clusters.back();
    Cluster.push_back(BBSrc);
    Cluster.push_back(BBDst);
    BBToClusterMap[BBSrc] = I;
    BBToClusterMap[BBDst] = I;
  }

  // Create an extra cluster for unvisited basic blocks
  std::vector<BinaryBasicBlock *> Unvisited;
  for (auto BB : BasicBlocksLayout) {
    if (BBToClusterMap[BB] == -1) {
      Unvisited.push_back(BB);
    }
  }

  // Define final function layout based on clusters
  BasicBlocksLayout.clear();
  for (auto &Cluster : Clusters) {
    BasicBlocksLayout.insert(BasicBlocksLayout.end(), Cluster.begin(),
                             Cluster.end());
  }

  // Finalize layout with BBs that weren't assigned to any cluster, preserving
  // their relative order
  BasicBlocksLayout.insert(BasicBlocksLayout.end(), Unvisited.begin(),
                           Unvisited.end());

  fixBranches();
}

void BinaryFunction::solveOptimalLayout() {
  std::vector<std::vector<uint64_t>> Weight;
  std::map<BinaryBasicBlock *, int> BBToIndex;
  std::vector<BinaryBasicBlock *> IndexToBB;

  DEBUG(dbgs() << "finding optimal block layout for " << getName() << "\n");

  unsigned N = BasicBlocksLayout.size();
  // Populating weight map and index map
  for (auto BB : BasicBlocksLayout) {
    BBToIndex[BB] = IndexToBB.size();
    IndexToBB.push_back(BB);
  }
  Weight.resize(N);
  for (auto BB : BasicBlocksLayout) {
    auto BI = BB->BranchInfo.begin();
    Weight[BBToIndex[BB]].resize(N);
    for (auto I : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Weight[BBToIndex[BB]][BBToIndex[I]] = BI->Count;
      ++BI;
    }
  }

  std::vector<std::vector<int64_t>> DP;
  DP.resize(1 << N);
  for (auto &Elmt : DP) {
    Elmt.resize(N, -1);
  }
  // Start with the entry basic block being allocated with cost zero
  DP[1][0] = 0;
  // Walk through TSP solutions using a bitmask to represent state (current set
  // of BBs in the layout)
  unsigned BestSet = 1;
  unsigned BestLast = 0;
  int64_t BestWeight = 0;
  for (unsigned Set = 1; Set < (1U << N); ++Set) {
    // Traverse each possibility of Last BB visited in this layout
    for (unsigned Last = 0; Last < N; ++Last) {
      // Case 1: There is no possible layout with this BB as Last
      if (DP[Set][Last] == -1)
        continue;

      // Case 2: There is a layout with this Set and this Last, and we try
      // to expand this set with New
      for (unsigned New = 1; New < N; ++New) {
        // Case 2a: BB "New" is already in this Set
        if ((Set & (1 << New)) != 0)
          continue;

        // Case 2b: BB "New" is not in this set and we add it to this Set and
        // record total weight of this layout with "New" as the last BB.
        unsigned NewSet = (Set | (1 << New));
        if (DP[NewSet][New] == -1)
          DP[NewSet][New] = DP[Set][Last] + (int64_t)Weight[Last][New];
        DP[NewSet][New] = std::max(DP[NewSet][New],
                                   DP[Set][Last] + (int64_t)Weight[Last][New]);

        if (DP[NewSet][New] > BestWeight) {
          BestWeight = DP[NewSet][New];
          BestSet = NewSet;
          BestLast = New;
        }
      }
    }
  }

  std::vector<BinaryBasicBlock *> PastLayout = BasicBlocksLayout;

  // Define final function layout based on layout that maximizes weight
  BasicBlocksLayout.clear();
  unsigned Last = BestLast;
  unsigned Set = BestSet;
  std::vector<bool> Visited;
  Visited.resize(N);
  Visited[Last] = true;
  BasicBlocksLayout.push_back(IndexToBB[Last]);
  Set = Set & ~(1U << Last);
  while (Set != 0) {
    int64_t Best = -1;
    for (unsigned I = 0; I < N; ++I) {
      if (DP[Set][I] == -1)
        continue;
      if (DP[Set][I] > Best) {
        Last = I;
        Best = DP[Set][I];
      }
    }
    Visited[Last] = true;
    BasicBlocksLayout.push_back(IndexToBB[Last]);
    Set = Set & ~(1U << Last);
  }
  std::reverse(BasicBlocksLayout.begin(), BasicBlocksLayout.end());

  // Finalize layout with BBs that weren't assigned to the layout
  for (auto BB : PastLayout) {
    if (Visited[BBToIndex[BB]] == false)
      BasicBlocksLayout.push_back(BB);
  }

  fixBranches();
}

const BinaryBasicBlock *
BinaryFunction::getOriginalLayoutSuccessor(const BinaryBasicBlock *BB) const {
  auto I = std::upper_bound(BasicBlocks.begin(), BasicBlocks.end(), *BB);
  assert(I != BasicBlocks.begin() && "first basic block not at offset 0");

  if (I == BasicBlocks.end())
    return nullptr;
  return &*I;
}

void BinaryFunction::fixBranches() {
  auto &MIA = BC.MIA;

  for (unsigned I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    BinaryBasicBlock *BB = BasicBlocksLayout[I];

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (!MIA->analyzeBranch(BB->Instructions, TBB, FBB, CondBranch,
                            UncondBranch)) {
      continue;
    }

    // Check if the original fall-through for this block has been moved
    const MCSymbol *FT = nullptr;
    if (I + 1 != BasicBlocksLayout.size())
      FT = BasicBlocksLayout[I + 1]->getLabel();
    const BinaryBasicBlock *OldFTBB = getOriginalLayoutSuccessor(BB);
    const MCSymbol *OldFT = nullptr;
    if (OldFTBB != nullptr)
      OldFT = OldFTBB->getLabel();

    // Case 1: There are no branches in this basic block and it just falls
    // through
    if (CondBranch == nullptr && UncondBranch == nullptr) {
      // Case 1a: Last instruction is a return, so it does *not* fall through to
      // the next block.
      if (!BB->empty() && MIA->isReturn(BB->back()))
        continue;
      // Case 1b: Layout has changed and the fallthrough is not the same. Need
      // to add a new unconditional branch to jump to the old fallthrough.
      if (FT != OldFT && OldFT != nullptr) {
        MCInst NewInst;
        if (!MIA->createUncondBranch(NewInst, OldFT, BC.Ctx.get()))
          llvm_unreachable("Target does not support creating new branches");
        BB->Instructions.emplace_back(std::move(NewInst));
      }
      // Case 1c: Layout hasn't changed, nothing to do.
      continue;
    }

    // Case 2: There is a single jump, unconditional, in this basic block
    if (CondBranch == nullptr) {
      // Case 2a: It jumps to the new fall-through, so we can delete it
      if (TBB == FT) {
        BB->eraseInstruction(UncondBranch);
      }
      // Case 2b: If 2a doesn't happen, there is nothing we can do
      continue;
    }

    // Case 3: There is a single jump, conditional, in this basic block
    if (UncondBranch == nullptr) {
      // Case 3a: If the taken branch goes to the next block in the new layout,
      // invert this conditional branch logic so we can make this a fallthrough.
      if (TBB == FT) {
        assert(OldFT != nullptr && "malformed CFG");
        if (!MIA->reverseBranchCondition(*CondBranch, OldFT, BC.Ctx.get()))
          llvm_unreachable("Target does not support reversing branches");
        continue;
      }
      // Case 3b: Need to add a new unconditional branch because layout
      // has changed
      if (FT != OldFT && OldFT != nullptr) {
        MCInst NewInst;
        if (!MIA->createUncondBranch(NewInst, OldFT, BC.Ctx.get()))
          llvm_unreachable("Target does not support creating new branches");
        BB->Instructions.emplace_back(std::move(NewInst));
        continue;
      }
      // Case 3c: Old fall-through is the same as the new one, no need to change
      continue;
    }

    // Case 4: There are two jumps in this basic block, one conditional followed
    // by another unconditional.
    // Case 4a: If the unconditional jump target is the new fall through,
    // delete it.
    if (FBB == FT) {
      BB->eraseInstruction(UncondBranch);
      continue;
    }
    // Case 4b: If the taken branch goes to the next block in the new layout,
    // invert this conditional branch logic so we can make this a fallthrough.
    // Now we don't need the unconditional jump anymore, so we also delete it.
    if (TBB == FT) {
      if (!MIA->reverseBranchCondition(*CondBranch, FBB, BC.Ctx.get()))
        llvm_unreachable("Target does not support reversing branches");
      BB->eraseInstruction(UncondBranch);
      continue;
    }
    // Case 4c: Nothing interesting happening.
  }
}

} // namespace flo
} // namespace llvm
