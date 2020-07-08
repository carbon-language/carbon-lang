#include "Passes/FeatureMiner.h"
#include "Passes/DataflowInfoManager.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-feature-miner"

namespace llvm {
namespace bolt {

class BinaryFunction;

int8_t FeatureMiner::getProcedureType(BinaryFunction &Function,
                                      BinaryContext &BC) {
  int8_t ProcedureType = 1;
  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      if (BC.MIB->isCall(Inst)) {
        ProcedureType = 0; // non-leaf type
        if (const auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst)) {
          const auto *Callee = BC.getFunctionForSymbol(CalleeSymbol);
          if (Callee &&
              Callee->getFunctionNumber() == Function.getFunctionNumber()) {
            return 2; // call self type
          }
        }
      }
    }
  }
  return ProcedureType; // leaf type
}

void FeatureMiner::addSuccessorInfo(DominatorAnalysis<false> &DA,
                                    DominatorAnalysis<true> &PDA,
                                    SBIPtr const &SBI, BinaryFunction &Function,
                                    BinaryContext &BC, MCInst &Inst,
                                    BinaryBasicBlock &BB, bool SuccType) {

  BinaryBasicBlock *Successor = BB.getConditionalSuccessor(SuccType);

  if (!Successor)
    return;

  unsigned NumLoads{0};
  unsigned NumStores{0};
  unsigned NumCallsExit{0};
  unsigned NumCalls{0};
  unsigned NumCallsInvoke{0};
  unsigned NumTailCalls{0};
  unsigned NumIndirectCalls{0};

  for (auto &Inst : BB) {
    if (BC.MIB->isLoad(Inst)) {
      ++NumLoads;
    } else if (BC.MIB->isStore(Inst)) {
      ++NumStores;
    } else if (BC.MIB->isCall(Inst)) {
      ++NumCalls;

      if (BC.MIB->isIndirectCall(Inst))
        ++NumIndirectCalls;

      if (BC.MIB->isInvoke(Inst))
        ++NumCallsInvoke;

      if (BC.MIB->isTailCall(Inst))
        ++NumTailCalls;

      if (const auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst)) {
        StringRef CalleeName = CalleeSymbol->getName();
        if (CalleeName == "__cxa_throw@PLT" ||
            CalleeName == "_Unwind_Resume@PLT" ||
            CalleeName == "__cxa_rethrow@PLT" || CalleeName == "exit@PLT" ||
            CalleeName == "abort@PLT")
          ++NumCallsExit;
      }
    }
  }

  BBIPtr SuccBBInfo = std::make_unique<struct BasicBlockInfo>();

  // Check if the successor basic block is a loop header and store it.
  SuccBBInfo->LoopHeader = BPI->isLoopHeader(Successor);

  SuccBBInfo->BasicBlockSize = Successor->size();

  // Check if the edge getting to the successor basic block is a loop
  // exit edge and store it.
  SuccBBInfo->Exit = BPI->isExitEdge(&BB, Successor);

  // Check if the edge getting to the successor basic block is a loop
  // back edge and store it.
  SuccBBInfo->Backedge = BPI->isBackEdge(&BB, Successor);

  MCInst *SuccInst = Successor->getTerminatorBefore(nullptr);
  // Store information about the branch type ending sucessor basic block
  SuccBBInfo->EndOpcode = (SuccInst && BC.MIA->isBranch(*SuccInst))
                              ? SuccInst->getOpcode()
                              : 0; // 0 = NOTHING
  if (SuccBBInfo->EndOpcode != 0)
    SuccBBInfo->EndOpcodeStr = BC.MII->getName(SuccInst->getOpcode());
  else
    SuccBBInfo->EndOpcodeStr = "NOTHING";

  // Check if the successor basic block contains
  // a procedure call and store it.
  SuccBBInfo->Call = (NumCalls > 0) ? 1  // Contains a call instruction
                                    : 0; // Does not contain a call instruction

  SuccBBInfo->NumStores = NumStores;
  SuccBBInfo->NumLoads = NumLoads;
  SuccBBInfo->NumCallsExit = NumCallsExit;
  SuccBBInfo->NumCalls = NumCalls;

  SuccBBInfo->NumCallsInvoke = NumCallsInvoke;
  SuccBBInfo->NumIndirectCalls = NumIndirectCalls;
  SuccBBInfo->NumTailCalls = NumTailCalls;

  auto InstSucc = Successor->getLastNonPseudoInstr();
  if (InstSucc) {
    // Check if the source basic block dominates its
    // target basic block and store it.
    SuccBBInfo->BranchDominates = (DA.doesADominateB(Inst, *InstSucc) == true)
                                      ? 1  // Dominates
                                      : 0; // Does not dominate

    // Check if the target basic block postdominates
    // the source basic block and store it.
    SuccBBInfo->BranchPostdominates =
        (PDA.doesADominateB(*InstSucc, Inst) == true)
            ? 1  // Postdominates
            : 0; // Does not postdominate
  }

  /// The follwoing information is used as an identifier only for
  /// the purpose of matching the inferred probabilities with the branches
  /// in the binary.
  SuccBBInfo->FromFunName = Function.getPrintName();
  SuccBBInfo->FromBb = BB.getInputOffset();
  BinaryFunction *ToFun = Successor->getFunction();
  SuccBBInfo->ToFunName = ToFun->getPrintName();
  SuccBBInfo->ToBb = Successor->getInputOffset();

  auto Offset = BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "Offset");
  if (Offset) {
    int64_t Delta = Successor->getInputOffset() - Offset.get();
    SBI->DeltaTaken = std::abs(Delta);
  }

  if (SuccType) {
    SBI->TrueSuccessor = std::move(SuccBBInfo);

    // Check if the taken branch is a forward
    // or a backwards branch and store it.
    SBI->Direction = (Function.isForwardBranch(&BB, Successor) == true)
                         ? 1  // Forward branch
                         : 0; // Backwards branch

    auto TakenBranchInfo = BB.getTakenBranchInfo();
    SBI->Count = TakenBranchInfo.Count;
    SBI->MissPredicted = TakenBranchInfo.MispredictedCount;
  } else {
    SBI->FalseSuccessor = std::move(SuccBBInfo);

    auto FallthroughBranchInfo = BB.getFallthroughBranchInfo();
    SBI->FallthroughCount = FallthroughBranchInfo.Count;
    SBI->FallthroughMissPredicted = FallthroughBranchInfo.MispredictedCount;
  }
}

void FeatureMiner::extractFeatures(BinaryFunction &Function,
                                   BinaryContext &BC) {
  int8_t ProcedureType = getProcedureType(Function, BC);
  auto Info = DataflowInfoManager(BC, Function, nullptr, nullptr);
  auto &DA = Info.getDominatorAnalysis();
  auto &PDA = Info.getPostDominatorAnalysis();
  const BinaryLoopInfo &LoopsInfo = Function.getLoopInfo();
  bool Simple = Function.isSimple();

  for (auto &BB : Function) {

    unsigned NumOuterLoops{0};
    unsigned TotalLoops{0};
    unsigned MaximumLoopDepth{0};
    unsigned LoopDepth{0};
    unsigned LoopNumExitEdges{0};
    unsigned LoopNumExitBlocks{0};
    unsigned LoopNumExitingBlocks{0};
    unsigned LoopNumLatches{0};
    unsigned LoopNumBlocks{0};
    unsigned LoopNumBackEdges{0};

    bool LocalExitingBlock{false};
    bool LocalLatchBlock{false};
    bool LocalLoopHeader{false};

    BinaryLoop *Loop = LoopsInfo.getLoopFor(&BB);
    if (Loop) {
      SmallVector<BinaryBasicBlock *, 1> ExitingBlocks;
      Loop->getExitingBlocks(ExitingBlocks);

      SmallVector<BinaryBasicBlock *, 1> ExitBlocks;
      Loop->getExitBlocks(ExitBlocks);

      SmallVector<BinaryLoop::Edge, 1> ExitEdges;
      Loop->getExitEdges(ExitEdges);

      SmallVector<BinaryBasicBlock *, 1> Latches;
      Loop->getLoopLatches(Latches);

      NumOuterLoops = LoopsInfo.OuterLoops;
      TotalLoops = LoopsInfo.TotalLoops;
      MaximumLoopDepth = LoopsInfo.MaximumDepth;
      LoopDepth = Loop->getLoopDepth();
      LoopNumExitEdges = ExitEdges.size();
      LoopNumExitBlocks = ExitBlocks.size();
      LoopNumExitingBlocks = ExitingBlocks.size();
      LoopNumLatches = Latches.size();
      LoopNumBlocks = Loop->getNumBlocks();
      LoopNumBackEdges = Loop->getNumBackEdges();

      LocalExitingBlock = Loop->isLoopExiting(&BB);
      LocalLatchBlock = Loop->isLoopLatch(&BB);
      LocalLoopHeader = ((Loop->getHeader() == (&BB)) ? 1 : 0);
    }

    unsigned NumLoads{0};
    unsigned NumStores{0};
    unsigned NumCallsExit{0};
    unsigned NumCalls{0};
    unsigned NumCallsInvoke{0};
    unsigned NumTailCalls{0};
    unsigned NumIndirectCalls{0};
    unsigned NumSelfCalls{0};

    for (auto &Inst : BB) {
      if (BC.MIB->isLoad(Inst)) {
        ++NumLoads;
      } else if (BC.MIB->isStore(Inst)) {
        ++NumStores;
      } else if (BC.MIB->isCall(Inst)) {
        ++NumCalls;

        if (BC.MIB->isIndirectCall(Inst))
          ++NumIndirectCalls;

        if (BC.MIB->isInvoke(Inst))
          ++NumCallsInvoke;

        if (BC.MIB->isTailCall(Inst))
          ++NumTailCalls;

        if (const auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst)) {
          StringRef CalleeName = CalleeSymbol->getName();
          if (CalleeName == "__cxa_throw@PLT" ||
              CalleeName == "_Unwind_Resume@PLT" ||
              CalleeName == "__cxa_rethrow@PLT" || CalleeName == "exit@PLT" ||
              CalleeName == "abort@PLT")
            ++NumCallsExit;
          else if (CalleeName == Function.getPrintName()) {
            ++NumSelfCalls;
          }
        }
      }
    }

    int Index = -2;
    bool LoopHeader = BPI->isLoopHeader(&BB);
    for (auto &Inst : BB) {
      ++Index;

      if (!BC.MIA->isConditionalBranch(Inst))
        continue;

      SBIPtr SBI = std::make_unique<struct StaticBranchInfo>();

      SBI->Simple = Simple;
      SBI->NumOuterLoops = NumOuterLoops;
      SBI->TotalLoops = TotalLoops;
      SBI->MaximumLoopDepth = MaximumLoopDepth;
      SBI->LoopDepth = LoopDepth;
      SBI->LoopNumExitEdges = LoopNumExitEdges;
      SBI->LoopNumExitBlocks = LoopNumExitBlocks;
      SBI->LoopNumExitingBlocks = LoopNumExitingBlocks;
      SBI->LoopNumLatches = LoopNumLatches;
      SBI->LoopNumBlocks = LoopNumBlocks;
      SBI->LoopNumBackEdges = LoopNumBackEdges;

      SBI->LocalExitingBlock = LocalExitingBlock;
      SBI->LocalLatchBlock = LocalLatchBlock;
      SBI->LocalLoopHeader = LocalLoopHeader;

      SBI->Call = ((NumCalls > 0) ? 1 : 0);
      SBI->NumCalls = NumCalls;

      SBI->BasicBlockSize = BB.size();
      SBI->NumBasicBlocks = Function.size();
      SBI->NumSelfCalls = NumSelfCalls;

      SBI->NumLoads = NumLoads;
      SBI->NumStores = NumStores;
      SBI->NumCallsExit = NumCallsExit;

      SBI->NumCallsInvoke = NumCallsInvoke;
      SBI->NumIndirectCalls = NumIndirectCalls;
      SBI->NumTailCalls = NumTailCalls;

      // Check if branch's basic block is a loop header and store it.
      SBI->LoopHeader = LoopHeader;

      // Adding taken successor info.
      addSuccessorInfo(DA, PDA, SBI, Function, BC, Inst, BB, true);
      // Adding fall through successor info.
      addSuccessorInfo(DA, PDA, SBI, Function, BC, Inst, BB, false);

      // Holds the branch opcode info.
      SBI->Opcode = Inst.getOpcode();
      SBI->OpcodeStr = BC.MII->getName(Inst.getOpcode());

      // Holds the branch's procedure type.
      SBI->ProcedureType = ProcedureType;

      SBI->CmpOpcode = 0;
      if (Index > -1) {
        auto Cmp = BB.begin() + Index;

        if (BC.MII->get((*Cmp).getOpcode()).isCompare()) {
          // Holding the branch comparison opcode info.
          SBI->CmpOpcode = (*Cmp).getOpcode();

          SBI->CmpOpcodeStr = BC.MII->getName((*Cmp).getOpcode());

          auto getOperandType = [&](const MCOperand &Operand) -> int32_t {
            if (Operand.isReg())
              return 0;
            else if (Operand.isImm())
              return 1;
            else if (Operand.isFPImm())
              return 2;
            else if (Operand.isExpr())
              return 3;
            else
              return -1;
          };

          const auto InstInfo = BC.MII->get((*Cmp).getOpcode());
          unsigned NumDefs = InstInfo.getNumDefs();
          int32_t NumPrimeOperands =
              MCPlus::getNumPrimeOperands(*Cmp) - NumDefs;
          switch (NumPrimeOperands) {
          case 6: {
            int32_t RBType = getOperandType((*Cmp).getOperand(NumDefs));
            int32_t RAType = getOperandType((*Cmp).getOperand(NumDefs + 1));

            if (RBType == 0 && RAType == 0) {
              SBI->OperandRBType = RBType;
              SBI->OperandRAType = RAType;
            } else if (RBType == 0 && (RAType == 1 || RAType == 2)) {
              RAType = getOperandType((*Cmp).getOperand(NumPrimeOperands - 1));

              if (RAType != 1 && RAType != 2) {
                RAType = -1;
              }

              SBI->OperandRBType = RBType;
              SBI->OperandRAType = RAType;
            } else {
              SBI->OperandRAType = -1;
              SBI->OperandRBType = -1;
            }
            break;
          }
          case 2:
            SBI->OperandRBType = getOperandType((*Cmp).getOperand(NumDefs));
            SBI->OperandRAType = getOperandType((*Cmp).getOperand(NumDefs + 1));
            break;
          case 3:
            SBI->OperandRBType = getOperandType((*Cmp).getOperand(NumDefs));
            SBI->OperandRAType = getOperandType((*Cmp).getOperand(NumDefs + 2));
            break;
          case 1:
            SBI->OperandRAType = getOperandType((*Cmp).getOperand(NumDefs));
            break;
          default:
            SBI->OperandRAType = -1;
            SBI->OperandRBType = -1;
            break;
          }

        } else {
          Index -= 1;
          for (int Idx = Index; Idx > -1; Idx--) {
            auto Cmp = BB.begin() + Idx;
            if (BC.MII->get((*Cmp).getOpcode()).isCompare()) {
              // Holding the branch comparison opcode info.
              SBI->CmpOpcode = (*Cmp).getOpcode();
              SBI->CmpOpcodeStr = BC.MII->getName((*Cmp).getOpcode());
              break;
            }
          }
        }
      }

      this->BranchesInfoSet.push_back(std::move(SBI));
    }
  }
}

void FeatureMiner::dumpSuccessorFeatures(raw_ostream &Printer,
                                         BBIPtr &Successor) {
  int16_t BranchDominates =
      (Successor->BranchDominates.hasValue())
          ? static_cast<bool>(*(Successor->BranchDominates))
          : -1;

  int16_t BranchPostdominates =
      (Successor->BranchPostdominates.hasValue())
          ? static_cast<bool>(*(Successor->BranchPostdominates))
          : -1;

  int16_t LoopHeader = (Successor->LoopHeader.hasValue())
                           ? static_cast<bool>(*(Successor->LoopHeader))
                           : -1;

  int16_t Backedge = (Successor->Backedge.hasValue())
                         ? static_cast<bool>(*(Successor->Backedge))
                         : -1;

  int16_t Exit =
      (Successor->Exit.hasValue()) ? static_cast<bool>(*(Successor->Exit)) : -1;

  int16_t Call =
      (Successor->Call.hasValue()) ? static_cast<bool>(*(Successor->Call)) : -1;

  int32_t EndOpcode = (Successor->EndOpcode.hasValue())
                          ? static_cast<int32_t>(*(Successor->EndOpcode))
                          : -1;

  int64_t NumLoads = (Successor->NumLoads.hasValue())
                         ? static_cast<int64_t>(*(Successor->NumLoads))
                         : -1;

  int64_t NumStores = (Successor->NumStores.hasValue())
                          ? static_cast<int64_t>(*(Successor->NumStores))
                          : -1;

  int64_t BasicBlockSize =
      (Successor->BasicBlockSize.hasValue())
          ? static_cast<int64_t>(*(Successor->BasicBlockSize))
          : -1;

  int64_t NumCalls = (Successor->NumCalls.hasValue())
                         ? static_cast<int64_t>(*(Successor->NumCalls))
                         : -1;

  int64_t NumCallsExit = (Successor->NumCallsExit.hasValue())
                             ? static_cast<int64_t>(*(Successor->NumCallsExit))
                             : -1;

  int64_t NumCallsInvoke =
      (Successor->NumCallsInvoke.hasValue())
          ? static_cast<int64_t>(*(Successor->NumCallsInvoke))
          : -1;

  int64_t NumIndirectCalls =
      (Successor->NumIndirectCalls.hasValue())
          ? static_cast<int64_t>(*(Successor->NumIndirectCalls))
          : -1;

  int64_t NumTailCalls = (Successor->NumTailCalls.hasValue())
                             ? static_cast<int64_t>(*(Successor->NumTailCalls))
                             : -1;

  Printer << "," << BranchDominates << "," << BranchPostdominates << ","
          << EndOpcode << "," << Successor->EndOpcodeStr << "," << LoopHeader
          << "," << Backedge << "," << Exit << "," << Call << ","
          << Successor->FromFunName << ","
          << Twine::utohexstr(Successor->FromBb) << "," << Successor->ToFunName
          << "," << Twine::utohexstr(Successor->ToBb) << "," << NumLoads << ","
          << NumStores << "," << BasicBlockSize << "," << NumCalls << ","
          << NumCallsExit << "," << NumIndirectCalls << "," << NumCallsInvoke
          << "," << NumTailCalls;
}

void FeatureMiner::dumpFeatures(raw_ostream &Printer,
                                uint64_t FunctionAddress) {

  for (auto const &SBI : BranchesInfoSet) {
    auto &FalseSuccessor = SBI->FalseSuccessor;
    auto &TrueSuccessor = SBI->TrueSuccessor;

    if (!FalseSuccessor && !TrueSuccessor)
      continue;

    int16_t ProcedureType = (SBI->ProcedureType.hasValue())
                                ? static_cast<int16_t>(*(SBI->ProcedureType))
                                : -1;

    int16_t Direction =
        (SBI->Direction.hasValue()) ? static_cast<bool>(*(SBI->Direction)) : -1;

    int16_t LoopHeader = (SBI->LoopHeader.hasValue())
                             ? static_cast<bool>(*(SBI->LoopHeader))
                             : -1;

    int32_t Opcode =
        (SBI->Opcode.hasValue()) ? static_cast<int32_t>(*(SBI->Opcode)) : -1;

    int32_t CmpOpcode = (SBI->CmpOpcode.hasValue())
                            ? static_cast<int32_t>(*(SBI->CmpOpcode))
                            : -1;

    int64_t Count =
        (SBI->Count.hasValue()) ? static_cast<int64_t>(*(SBI->Count)) : -1;

    int64_t MissPredicted = (SBI->MissPredicted.hasValue())
                                ? static_cast<int64_t>(*(SBI->MissPredicted))
                                : -1;

    int64_t FallthroughCount =
        (SBI->FallthroughCount.hasValue())
            ? static_cast<int64_t>(*(SBI->FallthroughCount))
            : -1;

    int64_t FallthroughMissPredicted =
        (SBI->FallthroughMissPredicted.hasValue())
            ? static_cast<int64_t>(*(SBI->FallthroughMissPredicted))
            : -1;

    int64_t NumOuterLoops = (SBI->NumOuterLoops.hasValue())
                                ? static_cast<int64_t>(*(SBI->NumOuterLoops))
                                : -1;
    int64_t TotalLoops = (SBI->TotalLoops.hasValue())
                             ? static_cast<int64_t>(*(SBI->TotalLoops))
                             : -1;
    int64_t MaximumLoopDepth =
        (SBI->MaximumLoopDepth.hasValue())
            ? static_cast<int64_t>(*(SBI->MaximumLoopDepth))
            : -1;
    int64_t LoopDepth = (SBI->LoopDepth.hasValue())
                            ? static_cast<int64_t>(*(SBI->LoopDepth))
                            : -1;
    int64_t LoopNumExitEdges =
        (SBI->LoopNumExitEdges.hasValue())
            ? static_cast<int64_t>(*(SBI->LoopNumExitEdges))
            : -1;
    int64_t LoopNumExitBlocks =
        (SBI->LoopNumExitBlocks.hasValue())
            ? static_cast<int64_t>(*(SBI->LoopNumExitBlocks))
            : -1;
    int64_t LoopNumExitingBlocks =
        (SBI->LoopNumExitingBlocks.hasValue())
            ? static_cast<int64_t>(*(SBI->LoopNumExitingBlocks))
            : -1;
    int64_t LoopNumLatches = (SBI->LoopNumLatches.hasValue())
                                 ? static_cast<int64_t>(*(SBI->LoopNumLatches))
                                 : -1;
    int64_t LoopNumBlocks = (SBI->LoopNumBlocks.hasValue())
                                ? static_cast<int64_t>(*(SBI->LoopNumBlocks))
                                : -1;
    int64_t LoopNumBackEdges =
        (SBI->LoopNumBackEdges.hasValue())
            ? static_cast<int64_t>(*(SBI->LoopNumBackEdges))
            : -1;

    int64_t LocalExitingBlock =
        (SBI->LocalExitingBlock.hasValue())
            ? static_cast<bool>(*(SBI->LocalExitingBlock))
            : -1;

    int64_t LocalLatchBlock = (SBI->LocalLatchBlock.hasValue())
                                  ? static_cast<bool>(*(SBI->LocalLatchBlock))
                                  : -1;

    int64_t LocalLoopHeader = (SBI->LocalLoopHeader.hasValue())
                                  ? static_cast<bool>(*(SBI->LocalLoopHeader))
                                  : -1;

    int64_t Call =
        (SBI->Call.hasValue()) ? static_cast<bool>(*(SBI->Call)) : -1;

    int64_t DeltaTaken = (SBI->DeltaTaken.hasValue())
                             ? static_cast<int64_t>(*(SBI->DeltaTaken))
                             : -1;

    int64_t NumLoads = (SBI->NumLoads.hasValue())
                           ? static_cast<int64_t>(*(SBI->NumLoads))
                           : -1;

    int64_t NumStores = (SBI->NumStores.hasValue())
                            ? static_cast<int64_t>(*(SBI->NumStores))
                            : -1;

    int64_t BasicBlockSize = (SBI->BasicBlockSize.hasValue())
                                 ? static_cast<int64_t>(*(SBI->BasicBlockSize))
                                 : -1;

    int64_t NumBasicBlocks = (SBI->NumBasicBlocks.hasValue())
                                 ? static_cast<int64_t>(*(SBI->NumBasicBlocks))
                                 : -1;

    int64_t NumCalls = (SBI->NumCalls.hasValue())
                           ? static_cast<int64_t>(*(SBI->NumCalls))
                           : -1;

    int64_t NumSelfCalls = (SBI->NumSelfCalls.hasValue())
                               ? static_cast<int64_t>(*(SBI->NumSelfCalls))
                               : -1;

    int64_t NumCallsExit = (SBI->NumCallsExit.hasValue())
                               ? static_cast<int64_t>(*(SBI->NumCallsExit))
                               : -1;

    int64_t OperandRAType = (SBI->OperandRAType.hasValue())
                                ? static_cast<int32_t>(*(SBI->OperandRAType))
                                : -1;

    int64_t OperandRBType = (SBI->OperandRBType.hasValue())
                                ? static_cast<int32_t>(*(SBI->OperandRBType))
                                : -1;

    int64_t NumCallsInvoke = (SBI->NumCallsInvoke.hasValue())
                                 ? static_cast<int64_t>(*(SBI->NumCallsInvoke))
                                 : -1;

    int64_t NumIndirectCalls =
        (SBI->NumIndirectCalls.hasValue())
            ? static_cast<int64_t>(*(SBI->NumIndirectCalls))
            : -1;

    int64_t NumTailCalls = (SBI->NumTailCalls.hasValue())
                               ? static_cast<int64_t>(*(SBI->NumTailCalls))
                               : -1;

    Printer << SBI->Simple << "," << Opcode << "," << SBI->OpcodeStr << ","
            << Direction << "," << CmpOpcode << "," << SBI->CmpOpcodeStr << ","
            << LoopHeader << "," << ProcedureType << "," << Count << ","
            << MissPredicted << "," << FallthroughCount << ","
            << FallthroughMissPredicted << "," << NumOuterLoops << ","
            << NumCallsExit << "," << TotalLoops << "," << MaximumLoopDepth
            << "," << LoopDepth << "," << LoopNumExitEdges << ","
            << LoopNumExitBlocks << "," << LoopNumExitingBlocks << ","
            << LoopNumLatches << "," << LoopNumBlocks << "," << LoopNumBackEdges
            << "," << LocalExitingBlock << "," << LocalLatchBlock << ","
            << LocalLoopHeader << "," << Call << "," << DeltaTaken << ","
            << NumLoads << "," << NumStores << "," << NumCalls << ","
            << OperandRAType << "," << OperandRBType << "," << BasicBlockSize
            << "," << NumBasicBlocks << "," << NumCallsInvoke << ","
            << NumIndirectCalls << "," << NumTailCalls << "," << NumSelfCalls;

    if (FalseSuccessor && TrueSuccessor) {
      dumpSuccessorFeatures(Printer, TrueSuccessor);
      dumpSuccessorFeatures(Printer, FalseSuccessor);
    }

    Printer << "," << Twine::utohexstr(FunctionAddress) << "\n";
  }
  BranchesInfoSet.clear();
}

void FeatureMiner::runOnFunctions(BinaryContext &BC) {
  auto FileName = "features.csv";
  outs() << "BOLT-DEBUG: Dumping Binary's Features to " << FileName << "\n";
  std::error_code EC;
  raw_fd_ostream Printer(FileName, EC, sys::fs::F_None);

  if (EC) {
    errs() << "BOLT-WARNING: " << EC.message() << ", unable to open "
           << FileName << " for output.\n";
    return;
  }

  auto FILENAME = "profile_data_regular.fdata";
  raw_fd_ostream Printer2(FILENAME, EC, sys::fs::F_None);
  if (EC) {
    dbgs() << "BOLT-WARNING: " << EC.message() << ", unable to open"
           << " " << FILENAME << " for output.\n";
    return;
  }

  // CSV file header
  Printer << "FUN_TYPE,OPCODE,OPCODE_STR,DIRECTION,CMP_OPCODE,CMP_OPCODE_STR,"
             "LOOP_HEADER,PROCEDURE_TYPE,"
             "COUNT_TAKEN,MISS_TAKEN,COUNT_NOT_TAKEN,MISS_NOT_TAKEN,"
             "NUM_OUTER_LOOPS,NUM_CALLS_EXIT,TOTAL_LOOPS,MAXIMUM_LOOP_DEPTH,"
             "LOOP_DEPTH,LOOP_NUM_EXIT_EDGES,LOOP_NUM_EXIT_BLOCKS,"
             "LOOP_NUM_EXITING_BLOCKS,LOOP_NUM_LATCHES,LOOP_NUM_BLOCKS,"
             "LOOP_NUM_BAKCEDGES,LOCAL_EXITING_BLOCK,LOCAL_LATCH_BLOCK,"
             "LOCAL_LOOP_HEADER,CALL,DELTA_TAKEN,NUM_LOADS,NUM_STORES,"
             "NUM_CALLS,OPERAND_RA_TYPE,OPERAND_RB_TYPE,BASIC_BLOCK_SIZE,"
             "NUM_BASIC_BLOCKS,NUM_CALLS_INVOKE,NUM_INDIRECT_CALLS,"
             "NUM_TAIL_CALLS,NUM_SELF_CALLS,TS_DOMINATES,TS_POSTDOMINATES,"
             "TS_END_OPCODE,TS_END_OPCODE_STR,TS_LOOP_HEADER,TS_BACKEDGE,TS_"
             "EXIT,TS_CALL,"
             "TS_FROM_FUN_NAME,TS_FROM_BB,TS_TO_FUN_NAME,TS_TO_BB,TS_NUM_LOADS,"
             "TS_NUM_STORES,TS_BASIC_BLOCK_SIZE,TS_NUM_CALLS,TS_NUM_CALLS_EXIT,"
             "TS_NUM_INDIRECT_CALL,TS_NUM_CALLS_INVOKE,TS_NUM_TAIL_CALLS,"
             "FS_DOMINATES,FS_POSTDOMINATES,FS_END_OPCODE,FS_END_OPCODE_STR,FS_"
             "LOOP_HEADER,"
             "FS_BACKEDGE,FS_EXIT,FS_CALL,FS_FROM_FUN_NAME,FS_FROM_BB,"
             "FS_TO_FUN_NAME,FS_TO_BB,FS_NUM_LOADS,FS_NUM_STORES,"
             "FS_BASIC_BLOCK_SIZE,FS_NUM_CALLS,FS_NUM_CALLS_EXIT,"
             "FS_NUM_INDIRECT_CALL,FS_NUM_CALLS_INVOKE,FS_NUM_TAIL_CALLS,"
             "FUN_ENTRY_ADDRESS\n";

  auto &BFs = BC.getBinaryFunctions();
  BPI = std::make_unique<BranchPredictionInfo>();
  for (auto &BFI : BFs) {
    BinaryFunction &Function = BFI.second;

    if (Function.empty()) // || !Function.isSimple())
      continue;

    if (!Function.isLoopFree()) {
      const BinaryLoopInfo &LoopsInfo = Function.getLoopInfo();
      BPI->findLoopEdgesInfo(LoopsInfo);
    }
    extractFeatures(Function, BC);

    BPI->clear();

    dumpFeatures(Printer, Function.getAddress());

    dumpProfileData(Function, Printer2);
  }
}

void FeatureMiner::dumpProfileData(BinaryFunction &Function,
                                   raw_ostream &Printer) {

  BinaryContext &BC = Function.getBinaryContext();

  std::string FromFunName = Function.getPrintName();
  for (auto &BB : Function) {
    auto LastInst = BB.getLastNonPseudoInstr();

    for (auto &Inst : BB) {
      if (!BC.MIB->isCall(Inst) && !BC.MIB->isBranch(Inst) &&
          LastInst != (&Inst))
        continue;

      auto Offset = BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "Offset");

      if (!Offset)
        continue;

      uint64_t TakenFreqEdge = 0;
      auto FromBb = Offset.get();
      std::string ToFunName;
      uint32_t ToBb;

      if (BC.MIB->isCall(Inst)) {
        auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst);
        if (!CalleeSymbol)
          continue;

        ToFunName = CalleeSymbol->getName();
        ToBb = 0;

        if (BC.MIB->getConditionalTailCall(Inst)) {

          if (BC.MIB->hasAnnotation(Inst, "CTCTakenCount")) {
            auto CountAnnt =
                BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "CTCTakenCount");
            if (CountAnnt) {
              TakenFreqEdge = (*CountAnnt);
            }
          }
        } else {
          if (BC.MIB->hasAnnotation(Inst, "Count")) {
            auto CountAnnt =
                BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "Count");
            if (CountAnnt) {
              TakenFreqEdge = (*CountAnnt);
            }
          }
        }

        if (TakenFreqEdge > 0)
          Printer << "1 " << FromFunName << " " << Twine::utohexstr(FromBb)
                  << " 1 " << ToFunName << " " << Twine::utohexstr(ToBb) << " "
                  << 0 << " " << TakenFreqEdge << "\n";
      } else {
        for (BinaryBasicBlock *SuccBB : BB.successors()) {
          TakenFreqEdge = BB.getBranchInfo(*SuccBB).Count;
          BinaryFunction *ToFun = SuccBB->getFunction();
          ToFunName = ToFun->getPrintName();
          ToBb = SuccBB->getInputOffset();

          if (TakenFreqEdge > 0)
            Printer << "1 " << FromFunName << " " << Twine::utohexstr(FromBb)
                    << " 1 " << ToFunName << " " << Twine::utohexstr(ToBb)
                    << " " << 0 << " " << TakenFreqEdge << "\n";
        }
      }
    }
  }
}

} // namespace bolt
} // namespace llvm
