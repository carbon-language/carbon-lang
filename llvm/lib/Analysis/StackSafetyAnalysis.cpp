//===- StackSafetyAnalysis.cpp - Stack memory safety analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "stack-safety"

namespace {

/// Rewrite an SCEV expression for a memory access address to an expression that
/// represents offset from the given alloca.
class AllocaOffsetRewriter : public SCEVRewriteVisitor<AllocaOffsetRewriter> {
  const Value *AllocaPtr;

public:
  AllocaOffsetRewriter(ScalarEvolution &SE, const Value *AllocaPtr)
      : SCEVRewriteVisitor(SE), AllocaPtr(AllocaPtr) {}

  const SCEV *visit(const SCEV *Expr) {
    // Only re-write the expression if the alloca is used in an addition
    // expression (it can be used in other types of expressions if it's cast to
    // an int and passed as an argument.)
    if (!isa<SCEVAddRecExpr>(Expr) && !isa<SCEVAddExpr>(Expr) &&
        !isa<SCEVUnknown>(Expr))
      return Expr;
    return SCEVRewriteVisitor<AllocaOffsetRewriter>::visit(Expr);
  }

  const SCEV *visitUnknown(const SCEVUnknown *Expr) {
    // FIXME: look through one or several levels of definitions?
    // This can be inttoptr(AllocaPtr) and SCEV would not unwrap
    // it for us.
    if (Expr->getValue() == AllocaPtr)
      return SE.getZero(Expr->getType());
    return Expr;
  }
};

/// Describes use of address in as a function call argument.
struct PassAsArgInfo {
  /// Function being called.
  const GlobalValue *Callee = nullptr;
  /// Index of argument which pass address.
  size_t ParamNo = 0;
  // Offset range of address from base address (alloca or calling function
  // argument).
  // Range should never set to empty-set, that is an invalid access range
  // that can cause empty-set to be propagated with ConstantRange::add
  ConstantRange Offset;
  PassAsArgInfo(const GlobalValue *Callee, size_t ParamNo, ConstantRange Offset)
      : Callee(Callee), ParamNo(ParamNo), Offset(Offset) {}

  StringRef getName() const { return Callee->getName(); }
};

raw_ostream &operator<<(raw_ostream &OS, const PassAsArgInfo &P) {
  return OS << "@" << P.getName() << "(arg" << P.ParamNo << ", " << P.Offset
            << ")";
}

/// Describe uses of address (alloca or parameter) inside of the function.
struct UseInfo {
  // Access range if the address (alloca or parameters).
  // It is allowed to be empty-set when there are no known accesses.
  ConstantRange Range;

  // List of calls which pass address as an argument.
  SmallVector<PassAsArgInfo, 4> Calls;

  explicit UseInfo(unsigned PointerSize) : Range{PointerSize, false} {}

  void updateRange(ConstantRange R) { Range = Range.unionWith(R); }
};

raw_ostream &operator<<(raw_ostream &OS, const UseInfo &U) {
  OS << U.Range;
  for (auto &Call : U.Calls)
    OS << ", " << Call;
  return OS;
}

struct AllocaInfo {
  const AllocaInst *AI = nullptr;
  uint64_t Size = 0;
  UseInfo Use;

  AllocaInfo(unsigned PointerSize, const AllocaInst *AI, uint64_t Size)
      : AI(AI), Size(Size), Use(PointerSize) {}

  StringRef getName() const { return AI->getName(); }
};

raw_ostream &operator<<(raw_ostream &OS, const AllocaInfo &A) {
  return OS << A.getName() << "[" << A.Size << "]: " << A.Use;
}

struct ParamInfo {
  const Argument *Arg = nullptr;
  UseInfo Use;

  explicit ParamInfo(unsigned PointerSize, const Argument *Arg)
      : Arg(Arg), Use(PointerSize) {}

  StringRef getName() const { return Arg ? Arg->getName() : "<N/A>"; }
};

raw_ostream &operator<<(raw_ostream &OS, const ParamInfo &P) {
  return OS << P.getName() << "[]: " << P.Use;
}

/// Calculate the allocation size of a given alloca. Returns 0 if the
/// size can not be statically determined.
uint64_t getStaticAllocaAllocationSize(const AllocaInst *AI) {
  const DataLayout &DL = AI->getModule()->getDataLayout();
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

} // end anonymous namespace

/// Describes uses of allocas and parameters inside of a single function.
struct StackSafetyInfo::FunctionInfo {
  // May be a Function or a GlobalAlias
  const GlobalValue *GV = nullptr;
  // Informations about allocas uses.
  SmallVector<AllocaInfo, 4> Allocas;
  // Informations about parameters uses.
  SmallVector<ParamInfo, 4> Params;
  // TODO: describe return value as depending on one or more of its arguments.

  FunctionInfo(const StackSafetyInfo &SSI) : FunctionInfo(*SSI.Info) {}

  explicit FunctionInfo(const Function *F) : GV(F){};

  FunctionInfo(FunctionInfo &&) = default;

  bool IsDSOLocal() const { return GV->isDSOLocal(); };

  bool IsInterposable() const { return GV->isInterposable(); };

  StringRef getName() const { return GV->getName(); }

  void print(raw_ostream &O) const {
    O << "  @" << getName() << (IsDSOLocal() ? "" : " dso_preemptable")
      << (IsInterposable() ? " interposable" : "") << "\n";
    O << "    args uses:\n";
    for (auto &P : Params)
      O << "      " << P << "\n";
    O << "    allocas uses:\n";
    for (auto &AS : Allocas)
      O << "      " << AS << "\n";
  }

private:
  FunctionInfo(const FunctionInfo &) = default;
};

namespace {

class StackSafetyLocalAnalysis {
  const Function &F;
  const DataLayout &DL;
  ScalarEvolution &SE;
  unsigned PointerSize = 0;

  const ConstantRange UnknownRange;

  ConstantRange offsetFromAlloca(Value *Addr, const Value *AllocaPtr);
  ConstantRange getAccessRange(Value *Addr, const Value *AllocaPtr,
                               uint64_t AccessSize);
  ConstantRange getMemIntrinsicAccessRange(const MemIntrinsic *MI, const Use &U,
                                           const Value *AllocaPtr);

  bool analyzeAllUses(const Value *Ptr, UseInfo &AS);

  ConstantRange getRange(uint64_t Lower, uint64_t Upper) const {
    return ConstantRange(APInt(PointerSize, Lower), APInt(PointerSize, Upper));
  }

public:
  StackSafetyLocalAnalysis(const Function &F, ScalarEvolution &SE)
      : F(F), DL(F.getParent()->getDataLayout()), SE(SE),
        PointerSize(DL.getPointerSizeInBits()),
        UnknownRange(PointerSize, true) {}

  // Run the transformation on the associated function.
  StackSafetyInfo run();
};

ConstantRange
StackSafetyLocalAnalysis::offsetFromAlloca(Value *Addr,
                                           const Value *AllocaPtr) {
  if (!SE.isSCEVable(Addr->getType()))
    return UnknownRange;

  AllocaOffsetRewriter Rewriter(SE, AllocaPtr);
  const SCEV *Expr = Rewriter.visit(SE.getSCEV(Addr));
  ConstantRange Offset = SE.getUnsignedRange(Expr).zextOrTrunc(PointerSize);
  assert(!Offset.isEmptySet());
  return Offset;
}

ConstantRange StackSafetyLocalAnalysis::getAccessRange(Value *Addr,
                                                       const Value *AllocaPtr,
                                                       uint64_t AccessSize) {
  if (!SE.isSCEVable(Addr->getType()))
    return UnknownRange;

  AllocaOffsetRewriter Rewriter(SE, AllocaPtr);
  const SCEV *Expr = Rewriter.visit(SE.getSCEV(Addr));

  ConstantRange AccessStartRange =
      SE.getUnsignedRange(Expr).zextOrTrunc(PointerSize);
  ConstantRange SizeRange = getRange(0, AccessSize);
  ConstantRange AccessRange = AccessStartRange.add(SizeRange);
  assert(!AccessRange.isEmptySet());
  return AccessRange;
}

ConstantRange StackSafetyLocalAnalysis::getMemIntrinsicAccessRange(
    const MemIntrinsic *MI, const Use &U, const Value *AllocaPtr) {
  if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
    if (MTI->getRawSource() != U && MTI->getRawDest() != U)
      return getRange(0, 1);
  } else {
    if (MI->getRawDest() != U)
      return getRange(0, 1);
  }
  const auto *Len = dyn_cast<ConstantInt>(MI->getLength());
  // Non-constant size => unsafe. FIXME: try SCEV getRange.
  if (!Len)
    return UnknownRange;
  ConstantRange AccessRange = getAccessRange(U, AllocaPtr, Len->getZExtValue());
  return AccessRange;
}

/// The function analyzes all local uses of Ptr (alloca or argument) and
/// calculates local access range and all function calls where it was used.
bool StackSafetyLocalAnalysis::analyzeAllUses(const Value *Ptr, UseInfo &US) {
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 8> WorkList;
  WorkList.push_back(Ptr);

  // A DFS search through all uses of the alloca in bitcasts/PHI/GEPs/etc.
  while (!WorkList.empty()) {
    const Value *V = WorkList.pop_back_val();
    for (const Use &UI : V->uses()) {
      auto I = cast<const Instruction>(UI.getUser());
      assert(V == UI.get());

      switch (I->getOpcode()) {
      case Instruction::Load: {
        US.updateRange(
            getAccessRange(UI, Ptr, DL.getTypeStoreSize(I->getType())));
        break;
      }

      case Instruction::VAArg:
        // "va-arg" from a pointer is safe.
        break;
      case Instruction::Store: {
        if (V == I->getOperand(0)) {
          // Stored the pointer - conservatively assume it may be unsafe.
          US.updateRange(UnknownRange);
          return false;
        }
        US.updateRange(getAccessRange(
            UI, Ptr, DL.getTypeStoreSize(I->getOperand(0)->getType())));
        break;
      }

      case Instruction::Ret:
        // Information leak.
        // FIXME: Process parameters correctly. This is a leak only if we return
        // alloca.
        US.updateRange(UnknownRange);
        return false;

      case Instruction::Call:
      case Instruction::Invoke: {
        ImmutableCallSite CS(I);

        if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end)
            break;
        }

        if (const MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I)) {
          US.updateRange(getMemIntrinsicAccessRange(MI, UI, Ptr));
          break;
        }

        // FIXME: consult devirt?
        // Do not follow aliases, otherwise we could inadvertently follow
        // dso_preemptable aliases or aliases with interposable linkage.
        const GlobalValue *Callee = dyn_cast<GlobalValue>(
            CS.getCalledValue()->stripPointerCastsNoFollowAliases());
        if (!Callee) {
          US.updateRange(UnknownRange);
          return false;
        }

        assert(isa<Function>(Callee) || isa<GlobalAlias>(Callee));

        ImmutableCallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
        for (ImmutableCallSite::arg_iterator A = B; A != E; ++A) {
          if (A->get() == V) {
            ConstantRange Offset = offsetFromAlloca(UI, Ptr);
            US.Calls.emplace_back(Callee, A - B, Offset);
          }
        }

        break;
      }

      default:
        if (Visited.insert(I).second)
          WorkList.push_back(cast<const Instruction>(I));
      }
    }
  }

  return true;
}

StackSafetyInfo StackSafetyLocalAnalysis::run() {
  StackSafetyInfo::FunctionInfo Info(&F);
  assert(!F.isDeclaration() &&
         "Can't run StackSafety on a function declaration");

  LLVM_DEBUG(dbgs() << "[StackSafety] " << F.getName() << "\n");

  for (auto &I : instructions(F)) {
    if (auto AI = dyn_cast<AllocaInst>(&I)) {
      Info.Allocas.emplace_back(PointerSize, AI,
                                getStaticAllocaAllocationSize(AI));
      AllocaInfo &AS = Info.Allocas.back();
      analyzeAllUses(AI, AS.Use);
    }
  }

  for (const Argument &A : make_range(F.arg_begin(), F.arg_end())) {
    Info.Params.emplace_back(PointerSize, &A);
    ParamInfo &PS = Info.Params.back();
    analyzeAllUses(&A, PS.Use);
  }

  LLVM_DEBUG(dbgs() << "[StackSafety] done\n");
  LLVM_DEBUG(Info.print(dbgs()));
  return StackSafetyInfo(std::move(Info));
}

void print(const StackSafetyGlobalInfo &SSI, raw_ostream &O, const Module &M) {
  O << "Not Implemented\n";
}

} // end anonymous namespace

StackSafetyInfo::StackSafetyInfo() = default;
StackSafetyInfo::StackSafetyInfo(StackSafetyInfo &&) = default;
StackSafetyInfo &StackSafetyInfo::operator=(StackSafetyInfo &&) = default;

StackSafetyInfo::StackSafetyInfo(FunctionInfo &&Info)
    : Info(new FunctionInfo(std::move(Info))) {}

StackSafetyInfo::~StackSafetyInfo() = default;

void StackSafetyInfo::print(raw_ostream &O) const { Info->print(O); }

AnalysisKey StackSafetyAnalysis::Key;

StackSafetyInfo StackSafetyAnalysis::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  StackSafetyLocalAnalysis SSLA(F, AM.getResult<ScalarEvolutionAnalysis>(F));
  return SSLA.run();
}

PreservedAnalyses StackSafetyPrinterPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  OS << "'Stack Safety Local Analysis' for function '" << F.getName() << "'\n";
  AM.getResult<StackSafetyAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

char StackSafetyInfoWrapperPass::ID = 0;

StackSafetyInfoWrapperPass::StackSafetyInfoWrapperPass() : FunctionPass(ID) {
  initializeStackSafetyInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

void StackSafetyInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

void StackSafetyInfoWrapperPass::print(raw_ostream &O, const Module *M) const {
  SSI.print(O);
}

bool StackSafetyInfoWrapperPass::runOnFunction(Function &F) {
  StackSafetyLocalAnalysis SSLA(
      F, getAnalysis<ScalarEvolutionWrapperPass>().getSE());
  SSI = StackSafetyInfo(SSLA.run());
  return false;
}

AnalysisKey StackSafetyGlobalAnalysis::Key;

StackSafetyGlobalInfo
StackSafetyGlobalAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return {};
}

PreservedAnalyses StackSafetyGlobalPrinterPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  OS << "'Stack Safety Analysis' for module '" << M.getName() << "'\n";
  print(AM.getResult<StackSafetyGlobalAnalysis>(M), OS, M);
  return PreservedAnalyses::all();
}

char StackSafetyGlobalInfoWrapperPass::ID = 0;

StackSafetyGlobalInfoWrapperPass::StackSafetyGlobalInfoWrapperPass()
    : ModulePass(ID) {
  initializeStackSafetyGlobalInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

void StackSafetyGlobalInfoWrapperPass::print(raw_ostream &O,
                                             const Module *M) const {
  ::print(SSI, O, *M);
}

void StackSafetyGlobalInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<StackSafetyInfoWrapperPass>();
}

bool StackSafetyGlobalInfoWrapperPass::runOnModule(Module &M) { return false; }

static const char LocalPassArg[] = "stack-safety-local";
static const char LocalPassName[] = "Stack Safety Local Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                      false, true)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                    false, true)

static const char GlobalPassName[] = "Stack Safety Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                      GlobalPassName, false, false)
INITIALIZE_PASS_DEPENDENCY(StackSafetyInfoWrapperPass)
INITIALIZE_PASS_END(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                    GlobalPassName, false, false)
