//===------ IslCodeGeneration.cpp - Code generate the Scops using ISL. ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The IslCodeGeneration pass takes a Scop created by ScopInfo and translates it
// back to LLVM-IR using the ISL code generator.
//
// The Scop describes the high level memory behaviour of a control flow region.
// Transformation passes can update the schedule (execution order) of statements
// in the Scop. ISL is used to generate an abstract syntax tree that reflects
// the updated execution order. This clast is used to create new LLVM-IR that is
// computationally equivalent to the original control flow region, but executes
// its code in the new execution order defined by the changed scattering.
//
//===----------------------------------------------------------------------===//
#include "polly/Config/config.h"
#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/LoopGenerators.h"
#include "polly/CodeGen/Utils.h"
#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "polly/TempScopInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "isl/union_map.h"
#include "isl/list.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/aff.h"

using namespace polly;
using namespace llvm;

#define DEBUG_TYPE "polly-codegen-isl"

class IslNodeBuilder {
public:
  IslNodeBuilder(PollyIRBuilder &Builder, LoopAnnotator &Annotator, Pass *P,
                 LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT)
      : Builder(Builder), Annotator(Annotator), ExprBuilder(Builder, IDToValue),
        P(P), LI(LI), SE(SE), DT(DT) {}

  /// @brief Add the mappings from array id's to array llvm::Value's.
  void addMemoryAccesses(Scop &S);
  void addParameters(__isl_take isl_set *Context);
  void create(__isl_take isl_ast_node *Node);
  IslExprBuilder &getExprBuilder() { return ExprBuilder; }

private:
  PollyIRBuilder &Builder;
  LoopAnnotator &Annotator;
  IslExprBuilder ExprBuilder;
  Pass *P;
  LoopInfo &LI;
  ScalarEvolution &SE;
  DominatorTree &DT;

  // This maps an isl_id* to the Value* it has in the generated program. For now
  // on, the only isl_ids that are stored here are the newly calculated loop
  // ivs.
  IslExprBuilder::IDToValueTy IDToValue;

  // Extract the upper bound of this loop
  //
  // The isl code generation can generate arbitrary expressions to check if the
  // upper bound of a loop is reached, but it provides an option to enforce
  // 'atomic' upper bounds. An 'atomic upper bound is always of the form
  // iv <= expr, where expr is an (arbitrary) expression not containing iv.
  //
  // This function extracts 'atomic' upper bounds. Polly, in general, requires
  // atomic upper bounds for the following reasons:
  //
  // 1. An atomic upper bound is loop invariant
  //
  //    It must not be calculated at each loop iteration and can often even be
  //    hoisted out further by the loop invariant code motion.
  //
  // 2. OpenMP needs a loop invarient upper bound to calculate the number
  //    of loop iterations.
  //
  // 3. With the existing code, upper bounds have been easier to implement.
  __isl_give isl_ast_expr *getUpperBound(__isl_keep isl_ast_node *For,
                                         CmpInst::Predicate &Predicate);

  unsigned getNumberOfIterations(__isl_keep isl_ast_node *For);

  void createFor(__isl_take isl_ast_node *For);
  void createForVector(__isl_take isl_ast_node *For, int VectorWidth);
  void createForSequential(__isl_take isl_ast_node *For);

  /// Generate LLVM-IR that computes the values of the original induction
  /// variables in function of the newly generated loop induction variables.
  ///
  /// Example:
  ///
  ///   // Original
  ///   for i
  ///     for j
  ///       S(i)
  ///
  ///   Schedule: [i,j] -> [i+j, j]
  ///
  ///   // New
  ///   for c0
  ///     for c1
  ///       S(c0 - c1, c1)
  ///
  /// Assuming the original code consists of two loops which are
  /// transformed according to a schedule [i,j] -> [c0=i+j,c1=j]. The resulting
  /// ast models the original statement as a call expression where each argument
  /// is an expression that computes the old induction variables from the new
  /// ones, ordered such that the first argument computes the value of induction
  /// variable that was outermost in the original code.
  ///
  /// @param Expr The call expression that represents the statement.
  /// @param Stmt The statement that is called.
  /// @param VMap The value map into which the mapping from the old induction
  ///             variable to the new one is inserted. This mapping is used
  ///             for the classical code generation (not scev-based) and
  ///             gives an explicit mapping from an original, materialized
  ///             induction variable. It consequently can only be expressed
  ///             if there was an explicit induction variable.
  /// @param LTS  The loop to SCEV map in which the mapping from the original
  ///             loop to a SCEV representing the new loop iv is added. This
  ///             mapping does not require an explicit induction variable.
  ///             Instead, we think in terms of an implicit induction variable
  ///             that counts the number of times a loop is executed. For each
  ///             original loop this count, expressed in function of the new
  ///             induction variables, is added to the LTS map.
  void createSubstitutions(__isl_take isl_ast_expr *Expr, ScopStmt *Stmt,
                           ValueMapT &VMap, LoopToScevMapT &LTS);
  void createSubstitutionsVector(__isl_take isl_ast_expr *Expr, ScopStmt *Stmt,
                                 VectorValueMapT &VMap,
                                 std::vector<LoopToScevMapT> &VLTS,
                                 std::vector<Value *> &IVS,
                                 __isl_take isl_id *IteratorID);
  void createIf(__isl_take isl_ast_node *If);
  void createUserVector(__isl_take isl_ast_node *User,
                        std::vector<Value *> &IVS,
                        __isl_take isl_id *IteratorID,
                        __isl_take isl_union_map *Schedule);
  void createUser(__isl_take isl_ast_node *User);
  void createBlock(__isl_take isl_ast_node *Block);
};

__isl_give isl_ast_expr *
IslNodeBuilder::getUpperBound(__isl_keep isl_ast_node *For,
                              ICmpInst::Predicate &Predicate) {
  isl_id *UBID, *IteratorID;
  isl_ast_expr *Cond, *Iterator, *UB, *Arg0;
  isl_ast_op_type Type;

  Cond = isl_ast_node_for_get_cond(For);
  Iterator = isl_ast_node_for_get_iterator(For);
  Type = isl_ast_expr_get_op_type(Cond);

  assert(isl_ast_expr_get_type(Cond) == isl_ast_expr_op &&
         "conditional expression is not an atomic upper bound");

  switch (Type) {
  case isl_ast_op_le:
    Predicate = ICmpInst::ICMP_SLE;
    break;
  case isl_ast_op_lt:
    Predicate = ICmpInst::ICMP_SLT;
    break;
  default:
    llvm_unreachable("Unexpected comparision type in loop conditon");
  }

  Arg0 = isl_ast_expr_get_op_arg(Cond, 0);

  assert(isl_ast_expr_get_type(Arg0) == isl_ast_expr_id &&
         "conditional expression is not an atomic upper bound");

  UBID = isl_ast_expr_get_id(Arg0);

  assert(isl_ast_expr_get_type(Iterator) == isl_ast_expr_id &&
         "Could not get the iterator");

  IteratorID = isl_ast_expr_get_id(Iterator);

  assert(UBID == IteratorID &&
         "conditional expression is not an atomic upper bound");

  UB = isl_ast_expr_get_op_arg(Cond, 1);

  isl_ast_expr_free(Cond);
  isl_ast_expr_free(Iterator);
  isl_ast_expr_free(Arg0);
  isl_id_free(IteratorID);
  isl_id_free(UBID);

  return UB;
}

unsigned IslNodeBuilder::getNumberOfIterations(__isl_keep isl_ast_node *For) {
  isl_union_map *Schedule = IslAstInfo::getSchedule(For);
  isl_set *LoopDomain = isl_set_from_union_set(isl_union_map_range(Schedule));
  int NumberOfIterations = polly::getNumberOfIterations(LoopDomain);
  if (NumberOfIterations == -1)
    return -1;
  return NumberOfIterations + 1;
}

void IslNodeBuilder::createUserVector(__isl_take isl_ast_node *User,
                                      std::vector<Value *> &IVS,
                                      __isl_take isl_id *IteratorID,
                                      __isl_take isl_union_map *Schedule) {
  isl_ast_expr *Expr = isl_ast_node_user_get_expr(User);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(StmtExpr);
  isl_ast_expr_free(StmtExpr);
  ScopStmt *Stmt = (ScopStmt *)isl_id_get_user(Id);
  VectorValueMapT VectorMap(IVS.size());
  std::vector<LoopToScevMapT> VLTS(IVS.size());

  isl_union_set *Domain = isl_union_set_from_set(Stmt->getDomain());
  Schedule = isl_union_map_intersect_domain(Schedule, Domain);
  isl_map *S = isl_map_from_union_map(Schedule);

  createSubstitutionsVector(Expr, Stmt, VectorMap, VLTS, IVS, IteratorID);
  VectorBlockGenerator::generate(Builder, *Stmt, VectorMap, VLTS, S, P, LI, SE);

  isl_map_free(S);
  isl_id_free(Id);
  isl_ast_node_free(User);
}

void IslNodeBuilder::createForVector(__isl_take isl_ast_node *For,
                                     int VectorWidth) {
  isl_ast_node *Body = isl_ast_node_for_get_body(For);
  isl_ast_expr *Init = isl_ast_node_for_get_init(For);
  isl_ast_expr *Inc = isl_ast_node_for_get_inc(For);
  isl_ast_expr *Iterator = isl_ast_node_for_get_iterator(For);
  isl_id *IteratorID = isl_ast_expr_get_id(Iterator);

  Value *ValueLB = ExprBuilder.create(Init);
  Value *ValueInc = ExprBuilder.create(Inc);

  Type *MaxType = ExprBuilder.getType(Iterator);
  MaxType = ExprBuilder.getWidestType(MaxType, ValueLB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueInc->getType());

  if (MaxType != ValueLB->getType())
    ValueLB = Builder.CreateSExt(ValueLB, MaxType);
  if (MaxType != ValueInc->getType())
    ValueInc = Builder.CreateSExt(ValueInc, MaxType);

  std::vector<Value *> IVS(VectorWidth);
  IVS[0] = ValueLB;

  for (int i = 1; i < VectorWidth; i++)
    IVS[i] = Builder.CreateAdd(IVS[i - 1], ValueInc, "p_vector_iv");

  isl_union_map *Schedule = IslAstInfo::getSchedule(For);
  assert(Schedule && "For statement annotation does not contain its schedule");

  IDToValue[IteratorID] = ValueLB;

  switch (isl_ast_node_get_type(Body)) {
  case isl_ast_node_user:
    createUserVector(Body, IVS, isl_id_copy(IteratorID),
                     isl_union_map_copy(Schedule));
    break;
  case isl_ast_node_block: {
    isl_ast_node_list *List = isl_ast_node_block_get_children(Body);

    for (int i = 0; i < isl_ast_node_list_n_ast_node(List); ++i)
      createUserVector(isl_ast_node_list_get_ast_node(List, i), IVS,
                       isl_id_copy(IteratorID), isl_union_map_copy(Schedule));

    isl_ast_node_free(Body);
    isl_ast_node_list_free(List);
    break;
  }
  default:
    isl_ast_node_dump(Body);
    llvm_unreachable("Unhandled isl_ast_node in vectorizer");
  }

  IDToValue.erase(IteratorID);
  isl_id_free(IteratorID);
  isl_union_map_free(Schedule);

  isl_ast_node_free(For);
  isl_ast_expr_free(Iterator);
}

void IslNodeBuilder::createForSequential(__isl_take isl_ast_node *For) {
  isl_ast_node *Body;
  isl_ast_expr *Init, *Inc, *Iterator, *UB;
  isl_id *IteratorID;
  Value *ValueLB, *ValueUB, *ValueInc;
  Type *MaxType;
  BasicBlock *ExitBlock;
  Value *IV;
  CmpInst::Predicate Predicate;
  bool Parallel;

  Parallel = IslAstInfo::isInnermostParallel(For) &&
             !IslAstInfo::isReductionParallel(For);

  Body = isl_ast_node_for_get_body(For);

  // isl_ast_node_for_is_degenerate(For)
  //
  // TODO: For degenerated loops we could generate a plain assignment.
  //       However, for now we just reuse the logic for normal loops, which will
  //       create a loop with a single iteration.

  Init = isl_ast_node_for_get_init(For);
  Inc = isl_ast_node_for_get_inc(For);
  Iterator = isl_ast_node_for_get_iterator(For);
  IteratorID = isl_ast_expr_get_id(Iterator);
  UB = getUpperBound(For, Predicate);

  ValueLB = ExprBuilder.create(Init);
  ValueUB = ExprBuilder.create(UB);
  ValueInc = ExprBuilder.create(Inc);

  MaxType = ExprBuilder.getType(Iterator);
  MaxType = ExprBuilder.getWidestType(MaxType, ValueLB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueUB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueInc->getType());

  if (MaxType != ValueLB->getType())
    ValueLB = Builder.CreateSExt(ValueLB, MaxType);
  if (MaxType != ValueUB->getType())
    ValueUB = Builder.CreateSExt(ValueUB, MaxType);
  if (MaxType != ValueInc->getType())
    ValueInc = Builder.CreateSExt(ValueInc, MaxType);

  // If we can show that LB <Predicate> UB holds at least once, we can
  // omit the GuardBB in front of the loop.
  bool UseGuardBB =
      !SE.isKnownPredicate(Predicate, SE.getSCEV(ValueLB), SE.getSCEV(ValueUB));
  IV = createLoop(ValueLB, ValueUB, ValueInc, Builder, P, LI, DT, ExitBlock,
                  Predicate, &Annotator, Parallel, UseGuardBB);
  IDToValue[IteratorID] = IV;

  create(Body);

  Annotator.End();

  IDToValue.erase(IteratorID);

  Builder.SetInsertPoint(ExitBlock->begin());

  isl_ast_node_free(For);
  isl_ast_expr_free(Iterator);
  isl_id_free(IteratorID);
}

void IslNodeBuilder::createFor(__isl_take isl_ast_node *For) {
  bool Vector = PollyVectorizerChoice != VECTORIZER_NONE;

  if (Vector && IslAstInfo::isInnermostParallel(For) &&
      !IslAstInfo::isReductionParallel(For)) {
    int VectorWidth = getNumberOfIterations(For);
    if (1 < VectorWidth && VectorWidth <= 16) {
      createForVector(For, VectorWidth);
      return;
    }
  }
  createForSequential(For);
}

void IslNodeBuilder::createIf(__isl_take isl_ast_node *If) {
  isl_ast_expr *Cond = isl_ast_node_if_get_cond(If);

  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *CondBB =
      SplitBlock(Builder.GetInsertBlock(), Builder.GetInsertPoint(), P);
  CondBB->setName("polly.cond");
  BasicBlock *MergeBB = SplitBlock(CondBB, CondBB->begin(), P);
  MergeBB->setName("polly.merge");
  BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);
  BasicBlock *ElseBB = BasicBlock::Create(Context, "polly.else", F);

  DT.addNewBlock(ThenBB, CondBB);
  DT.addNewBlock(ElseBB, CondBB);
  DT.changeImmediateDominator(MergeBB, CondBB);

  Loop *L = LI.getLoopFor(CondBB);
  if (L) {
    L->addBasicBlockToLoop(ThenBB, LI.getBase());
    L->addBasicBlockToLoop(ElseBB, LI.getBase());
  }

  CondBB->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(CondBB);
  Value *Predicate = ExprBuilder.create(Cond);
  Builder.CreateCondBr(Predicate, ThenBB, ElseBB);
  Builder.SetInsertPoint(ThenBB);
  Builder.CreateBr(MergeBB);
  Builder.SetInsertPoint(ElseBB);
  Builder.CreateBr(MergeBB);
  Builder.SetInsertPoint(ThenBB->begin());

  create(isl_ast_node_if_get_then(If));

  Builder.SetInsertPoint(ElseBB->begin());

  if (isl_ast_node_if_has_else(If))
    create(isl_ast_node_if_get_else(If));

  Builder.SetInsertPoint(MergeBB->begin());

  isl_ast_node_free(If);
}

void IslNodeBuilder::createSubstitutions(isl_ast_expr *Expr, ScopStmt *Stmt,
                                         ValueMapT &VMap, LoopToScevMapT &LTS) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expression of type 'op' expected");
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_call &&
         "Opertation of type 'call' expected");
  for (int i = 0; i < isl_ast_expr_get_op_n_arg(Expr) - 1; ++i) {
    isl_ast_expr *SubExpr;
    Value *V;

    SubExpr = isl_ast_expr_get_op_arg(Expr, i + 1);
    V = ExprBuilder.create(SubExpr);
    ScalarEvolution *SE = Stmt->getParent()->getSE();
    LTS[Stmt->getLoopForDimension(i)] = SE->getUnknown(V);

    // CreateIntCast can introduce trunc expressions. This is correct, as the
    // result will always fit into the type of the original induction variable
    // (because we calculate a value of the original induction variable).
    const Value *OldIV = Stmt->getInductionVariableForDimension(i);
    if (OldIV) {
      V = Builder.CreateIntCast(V, OldIV->getType(), true);
      VMap[OldIV] = V;
    }
  }

  isl_ast_expr_free(Expr);
}

void IslNodeBuilder::createSubstitutionsVector(
    __isl_take isl_ast_expr *Expr, ScopStmt *Stmt, VectorValueMapT &VMap,
    std::vector<LoopToScevMapT> &VLTS, std::vector<Value *> &IVS,
    __isl_take isl_id *IteratorID) {
  int i = 0;

  Value *OldValue = IDToValue[IteratorID];
  for (Value *IV : IVS) {
    IDToValue[IteratorID] = IV;
    createSubstitutions(isl_ast_expr_copy(Expr), Stmt, VMap[i], VLTS[i]);
    i++;
  }

  IDToValue[IteratorID] = OldValue;
  isl_id_free(IteratorID);
  isl_ast_expr_free(Expr);
}

void IslNodeBuilder::createUser(__isl_take isl_ast_node *User) {
  ValueMapT VMap;
  LoopToScevMapT LTS;
  isl_id *Id;
  ScopStmt *Stmt;

  isl_ast_expr *Expr = isl_ast_node_user_get_expr(User);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  Id = isl_ast_expr_get_id(StmtExpr);
  isl_ast_expr_free(StmtExpr);

  Stmt = (ScopStmt *)isl_id_get_user(Id);

  createSubstitutions(Expr, Stmt, VMap, LTS);
  BlockGenerator::generate(Builder, *Stmt, VMap, LTS, P, LI, SE,
                           IslAstInfo::getBuild(User), &ExprBuilder);

  isl_ast_node_free(User);
  isl_id_free(Id);
}

void IslNodeBuilder::createBlock(__isl_take isl_ast_node *Block) {
  isl_ast_node_list *List = isl_ast_node_block_get_children(Block);

  for (int i = 0; i < isl_ast_node_list_n_ast_node(List); ++i)
    create(isl_ast_node_list_get_ast_node(List, i));

  isl_ast_node_free(Block);
  isl_ast_node_list_free(List);
}

void IslNodeBuilder::create(__isl_take isl_ast_node *Node) {
  switch (isl_ast_node_get_type(Node)) {
  case isl_ast_node_error:
    llvm_unreachable("code  generation error");
  case isl_ast_node_for:
    createFor(Node);
    return;
  case isl_ast_node_if:
    createIf(Node);
    return;
  case isl_ast_node_user:
    createUser(Node);
    return;
  case isl_ast_node_block:
    createBlock(Node);
    return;
  }

  llvm_unreachable("Unknown isl_ast_node type");
}

void IslNodeBuilder::addParameters(__isl_take isl_set *Context) {
  SCEVExpander Rewriter(SE, "polly");

  for (unsigned i = 0; i < isl_set_dim(Context, isl_dim_param); ++i) {
    isl_id *Id;
    const SCEV *Scev;
    IntegerType *T;
    Instruction *InsertLocation;

    Id = isl_set_get_dim_id(Context, isl_dim_param, i);
    Scev = (const SCEV *)isl_id_get_user(Id);
    T = dyn_cast<IntegerType>(Scev->getType());
    InsertLocation = --(Builder.GetInsertBlock()->end());
    Value *V = Rewriter.expandCodeFor(Scev, T, InsertLocation);
    IDToValue[Id] = V;

    isl_id_free(Id);
  }

  isl_set_free(Context);
}

void IslNodeBuilder::addMemoryAccesses(Scop &S) {
  for (ScopStmt *Stmt : S)
    for (MemoryAccess *MA : *Stmt) {
      isl_id *Id = MA->getArrayId();
      IDToValue[Id] = MA->getBaseAddr();
      isl_id_free(Id);
    }
}

namespace {
class IslCodeGeneration : public ScopPass {
public:
  static char ID;

  IslCodeGeneration() : ScopPass(ID) {}

  /// @name The analysis passes we need to generate code.
  ///
  ///{
  LoopInfo *LI;
  IslAstInfo *AI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  ///}

  /// @brief The loop annotator to generate llvm.loop metadata.
  LoopAnnotator Annotator;

  /// @brief Build the runtime condition.
  ///
  /// Build the condition that evaluates at run-time to true iff all
  /// assumptions taken for the SCoP hold, and to false otherwise.
  ///
  /// @return A value evaluating to true/false if execution is save/unsafe.
  Value *buildRTC(PollyIRBuilder &Builder, IslExprBuilder &ExprBuilder) {
    Builder.SetInsertPoint(Builder.GetInsertBlock()->getTerminator());
    Value *RTC = ExprBuilder.create(AI->getRunCondition());
    return Builder.CreateIsNotNull(RTC);
  }

  bool runOnScop(Scop &S) {
    LI = &getAnalysis<LoopInfo>();
    AI = &getAnalysis<IslAstInfo>();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    SE = &getAnalysis<ScalarEvolution>();

    assert(!S.getRegion().isTopLevelRegion() &&
           "Top level regions are not supported");

    BasicBlock *EnteringBB = simplifyRegion(&S, this);
    PollyIRBuilder Builder = createPollyIRBuilder(EnteringBB, Annotator);

    IslNodeBuilder NodeBuilder(Builder, Annotator, this, *LI, *SE, *DT);
    NodeBuilder.addMemoryAccesses(S);
    NodeBuilder.addParameters(S.getContext());

    Value *RTC = buildRTC(Builder, NodeBuilder.getExprBuilder());
    BasicBlock *StartBlock = executeScopConditionally(S, this, RTC);
    Builder.SetInsertPoint(StartBlock->begin());

    NodeBuilder.create(AI->getAst());
    return true;
  }

  virtual void printScop(raw_ostream &OS) const {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<IslAstInfo>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfo>();
    AU.addRequired<LoopInfo>();

    AU.addPreserved<Dependences>();

    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<IslAstInfo>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolution>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
    AU.addPreserved<RegionInfoPass>();
    AU.addPreserved<TempScopInfo>();
    AU.addPreserved<ScopInfo>();
    AU.addPreservedID(IndependentBlocksID);
  }
};
}

char IslCodeGeneration::ID = 1;

Pass *polly::createIslCodeGenerationPass() { return new IslCodeGeneration(); }

INITIALIZE_PASS_BEGIN(IslCodeGeneration, "polly-codegen-isl",
                      "Polly - Create LLVM-IR from SCoPs", false, false);
INITIALIZE_PASS_DEPENDENCY(Dependences);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfo);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(IslCodeGeneration, "polly-codegen-isl",
                    "Polly - Create LLVM-IR from SCoPs", false, false)
