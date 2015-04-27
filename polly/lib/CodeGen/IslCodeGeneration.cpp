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
// its code in the new execution order defined by the changed schedule.
//
//===----------------------------------------------------------------------===//
#include "polly/Config/config.h"
#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/LoopGenerators.h"
#include "polly/CodeGen/Utils.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/TempScopInfo.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Verifier.h"
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
  IslNodeBuilder(PollyIRBuilder &Builder, ScopAnnotator &Annotator, Pass *P,
                 const DataLayout &DL, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, Scop &S)
      : S(S), Builder(Builder), Annotator(Annotator), Rewriter(SE, DL, "polly"),
        ExprBuilder(Builder, IDToValue, Rewriter, DT, LI),
        BlockGen(Builder, LI, SE, DT, &ExprBuilder), RegionGen(BlockGen), P(P),
        DL(DL), LI(LI), SE(SE), DT(DT) {}

  ~IslNodeBuilder() {}

  void addParameters(__isl_take isl_set *Context);
  void create(__isl_take isl_ast_node *Node);
  IslExprBuilder &getExprBuilder() { return ExprBuilder; }

private:
  Scop &S;
  PollyIRBuilder &Builder;
  ScopAnnotator &Annotator;

  /// @brief A SCEVExpander to create llvm values from SCEVs.
  SCEVExpander Rewriter;

  IslExprBuilder ExprBuilder;
  BlockGenerator BlockGen;

  /// @brief Generator for region statements.
  RegionGenerator RegionGen;

  Pass *const P;
  const DataLayout &DL;
  LoopInfo &LI;
  ScalarEvolution &SE;
  DominatorTree &DT;

  /// @brief The current iteration of out-of-scop loops
  ///
  /// This map provides for a given loop a llvm::Value that contains the current
  /// loop iteration.
  LoopToScevMapT OutsideLoopIterations;

  // This maps an isl_id* to the Value* it has in the generated program. For now
  // on, the only isl_ids that are stored here are the newly calculated loop
  // ivs.
  IslExprBuilder::IDToValueTy IDToValue;

  /// Generate code for a given SCEV*
  ///
  /// This function generates code for a given SCEV expression. It generated
  /// code is emmitted at the end of the basic block our Builder currently
  /// points to and the resulting value is returned.
  ///
  /// @param Expr The expression to code generate.
  Value *generateSCEV(const SCEV *Expr);

  /// A set of Value -> Value remappings to apply when generating new code.
  ///
  /// When generating new code for a ScopStmt this map is used to map certain
  /// llvm::Values to new llvm::Values.
  ValueMapT ValueMap;

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

  /// Compute the values and loops referenced in this subtree.
  ///
  /// This function looks at all ScopStmts scheduled below the provided For node
  /// and finds the llvm::Value[s] and llvm::Loops[s] which are referenced but
  /// not locally defined.
  ///
  /// Values that can be synthesized or that are available as globals are
  /// considered locally defined.
  ///
  /// Loops that contain the scop or that are part of the scop are considered
  /// locally defined. Loops that are before the scop, but do not contain the
  /// scop itself are considered not locally defined.
  ///
  /// @param For    The node defining the subtree.
  /// @param Values A vector that will be filled with the Values referenced in
  ///               this subtree.
  /// @param Loops  A vector that will be filled with the Loops referenced in
  ///               this subtree.
  void getReferencesInSubtree(__isl_keep isl_ast_node *For,
                              SetVector<Value *> &Values,
                              SetVector<const Loop *> &Loops);

  /// Change the llvm::Value(s) used for code generation.
  ///
  /// When generating code certain values (e.g., references to induction
  /// variables or array base pointers) in the original code may be replaced by
  /// new values. This function allows to (partially) update the set of values
  /// used. A typical use case for this function is the case when we continue
  /// code generation in a subfunction/kernel function and need to explicitly
  /// pass down certain values.
  ///
  /// @param NewValues A map that maps certain llvm::Values to new llvm::Values.
  void updateValues(ParallelLoopGenerator::ValueToValueMapTy &NewValues);

  void createFor(__isl_take isl_ast_node *For);
  void createForVector(__isl_take isl_ast_node *For, int VectorWidth);
  void createForSequential(__isl_take isl_ast_node *For);

  /// Create LLVM-IR that executes a for node thread parallel.
  ///
  /// @param For The FOR isl_ast_node for which code is generated.
  void createForParallel(__isl_take isl_ast_node *For);

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
  isl_ast_expr_get_type(Cond);
  assert(isl_ast_expr_get_type(Cond) == isl_ast_expr_op &&
         "conditional expression is not an atomic upper bound");

  Type = isl_ast_expr_get_op_type(Cond);

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
  int Dim = isl_set_dim(LoopDomain, isl_dim_set);

  // Calculate a map similar to the identity map, but with the last input
  // and output dimension not related.
  //  [i0, i1, i2, i3] -> [i0, i1, i2, o0]
  isl_space *Space = isl_set_get_space(LoopDomain);
  Space = isl_space_drop_dims(Space, isl_dim_out, Dim - 1, 1);
  Space = isl_space_map_from_set(Space);
  isl_map *Identity = isl_map_identity(Space);
  Identity = isl_map_add_dims(Identity, isl_dim_in, 1);
  Identity = isl_map_add_dims(Identity, isl_dim_out, 1);

  LoopDomain = isl_set_reset_tuple_id(LoopDomain);

  isl_map *Map = isl_map_from_domain_and_range(isl_set_copy(LoopDomain),
                                               isl_set_copy(LoopDomain));
  isl_set_free(LoopDomain);
  Map = isl_map_intersect(Map, Identity);

  isl_map *LexMax = isl_map_lexmax(isl_map_copy(Map));
  isl_map *LexMin = isl_map_lexmin(Map);
  isl_map *Sub = isl_map_sum(LexMax, isl_map_neg(LexMin));

  isl_set *Elements = isl_map_range(Sub);

  if (!isl_set_is_singleton(Elements)) {
    isl_set_free(Elements);
    return -1;
  }

  isl_point *P = isl_set_sample_point(Elements);

  isl_val *V;
  V = isl_point_get_coordinate_val(P, isl_dim_set, Dim - 1);
  int NumberIterations = isl_val_get_num_si(V);
  isl_val_free(V);
  isl_point_free(P);
  if (NumberIterations == -1)
    return -1;
  return NumberIterations + 1;
}

struct FindValuesUser {
  LoopInfo &LI;
  ScalarEvolution &SE;
  Region &R;
  SetVector<Value *> &Values;
  SetVector<const SCEV *> &SCEVs;
};

/// @brief Extract the values and SCEVs needed to generate code for a block.
static int findValuesInBlock(struct FindValuesUser &User, const ScopStmt *Stmt,
                             const BasicBlock *BB) {
  // Check all the operands of instructions in the basic block.
  for (const Instruction &Inst : *BB) {
    for (Value *SrcVal : Inst.operands()) {
      if (Instruction *OpInst = dyn_cast<Instruction>(SrcVal))
        if (canSynthesize(OpInst, &User.LI, &User.SE, &User.R)) {
          User.SCEVs.insert(
              User.SE.getSCEVAtScope(OpInst, User.LI.getLoopFor(BB)));
          continue;
        }
      if (Instruction *OpInst = dyn_cast<Instruction>(SrcVal))
        if (Stmt->getParent()->getRegion().contains(OpInst))
          continue;

      if (isa<Instruction>(SrcVal) || isa<Argument>(SrcVal))
        User.Values.insert(SrcVal);
    }
  }
  return 0;
}

/// Extract the values and SCEVs needed to generate code for a ScopStmt.
///
/// This function extracts a ScopStmt from a given isl_set and computes the
/// Values this statement depends on as well as a set of SCEV expressions that
/// need to be synthesized when generating code for this statment.
static int findValuesInStmt(isl_set *Set, void *UserPtr) {
  isl_id *Id = isl_set_get_tuple_id(Set);
  struct FindValuesUser &User = *static_cast<struct FindValuesUser *>(UserPtr);
  const ScopStmt *Stmt = static_cast<const ScopStmt *>(isl_id_get_user(Id));

  if (Stmt->isBlockStmt())
    findValuesInBlock(User, Stmt, Stmt->getBasicBlock());
  else {
    assert(Stmt->isRegionStmt() &&
           "Stmt was neither block nor region statement");
    for (const BasicBlock *BB : Stmt->getRegion()->blocks())
      findValuesInBlock(User, Stmt, BB);
  }

  isl_id_free(Id);
  isl_set_free(Set);
  return 0;
}

void IslNodeBuilder::getReferencesInSubtree(__isl_keep isl_ast_node *For,
                                            SetVector<Value *> &Values,
                                            SetVector<const Loop *> &Loops) {

  SetVector<const SCEV *> SCEVs;
  struct FindValuesUser FindValues = {LI, SE, S.getRegion(), Values, SCEVs};

  for (const auto &I : IDToValue)
    Values.insert(I.second);

  for (const auto &I : OutsideLoopIterations)
    Values.insert(cast<SCEVUnknown>(I.second)->getValue());

  isl_union_set *Schedule = isl_union_map_domain(IslAstInfo::getSchedule(For));

  isl_union_set_foreach_set(Schedule, findValuesInStmt, &FindValues);
  isl_union_set_free(Schedule);

  for (const SCEV *Expr : SCEVs) {
    findValues(Expr, Values);
    findLoops(Expr, Loops);
  }

  Values.remove_if([](const Value *V) { return isa<GlobalValue>(V); });

  /// Remove loops that contain the scop or that are part of the scop, as they
  /// are considered local. This leaves only loops that are before the scop, but
  /// do not contain the scop itself.
  Loops.remove_if([this](const Loop *L) {
    return this->S.getRegion().contains(L) ||
           L->contains(S.getRegion().getEntry());
  });
}

void IslNodeBuilder::updateValues(
    ParallelLoopGenerator::ValueToValueMapTy &NewValues) {
  SmallPtrSet<Value *, 5> Inserted;

  for (const auto &I : IDToValue) {
    IDToValue[I.first] = NewValues[I.second];
    Inserted.insert(I.second);
  }

  for (const auto &I : NewValues) {
    if (Inserted.count(I.first))
      continue;

    ValueMap[I.first] = I.second;
  }
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
  Stmt->setAstBuild(IslAstInfo::getBuild(User));
  VectorValueMapT VectorMap(IVS.size());
  std::vector<LoopToScevMapT> VLTS(IVS.size());

  isl_union_set *Domain = isl_union_set_from_set(Stmt->getDomain());
  Schedule = isl_union_map_intersect_domain(Schedule, Domain);
  isl_map *S = isl_map_from_union_map(Schedule);

  createSubstitutionsVector(Expr, Stmt, VectorMap, VLTS, IVS, IteratorID);
  VectorBlockGenerator::generate(BlockGen, *Stmt, VectorMap, VLTS, S);

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

  IDToValue.erase(IDToValue.find(IteratorID));
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

  Parallel =
      IslAstInfo::isParallel(For) && !IslAstInfo::isReductionParallel(For);

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

  Annotator.popLoop(Parallel);

  IDToValue.erase(IDToValue.find(IteratorID));

  Builder.SetInsertPoint(ExitBlock->begin());

  isl_ast_node_free(For);
  isl_ast_expr_free(Iterator);
  isl_id_free(IteratorID);
}

/// @brief Remove the BBs contained in a (sub)function from the dominator tree.
///
/// This function removes the basic blocks that are part of a subfunction from
/// the dominator tree. Specifically, when generating code it may happen that at
/// some point the code generation continues in a new sub-function (e.g., when
/// generating OpenMP code). The basic blocks that are created in this
/// sub-function are then still part of the dominator tree of the original
/// function, such that the dominator tree reaches over function boundaries.
/// This is not only incorrect, but also causes crashes. This function now
/// removes from the dominator tree all basic blocks that are dominated (and
/// consequently reachable) from the entry block of this (sub)function.
///
/// FIXME: A LLVM (function or region) pass should not touch anything outside of
/// the function/region it runs on. Hence, the pure need for this function shows
/// that we do not comply to this rule. At the moment, this does not cause any
/// issues, but we should be aware that such issues may appear. Unfortunately
/// the current LLVM pass infrastructure does not allow to make Polly a module
/// or call-graph pass to solve this issue, as such a pass would not have access
/// to the per-function analyses passes needed by Polly. A future pass manager
/// infrastructure is supposed to enable such kind of access possibly allowing
/// us to create a cleaner solution here.
///
/// FIXME: Instead of adding the dominance information and then dropping it
/// later on, we should try to just not add it in the first place. This requires
/// some careful testing to make sure this does not break in interaction with
/// the SCEVBuilder and SplitBlock which may rely on the dominator tree or
/// which may try to update it.
///
/// @param F The function which contains the BBs to removed.
/// @param DT The dominator tree from which to remove the BBs.
static void removeSubFuncFromDomTree(Function *F, DominatorTree &DT) {
  DomTreeNode *N = DT.getNode(&F->getEntryBlock());
  std::vector<BasicBlock *> Nodes;

  // We can only remove an element from the dominator tree, if all its children
  // have been removed. To ensure this we obtain the list of nodes to remove
  // using a post-order tree traversal.
  for (po_iterator<DomTreeNode *> I = po_begin(N), E = po_end(N); I != E; ++I)
    Nodes.push_back(I->getBlock());

  for (BasicBlock *BB : Nodes)
    DT.eraseNode(BB);
}

void IslNodeBuilder::createForParallel(__isl_take isl_ast_node *For) {
  isl_ast_node *Body;
  isl_ast_expr *Init, *Inc, *Iterator, *UB;
  isl_id *IteratorID;
  Value *ValueLB, *ValueUB, *ValueInc;
  Type *MaxType;
  Value *IV;
  CmpInst::Predicate Predicate;

  Body = isl_ast_node_for_get_body(For);
  Init = isl_ast_node_for_get_init(For);
  Inc = isl_ast_node_for_get_inc(For);
  Iterator = isl_ast_node_for_get_iterator(For);
  IteratorID = isl_ast_expr_get_id(Iterator);
  UB = getUpperBound(For, Predicate);

  ValueLB = ExprBuilder.create(Init);
  ValueUB = ExprBuilder.create(UB);
  ValueInc = ExprBuilder.create(Inc);

  // OpenMP always uses SLE. In case the isl generated AST uses a SLT
  // expression, we need to adjust the loop blound by one.
  if (Predicate == CmpInst::ICMP_SLT)
    ValueUB = Builder.CreateAdd(
        ValueUB, Builder.CreateSExt(Builder.getTrue(), ValueUB->getType()));

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

  BasicBlock::iterator LoopBody;

  SetVector<Value *> SubtreeValues;
  SetVector<const Loop *> Loops;

  getReferencesInSubtree(For, SubtreeValues, Loops);

  // Create for all loops we depend on values that contain the current loop
  // iteration. These values are necessary to generate code for SCEVs that
  // depend on such loops. As a result we need to pass them to the subfunction.
  for (const Loop *L : Loops) {
    const SCEV *OuterLIV = SE.getAddRecExpr(SE.getUnknown(Builder.getInt64(0)),
                                            SE.getUnknown(Builder.getInt64(1)),
                                            L, SCEV::FlagAnyWrap);
    Value *V = generateSCEV(OuterLIV);
    OutsideLoopIterations[L] = SE.getUnknown(V);
    SubtreeValues.insert(V);
  }

  ParallelLoopGenerator::ValueToValueMapTy NewValues;
  ParallelLoopGenerator ParallelLoopGen(Builder, P, LI, DT, DL);

  IV = ParallelLoopGen.createParallelLoop(ValueLB, ValueUB, ValueInc,
                                          SubtreeValues, NewValues, &LoopBody);
  BasicBlock::iterator AfterLoop = Builder.GetInsertPoint();
  Builder.SetInsertPoint(LoopBody);

  // Save the current values.
  ValueMapT ValueMapCopy = ValueMap;
  IslExprBuilder::IDToValueTy IDToValueCopy = IDToValue;

  updateValues(NewValues);
  IDToValue[IteratorID] = IV;

  create(Body);

  // Restore the original values.
  ValueMap = ValueMapCopy;
  IDToValue = IDToValueCopy;

  Builder.SetInsertPoint(AfterLoop);
  removeSubFuncFromDomTree((*LoopBody).getParent()->getParent(), DT);

  for (const Loop *L : Loops)
    OutsideLoopIterations.erase(L);

  isl_ast_node_free(For);
  isl_ast_expr_free(Iterator);
  isl_id_free(IteratorID);
}

void IslNodeBuilder::createFor(__isl_take isl_ast_node *For) {
  bool Vector = PollyVectorizerChoice == VECTORIZER_POLLY;

  if (Vector && IslAstInfo::isInnermostParallel(For) &&
      !IslAstInfo::isReductionParallel(For)) {
    int VectorWidth = getNumberOfIterations(For);
    if (1 < VectorWidth && VectorWidth <= 16) {
      createForVector(For, VectorWidth);
      return;
    }
  }

  if (IslAstInfo::isExecutedInParallel(For)) {
    createForParallel(For);
    return;
  }
  createForSequential(For);
}

void IslNodeBuilder::createIf(__isl_take isl_ast_node *If) {
  isl_ast_expr *Cond = isl_ast_node_if_get_cond(If);

  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *CondBB =
      SplitBlock(Builder.GetInsertBlock(), Builder.GetInsertPoint(), &DT, &LI);
  CondBB->setName("polly.cond");
  BasicBlock *MergeBB = SplitBlock(CondBB, CondBB->begin(), &DT, &LI);
  MergeBB->setName("polly.merge");
  BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);
  BasicBlock *ElseBB = BasicBlock::Create(Context, "polly.else", F);

  DT.addNewBlock(ThenBB, CondBB);
  DT.addNewBlock(ElseBB, CondBB);
  DT.changeImmediateDominator(MergeBB, CondBB);

  Loop *L = LI.getLoopFor(CondBB);
  if (L) {
    L->addBasicBlockToLoop(ThenBB, LI);
    L->addBasicBlockToLoop(ElseBB, LI);
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
  }

  // Add the current ValueMap to our per-statement value map.
  //
  // This is needed e.g. to rewrite array base addresses when moving code
  // into a parallely executed subfunction.
  VMap.insert(ValueMap.begin(), ValueMap.end());

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

  LTS.insert(OutsideLoopIterations.begin(), OutsideLoopIterations.end());

  Stmt = (ScopStmt *)isl_id_get_user(Id);
  Stmt->setAstBuild(IslAstInfo::getBuild(User));

  createSubstitutions(Expr, Stmt, VMap, LTS);
  if (Stmt->isBlockStmt())
    BlockGen.copyStmt(*Stmt, VMap, LTS);
  else
    RegionGen.copyStmt(*Stmt, VMap, LTS);

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
    llvm_unreachable("code generation error");
  case isl_ast_node_mark:
    llvm_unreachable("Mark node unexpected");
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

  for (unsigned i = 0; i < isl_set_dim(Context, isl_dim_param); ++i) {
    isl_id *Id;

    Id = isl_set_get_dim_id(Context, isl_dim_param, i);
    IDToValue[Id] = generateSCEV((const SCEV *)isl_id_get_user(Id));

    isl_id_free(Id);
  }

  // Generate values for the current loop iteration for all surrounding loops.
  //
  // We may also reference loops outside of the scop which do not contain the
  // scop itself, but as the number of such scops may be arbitrarily large we do
  // not generate code for them here, but only at the point of code generation
  // where these values are needed.
  Region &R = S.getRegion();
  Loop *L = LI.getLoopFor(R.getEntry());

  while (L != nullptr && R.contains(L))
    L = L->getParentLoop();

  while (L != nullptr) {
    const SCEV *OuterLIV = SE.getAddRecExpr(SE.getUnknown(Builder.getInt64(0)),
                                            SE.getUnknown(Builder.getInt64(1)),
                                            L, SCEV::FlagAnyWrap);
    Value *V = generateSCEV(OuterLIV);
    OutsideLoopIterations[L] = SE.getUnknown(V);
    L = L->getParentLoop();
  }

  isl_set_free(Context);
}

Value *IslNodeBuilder::generateSCEV(const SCEV *Expr) {
  Instruction *InsertLocation = --(Builder.GetInsertBlock()->end());
  return Rewriter.expandCodeFor(Expr, Expr->getType(), InsertLocation);
}

namespace {
class IslCodeGeneration : public ScopPass {
public:
  static char ID;

  IslCodeGeneration() : ScopPass(ID) {}

  /// @brief The datalayout used
  const DataLayout *DL;

  /// @name The analysis passes we need to generate code.
  ///
  ///{
  LoopInfo *LI;
  IslAstInfo *AI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  ///}

  /// @brief The loop annotator to generate llvm.loop metadata.
  ScopAnnotator Annotator;

  /// @brief Build the runtime condition.
  ///
  /// Build the condition that evaluates at run-time to true iff all
  /// assumptions taken for the SCoP hold, and to false otherwise.
  ///
  /// @return A value evaluating to true/false if execution is save/unsafe.
  Value *buildRTC(PollyIRBuilder &Builder, IslExprBuilder &ExprBuilder) {
    Builder.SetInsertPoint(Builder.GetInsertBlock()->getTerminator());
    Value *RTC = ExprBuilder.create(AI->getRunCondition());
    if (!RTC->getType()->isIntegerTy(1))
      RTC = Builder.CreateIsNotNull(RTC);
    return RTC;
  }

  bool verifyGeneratedFunction(Scop &S, Function &F) {
    if (!verifyFunction(F))
      return false;

    DEBUG({
      errs() << "== ISL Codegen created an invalid function ==\n\n== The "
                "SCoP ==\n";
      S.print(errs());
      errs() << "\n== The isl AST ==\n";
      AI->printScop(errs(), S);
      errs() << "\n== The invalid function ==\n";
      F.print(errs());
      errs() << "\n== The errors ==\n";
      verifyFunction(F, &errs());
    });

    return true;
  }

  bool runOnScop(Scop &S) override {
    AI = &getAnalysis<IslAstInfo>();

    // Check if we created an isl_ast root node, otherwise exit.
    isl_ast_node *AstRoot = AI->getAst();
    if (!AstRoot)
      return false;

    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    SE = &getAnalysis<ScalarEvolution>();
    DL = &S.getRegion().getEntry()->getParent()->getParent()->getDataLayout();

    assert(!S.getRegion().isTopLevelRegion() &&
           "Top level regions are not supported");

    // Build the alias scopes for annotations first.
    if (PollyAnnotateAliasScopes)
      Annotator.buildAliasScopes(S);

    BasicBlock *EnteringBB = simplifyRegion(&S, this);
    PollyIRBuilder Builder = createPollyIRBuilder(EnteringBB, Annotator);

    IslNodeBuilder NodeBuilder(Builder, Annotator, this, *DL, *LI, *SE, *DT, S);

    // Only build the run-time condition and parameters _after_ having
    // introduced the conditional branch. This is important as the conditional
    // branch will guard the original scop from new induction variables that
    // the SCEVExpander may introduce while code generating the parameters and
    // which may introduce scalar dependences that prevent us from correctly
    // code generating this scop.
    BasicBlock *StartBlock =
        executeScopConditionally(S, this, Builder.getTrue());
    auto SplitBlock = StartBlock->getSinglePredecessor();
    Builder.SetInsertPoint(SplitBlock->getTerminator());
    NodeBuilder.addParameters(S.getContext());
    Value *RTC = buildRTC(Builder, NodeBuilder.getExprBuilder());
    SplitBlock->getTerminator()->setOperand(0, RTC);
    Builder.SetInsertPoint(StartBlock->begin());

    NodeBuilder.create(AstRoot);

    assert(!verifyGeneratedFunction(S, *EnteringBB->getParent()) &&
           "Verification of generated function failed");
    return true;
  }

  void printScop(raw_ostream &, Scop &) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<IslAstInfo>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfo>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<DependenceInfo>();

    AU.addPreserved<LoopInfoWrapperPass>();
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
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(IslCodeGeneration, "polly-codegen-isl",
                    "Polly - Create LLVM-IR from SCoPs", false, false)
