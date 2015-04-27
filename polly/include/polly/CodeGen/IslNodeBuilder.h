//===------ IslNodeBuilder.cpp - Translate an isl AST into a LLVM-IR AST---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file contains the IslNodeBuilder, a class to translate an isl AST into
// a LLVM-IR AST.
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISL_NODE_BUILDER_H
#define POLLY_ISL_NODE_BUILDER_H

#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/LoopGenerators.h"

#include "isl/ctx.h"

#include "llvm/Analysis/ScalarEvolutionExpander.h"

using namespace polly;
using namespace llvm;

struct isl_ast_node;

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
  llvm::Value *generateSCEV(const SCEV *Expr);

  /// A set of Value -> Value remappings to apply when generating new code.
  ///
  /// When generating new code for a ScopStmt this map is used to map certain
  /// llvm::Values to new llvm::Values.
  polly::ValueMapT ValueMap;

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

#endif
