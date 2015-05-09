//===-BlockGenerators.h - Helper to generate code for statements-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the BlockGenerator and VectorBlockGenerator classes, which
// generate sequential code and vectorized code for a polyhedral statement,
// respectively.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_BLOCK_GENERATORS_H
#define POLLY_BLOCK_GENERATORS_H

#include "polly/CodeGen/IRBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "isl/map.h"
#include <vector>

struct isl_ast_build;

namespace llvm {
class Pass;
class Region;
class ScalarEvolution;
}

namespace polly {
using namespace llvm;
class ScopStmt;
class MemoryAccess;
class IslExprBuilder;

typedef DenseMap<const Value *, Value *> ValueMapT;
typedef std::vector<ValueMapT> VectorValueMapT;

/// @brief Check whether an instruction can be synthesized by the code
///        generator.
///
/// Some instructions will be recalculated only from information that is code
/// generated from the polyhedral representation. For such instructions we do
/// not need to ensure that their operands are available during code generation.
///
/// @param I The instruction to check.
/// @param LI The LoopInfo analysis.
/// @param SE The scalar evolution database.
/// @param R The region out of which SSA names are parameters.
/// @return If the instruction I can be regenerated from its
///         scalar evolution representation, return true,
///         otherwise return false.
bool canSynthesize(const llvm::Instruction *I, const llvm::LoopInfo *LI,
                   llvm::ScalarEvolution *SE, const llvm::Region *R);

/// @brief Return true iff @p V is an intrinsic that we ignore during code
///        generation.
bool isIgnoredIntrinsic(const llvm::Value *V);

/// @brief Generate a new basic block for a polyhedral statement.
class BlockGenerator {
public:
  /// @brief Create a generator for basic blocks.
  ///
  /// @param Builder     The LLVM-IR Builder used to generate the statement. The
  ///                    code is generated at the location, the Builder points
  ///                    to.
  /// @param LI          The loop info for the current function
  /// @param SE          The scalar evolution info for the current function
  /// @param DT          The dominator tree of this function.
  /// @param ExprBuilder An expression builder to generate new access functions.
  BlockGenerator(PollyIRBuilder &Builder, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, IslExprBuilder *ExprBuilder = nullptr);

  /// @brief Copy the basic block.
  ///
  /// This copies the entire basic block and updates references to old values
  /// with references to new values, as defined by GlobalMap.
  ///
  /// @param Stmt      The block statement to code generate.
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  void copyStmt(ScopStmt &Stmt, ValueMapT &GlobalMap, LoopToScevMapT &LTS);

protected:
  PollyIRBuilder &Builder;
  LoopInfo &LI;
  ScalarEvolution &SE;
  IslExprBuilder *ExprBuilder;

  /// @brief The dominator tree of this function.
  DominatorTree &DT;

  /// @brief Split @p BB to create a new one we can use to clone @p BB in.
  BasicBlock *splitBB(BasicBlock *BB);

  /// @brief Copy the given basic block.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param BB        The basic block to code generate.
  /// @param BBMap     A mapping from old values to their new values in this
  /// block.
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  ///
  /// @returns The copy of the basic block.
  BasicBlock *copyBB(ScopStmt &Stmt, BasicBlock *BB, ValueMapT &BBMap,
                     ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  /// @brief Copy the given basic block.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param BB        The basic block to code generate.
  /// @param BBCopy    The new basic block to generate code in.
  /// @param BBMap     A mapping from old values to their new values in this
  /// block.
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  void copyBB(ScopStmt &Stmt, BasicBlock *BB, BasicBlock *BBCopy,
              ValueMapT &BBMap, ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  /// @brief Get the new version of a value.
  ///
  /// Given an old value, we first check if a new version of this value is
  /// available in the BBMap or GlobalMap. In case it is not and the value can
  /// be recomputed using SCEV, we do so. If we can not recompute a value
  /// using SCEV, but we understand that the value is constant within the scop,
  /// we return the old value.  If the value can still not be derived, this
  /// function will assert.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param Old       The old Value.
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block).
  /// @param LTS       A mapping from loops virtual canonical induction
  ///                  variable to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block).
  /// @param L         The loop that surrounded the instruction that referenced
  ///                  this value in the original code. This loop is used to
  ///                  evaluate the scalar evolution at the right scope.
  ///
  /// @returns  o The old value, if it is still valid.
  ///           o The new value, if available.
  ///           o NULL, if no value is found.
  Value *getNewValue(ScopStmt &Stmt, const Value *Old, ValueMapT &BBMap,
                     ValueMapT &GlobalMap, LoopToScevMapT &LTS, Loop *L) const;

  void copyInstScalar(ScopStmt &Stmt, const Instruction *Inst, ValueMapT &BBMap,
                      ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  /// @brief Get the innermost loop that surrounds an instruction.
  ///
  /// @param Inst The instruction for which we get the loop.
  /// @return The innermost loop that surrounds the instruction.
  Loop *getLoopForInst(const Instruction *Inst);

  /// @brief Get the new operand address according to access relation of @p MA.
  Value *getNewAccessOperand(ScopStmt &Stmt, const MemoryAccess &MA);

  /// @brief Generate the operand address
  Value *generateLocationAccessed(ScopStmt &Stmt, const Instruction *Inst,
                                  const Value *Pointer, ValueMapT &BBMap,
                                  ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  Value *generateScalarLoad(ScopStmt &Stmt, const LoadInst *load,
                            ValueMapT &BBMap, ValueMapT &GlobalMap,
                            LoopToScevMapT &LTS);

  Value *generateScalarStore(ScopStmt &Stmt, const StoreInst *store,
                             ValueMapT &BBMap, ValueMapT &GlobalMap,
                             LoopToScevMapT &LTS);

  /// @brief Copy a single Instruction.
  ///
  /// This copies a single Instruction and updates references to old values
  /// with references to new values, as defined by GlobalMap and BBMap.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param Inst      The instruction to copy.
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  /// @param LTS       A mapping from loops virtual canonical induction
  ///                  variable to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block).
  void copyInstruction(ScopStmt &Stmt, const Instruction *Inst,
                       ValueMapT &BBMap, ValueMapT &GlobalMap,
                       LoopToScevMapT &LTS);
};

/// @brief Generate a new vector basic block for a polyhedral statement.
///
/// The only public function exposed is generate().
class VectorBlockGenerator : BlockGenerator {
public:
  /// @brief Generate a new vector basic block for a ScoPStmt.
  ///
  /// This code generation is similar to the normal, scalar code generation,
  /// except that each instruction is code generated for several vector lanes
  /// at a time. If possible instructions are issued as actual vector
  /// instructions, but e.g. for address calculation instructions we currently
  /// generate scalar instructions for each vector lane.
  ///
  /// @param BlockGen   A block generator object used as parent.
  /// @param Stmt       The statement to code generate.
  /// @param GlobalMaps A vector of maps that define for certain Values
  ///                   referenced from the original code new Values they should
  ///                   be replaced with. Each map in the vector of maps is
  ///                   used for one vector lane. The number of elements in the
  ///                   vector defines the width of the generated vector
  ///                   instructions.
  /// @param VLTS       A mapping from loops virtual canonical induction
  ///                   variable to their new values
  ///                   (for values recalculated in the new ScoP, but not
  ///                    within this basic block), one for each lane.
  /// @param Schedule   A map from the statement to a schedule where the
  ///                   innermost dimension is the dimension of the innermost
  ///                   loop containing the statemenet.
  static void generate(BlockGenerator &BlockGen, ScopStmt &Stmt,
                       VectorValueMapT &GlobalMaps,
                       std::vector<LoopToScevMapT> &VLTS,
                       __isl_keep isl_map *Schedule) {
    VectorBlockGenerator Generator(BlockGen, GlobalMaps, VLTS, Schedule);
    Generator.copyStmt(Stmt);
  }

private:
  // This is a vector of global value maps.  The first map is used for the first
  // vector lane, ...
  // Each map, contains information about Instructions in the old ScoP, which
  // are recalculated in the new SCoP. When copying the basic block, we replace
  // all referenes to the old instructions with their recalculated values.
  VectorValueMapT &GlobalMaps;

  // This is a vector of loop->scev maps.  The first map is used for the first
  // vector lane, ...
  // Each map, contains information about Instructions in the old ScoP, which
  // are recalculated in the new SCoP. When copying the basic block, we replace
  // all referenes to the old instructions with their recalculated values.
  //
  // For example, when the code generator produces this AST:
  //
  //   for (int c1 = 0; c1 <= 1023; c1 += 1)
  //     for (int c2 = 0; c2 <= 1023; c2 += VF)
  //       for (int lane = 0; lane <= VF; lane += 1)
  //         Stmt(c2 + lane + 3, c1);
  //
  // VLTS[lane] contains a map:
  //   "outer loop in the old loop nest" -> SCEV("c2 + lane + 3"),
  //   "inner loop in the old loop nest" -> SCEV("c1").
  std::vector<LoopToScevMapT> &VLTS;

  // A map from the statement to a schedule where the innermost dimension is the
  // dimension of the innermost loop containing the statemenet.
  isl_map *Schedule;

  VectorBlockGenerator(BlockGenerator &BlockGen, VectorValueMapT &GlobalMaps,
                       std::vector<LoopToScevMapT> &VLTS,
                       __isl_keep isl_map *Schedule);

  int getVectorWidth();

  Value *getVectorValue(ScopStmt &Stmt, const Value *Old, ValueMapT &VectorMap,
                        VectorValueMapT &ScalarMaps, Loop *L);

  Type *getVectorPtrTy(const Value *V, int Width);

  /// @brief Load a vector from a set of adjacent scalars
  ///
  /// In case a set of scalars is known to be next to each other in memory,
  /// create a vector load that loads those scalars
  ///
  /// %vector_ptr= bitcast double* %p to <4 x double>*
  /// %vec_full = load <4 x double>* %vector_ptr
  ///
  /// @param Stmt           The statement to code generate.
  /// @param NegativeStride This is used to indicate a -1 stride. In such
  ///                       a case we load the end of a base address and
  ///                       shuffle the accesses in reverse order into the
  ///                       vector. By default we would do only positive
  ///                       strides.
  ///
  Value *generateStrideOneLoad(ScopStmt &Stmt, const LoadInst *Load,
                               VectorValueMapT &ScalarMaps,
                               bool NegativeStride);

  /// @brief Load a vector initialized from a single scalar in memory
  ///
  /// In case all elements of a vector are initialized to the same
  /// scalar value, this value is loaded and shuffeled into all elements
  /// of the vector.
  ///
  /// %splat_one = load <1 x double>* %p
  /// %splat = shufflevector <1 x double> %splat_one, <1 x
  ///       double> %splat_one, <4 x i32> zeroinitializer
  ///
  Value *generateStrideZeroLoad(ScopStmt &Stmt, const LoadInst *Load,
                                ValueMapT &BBMap);

  /// @brief Load a vector from scalars distributed in memory
  ///
  /// In case some scalars a distributed randomly in memory. Create a vector
  /// by loading each scalar and by inserting one after the other into the
  /// vector.
  ///
  /// %scalar_1= load double* %p_1
  /// %vec_1 = insertelement <2 x double> undef, double %scalar_1, i32 0
  /// %scalar 2 = load double* %p_2
  /// %vec_2 = insertelement <2 x double> %vec_1, double %scalar_1, i32 1
  ///
  Value *generateUnknownStrideLoad(ScopStmt &Stmt, const LoadInst *Load,
                                   VectorValueMapT &ScalarMaps);

  void generateLoad(ScopStmt &Stmt, const LoadInst *Load, ValueMapT &VectorMap,
                    VectorValueMapT &ScalarMaps);

  void copyUnaryInst(ScopStmt &Stmt, const UnaryInstruction *Inst,
                     ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  void copyBinaryInst(ScopStmt &Stmt, const BinaryOperator *Inst,
                      ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  void copyStore(ScopStmt &Stmt, const StoreInst *Store, ValueMapT &VectorMap,
                 VectorValueMapT &ScalarMaps);

  void copyInstScalarized(ScopStmt &Stmt, const Instruction *Inst,
                          ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  bool extractScalarValues(const Instruction *Inst, ValueMapT &VectorMap,
                           VectorValueMapT &ScalarMaps);

  bool hasVectorOperands(const Instruction *Inst, ValueMapT &VectorMap);

  void copyInstruction(ScopStmt &Stmt, const Instruction *Inst,
                       ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  void copyStmt(ScopStmt &Stmt);
};

/// @brief Generator for new versions of polyhedral region statements.
class RegionGenerator : public BlockGenerator {
public:
  /// @brief Create a generator for regions.
  ///
  /// @param BlockGen A generator for basic blocks.
  RegionGenerator(BlockGenerator &BlockGen) : BlockGenerator(BlockGen) {}

  /// @brief Copy the region statement @p Stmt.
  ///
  /// This copies the entire region represented by @p Stmt and updates
  /// references to old values with references to new values, as defined by
  /// GlobalMap.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  void copyStmt(ScopStmt &Stmt, ValueMapT &GlobalMap, LoopToScevMapT &LTS);

private:
  /// @brief Repair the dominance tree after we created a copy block for @p BB.
  ///
  /// @returns The immediate dominator in the DT for @p BBCopy if in the region.
  BasicBlock *repairDominance(BasicBlock *BB, BasicBlock *BBCopy,
                              DenseMap<BasicBlock *, BasicBlock *> &BlockMap);
};
}
#endif
