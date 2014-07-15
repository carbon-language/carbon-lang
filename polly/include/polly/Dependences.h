//===------ polly/Dependences.h - Polyhedral dependency analysis *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven, has a integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculate those dependences a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may writes a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_DEPENDENCES_H
#define POLLY_DEPENDENCES_H

#include "polly/ScopPass.h"

#include <map>
#include "isl/ctx.h"

struct isl_union_map;
struct isl_union_set;
struct isl_map;
struct isl_set;
struct clast_for;

using namespace llvm;

namespace polly {

class Scop;
class ScopStmt;

class Dependences : public ScopPass {
public:
  static char ID;

  /// @brief The type of the dependences.
  ///
  /// Reduction dependences are separated from RAW/WAW/WAR dependences because
  /// we can ignore them during the scheduling. This is the case since the order
  /// in which the reduction statements are executed does not matter. However,
  /// if they are executed in parallel we need to take additional measures
  /// (e.g, privatization) to ensure a correct result. The (reverse) transitive
  /// closure of the reduction dependences are used to check for parallel
  /// executed reduction statements during code generation. These dependences
  /// connect all instances of a reduction with each other, they are therefor
  /// cyclic and possibly "reversed".
  enum Type {
    // Write after read
    TYPE_WAR = 1 << 0,

    // Read after write
    TYPE_RAW = 1 << 1,

    // Write after write
    TYPE_WAW = 1 << 2,

    // Reduction dependences
    TYPE_RED = 1 << 3,

    // Transitive closure of the reduction dependences (& the reverse)
    TYPE_TC_RED = 1 << 4,
  };

  typedef std::map<ScopStmt *, isl_map *> StatementToIslMapTy;

  Dependences();

  // @brief Check if a new scattering is valid.
  //
  // @param NewScattering The new scatterings
  //
  // @return bool True if the new scattering is valid, false it it reverses
  //              dependences.
  bool isValidScattering(StatementToIslMapTy *NewScatterings);

  /// @brief Check if a dimension of the Scop can be executed in parallel.
  ///
  /// @param LoopDomain The subset of the scattering space that is executed in
  ///                   parallel.
  /// @param ParallelDimension The scattering dimension that is being executed
  ///                          in parallel.
  ///
  /// @return bool Returns true, if executing parallelDimension in parallel is
  ///              valid for the scattering domain subset given.
  bool isParallelDimension(__isl_take isl_set *LoopDomain,
                           unsigned ParallelDimension);

  /// @brief Get the dependences in this Scop.
  ///
  /// @param Kinds This integer defines the different kinds of dependences
  ///              that will be returned. To return more than one kind, the
  ///              different kinds are 'ored' together.
  isl_union_map *getDependences(int Kinds);

  /// @brief Report if valid dependences are available.
  bool hasValidDependences();

  bool runOnScop(Scop &S);
  void printScop(raw_ostream &OS) const;
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  // The different kinds of dependences we calculate.
  isl_union_map *RAW;
  isl_union_map *WAR;
  isl_union_map *WAW;

  /// @brief The map of reduction dependences
  isl_union_map *RED = nullptr;

  /// @brief The (reverse) transitive closure of reduction dependences
  isl_union_map *TC_RED = nullptr;

  /// @brief Collect information about the SCoP.
  void collectInfo(Scop &S, isl_union_map **Read, isl_union_map **Write,
                   isl_union_map **MayWrite, isl_union_map **AccessSchedule,
                   isl_union_map **StmtSchedule);

  /// @brief Calculate and add at the privatization dependences
  void addPrivatizationDependences();

  // @brief Calculate the dependences for a certain SCoP.
  void calculateDependences(Scop &S);
};

} // End polly namespace.

namespace llvm {
class PassRegistry;
void initializeDependencesPass(llvm::PassRegistry &);
}

#endif
