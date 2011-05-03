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

    isl_union_map *must_dep, *may_dep;
    isl_union_map *must_no_source, *may_no_source;

    isl_union_map *war_dep;
    isl_union_map *waw_dep;

    isl_union_map *sink;
    isl_union_map *must_source;
    isl_union_map *may_source;

  public:
    static char ID;
    typedef std::map<ScopStmt*, isl_map*> StatementToIslMapTy;

    Dependences();
    bool isValidScattering(StatementToIslMapTy *NewScatterings);

    /// @brief Check if a dimension of the Scop can be executed in parallel.
    ///
    /// @param loopDomain The subset of the scattering space that is executed in
    ///                   parallel.
    /// @param parallelDimension The scattering dimension that is being executed
    ///                          in parallel.
    ///
    /// @return bool Returns true, if executing parallelDimension in parallel is
    ///              valid for the scattering domain subset given.
    bool isParallelDimension(isl_set *loopDomain, unsigned parallelDimension);

    /// @brief Check if a loop is parallel
    ///
    /// Detect if a clast_for loop can be executed in parallel.
    ///
    /// @param f The clast for loop to check.
    ///
    /// @return bool Returns true if the incoming clast_for statement can
    ///              execute in parallel.
    bool isParallelFor(const clast_for *f);

    bool runOnScop(Scop &S);
    void printScop(raw_ostream &OS) const;
    virtual void releaseMemory();
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  };
} // End polly namespace.

#endif
