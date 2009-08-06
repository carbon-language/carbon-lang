#ifndef LLVM_CODEGEN_PBQP_SOLUTION_H
#define LLVM_CODEGEN_PBQP_SOLUTION_H

#include "PBQPMath.h"

namespace PBQP {

class Solution {

  friend class SolverImplementation;

private:

  std::vector<unsigned> selections;
  PBQPNum solutionCost;
  bool provedOptimal;
  unsigned r0Reductions, r1Reductions,
           r2Reductions, rNReductions;

public:

  Solution() :
    solutionCost(0.0), provedOptimal(false),
    r0Reductions(0), r1Reductions(0), r2Reductions(0), rNReductions(0) {}

  Solution(unsigned length, bool assumeOptimal) :
    selections(length), solutionCost(0.0), provedOptimal(assumeOptimal),
    r0Reductions(0), r1Reductions(0), r2Reductions(0), rNReductions(0) {}

  void setProvedOptimal(bool provedOptimal) {
    this->provedOptimal = provedOptimal;
  }

  void setSelection(unsigned nodeID, unsigned selection) {
    selections[nodeID] = selection;
  }

  void setSolutionCost(PBQPNum solutionCost) {
    this->solutionCost = solutionCost;
  }

  void incR0Reductions() { ++r0Reductions; }
  void incR1Reductions() { ++r1Reductions; }
  void incR2Reductions() { ++r2Reductions; }
  void incRNReductions() { ++rNReductions; }

  unsigned numNodes() const { return selections.size(); }

  unsigned getSelection(unsigned nodeID) const {
    return selections[nodeID];
  }

  PBQPNum getCost() const { return solutionCost; }

  bool isProvedOptimal() const { return provedOptimal; }

  unsigned getR0Reductions() const { return r0Reductions; }
  unsigned getR1Reductions() const { return r1Reductions; }
  unsigned getR2Reductions() const { return r2Reductions; }
  unsigned getRNReductions() const { return rNReductions; }

  bool operator==(const Solution &other) const {
    return (selections == other.selections);
  }

  bool operator!=(const Solution &other) const {
    return !(*this == other);
  }

};

}

#endif // LLVM_CODEGEN_PBQP_SOLUTION_H
