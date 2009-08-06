#ifndef LLVM_CODEGEN_PBQP_SOLVER_H
#define LLVM_CODEGEN_PBQP_SOLVER_H

#include "SimpleGraph.h"
#include "Solution.h"

namespace PBQP {

/// \brief Interface for solver classes.
class Solver {
public:

  virtual ~Solver() = 0;
  virtual Solution solve(const SimpleGraph &orig) const = 0;
};

Solver::~Solver() {}

}

#endif // LLVM_CODEGEN_PBQP_SOLVER_H
