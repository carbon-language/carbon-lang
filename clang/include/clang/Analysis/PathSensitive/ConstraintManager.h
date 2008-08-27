#ifndef CONSTRAINT_MANAGER_H
#define CONSTRAINT_MANAGER_H

namespace clang {

class GRState;
class GRStateManager;
class RVal;

class ConstraintManager {
public:
  virtual const GRState* Assume(const GRState* St, RVal Cond, bool Assumption,
                                bool& isFeasible) = 0;
};

ConstraintManager* CreateBasicConstraintManager(GRStateManager& statemgr);

} // end clang namespace

#endif
