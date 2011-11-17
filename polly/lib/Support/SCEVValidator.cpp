
#include "polly/Support/SCEVValidator.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/RegionInfo.h"

#include <vector>

using namespace llvm;

namespace SCEVType {
  /// @brief The type of a SCEV
  ///
  /// To check for the validity of a SCEV we assign to each SCEV a type. The
  /// possible types are INT, PARAM, IV and INVALID. The order of the types is
  /// important. The subexpressions of SCEV with a type X can only have a type
  /// that is smaller or equal than X.
  enum TYPE {
             // An integer value.
             INT,

             // An expression that is constant during the execution of the Scop,
             // but that may depend on parameters unknown at compile time.
             PARAM,

             // An expression that may change during the execution of the SCoP.
             IV,

             // An invalid expression.
             INVALID
  };
}

/// @brief The result the validator returns for a SCEV expression.
class ValidatorResult {
  /// @brief The type of the expression
  SCEVType::TYPE Type;

  /// @brief The set of Parameters in the expression.
  std::vector<const SCEV*> Parameters;

public:

  /// @brief Create an invalid result.
  ValidatorResult() : Type(SCEVType::INVALID) {};

  /// @brief The copy constructor
  ValidatorResult(const ValidatorResult &Source) {
    Type = Source.Type;
    Parameters = Source.Parameters;
  };

  /// @brief Construct a result with a certain type and no parameters.
  ValidatorResult(SCEVType::TYPE Type) : Type(Type) {};

  /// @brief Construct a result with a certain type and a single parameter.
  ValidatorResult(SCEVType::TYPE Type, const SCEV *Expr) : Type(Type) {
    Parameters.push_back(Expr);
  };

  /// @brief Is the analyzed SCEV constant during the execution of the SCoP.
  bool isConstant() {
    return Type == SCEVType::INT || Type == SCEVType::PARAM;
  }

  /// @brief Is the analyzed SCEV valid.
  bool isValid() {
    return Type != SCEVType::INVALID;
  }

  /// @brief Is the analyzed SCEV of Type IV.
  bool isIV() {
    return Type == SCEVType::IV;
  }

  /// @brief Is the analyzed SCEV of Type INT.
  bool isINT() {
    return Type == SCEVType::INT;
  }

  /// @brief Get the parameters of this validator result.
  std::vector<const SCEV*> getParameters() {
    return Parameters;
  }

  /// @brief Add the parameters of Source to this result.
  void addParamsFrom(class ValidatorResult &Source) {
    Parameters.insert(Parameters.end(),
                      Source.Parameters.begin(),
                      Source.Parameters.end());
  }

  /// @brief Merge a result.
  ///
  /// This means to merge the parameters and to set the Type to the most
  /// specific Type that matches both.
  void merge(class ValidatorResult &ToMerge) {
    Type = std::max(Type, ToMerge.Type);
    addParamsFrom(ToMerge);
  }
};

/// Check if a SCEV is valid in a SCoP.
struct SCEVValidator
  : public SCEVVisitor<SCEVValidator, class ValidatorResult> {
private:
  const Region *R;
  ScalarEvolution &SE;
  const Value *BaseAddress;

public:
  SCEVValidator(const Region *R, ScalarEvolution &SE,
                const Value *BaseAddress) : R(R), SE(SE),
    BaseAddress(BaseAddress) {};

  class ValidatorResult visitConstant(const SCEVConstant *Constant) {
    return ValidatorResult(SCEVType::INT);
  }

  class ValidatorResult visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    ValidatorResult Op = visit(Expr->getOperand());

    // We currently do not represent a truncate expression as an affine
    // expression. If it is constant during Scop execution, we treat it as a
    // parameter, otherwise we bail out.
    if (Op.isConstant())
      return ValidatorResult(SCEVType::PARAM, Expr);

    return ValidatorResult(SCEVType::INVALID);
  }

  class ValidatorResult visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    ValidatorResult Op = visit(Expr->getOperand());

    // We currently do not represent a zero extend expression as an affine
    // expression. If it is constant during Scop execution, we treat it as a
    // parameter, otherwise we bail out.
    if (Op.isConstant())
      return ValidatorResult(SCEVType::PARAM, Expr);

    return ValidatorResult(SCEVType::INVALID);
  }

  class ValidatorResult visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    // We currently allow only signed SCEV expressions. In the case of a
    // signed value, a sign extend is a noop.
    //
    // TODO: Reconsider this when we add support for unsigned values.
    return visit(Expr->getOperand());
  }

  class ValidatorResult visitAddExpr(const SCEVAddExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isValid())
        return ValidatorResult(SCEVType::INVALID);

      Return.merge(Op);
    }

    // TODO: Check for NSW and NUW.
    return Return;
  }

  class ValidatorResult visitMulExpr(const SCEVMulExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (Op.isINT())
        continue;

      if (!Op.isValid() || !Return.isINT())
        return ValidatorResult(SCEVType::INVALID);

      Return.merge(Op);
    }

    // TODO: Check for NSW and NUW.
    return Return;
  }

  class ValidatorResult visitUDivExpr(const SCEVUDivExpr *Expr) {
    ValidatorResult LHS = visit(Expr->getLHS());
    ValidatorResult RHS = visit(Expr->getRHS());

    // We currently do not represent an unsigned devision as an affine
    // expression. If the division is constant during Scop execution we treat it
    // as a parameter, otherwise we bail out.
    if (LHS.isConstant() && RHS.isConstant())
      return ValidatorResult(SCEVType::PARAM, Expr);

    return ValidatorResult(SCEVType::INVALID);
  }

  class ValidatorResult visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    if (!Expr->isAffine())
      return ValidatorResult(SCEVType::INVALID);

    ValidatorResult Start = visit(Expr->getStart());
    ValidatorResult Recurrence = visit(Expr->getStepRecurrence(SE));

    if (!Start.isValid() || !Recurrence.isConstant())
      return ValidatorResult(SCEVType::INVALID);

    if (R->contains(Expr->getLoop())) {
      if (Recurrence.isINT()) {
        ValidatorResult Result(SCEVType::IV);
        Result.addParamsFrom(Start);
        return Result;
      }

      return ValidatorResult(SCEVType::INVALID);
    }

    if (Start.isConstant())
      return ValidatorResult(SCEVType::PARAM, Expr);

    return ValidatorResult(SCEVType::INVALID);
  }

  class ValidatorResult visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isValid())
        return ValidatorResult(SCEVType::INVALID);

      Return.merge(Op);
    }

    return Return;
  }

  class ValidatorResult visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    ValidatorResult Return(SCEVType::PARAM);

    // We do not support unsigned operations. If 'Expr' is constant during Scop
    // execution we treat this as a parameter, otherwise we bail out.
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isConstant())
        return ValidatorResult(SCEVType::INVALID);

      Return.merge(Op);
    }

    return Return;
  }

  ValidatorResult visitUnknown(const SCEVUnknown *Expr) {
    Value *V = Expr->getValue();

    if (isa<UndefValue>(V))
      return ValidatorResult(SCEVType::INVALID);

    if (Instruction *I = dyn_cast<Instruction>(Expr->getValue()))
      if (R->contains(I))
        return ValidatorResult(SCEVType::INVALID);

    if (BaseAddress == V)
      return ValidatorResult(SCEVType::INVALID);

    return ValidatorResult(SCEVType::PARAM, Expr);
  }
};

namespace polly {
  bool isAffineExpr(const Region *R, const SCEV *Expr, ScalarEvolution &SE,
                    const Value *BaseAddress) {
    if (isa<SCEVCouldNotCompute>(Expr))
      return false;

    SCEVValidator Validator(R, SE, BaseAddress);
    ValidatorResult Result = Validator.visit(Expr);

    return Result.isValid();
  }

  std::vector<const SCEV*> getParamsInAffineExpr(const Region *R,
                                                 const SCEV *Expr,
                                                 ScalarEvolution &SE,
                                                 const Value *BaseAddress) {
    if (isa<SCEVCouldNotCompute>(Expr))
      return std::vector<const SCEV*>();

    SCEVValidator Validator(R, SE, BaseAddress);
    ValidatorResult Result = Validator.visit(Expr);

    return Result.getParameters();
  }
}


