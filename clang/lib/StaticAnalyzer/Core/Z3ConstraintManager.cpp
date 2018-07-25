//== Z3ConstraintManager.cpp --------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTContext.h"

#include "clang/Config/config.h"

using namespace clang;
using namespace ento;

#if CLANG_ANALYZER_WITH_Z3

#include <z3.h>

// Forward declarations
namespace {
class Z3Expr;
class ConstraintZ3 {};
} // end anonymous namespace

typedef llvm::ImmutableSet<std::pair<SymbolRef, Z3Expr>> ConstraintZ3Ty;

// Expansion of REGISTER_TRAIT_WITH_PROGRAMSTATE(ConstraintZ3, Z3SetPair)
namespace clang {
namespace ento {
template <>
struct ProgramStateTrait<ConstraintZ3>
    : public ProgramStatePartialTrait<ConstraintZ3Ty> {
  static void *GDMIndex() {
    static int Index;
    return &Index;
  }
};
} // end namespace ento
} // end namespace clang

namespace {

class Z3Config {
  friend class Z3Context;

  Z3_config Config;

public:
  Z3Config() : Config(Z3_mk_config()) {
    // Enable model finding
    Z3_set_param_value(Config, "model", "true");
    // Disable proof generation
    Z3_set_param_value(Config, "proof", "false");
    // Set timeout to 15000ms = 15s
    Z3_set_param_value(Config, "timeout", "15000");
  }

  ~Z3Config() { Z3_del_config(Config); }
}; // end class Z3Config

void Z3ErrorHandler(Z3_context Context, Z3_error_code Error) {
  llvm::report_fatal_error("Z3 error: " +
                           llvm::Twine(Z3_get_error_msg_ex(Context, Error)));
}

class Z3Context : public SMTContext {
public:
  Z3_context Context;

  Z3Context() : SMTContext() {
    Context = Z3_mk_context_rc(Z3Config().Config);
    Z3_set_error_handler(Context, Z3ErrorHandler);
  }

  virtual ~Z3Context() {
    Z3_del_context(Context);
    Context = nullptr;
  }
}; // end class Z3Context

class Z3Sort {
  friend class Z3Expr;
  friend class Z3Solver;

  Z3Context &Context;

  Z3_sort Sort;

  Z3Sort(Z3Context &C, Z3_sort ZS) : Context(C), Sort(ZS) {
    assert(C.Context != nullptr);
    Z3_inc_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
  }

public:
  /// Override implicit copy constructor for correct reference counting.
  Z3Sort(const Z3Sort &Copy) : Context(Copy.Context), Sort(Copy.Sort) {
    Z3_inc_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
  }

  /// Provide move constructor
  Z3Sort(Z3Sort &&Move) : Context(Move.Context), Sort(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Sort &operator=(Z3Sort &&Move) {
    if (this != &Move) {
      if (Sort)
        Z3_dec_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
      Sort = Move.Sort;
      Move.Sort = nullptr;
    }
    return *this;
  }

  ~Z3Sort() {
    if (Sort)
      Z3_dec_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
  }

  Z3_sort_kind getSortKind() const {
    return Z3_get_sort_kind(Context.Context, Sort);
  }

  unsigned getBitvectorSortSize() const {
    assert(getSortKind() == Z3_BV_SORT && "Not a bitvector sort!");
    return Z3_get_bv_sort_size(Context.Context, Sort);
  }

  unsigned getFloatSortSize() const {
    assert(getSortKind() == Z3_FLOATING_POINT_SORT &&
           "Not a floating-point sort!");
    return Z3_fpa_get_ebits(Context.Context, Sort) +
           Z3_fpa_get_sbits(Context.Context, Sort);
  }

  bool operator==(const Z3Sort &Other) const {
    return Z3_is_eq_sort(Context.Context, Sort, Other.Sort);
  }

  Z3Sort &operator=(const Z3Sort &Move) {
    Z3_inc_ref(Context.Context, reinterpret_cast<Z3_ast>(Move.Sort));
    Z3_dec_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
    Sort = Move.Sort;
    return *this;
  }

  void print(raw_ostream &OS) const {
    OS << Z3_sort_to_string(Context.Context, Sort);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Sort

class Z3Expr {
  friend class Z3Model;
  friend class Z3Solver;

  Z3Context &Context;

  Z3_ast AST;

  Z3Expr(Z3Context &C, Z3_ast ZA) : Context(C), AST(ZA) {
    assert(C.Context != nullptr);
    Z3_inc_ref(Context.Context, AST);
  }

  // Determine whether two float semantics are equivalent
  static bool areEquivalent(const llvm::fltSemantics &LHS,
                            const llvm::fltSemantics &RHS) {
    return (llvm::APFloat::semanticsPrecision(LHS) ==
            llvm::APFloat::semanticsPrecision(RHS)) &&
           (llvm::APFloat::semanticsMinExponent(LHS) ==
            llvm::APFloat::semanticsMinExponent(RHS)) &&
           (llvm::APFloat::semanticsMaxExponent(LHS) ==
            llvm::APFloat::semanticsMaxExponent(RHS)) &&
           (llvm::APFloat::semanticsSizeInBits(LHS) ==
            llvm::APFloat::semanticsSizeInBits(RHS));
  }

public:
  /// Override implicit copy constructor for correct reference counting.
  Z3Expr(const Z3Expr &Copy) : Context(Copy.Context), AST(Copy.AST) {
    Z3_inc_ref(Context.Context, AST);
  }

  /// Provide move constructor
  Z3Expr(Z3Expr &&Move) : Context(Move.Context), AST(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Expr &operator=(Z3Expr &&Move) {
    if (this != &Move) {
      if (AST)
        Z3_dec_ref(Context.Context, AST);
      AST = Move.AST;
      Move.AST = nullptr;
    }
    return *this;
  }

  ~Z3Expr() {
    if (AST)
      Z3_dec_ref(Context.Context, AST);
  }

  /// Get the corresponding IEEE floating-point type for a given bitwidth.
  static const llvm::fltSemantics &getFloatSemantics(unsigned BitWidth) {
    switch (BitWidth) {
    default:
      llvm_unreachable("Unsupported floating-point semantics!");
      break;
    case 16:
      return llvm::APFloat::IEEEhalf();
    case 32:
      return llvm::APFloat::IEEEsingle();
    case 64:
      return llvm::APFloat::IEEEdouble();
    case 128:
      return llvm::APFloat::IEEEquad();
    }
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Z3_get_ast_hash(Context.Context, AST));
  }

  bool operator<(const Z3Expr &Other) const {
    llvm::FoldingSetNodeID ID1, ID2;
    Profile(ID1);
    Other.Profile(ID2);
    return ID1 < ID2;
  }

  /// Comparison of AST equality, not model equivalence.
  bool operator==(const Z3Expr &Other) const {
    assert(Z3_is_eq_sort(Context.Context, Z3_get_sort(Context.Context, AST),
                         Z3_get_sort(Context.Context, Other.AST)) &&
           "AST's must have the same sort");
    return Z3_is_eq_ast(Context.Context, AST, Other.AST);
  }

  /// Override implicit move constructor for correct reference counting.
  Z3Expr &operator=(const Z3Expr &Move) {
    Z3_inc_ref(Context.Context, Move.AST);
    Z3_dec_ref(Context.Context, AST);
    AST = Move.AST;
    return *this;
  }

  void print(raw_ostream &OS) const {
    OS << Z3_ast_to_string(Context.Context, AST);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Expr

class Z3Model {
  friend class Z3Solver;

  Z3Context &Context;

  Z3_model Model;

public:
  Z3Model(Z3Context &C, Z3_model ZM) : Context(C), Model(ZM) {
    assert(C.Context != nullptr);
    Z3_model_inc_ref(Context.Context, Model);
  }

  /// Override implicit copy constructor for correct reference counting.
  Z3Model(const Z3Model &Copy) : Context(Copy.Context), Model(Copy.Model) {
    Z3_model_inc_ref(Context.Context, Model);
  }

  /// Provide move constructor
  Z3Model(Z3Model &&Move) : Context(Move.Context), Model(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Model &operator=(Z3Model &&Move) {
    if (this != &Move) {
      if (Model)
        Z3_model_dec_ref(Context.Context, Model);
      Model = Move.Model;
      Move.Model = nullptr;
    }
    return *this;
  }

  ~Z3Model() {
    if (Model)
      Z3_model_dec_ref(Context.Context, Model);
  }

  void print(raw_ostream &OS) const {
    OS << Z3_model_to_string(Context.Context, Model);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Model

class Z3Solver {
  friend class Z3ConstraintManager;

  Z3Context Context;

  Z3_solver Solver;

  Z3Solver() : Solver(Z3_mk_simple_solver(Context.Context)) {
    Z3_solver_inc_ref(Context.Context, Solver);
  }

public:
  /// Override implicit copy constructor for correct reference counting.
  Z3Solver(const Z3Solver &Copy) : Context(Copy.Context), Solver(Copy.Solver) {
    Z3_solver_inc_ref(Context.Context, Solver);
  }

  /// Provide move constructor
  Z3Solver(Z3Solver &&Move) : Context(Move.Context), Solver(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Solver &operator=(Z3Solver &&Move) {
    if (this != &Move) {
      if (Solver)
        Z3_solver_dec_ref(Context.Context, Solver);
      Solver = Move.Solver;
      Move.Solver = nullptr;
    }
    return *this;
  }

  ~Z3Solver() {
    if (Solver)
      Z3_solver_dec_ref(Context.Context, Solver);
  }

  /// Given a constraint, add it to the solver
  void addConstraint(const Z3Expr &Exp) {
    Z3_solver_assert(Context.Context, Solver, Exp.AST);
  }

  // Return a boolean sort.
  Z3Sort getBoolSort() {
    return Z3Sort(Context, Z3_mk_bool_sort(Context.Context));
  }

  // Return an appropriate bitvector sort for the given bitwidth.
  Z3Sort getBitvectorSort(unsigned BitWidth) {
    return Z3Sort(Context, Z3_mk_bv_sort(Context.Context, BitWidth));
  }

  // Return an appropriate floating-point sort for the given bitwidth.
  Z3Sort getFloatSort(unsigned BitWidth) {
    Z3_sort Sort;

    switch (BitWidth) {
    default:
      llvm_unreachable("Unsupported floating-point bitwidth!");
      break;
    case 16:
      Sort = Z3_mk_fpa_sort_16(Context.Context);
      break;
    case 32:
      Sort = Z3_mk_fpa_sort_32(Context.Context);
      break;
    case 64:
      Sort = Z3_mk_fpa_sort_64(Context.Context);
      break;
    case 128:
      Sort = Z3_mk_fpa_sort_128(Context.Context);
      break;
    }
    return Z3Sort(Context, Sort);
  }

  // Return an appropriate sort, given a QualType
  Z3Sort MkSort(const QualType &Ty, unsigned BitWidth) {
    if (Ty->isBooleanType())
      return getBoolSort();

    if (Ty->isRealFloatingType())
      return getFloatSort(BitWidth);

    return getBitvectorSort(BitWidth);
  }

  // Return an appropriate sort for the given AST.
  Z3Sort getSort(Z3_ast AST) {
    return Z3Sort(Context, Z3_get_sort(Context.Context, AST));
  }

  /// Given a program state, construct the logical conjunction and add it to
  /// the solver
  void addStateConstraints(ProgramStateRef State) {
    // TODO: Don't add all the constraints, only the relevant ones
    ConstraintZ3Ty CZ = State->get<ConstraintZ3>();
    ConstraintZ3Ty::iterator I = CZ.begin(), IE = CZ.end();

    // Construct the logical AND of all the constraints
    if (I != IE) {
      std::vector<Z3_ast> ASTs;

      while (I != IE)
        ASTs.push_back(I++->second.AST);

      Z3Expr Conj = fromNBinOp(BO_LAnd, ASTs);
      addConstraint(Conj);
    }
  }

  // Return an appropriate floating-point rounding mode.
  Z3Expr getFloatRoundingMode() {
    // TODO: Don't assume nearest ties to even rounding mode
    return Z3Expr(Context, Z3_mk_fpa_rne(Context.Context));
  }

  /// Construct a Z3Expr from a unary operator, given a Z3_context.
  Z3Expr fromUnOp(const UnaryOperator::Opcode Op, const Z3Expr &Exp) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    case UO_Minus:
      AST = Z3_mk_bvneg(Context.Context, Exp.AST);
      break;

    case UO_Not:
      AST = Z3_mk_bvnot(Context.Context, Exp.AST);
      break;

    case UO_LNot:
      AST = Z3_mk_not(Context.Context, Exp.AST);
      break;
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a floating-point unary operator, given a
  /// Z3_context.
  Z3Expr fromFloatUnOp(const UnaryOperator::Opcode Op, const Z3Expr &Exp) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    case UO_Minus:
      AST = Z3_mk_fpa_neg(Context.Context, Exp.AST);
      break;

    case UO_LNot:
      return fromUnOp(Op, Exp);
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a n-ary binary operator.
  Z3Expr fromNBinOp(const BinaryOperator::Opcode Op,
                    const std::vector<Z3_ast> &ASTs) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    case BO_LAnd:
      AST = Z3_mk_and(Context.Context, ASTs.size(), ASTs.data());
      break;

    case BO_LOr:
      AST = Z3_mk_or(Context.Context, ASTs.size(), ASTs.data());
      break;
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a binary operator, given a Z3_context.
  Z3Expr fromBinOp(const Z3Expr &LHS, const BinaryOperator::Opcode Op,
                   const Z3Expr &RHS, bool isSigned) {
    Z3_ast AST;

    assert(getSort(LHS.AST) == getSort(RHS.AST) &&
           "AST's must have the same sort!");

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

      // Multiplicative operators
    case BO_Mul:
      AST = Z3_mk_bvmul(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_Div:
      AST = isSigned ? Z3_mk_bvsdiv(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvudiv(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_Rem:
      AST = isSigned ? Z3_mk_bvsrem(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvurem(Context.Context, LHS.AST, RHS.AST);
      break;

      // Additive operators
    case BO_Add:
      AST = Z3_mk_bvadd(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_Sub:
      AST = Z3_mk_bvsub(Context.Context, LHS.AST, RHS.AST);
      break;

      // Bitwise shift operators
    case BO_Shl:
      AST = Z3_mk_bvshl(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_Shr:
      AST = isSigned ? Z3_mk_bvashr(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvlshr(Context.Context, LHS.AST, RHS.AST);
      break;

      // Relational operators
    case BO_LT:
      AST = isSigned ? Z3_mk_bvslt(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvult(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_GT:
      AST = isSigned ? Z3_mk_bvsgt(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvugt(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_LE:
      AST = isSigned ? Z3_mk_bvsle(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvule(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_GE:
      AST = isSigned ? Z3_mk_bvsge(Context.Context, LHS.AST, RHS.AST)
                     : Z3_mk_bvuge(Context.Context, LHS.AST, RHS.AST);
      break;

      // Equality operators
    case BO_EQ:
      AST = Z3_mk_eq(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_NE:
      return fromUnOp(UO_LNot, fromBinOp(LHS, BO_EQ, RHS, isSigned));
      break;

      // Bitwise operators
    case BO_And:
      AST = Z3_mk_bvand(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_Xor:
      AST = Z3_mk_bvxor(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_Or:
      AST = Z3_mk_bvor(Context.Context, LHS.AST, RHS.AST);
      break;

      // Logical operators
    case BO_LAnd:
    case BO_LOr: {
      std::vector<Z3_ast> Args = {LHS.AST, RHS.AST};
      return fromNBinOp(Op, Args);
    }
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a special floating-point binary operator, given
  /// a Z3_context.
  Z3Expr fromFloatSpecialBinOp(const Z3Expr &LHS,
                               const BinaryOperator::Opcode Op,
                               const llvm::APFloat::fltCategory &RHS) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

      // Equality operators
    case BO_EQ:
      switch (RHS) {
      case llvm::APFloat::fcInfinity:
        AST = Z3_mk_fpa_is_infinite(Context.Context, LHS.AST);
        break;
      case llvm::APFloat::fcNaN:
        AST = Z3_mk_fpa_is_nan(Context.Context, LHS.AST);
        break;
      case llvm::APFloat::fcNormal:
        AST = Z3_mk_fpa_is_normal(Context.Context, LHS.AST);
        break;
      case llvm::APFloat::fcZero:
        AST = Z3_mk_fpa_is_zero(Context.Context, LHS.AST);
        break;
      }
      break;
    case BO_NE:
      return fromFloatUnOp(UO_LNot, fromFloatSpecialBinOp(LHS, BO_EQ, RHS));
      break;
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a floating-point binary operator, given a
  /// Z3_context.
  Z3Expr fromFloatBinOp(const Z3Expr &LHS, const BinaryOperator::Opcode Op,
                        const Z3Expr &RHS) {
    Z3_ast AST;

    assert(getSort(LHS.AST) == getSort(RHS.AST) &&
           "AST's must have the same sort!");

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

      // Multiplicative operators
    case BO_Mul: {
      Z3Expr RoundingMode = getFloatRoundingMode();
      AST = Z3_mk_fpa_mul(Context.Context, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }
    case BO_Div: {
      Z3Expr RoundingMode = getFloatRoundingMode();
      AST = Z3_mk_fpa_div(Context.Context, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }
    case BO_Rem:
      AST = Z3_mk_fpa_rem(Context.Context, LHS.AST, RHS.AST);
      break;

      // Additive operators
    case BO_Add: {
      Z3Expr RoundingMode = getFloatRoundingMode();
      AST = Z3_mk_fpa_add(Context.Context, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }
    case BO_Sub: {
      Z3Expr RoundingMode = getFloatRoundingMode();
      AST = Z3_mk_fpa_sub(Context.Context, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }

      // Relational operators
    case BO_LT:
      AST = Z3_mk_fpa_lt(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_GT:
      AST = Z3_mk_fpa_gt(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_LE:
      AST = Z3_mk_fpa_leq(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_GE:
      AST = Z3_mk_fpa_geq(Context.Context, LHS.AST, RHS.AST);
      break;

      // Equality operators
    case BO_EQ:
      AST = Z3_mk_fpa_eq(Context.Context, LHS.AST, RHS.AST);
      break;
    case BO_NE:
      return fromFloatUnOp(UO_LNot, fromFloatBinOp(LHS, BO_EQ, RHS));
      break;

      // Logical operators
    case BO_LAnd:
    case BO_LOr:
      return fromBinOp(LHS, Op, RHS, false);
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a SymbolData, given a Z3_context.
  Z3Expr fromData(const SymbolID ID, const QualType &Ty, uint64_t BitWidth) {
    llvm::Twine Name = "$" + llvm::Twine(ID);

    Z3Sort Sort = MkSort(Ty, BitWidth);

    Z3_symbol Symbol = Z3_mk_string_symbol(Context.Context, Name.str().c_str());
    Z3_ast AST = Z3_mk_const(Context.Context, Symbol, Sort.Sort);
    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a SymbolCast, given a Z3_context.
  Z3Expr fromCast(const Z3Expr &Exp, QualType ToTy, uint64_t ToBitWidth,
                  QualType FromTy, uint64_t FromBitWidth) {
    Z3_ast AST;

    if ((FromTy->isIntegralOrEnumerationType() &&
         ToTy->isIntegralOrEnumerationType()) ||
        (FromTy->isAnyPointerType() ^ ToTy->isAnyPointerType()) ||
        (FromTy->isBlockPointerType() ^ ToTy->isBlockPointerType()) ||
        (FromTy->isReferenceType() ^ ToTy->isReferenceType())) {
      // Special case: Z3 boolean type is distinct from bitvector type, so
      // must use if-then-else expression instead of direct cast
      if (FromTy->isBooleanType()) {
        assert(ToBitWidth > 0 && "BitWidth must be positive!");
        Z3Expr Zero = fromInt("0", ToBitWidth);
        Z3Expr One = fromInt("1", ToBitWidth);
        AST = Z3_mk_ite(Context.Context, Exp.AST, One.AST, Zero.AST);
      } else if (ToBitWidth > FromBitWidth) {
        AST = FromTy->isSignedIntegerOrEnumerationType()
                  ? Z3_mk_sign_ext(Context.Context, ToBitWidth - FromBitWidth,
                                   Exp.AST)
                  : Z3_mk_zero_ext(Context.Context, ToBitWidth - FromBitWidth,
                                   Exp.AST);
      } else if (ToBitWidth < FromBitWidth) {
        AST = Z3_mk_extract(Context.Context, ToBitWidth - 1, 0, Exp.AST);
      } else {
        // Both are bitvectors with the same width, ignore the type cast
        return Exp;
      }
    } else if (FromTy->isRealFloatingType() && ToTy->isRealFloatingType()) {
      if (ToBitWidth != FromBitWidth) {
        Z3Expr RoundingMode = getFloatRoundingMode();
        Z3Sort Sort = getFloatSort(ToBitWidth);
        AST = Z3_mk_fpa_to_fp_float(Context.Context, RoundingMode.AST, Exp.AST,
                                    Sort.Sort);
      } else {
        return Exp;
      }
    } else if (FromTy->isIntegralOrEnumerationType() &&
               ToTy->isRealFloatingType()) {
      Z3Expr RoundingMode = getFloatRoundingMode();
      Z3Sort Sort = getFloatSort(ToBitWidth);
      AST = FromTy->isSignedIntegerOrEnumerationType()
                ? Z3_mk_fpa_to_fp_signed(Context.Context, RoundingMode.AST,
                                         Exp.AST, Sort.Sort)
                : Z3_mk_fpa_to_fp_unsigned(Context.Context, RoundingMode.AST,
                                           Exp.AST, Sort.Sort);
    } else if (FromTy->isRealFloatingType() &&
               ToTy->isIntegralOrEnumerationType()) {
      Z3Expr RoundingMode = getFloatRoundingMode();
      AST = ToTy->isSignedIntegerOrEnumerationType()
                ? Z3_mk_fpa_to_sbv(Context.Context, RoundingMode.AST, Exp.AST,
                                   ToBitWidth)
                : Z3_mk_fpa_to_ubv(Context.Context, RoundingMode.AST, Exp.AST,
                                   ToBitWidth);
    } else {
      llvm_unreachable("Unsupported explicit type cast!");
    }

    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a boolean, given a Z3_context.
  Z3Expr fromBoolean(const bool Bool) {
    Z3_ast AST =
        Bool ? Z3_mk_true(Context.Context) : Z3_mk_false(Context.Context);
    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from a finite APFloat, given a Z3_context.
  Z3Expr fromAPFloat(const llvm::APFloat &Float) {
    Z3_ast AST;
    Z3Sort Sort =
        getFloatSort(llvm::APFloat::semanticsSizeInBits(Float.getSemantics()));

    llvm::APSInt Int = llvm::APSInt(Float.bitcastToAPInt(), false);
    Z3Expr Z3Int = fromAPSInt(Int);
    AST = Z3_mk_fpa_to_fp_bv(Context.Context, Z3Int.AST, Sort.Sort);
    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from an APSInt, given a Z3_context.
  Z3Expr fromAPSInt(const llvm::APSInt &Int) {
    Z3Sort Sort = getBitvectorSort(Int.getBitWidth());
    Z3_ast AST =
        Z3_mk_numeral(Context.Context, Int.toString(10).c_str(), Sort.Sort);
    return Z3Expr(Context, AST);
  }

  /// Construct a Z3Expr from an integer, given a Z3_context.
  Z3Expr fromInt(const char *Int, uint64_t BitWidth) {
    Z3Sort Sort = getBitvectorSort(BitWidth);
    Z3_ast AST = Z3_mk_numeral(Context.Context, Int, Sort.Sort);
    return Z3Expr(Context, AST);
  }

  /// Construct an APFloat from a Z3Expr, given the AST representation
  bool toAPFloat(const Z3Sort &Sort, const Z3_ast &AST, llvm::APFloat &Float,
                 bool useSemantics = true) {
    assert(Sort.getSortKind() == Z3_FLOATING_POINT_SORT &&
           "Unsupported sort to floating-point!");

    llvm::APSInt Int(Sort.getFloatSortSize(), true);
    const llvm::fltSemantics &Semantics =
        Z3Expr::getFloatSemantics(Sort.getFloatSortSize());
    Z3Sort BVSort = getBitvectorSort(Sort.getFloatSortSize());
    if (!toAPSInt(BVSort, AST, Int, true)) {
      return false;
    }

    if (useSemantics &&
        !Z3Expr::areEquivalent(Float.getSemantics(), Semantics)) {
      assert(false && "Floating-point types don't match!");
      return false;
    }

    Float = llvm::APFloat(Semantics, Int);
    return true;
  }

  /// Construct an APSInt from a Z3Expr, given the AST representation
  bool toAPSInt(const Z3Sort &Sort, const Z3_ast &AST, llvm::APSInt &Int,
                bool useSemantics = true) {
    switch (Sort.getSortKind()) {
    default:
      llvm_unreachable("Unsupported sort to integer!");
    case Z3_BV_SORT: {
      if (useSemantics && Int.getBitWidth() != Sort.getBitvectorSortSize()) {
        assert(false && "Bitvector types don't match!");
        return false;
      }

      uint64_t Value[2];
      // Force cast because Z3 defines __uint64 to be a unsigned long long
      // type, which isn't compatible with a unsigned long type, even if they
      // are the same size.
      Z3_get_numeral_uint64(Context.Context, AST,
                            reinterpret_cast<__uint64 *>(&Value[0]));
      if (Sort.getBitvectorSortSize() <= 64) {
        Int = llvm::APSInt(llvm::APInt(Int.getBitWidth(), Value[0]),
                           Int.isUnsigned());
      } else if (Sort.getBitvectorSortSize() == 128) {
        Z3Expr ASTHigh =
            Z3Expr(Context, Z3_mk_extract(Context.Context, 127, 64, AST));
        Z3_get_numeral_uint64(Context.Context, AST,
                              reinterpret_cast<__uint64 *>(&Value[1]));
        Int = llvm::APSInt(llvm::APInt(Int.getBitWidth(), Value),
                           Int.isUnsigned());
      } else {
        assert(false && "Bitwidth not supported!");
        return false;
      }
      return true;
    }
    case Z3_BOOL_SORT:
      if (useSemantics && Int.getBitWidth() < 1) {
        assert(false && "Boolean type doesn't match!");
        return false;
      }
      Int = llvm::APSInt(
          llvm::APInt(Int.getBitWidth(),
                      Z3_get_bool_value(Context.Context, AST) == Z3_L_TRUE ? 1
                                                                           : 0),
          Int.isUnsigned());
      return true;
    }
  }

  /// Given an expression and a model, extract the value of this operand in
  /// the model.
  bool getInterpretation(const Z3Model Model, const Z3Expr &Exp,
                         llvm::APSInt &Int) {
    Z3_func_decl Func =
        Z3_get_app_decl(Context.Context, Z3_to_app(Context.Context, Exp.AST));
    if (Z3_model_has_interp(Context.Context, Model.Model, Func) != Z3_L_TRUE)
      return false;

    Z3_ast Assign =
        Z3_model_get_const_interp(Context.Context, Model.Model, Func);
    Z3Sort Sort = getSort(Assign);
    return toAPSInt(Sort, Assign, Int, true);
  }

  /// Given an expression and a model, extract the value of this operand in
  /// the model.
  bool getInterpretation(const Z3Model Model, const Z3Expr &Exp,
                         llvm::APFloat &Float) {
    Z3_func_decl Func =
        Z3_get_app_decl(Context.Context, Z3_to_app(Context.Context, Exp.AST));
    if (Z3_model_has_interp(Context.Context, Model.Model, Func) != Z3_L_TRUE)
      return false;

    Z3_ast Assign =
        Z3_model_get_const_interp(Context.Context, Model.Model, Func);
    Z3Sort Sort = getSort(Assign);
    return toAPFloat(Sort, Assign, Float, true);
  }

  // Callback function for doCast parameter on APSInt type.
  llvm::APSInt castAPSInt(const llvm::APSInt &V, QualType ToTy,
                          uint64_t ToWidth, QualType FromTy,
                          uint64_t FromWidth) {
    APSIntType TargetType(ToWidth, !ToTy->isSignedIntegerOrEnumerationType());
    return TargetType.convert(V);
  }

  /// Check if the constraints are satisfiable
  Z3_lbool check() { return Z3_solver_check(Context.Context, Solver); }

  /// Push the current solver state
  void push() { return Z3_solver_push(Context.Context, Solver); }

  /// Pop the previous solver state
  void pop(unsigned NumStates = 1) {
    assert(Z3_solver_get_num_scopes(Context.Context, Solver) >= NumStates);
    return Z3_solver_pop(Context.Context, Solver, NumStates);
  }

  /// Get a model from the solver. Caller should check the model is
  /// satisfiable.
  Z3Model getModel() {
    return Z3Model(Context, Z3_solver_get_model(Context.Context, Solver));
  }

  /// Reset the solver and remove all constraints.
  void reset() { Z3_solver_reset(Context.Context, Solver); }

  void print(raw_ostream &OS) const {
    OS << Z3_solver_to_string(Context.Context, Solver);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Solver

class Z3ConstraintManager : public SMTConstraintManager {
  mutable Z3Solver Solver;

public:
  Z3ConstraintManager(SubEngine *SE, SValBuilder &SB)
      : SMTConstraintManager(SE, SB) {}

  //===------------------------------------------------------------------===//
  // Implementation for Refutation.
  //===------------------------------------------------------------------===//

  void addRangeConstraints(clang::ento::ConstraintRangeTy CR) override;

  ConditionTruthVal isModelFeasible() override;

  LLVM_DUMP_METHOD void dump() const override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from ConstraintManager.
  //===------------------------------------------------------------------===//

  bool canReasonAbout(SVal X) const override;

  ConditionTruthVal checkNull(ProgramStateRef State, SymbolRef Sym) override;

  const llvm::APSInt *getSymVal(ProgramStateRef State,
                                SymbolRef Sym) const override;

  ProgramStateRef removeDeadBindings(ProgramStateRef St,
                                     SymbolReaper &SymReaper) override;

  void print(ProgramStateRef St, raw_ostream &Out, const char *nl,
             const char *sep) override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from SimpleConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSym(ProgramStateRef state, SymbolRef Sym,
                            bool Assumption) override;

  ProgramStateRef assumeSymInclusiveRange(ProgramStateRef State, SymbolRef Sym,
                                          const llvm::APSInt &From,
                                          const llvm::APSInt &To,
                                          bool InRange) override;

  ProgramStateRef assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                       bool Assumption) override;

private:
  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//

  // Check whether a new model is satisfiable, and update the program state.
  ProgramStateRef assumeZ3Expr(ProgramStateRef State, SymbolRef Sym,
                               const Z3Expr &Exp);

  // Generate and check a Z3 model, using the given constraint.
  Z3_lbool checkZ3Model(ProgramStateRef State, const Z3Expr &Exp) const;

  // Generate a Z3Expr that represents the given symbolic expression.
  // Sets the hasComparison parameter if the expression has a comparison
  // operator.
  // Sets the RetTy parameter to the final return type after promotions and
  // casts.
  Z3Expr getZ3Expr(SymbolRef Sym, QualType *RetTy = nullptr,
                   bool *hasComparison = nullptr) const;

  // Generate a Z3Expr that takes the logical not of an expression.
  Z3Expr getZ3NotExpr(const Z3Expr &Exp) const;

  // Generate a Z3Expr that compares the expression to zero.
  Z3Expr getZ3ZeroExpr(const Z3Expr &Exp, QualType RetTy,
                       bool Assumption) const;

  // Recursive implementation to unpack and generate symbolic expression.
  // Sets the hasComparison and RetTy parameters. See getZ3Expr().
  Z3Expr getZ3SymExpr(SymbolRef Sym, QualType *RetTy,
                      bool *hasComparison) const;

  // Wrapper to generate Z3Expr from SymbolData.
  Z3Expr getZ3DataExpr(const SymbolID ID, QualType Ty) const;

  // Wrapper to generate Z3Expr from SymbolCast.
  Z3Expr getZ3CastExpr(const Z3Expr &Exp, QualType FromTy, QualType Ty) const;

  // Wrapper to generate Z3Expr from BinarySymExpr.
  // Sets the hasComparison and RetTy parameters. See getZ3Expr().
  Z3Expr getZ3SymBinExpr(const BinarySymExpr *BSE, bool *hasComparison,
                         QualType *RetTy) const;

  // Wrapper to generate Z3Expr from unpacked binary symbolic expression.
  // Sets the RetTy parameter. See getZ3Expr().
  Z3Expr getZ3BinExpr(const Z3Expr &LHS, QualType LTy,
                      BinaryOperator::Opcode Op, const Z3Expr &RHS,
                      QualType RTy, QualType *RetTy) const;

  // Wrapper to generate Z3Expr from a range. If From == To, an equality will
  // be created instead.
  Z3Expr getZ3RangeExpr(SymbolRef Sym, const llvm::APSInt &From,
                        const llvm::APSInt &To, bool InRange);

  //===------------------------------------------------------------------===//
  // Helper functions.
  //===------------------------------------------------------------------===//

  // Recover the QualType of an APSInt.
  // TODO: Refactor to put elsewhere
  QualType getAPSIntType(const llvm::APSInt &Int) const;

  // Get the QualTy for the input APSInt, and fix it if it has a bitwidth of 1.
  std::pair<llvm::APSInt, QualType> fixAPSInt(const llvm::APSInt &Int) const;

  // Perform implicit type conversion on binary symbolic expressions.
  // May modify all input parameters.
  // TODO: Refactor to use built-in conversion functions
  void doTypeConversion(Z3Expr &LHS, Z3Expr &RHS, QualType &LTy,
                        QualType &RTy) const;

  // Perform implicit integer type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleIntegerConversion()
  template <typename T, T (Z3Solver::*doCast)(const T &, QualType, uint64_t,
                                              QualType, uint64_t)>
  void doIntTypeConversion(T &LHS, QualType &LTy, T &RHS, QualType &RTy) const;

  // Perform implicit floating-point type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleFloatConversion()
  template <typename T, T (Z3Solver::*doCast)(const T &, QualType, uint64_t,
                                              QualType, uint64_t)>
  void doFloatTypeConversion(T &LHS, QualType &LTy, T &RHS,
                             QualType &RTy) const;
}; // end class Z3ConstraintManager

} // end anonymous namespace

ProgramStateRef Z3ConstraintManager::assumeSym(ProgramStateRef State,
                                               SymbolRef Sym, bool Assumption) {
  QualType RetTy;
  bool hasComparison;

  Z3Expr Exp = getZ3Expr(Sym, &RetTy, &hasComparison);
  // Create zero comparison for implicit boolean cast, with reversed assumption
  if (!hasComparison && !RetTy->isBooleanType())
    return assumeZ3Expr(State, Sym, getZ3ZeroExpr(Exp, RetTy, !Assumption));

  return assumeZ3Expr(State, Sym, Assumption ? Exp : getZ3NotExpr(Exp));
}

ProgramStateRef Z3ConstraintManager::assumeSymInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, bool InRange) {
  return assumeZ3Expr(State, Sym, getZ3RangeExpr(Sym, From, To, InRange));
}

ProgramStateRef Z3ConstraintManager::assumeSymUnsupported(ProgramStateRef State,
                                                          SymbolRef Sym,
                                                          bool Assumption) {
  // Skip anything that is unsupported
  return State;
}

bool Z3ConstraintManager::canReasonAbout(SVal X) const {
  const TargetInfo &TI = getBasicVals().getContext().getTargetInfo();

  Optional<nonloc::SymbolVal> SymVal = X.getAs<nonloc::SymbolVal>();
  if (!SymVal)
    return true;

  const SymExpr *Sym = SymVal->getSymbol();
  QualType Ty = Sym->getType();

  // Complex types are not modeled
  if (Ty->isComplexType() || Ty->isComplexIntegerType())
    return false;

  // Non-IEEE 754 floating-point types are not modeled
  if ((Ty->isSpecificBuiltinType(BuiltinType::LongDouble) &&
       (&TI.getLongDoubleFormat() == &llvm::APFloat::x87DoubleExtended() ||
        &TI.getLongDoubleFormat() == &llvm::APFloat::PPCDoubleDouble())))
    return false;

  if (isa<SymbolData>(Sym))
    return true;

  SValBuilder &SVB = getSValBuilder();

  if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym))
    return canReasonAbout(SVB.makeSymbolVal(SC->getOperand()));

  if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE))
      return canReasonAbout(SVB.makeSymbolVal(SIE->getLHS()));

    if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE))
      return canReasonAbout(SVB.makeSymbolVal(ISE->getRHS()));

    if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(BSE))
      return canReasonAbout(SVB.makeSymbolVal(SSE->getLHS())) &&
             canReasonAbout(SVB.makeSymbolVal(SSE->getRHS()));
  }

  llvm_unreachable("Unsupported expression to reason about!");
}

ConditionTruthVal Z3ConstraintManager::checkNull(ProgramStateRef State,
                                                 SymbolRef Sym) {
  QualType RetTy;
  // The expression may be casted, so we cannot call getZ3DataExpr() directly
  Z3Expr VarExp = getZ3Expr(Sym, &RetTy);
  Z3Expr Exp = getZ3ZeroExpr(VarExp, RetTy, true);
  // Negate the constraint
  Z3Expr NotExp = getZ3ZeroExpr(VarExp, RetTy, false);

  Solver.reset();
  Solver.addStateConstraints(State);

  Solver.push();
  Solver.addConstraint(Exp);
  Z3_lbool isSat = Solver.check();

  Solver.pop();
  Solver.addConstraint(NotExp);
  Z3_lbool isNotSat = Solver.check();

  // Zero is the only possible solution
  if (isSat == Z3_L_TRUE && isNotSat == Z3_L_FALSE)
    return true;
  // Zero is not a solution
  else if (isSat == Z3_L_FALSE && isNotSat == Z3_L_TRUE)
    return false;

  // Zero may be a solution
  return ConditionTruthVal();
}

const llvm::APSInt *Z3ConstraintManager::getSymVal(ProgramStateRef State,
                                                   SymbolRef Sym) const {
  BasicValueFactory &BVF = getBasicVals();
  ASTContext &Ctx = BVF.getContext();

  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    QualType Ty = Sym->getType();
    assert(!Ty->isRealFloatingType());
    llvm::APSInt Value(Ctx.getTypeSize(Ty),
                       !Ty->isSignedIntegerOrEnumerationType());

    Z3Expr Exp = getZ3DataExpr(SD->getSymbolID(), Ty);

    Solver.reset();
    Solver.addStateConstraints(State);

    // Constraints are unsatisfiable
    if (Solver.check() != Z3_L_TRUE)
      return nullptr;

    Z3Model Model = Solver.getModel();
    // Model does not assign interpretation
    if (!Solver.getInterpretation(Model, Exp, Value))
      return nullptr;

    // A value has been obtained, check if it is the only value
    Z3Expr NotExp = Solver.fromBinOp(
        Exp, BO_NE,
        Ty->isBooleanType() ? Solver.fromBoolean(Value.getBoolValue())
                            : Solver.fromAPSInt(Value),
        false);

    Solver.addConstraint(NotExp);
    if (Solver.check() == Z3_L_TRUE)
      return nullptr;

    // This is the only solution, store it
    return &BVF.getValue(Value);
  } else if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    SymbolRef CastSym = SC->getOperand();
    QualType CastTy = SC->getType();
    // Skip the void type
    if (CastTy->isVoidType())
      return nullptr;

    const llvm::APSInt *Value;
    if (!(Value = getSymVal(State, CastSym)))
      return nullptr;
    return &BVF.Convert(SC->getType(), *Value);
  } else if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    const llvm::APSInt *LHS, *RHS;
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
      LHS = getSymVal(State, SIE->getLHS());
      RHS = &SIE->getRHS();
    } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
      LHS = &ISE->getLHS();
      RHS = getSymVal(State, ISE->getRHS());
    } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
      // Early termination to avoid expensive call
      LHS = getSymVal(State, SSM->getLHS());
      RHS = LHS ? getSymVal(State, SSM->getRHS()) : nullptr;
    } else {
      llvm_unreachable("Unsupported binary expression to get symbol value!");
    }

    if (!LHS || !RHS)
      return nullptr;

    llvm::APSInt ConvertedLHS, ConvertedRHS;
    QualType LTy, RTy;
    std::tie(ConvertedLHS, LTy) = fixAPSInt(*LHS);
    std::tie(ConvertedRHS, RTy) = fixAPSInt(*RHS);
    doIntTypeConversion<llvm::APSInt, &Z3Solver::castAPSInt>(ConvertedLHS, LTy,
                                                             ConvertedRHS, RTy);
    return BVF.evalAPSInt(BSE->getOpcode(), ConvertedLHS, ConvertedRHS);
  }

  llvm_unreachable("Unsupported expression to get symbol value!");
}

ProgramStateRef
Z3ConstraintManager::removeDeadBindings(ProgramStateRef State,
                                        SymbolReaper &SymReaper) {
  ConstraintZ3Ty CZ = State->get<ConstraintZ3>();
  ConstraintZ3Ty::Factory &CZFactory = State->get_context<ConstraintZ3>();

  for (ConstraintZ3Ty::iterator I = CZ.begin(), E = CZ.end(); I != E; ++I) {
    if (SymReaper.maybeDead(I->first))
      CZ = CZFactory.remove(CZ, *I);
  }

  return State->set<ConstraintZ3>(CZ);
}

void Z3ConstraintManager::addRangeConstraints(ConstraintRangeTy CR) {
  Solver.reset();

  for (const auto &I : CR) {
    SymbolRef Sym = I.first;

    Z3Expr Constraints = Solver.fromBoolean(false);
    for (const auto &Range : I.second) {
      Z3Expr SymRange =
          getZ3RangeExpr(Sym, Range.From(), Range.To(), /*InRange=*/true);

      // FIXME: the last argument (isSigned) is not used when generating the
      // or expression, as both arguments are booleans
      Constraints =
          Solver.fromBinOp(Constraints, BO_LOr, SymRange, /*IsSigned=*/true);
    }
    Solver.addConstraint(Constraints);
  }
}

clang::ento::ConditionTruthVal Z3ConstraintManager::isModelFeasible() {
  if (Solver.check() == Z3_L_FALSE)
    return false;

  return ConditionTruthVal();
}

LLVM_DUMP_METHOD void Z3ConstraintManager::dump() const { Solver.dump(); }

//===------------------------------------------------------------------===//
// Internal implementation.
//===------------------------------------------------------------------===//

ProgramStateRef Z3ConstraintManager::assumeZ3Expr(ProgramStateRef State,
                                                  SymbolRef Sym,
                                                  const Z3Expr &Exp) {
  // Check the model, avoid simplifying AST to save time
  if (checkZ3Model(State, Exp) == Z3_L_TRUE)
    return State->add<ConstraintZ3>(std::make_pair(Sym, Exp));

  return nullptr;
}

Z3_lbool Z3ConstraintManager::checkZ3Model(ProgramStateRef State,
                                           const Z3Expr &Exp) const {
  Solver.reset();
  Solver.addConstraint(Exp);
  Solver.addStateConstraints(State);
  return Solver.check();
}

Z3Expr Z3ConstraintManager::getZ3Expr(SymbolRef Sym, QualType *RetTy,
                                      bool *hasComparison) const {
  if (hasComparison) {
    *hasComparison = false;
  }

  return getZ3SymExpr(Sym, RetTy, hasComparison);
}

Z3Expr Z3ConstraintManager::getZ3NotExpr(const Z3Expr &Exp) const {
  return Solver.fromUnOp(UO_LNot, Exp);
}

Z3Expr Z3ConstraintManager::getZ3ZeroExpr(const Z3Expr &Exp, QualType Ty,
                                          bool Assumption) const {
  ASTContext &Ctx = getBasicVals().getContext();
  if (Ty->isRealFloatingType()) {
    llvm::APFloat Zero = llvm::APFloat::getZero(Ctx.getFloatTypeSemantics(Ty));
    return Solver.fromFloatBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                                 Solver.fromAPFloat(Zero));
  } else if (Ty->isIntegralOrEnumerationType() || Ty->isAnyPointerType() ||
             Ty->isBlockPointerType() || Ty->isReferenceType()) {
    bool isSigned = Ty->isSignedIntegerOrEnumerationType();
    // Skip explicit comparison for boolean types
    if (Ty->isBooleanType())
      return Assumption ? getZ3NotExpr(Exp) : Exp;
    return Solver.fromBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                            Solver.fromInt("0", Ctx.getTypeSize(Ty)), isSigned);
  }

  llvm_unreachable("Unsupported type for zero value!");
}

Z3Expr Z3ConstraintManager::getZ3SymExpr(SymbolRef Sym, QualType *RetTy,
                                         bool *hasComparison) const {
  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    if (RetTy)
      *RetTy = Sym->getType();

    return getZ3DataExpr(SD->getSymbolID(), Sym->getType());
  } else if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    if (RetTy)
      *RetTy = Sym->getType();

    QualType FromTy;
    Z3Expr Exp = getZ3SymExpr(SC->getOperand(), &FromTy, hasComparison);
    // Casting an expression with a comparison invalidates it. Note that this
    // must occur after the recursive call above.
    // e.g. (signed char) (x > 0)
    if (hasComparison)
      *hasComparison = false;
    return getZ3CastExpr(Exp, FromTy, Sym->getType());
  } else if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    Z3Expr Exp = getZ3SymBinExpr(BSE, hasComparison, RetTy);
    // Set the hasComparison parameter, in post-order traversal order.
    if (hasComparison)
      *hasComparison = BinaryOperator::isComparisonOp(BSE->getOpcode());
    return Exp;
  }

  llvm_unreachable("Unsupported SymbolRef type!");
}

Z3Expr Z3ConstraintManager::getZ3DataExpr(const SymbolID ID,
                                          QualType Ty) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Solver.fromData(ID, Ty, Ctx.getTypeSize(Ty));
}

Z3Expr Z3ConstraintManager::getZ3CastExpr(const Z3Expr &Exp, QualType FromTy,
                                          QualType ToTy) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Solver.fromCast(Exp, ToTy, Ctx.getTypeSize(ToTy), FromTy,
                         Ctx.getTypeSize(FromTy));
}

Z3Expr Z3ConstraintManager::getZ3SymBinExpr(const BinarySymExpr *BSE,
                                            bool *hasComparison,
                                            QualType *RetTy) const {
  QualType LTy, RTy;
  BinaryOperator::Opcode Op = BSE->getOpcode();

  if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
    Z3Expr LHS = getZ3SymExpr(SIE->getLHS(), &LTy, hasComparison);
    llvm::APSInt NewRInt;
    std::tie(NewRInt, RTy) = fixAPSInt(SIE->getRHS());
    Z3Expr RHS = Solver.fromAPSInt(NewRInt);
    return getZ3BinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
    llvm::APSInt NewLInt;
    std::tie(NewLInt, LTy) = fixAPSInt(ISE->getLHS());
    Z3Expr LHS = Solver.fromAPSInt(NewLInt);
    Z3Expr RHS = getZ3SymExpr(ISE->getRHS(), &RTy, hasComparison);
    return getZ3BinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
    Z3Expr LHS = getZ3SymExpr(SSM->getLHS(), &LTy, hasComparison);
    Z3Expr RHS = getZ3SymExpr(SSM->getRHS(), &RTy, hasComparison);
    return getZ3BinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  } else {
    llvm_unreachable("Unsupported BinarySymExpr type!");
  }
}

Z3Expr Z3ConstraintManager::getZ3BinExpr(const Z3Expr &LHS, QualType LTy,
                                         BinaryOperator::Opcode Op,
                                         const Z3Expr &RHS, QualType RTy,
                                         QualType *RetTy) const {
  Z3Expr NewLHS = LHS;
  Z3Expr NewRHS = RHS;
  doTypeConversion(NewLHS, NewRHS, LTy, RTy);
  // Update the return type parameter if the output type has changed.
  if (RetTy) {
    // A boolean result can be represented as an integer type in C/C++, but at
    // this point we only care about the Z3 type. Set it as a boolean type to
    // avoid subsequent Z3 errors.
    if (BinaryOperator::isComparisonOp(Op) || BinaryOperator::isLogicalOp(Op)) {
      ASTContext &Ctx = getBasicVals().getContext();
      *RetTy = Ctx.BoolTy;
    } else {
      *RetTy = LTy;
    }

    // If the two operands are pointers and the operation is a subtraction, the
    // result is of type ptrdiff_t, which is signed
    if (LTy->isAnyPointerType() && RTy->isAnyPointerType() && Op == BO_Sub) {
      *RetTy = getBasicVals().getContext().getPointerDiffType();
    }
  }

  return LTy->isRealFloatingType()
             ? Solver.fromFloatBinOp(NewLHS, Op, NewRHS)
             : Solver.fromBinOp(NewLHS, Op, NewRHS,
                                LTy->isSignedIntegerOrEnumerationType());
}

Z3Expr Z3ConstraintManager::getZ3RangeExpr(SymbolRef Sym,
                                           const llvm::APSInt &From,
                                           const llvm::APSInt &To,
                                           bool InRange) {
  // Convert lower bound
  QualType FromTy;
  llvm::APSInt NewFromInt;
  std::tie(NewFromInt, FromTy) = fixAPSInt(From);
  Z3Expr FromExp = Solver.fromAPSInt(NewFromInt);

  // Convert symbol
  QualType SymTy;
  Z3Expr Exp = getZ3Expr(Sym, &SymTy);

  // Construct single (in)equality
  if (From == To)
    return getZ3BinExpr(Exp, SymTy, InRange ? BO_EQ : BO_NE, FromExp, FromTy,
                        /*RetTy=*/nullptr);

  QualType ToTy;
  llvm::APSInt NewToInt;
  std::tie(NewToInt, ToTy) = fixAPSInt(To);
  Z3Expr ToExp = Solver.fromAPSInt(NewToInt);
  assert(FromTy == ToTy && "Range values have different types!");

  // Construct two (in)equalities, and a logical and/or
  Z3Expr LHS = getZ3BinExpr(Exp, SymTy, InRange ? BO_GE : BO_LT, FromExp,
                            FromTy, /*RetTy=*/nullptr);
  Z3Expr RHS = getZ3BinExpr(Exp, SymTy, InRange ? BO_LE : BO_GT, ToExp, ToTy,
                            /*RetTy=*/nullptr);

  return Solver.fromBinOp(LHS, InRange ? BO_LAnd : BO_LOr, RHS,
                          SymTy->isSignedIntegerOrEnumerationType());
}

//===------------------------------------------------------------------===//
// Helper functions.
//===------------------------------------------------------------------===//

QualType Z3ConstraintManager::getAPSIntType(const llvm::APSInt &Int) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Ctx.getIntTypeForBitwidth(Int.getBitWidth(), Int.isSigned());
}

std::pair<llvm::APSInt, QualType>
Z3ConstraintManager::fixAPSInt(const llvm::APSInt &Int) const {
  llvm::APSInt NewInt;

  // FIXME: This should be a cast from a 1-bit integer type to a boolean type,
  // but the former is not available in Clang. Instead, extend the APSInt
  // directly.
  if (Int.getBitWidth() == 1 && getAPSIntType(Int).isNull()) {
    ASTContext &Ctx = getBasicVals().getContext();
    NewInt = Int.extend(Ctx.getTypeSize(Ctx.BoolTy));
  } else
    NewInt = Int;

  return std::make_pair(NewInt, getAPSIntType(NewInt));
}

void Z3ConstraintManager::doTypeConversion(Z3Expr &LHS, Z3Expr &RHS,
                                           QualType &LTy, QualType &RTy) const {
  ASTContext &Ctx = getBasicVals().getContext();

  assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");
  // Perform type conversion
  if (LTy->isIntegralOrEnumerationType() &&
      RTy->isIntegralOrEnumerationType()) {
    if (LTy->isArithmeticType() && RTy->isArithmeticType())
      return doIntTypeConversion<Z3Expr, &Z3Solver::fromCast>(LHS, LTy, RHS,
                                                              RTy);
  } else if (LTy->isRealFloatingType() || RTy->isRealFloatingType()) {
    return doFloatTypeConversion<Z3Expr, &Z3Solver::fromCast>(LHS, LTy, RHS,
                                                              RTy);
  } else if ((LTy->isAnyPointerType() || RTy->isAnyPointerType()) ||
             (LTy->isBlockPointerType() || RTy->isBlockPointerType()) ||
             (LTy->isReferenceType() || RTy->isReferenceType())) {
    // TODO: Refactor to Sema::FindCompositePointerType(), and
    // Sema::CheckCompareOperands().

    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    // Cast the non-pointer type to the pointer type.
    // TODO: Be more strict about this.
    if ((LTy->isAnyPointerType() ^ RTy->isAnyPointerType()) ||
        (LTy->isBlockPointerType() ^ RTy->isBlockPointerType()) ||
        (LTy->isReferenceType() ^ RTy->isReferenceType())) {
      if (LTy->isNullPtrType() || LTy->isBlockPointerType() ||
          LTy->isReferenceType()) {
        LHS = Solver.fromCast(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      } else {
        RHS = Solver.fromCast(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      }
    }

    // Cast the void pointer type to the non-void pointer type.
    // For void types, this assumes that the casted value is equal to the value
    // of the original pointer, and does not account for alignment requirements.
    if (LTy->isVoidPointerType() ^ RTy->isVoidPointerType()) {
      assert((Ctx.getTypeSize(LTy) == Ctx.getTypeSize(RTy)) &&
             "Pointer types have different bitwidths!");
      if (RTy->isVoidPointerType())
        RTy = LTy;
      else
        LTy = RTy;
    }

    if (LTy == RTy)
      return;
  }

  // Fallback: for the solver, assume that these types don't really matter
  if ((LTy.getCanonicalType() == RTy.getCanonicalType()) ||
      (LTy->isObjCObjectPointerType() && RTy->isObjCObjectPointerType())) {
    LTy = RTy;
    return;
  }

  // TODO: Refine behavior for invalid type casts
}

template <typename T, T (Z3Solver::*doCast)(const T &, QualType, uint64_t,
                                            QualType, uint64_t)>
void Z3ConstraintManager::doIntTypeConversion(T &LHS, QualType &LTy, T &RHS,
                                              QualType &RTy) const {
  ASTContext &Ctx = getBasicVals().getContext();
  uint64_t LBitWidth = Ctx.getTypeSize(LTy);
  uint64_t RBitWidth = Ctx.getTypeSize(RTy);

  assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");
  // Always perform integer promotion before checking type equality.
  // Otherwise, e.g. (bool) a + (bool) b could trigger a backend assertion
  if (LTy->isPromotableIntegerType()) {
    QualType NewTy = Ctx.getPromotedIntegerType(LTy);
    uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
    LHS = (Solver.*doCast)(LHS, NewTy, NewBitWidth, LTy, LBitWidth);
    LTy = NewTy;
    LBitWidth = NewBitWidth;
  }
  if (RTy->isPromotableIntegerType()) {
    QualType NewTy = Ctx.getPromotedIntegerType(RTy);
    uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
    RHS = (Solver.*doCast)(RHS, NewTy, NewBitWidth, RTy, RBitWidth);
    RTy = NewTy;
    RBitWidth = NewBitWidth;
  }

  if (LTy == RTy)
    return;

  // Perform integer type conversion
  // Note: Safe to skip updating bitwidth because this must terminate
  bool isLSignedTy = LTy->isSignedIntegerOrEnumerationType();
  bool isRSignedTy = RTy->isSignedIntegerOrEnumerationType();

  int order = Ctx.getIntegerTypeOrder(LTy, RTy);
  if (isLSignedTy == isRSignedTy) {
    // Same signedness; use the higher-ranked type
    if (order == 1) {
      RHS = (Solver.*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else {
      LHS = (Solver.*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    }
  } else if (order != (isLSignedTy ? 1 : -1)) {
    // The unsigned type has greater than or equal rank to the
    // signed type, so use the unsigned type
    if (isRSignedTy) {
      RHS = (Solver.*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else {
      LHS = (Solver.*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    }
  } else if (LBitWidth != RBitWidth) {
    // The two types are different widths; if we are here, that
    // means the signed type is larger than the unsigned type, so
    // use the signed type.
    if (isLSignedTy) {
      RHS = (Solver.*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else {
      LHS = (Solver.*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    }
  } else {
    // The signed type is higher-ranked than the unsigned type,
    // but isn't actually any bigger (like unsigned int and long
    // on most 32-bit systems).  Use the unsigned type corresponding
    // to the signed type.
    QualType NewTy = Ctx.getCorrespondingUnsignedType(isLSignedTy ? LTy : RTy);
    RHS = (Solver.*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
    RTy = NewTy;
    LHS = (Solver.*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
    LTy = NewTy;
  }
}

template <typename T, T (Z3Solver::*doCast)(const T &, QualType, uint64_t,
                                            QualType, uint64_t)>
void Z3ConstraintManager::doFloatTypeConversion(T &LHS, QualType &LTy, T &RHS,
                                                QualType &RTy) const {
  ASTContext &Ctx = getBasicVals().getContext();

  uint64_t LBitWidth = Ctx.getTypeSize(LTy);
  uint64_t RBitWidth = Ctx.getTypeSize(RTy);

  // Perform float-point type promotion
  if (!LTy->isRealFloatingType()) {
    LHS = (Solver.*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
    LTy = RTy;
    LBitWidth = RBitWidth;
  }
  if (!RTy->isRealFloatingType()) {
    RHS = (Solver.*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
    RTy = LTy;
    RBitWidth = LBitWidth;
  }

  if (LTy == RTy)
    return;

  // If we have two real floating types, convert the smaller operand to the
  // bigger result
  // Note: Safe to skip updating bitwidth because this must terminate
  int order = Ctx.getFloatingTypeOrder(LTy, RTy);
  if (order > 0) {
    RHS = (Solver.*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
    RTy = LTy;
  } else if (order == 0) {
    LHS = (Solver.*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
    LTy = RTy;
  } else {
    llvm_unreachable("Unsupported floating-point type cast!");
  }
}

//==------------------------------------------------------------------------==/
// Pretty-printing.
//==------------------------------------------------------------------------==/

void Z3ConstraintManager::print(ProgramStateRef St, raw_ostream &OS,
                                const char *nl, const char *sep) {

  ConstraintZ3Ty CZ = St->get<ConstraintZ3>();

  OS << nl << sep << "Constraints:";
  for (ConstraintZ3Ty::iterator I = CZ.begin(), E = CZ.end(); I != E; ++I) {
    OS << nl << ' ' << I->first << " : ";
    I->second.print(OS);
  }
  OS << nl;
}

#endif

std::unique_ptr<ConstraintManager>
ento::CreateZ3ConstraintManager(ProgramStateManager &StMgr, SubEngine *Eng) {
#if CLANG_ANALYZER_WITH_Z3
  return llvm::make_unique<Z3ConstraintManager>(Eng, StMgr.getSValBuilder());
#else
  llvm::report_fatal_error("Clang was not compiled with Z3 support, rebuild "
                           "with -DCLANG_ANALYZER_BUILD_Z3=ON",
                           false);
  return nullptr;
#endif
}
