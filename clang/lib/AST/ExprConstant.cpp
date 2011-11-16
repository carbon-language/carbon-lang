//===--- ExprConstant.cpp - Expression Constant Evaluator -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr constant evaluator.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include <cstring>

using namespace clang;
using llvm::APSInt;
using llvm::APFloat;

/// EvalInfo - This is a private struct used by the evaluator to capture
/// information about a subexpression as it is folded.  It retains information
/// about the AST context, but also maintains information about the folded
/// expression.
///
/// If an expression could be evaluated, it is still possible it is not a C
/// "integer constant expression" or constant expression.  If not, this struct
/// captures information about how and why not.
///
/// One bit of information passed *into* the request for constant folding
/// indicates whether the subexpression is "evaluated" or not according to C
/// rules.  For example, the RHS of (0 && foo()) is not evaluated.  We can
/// evaluate the expression regardless of what the RHS is, but C only allows
/// certain things in certain situations.
namespace {
  struct LValue;
  struct CallStackFrame;
  struct EvalInfo;

  QualType getType(APValue::LValueBase B) {
    if (!B) return QualType();
    if (const ValueDecl *D = B.dyn_cast<const ValueDecl*>())
      return D->getType();
    return B.get<const Expr*>()->getType();
  }

  /// Get an LValue path entry, which is known to not be an array index, as a
  /// field declaration.
  const FieldDecl *getAsField(APValue::LValuePathEntry E) {
    APValue::BaseOrMemberType Value;
    Value.setFromOpaqueValue(E.BaseOrMember);
    return dyn_cast<FieldDecl>(Value.getPointer());
  }
  /// Get an LValue path entry, which is known to not be an array index, as a
  /// base class declaration.
  const CXXRecordDecl *getAsBaseClass(APValue::LValuePathEntry E) {
    APValue::BaseOrMemberType Value;
    Value.setFromOpaqueValue(E.BaseOrMember);
    return dyn_cast<CXXRecordDecl>(Value.getPointer());
  }
  /// Determine whether this LValue path entry for a base class names a virtual
  /// base class.
  bool isVirtualBaseClass(APValue::LValuePathEntry E) {
    APValue::BaseOrMemberType Value;
    Value.setFromOpaqueValue(E.BaseOrMember);
    return Value.getInt();
  }

  /// Determine whether the described subobject is an array element.
  static bool SubobjectIsArrayElement(QualType Base,
                                      ArrayRef<APValue::LValuePathEntry> Path) {
    bool IsArrayElement = false;
    const Type *T = Base.getTypePtr();
    for (unsigned I = 0, N = Path.size(); I != N; ++I) {
      IsArrayElement = T && T->isArrayType();
      if (IsArrayElement)
        T = T->getBaseElementTypeUnsafe();
      else if (const FieldDecl *FD = getAsField(Path[I]))
        T = FD->getType().getTypePtr();
      else
        // Path[I] describes a base class.
        T = 0;
    }
    return IsArrayElement;
  }

  /// A path from a glvalue to a subobject of that glvalue.
  struct SubobjectDesignator {
    /// True if the subobject was named in a manner not supported by C++11. Such
    /// lvalues can still be folded, but they are not core constant expressions
    /// and we cannot perform lvalue-to-rvalue conversions on them.
    bool Invalid : 1;

    /// Whether this designates an array element.
    bool ArrayElement : 1;

    /// Whether this designates 'one past the end' of the current subobject.
    bool OnePastTheEnd : 1;

    typedef APValue::LValuePathEntry PathEntry;

    /// The entries on the path from the glvalue to the designated subobject.
    SmallVector<PathEntry, 8> Entries;

    SubobjectDesignator() :
      Invalid(false), ArrayElement(false), OnePastTheEnd(false) {}

    SubobjectDesignator(const APValue &V) :
      Invalid(!V.isLValue() || !V.hasLValuePath()), ArrayElement(false),
      OnePastTheEnd(false) {
      if (!Invalid) {
        ArrayRef<PathEntry> VEntries = V.getLValuePath();
        Entries.insert(Entries.end(), VEntries.begin(), VEntries.end());
        if (V.getLValueBase())
          ArrayElement = SubobjectIsArrayElement(getType(V.getLValueBase()),
                                                 V.getLValuePath());
        else
          assert(V.getLValuePath().empty() &&"Null pointer with nonempty path");
      }
    }

    void setInvalid() {
      Invalid = true;
      Entries.clear();
    }
    /// Update this designator to refer to the given element within this array.
    void addIndex(uint64_t N) {
      if (Invalid) return;
      if (OnePastTheEnd) {
        setInvalid();
        return;
      }
      PathEntry Entry;
      Entry.ArrayIndex = N;
      Entries.push_back(Entry);
      ArrayElement = true;
    }
    /// Update this designator to refer to the given base or member of this
    /// object.
    void addDecl(const Decl *D, bool Virtual = false) {
      if (Invalid) return;
      if (OnePastTheEnd) {
        setInvalid();
        return;
      }
      PathEntry Entry;
      APValue::BaseOrMemberType Value(D, Virtual);
      Entry.BaseOrMember = Value.getOpaqueValue();
      Entries.push_back(Entry);
      ArrayElement = false;
    }
    /// Add N to the address of this subobject.
    void adjustIndex(uint64_t N) {
      if (Invalid) return;
      if (ArrayElement) {
        // FIXME: Make sure the index stays within bounds, or one past the end.
        Entries.back().ArrayIndex += N;
        return;
      }
      if (OnePastTheEnd && N == (uint64_t)-1)
        OnePastTheEnd = false;
      else if (!OnePastTheEnd && N == 1)
        OnePastTheEnd = true;
      else if (N != 0)
        setInvalid();
    }
  };

  /// A core constant value. This can be the value of any constant expression,
  /// or a pointer or reference to a non-static object or function parameter.
  class CCValue : public APValue {
    typedef llvm::APSInt APSInt;
    typedef llvm::APFloat APFloat;
    /// If the value is a reference or pointer into a parameter or temporary,
    /// this is the corresponding call stack frame.
    CallStackFrame *CallFrame;
    /// If the value is a reference or pointer, this is a description of how the
    /// subobject was specified.
    SubobjectDesignator Designator;
  public:
    struct GlobalValue {};

    CCValue() {}
    explicit CCValue(const APSInt &I) : APValue(I) {}
    explicit CCValue(const APFloat &F) : APValue(F) {}
    CCValue(const APValue *E, unsigned N) : APValue(E, N) {}
    CCValue(const APSInt &R, const APSInt &I) : APValue(R, I) {}
    CCValue(const APFloat &R, const APFloat &I) : APValue(R, I) {}
    CCValue(const CCValue &V) : APValue(V), CallFrame(V.CallFrame) {}
    CCValue(LValueBase B, const CharUnits &O, CallStackFrame *F,
            const SubobjectDesignator &D) :
      APValue(B, O, APValue::NoLValuePath()), CallFrame(F), Designator(D) {}
    CCValue(const APValue &V, GlobalValue) :
      APValue(V), CallFrame(0), Designator(V) {}

    CallStackFrame *getLValueFrame() const {
      assert(getKind() == LValue);
      return CallFrame;
    }
    SubobjectDesignator &getLValueDesignator() {
      assert(getKind() == LValue);
      return Designator;
    }
    const SubobjectDesignator &getLValueDesignator() const {
      return const_cast<CCValue*>(this)->getLValueDesignator();
    }
  };

  /// A stack frame in the constexpr call stack.
  struct CallStackFrame {
    EvalInfo &Info;

    /// Parent - The caller of this stack frame.
    CallStackFrame *Caller;

    /// This - The binding for the this pointer in this call, if any.
    const LValue *This;

    /// ParmBindings - Parameter bindings for this function call, indexed by
    /// parameters' function scope indices.
    const CCValue *Arguments;

    typedef llvm::DenseMap<const Expr*, CCValue> MapTy;
    typedef MapTy::const_iterator temp_iterator;
    /// Temporaries - Temporary lvalues materialized within this stack frame.
    MapTy Temporaries;

    CallStackFrame(EvalInfo &Info, const LValue *This,
                   const CCValue *Arguments);
    ~CallStackFrame();
  };

  struct EvalInfo {
    const ASTContext &Ctx;

    /// EvalStatus - Contains information about the evaluation.
    Expr::EvalStatus &EvalStatus;

    /// CurrentCall - The top of the constexpr call stack.
    CallStackFrame *CurrentCall;

    /// NumCalls - The number of calls we've evaluated so far.
    unsigned NumCalls;

    /// CallStackDepth - The number of calls in the call stack right now.
    unsigned CallStackDepth;

    typedef llvm::DenseMap<const OpaqueValueExpr*, CCValue> MapTy;
    /// OpaqueValues - Values used as the common expression in a
    /// BinaryConditionalOperator.
    MapTy OpaqueValues;

    /// BottomFrame - The frame in which evaluation started. This must be
    /// initialized last.
    CallStackFrame BottomFrame;

    /// EvaluatingDecl - This is the declaration whose initializer is being
    /// evaluated, if any.
    const VarDecl *EvaluatingDecl;

    /// EvaluatingDeclValue - This is the value being constructed for the
    /// declaration whose initializer is being evaluated, if any.
    APValue *EvaluatingDeclValue;


    EvalInfo(const ASTContext &C, Expr::EvalStatus &S)
      : Ctx(C), EvalStatus(S), CurrentCall(0), NumCalls(0), CallStackDepth(0),
        BottomFrame(*this, 0, 0), EvaluatingDecl(0), EvaluatingDeclValue(0) {}

    const CCValue *getOpaqueValue(const OpaqueValueExpr *e) const {
      MapTy::const_iterator i = OpaqueValues.find(e);
      if (i == OpaqueValues.end()) return 0;
      return &i->second;
    }

    void setEvaluatingDecl(const VarDecl *VD, APValue &Value) {
      EvaluatingDecl = VD;
      EvaluatingDeclValue = &Value;
    }

    const LangOptions &getLangOpts() { return Ctx.getLangOptions(); }
  };

  CallStackFrame::CallStackFrame(EvalInfo &Info, const LValue *This,
                                 const CCValue *Arguments)
      : Info(Info), Caller(Info.CurrentCall), This(This), Arguments(Arguments) {
    Info.CurrentCall = this;
    ++Info.CallStackDepth;
  }

  CallStackFrame::~CallStackFrame() {
    assert(Info.CurrentCall == this && "calls retired out of order");
    --Info.CallStackDepth;
    Info.CurrentCall = Caller;
  }

  struct ComplexValue {
  private:
    bool IsInt;

  public:
    APSInt IntReal, IntImag;
    APFloat FloatReal, FloatImag;

    ComplexValue() : FloatReal(APFloat::Bogus), FloatImag(APFloat::Bogus) {}

    void makeComplexFloat() { IsInt = false; }
    bool isComplexFloat() const { return !IsInt; }
    APFloat &getComplexFloatReal() { return FloatReal; }
    APFloat &getComplexFloatImag() { return FloatImag; }

    void makeComplexInt() { IsInt = true; }
    bool isComplexInt() const { return IsInt; }
    APSInt &getComplexIntReal() { return IntReal; }
    APSInt &getComplexIntImag() { return IntImag; }

    void moveInto(CCValue &v) const {
      if (isComplexFloat())
        v = CCValue(FloatReal, FloatImag);
      else
        v = CCValue(IntReal, IntImag);
    }
    void setFrom(const CCValue &v) {
      assert(v.isComplexFloat() || v.isComplexInt());
      if (v.isComplexFloat()) {
        makeComplexFloat();
        FloatReal = v.getComplexFloatReal();
        FloatImag = v.getComplexFloatImag();
      } else {
        makeComplexInt();
        IntReal = v.getComplexIntReal();
        IntImag = v.getComplexIntImag();
      }
    }
  };

  struct LValue {
    APValue::LValueBase Base;
    CharUnits Offset;
    CallStackFrame *Frame;
    SubobjectDesignator Designator;

    const APValue::LValueBase getLValueBase() const { return Base; }
    CharUnits &getLValueOffset() { return Offset; }
    const CharUnits &getLValueOffset() const { return Offset; }
    CallStackFrame *getLValueFrame() const { return Frame; }
    SubobjectDesignator &getLValueDesignator() { return Designator; }
    const SubobjectDesignator &getLValueDesignator() const { return Designator;}

    void moveInto(CCValue &V) const {
      V = CCValue(Base, Offset, Frame, Designator);
    }
    void setFrom(const CCValue &V) {
      assert(V.isLValue());
      Base = V.getLValueBase();
      Offset = V.getLValueOffset();
      Frame = V.getLValueFrame();
      Designator = V.getLValueDesignator();
    }

    void set(APValue::LValueBase B, CallStackFrame *F = 0) {
      Base = B;
      Offset = CharUnits::Zero();
      Frame = F;
      Designator = SubobjectDesignator();
    }
  };
}

static bool Evaluate(CCValue &Result, EvalInfo &Info, const Expr *E);
static bool EvaluateConstantExpression(APValue &Result, EvalInfo &Info,
                                       const LValue &This, const Expr *E);
static bool EvaluateLValue(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateIntegerOrLValue(const Expr *E, CCValue &Result,
                                    EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);
static bool EvaluateComplex(const Expr *E, ComplexValue &Res, EvalInfo &Info);

//===----------------------------------------------------------------------===//
// Misc utilities
//===----------------------------------------------------------------------===//

/// Should this call expression be treated as a string literal?
static bool IsStringLiteralCall(const CallExpr *E) {
  unsigned Builtin = E->isBuiltinCall();
  return (Builtin == Builtin::BI__builtin___CFStringMakeConstantString ||
          Builtin == Builtin::BI__builtin___NSStringMakeConstantString);
}

static bool IsGlobalLValue(APValue::LValueBase B) {
  // C++11 [expr.const]p3 An address constant expression is a prvalue core
  // constant expression of pointer type that evaluates to...

  // ... a null pointer value, or a prvalue core constant expression of type
  // std::nullptr_t.
  if (!B) return true;

  if (const ValueDecl *D = B.dyn_cast<const ValueDecl*>()) {
    // ... the address of an object with static storage duration,
    if (const VarDecl *VD = dyn_cast<VarDecl>(D))
      return VD->hasGlobalStorage();
    // ... the address of a function,
    return isa<FunctionDecl>(D);
  }

  const Expr *E = B.get<const Expr*>();
  switch (E->getStmtClass()) {
  default:
    return false;
  case Expr::CompoundLiteralExprClass:
    return cast<CompoundLiteralExpr>(E)->isFileScope();
  // A string literal has static storage duration.
  case Expr::StringLiteralClass:
  case Expr::PredefinedExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCEncodeExprClass:
    return true;
  case Expr::CallExprClass:
    return IsStringLiteralCall(cast<CallExpr>(E));
  // For GCC compatibility, &&label has static storage duration.
  case Expr::AddrLabelExprClass:
    return true;
  // A Block literal expression may be used as the initialization value for
  // Block variables at global or local static scope.
  case Expr::BlockExprClass:
    return !cast<BlockExpr>(E)->getBlockDecl()->hasCaptures();
  }
}

/// Check that this reference or pointer core constant expression is a valid
/// value for a constant expression. Type T should be either LValue or CCValue.
template<typename T>
static bool CheckLValueConstantExpression(const T &LVal, APValue &Value) {
  if (!IsGlobalLValue(LVal.getLValueBase()))
    return false;

  const SubobjectDesignator &Designator = LVal.getLValueDesignator();
  // A constant expression must refer to an object or be a null pointer.
  if (Designator.Invalid || Designator.OnePastTheEnd ||
      (!LVal.getLValueBase() && !Designator.Entries.empty())) {
    // FIXME: Check for out-of-bounds array indices.
    // FIXME: This is not a constant expression.
    Value = APValue(LVal.getLValueBase(), LVal.getLValueOffset(),
                    APValue::NoLValuePath());
    return true;
  }

  // FIXME: Null references are not constant expressions.

  Value = APValue(LVal.getLValueBase(), LVal.getLValueOffset(),
                  Designator.Entries);
  return true;
}

/// Check that this core constant expression value is a valid value for a
/// constant expression, and if it is, produce the corresponding constant value.
static bool CheckConstantExpression(const CCValue &CCValue, APValue &Value) {
  if (!CCValue.isLValue()) {
    Value = CCValue;
    return true;
  }
  return CheckLValueConstantExpression(CCValue, Value);
}

const ValueDecl *GetLValueBaseDecl(const LValue &LVal) {
  return LVal.Base.dyn_cast<const ValueDecl*>();
}

static bool IsLiteralLValue(const LValue &Value) {
  return Value.Base.dyn_cast<const Expr*>() && !Value.Frame;
}

static bool IsWeakDecl(const ValueDecl *Decl) {
  return Decl->hasAttr<WeakAttr>() ||
         Decl->hasAttr<WeakRefAttr>() ||
         Decl->isWeakImported();
}

static bool IsWeakLValue(const LValue &Value) {
  const ValueDecl *Decl = GetLValueBaseDecl(Value);
  return Decl && IsWeakDecl(Decl);
}

static bool EvalPointerValueAsBool(const LValue &Value, bool &Result) {
  // A null base expression indicates a null pointer.  These are always
  // evaluatable, and they are false unless the offset is zero.
  if (!Value.Base) {
    Result = !Value.Offset.isZero();
    return true;
  }

  // Require the base expression to be a global l-value.
  // FIXME: C++11 requires such conversions. Remove this check.
  if (!IsGlobalLValue(Value.Base)) return false;

  // We have a non-null base expression.  These are generally known to
  // be true, but if it'a decl-ref to a weak symbol it can be null at
  // runtime.
  Result = true;
  return !IsWeakLValue(Value);
}

static bool HandleConversionToBool(const CCValue &Val, bool &Result) {
  switch (Val.getKind()) {
  case APValue::Uninitialized:
    return false;
  case APValue::Int:
    Result = Val.getInt().getBoolValue();
    return true;
  case APValue::Float:
    Result = !Val.getFloat().isZero();
    return true;
  case APValue::ComplexInt:
    Result = Val.getComplexIntReal().getBoolValue() ||
             Val.getComplexIntImag().getBoolValue();
    return true;
  case APValue::ComplexFloat:
    Result = !Val.getComplexFloatReal().isZero() ||
             !Val.getComplexFloatImag().isZero();
    return true;
  case APValue::LValue: {
    LValue PointerResult;
    PointerResult.setFrom(Val);
    return EvalPointerValueAsBool(PointerResult, Result);
  }
  case APValue::Vector:
  case APValue::Array:
  case APValue::Struct:
  case APValue::Union:
    return false;
  }

  llvm_unreachable("unknown APValue kind");
}

static bool EvaluateAsBooleanCondition(const Expr *E, bool &Result,
                                       EvalInfo &Info) {
  assert(E->isRValue() && "missing lvalue-to-rvalue conv in bool condition");
  CCValue Val;
  if (!Evaluate(Val, Info, E))
    return false;
  return HandleConversionToBool(Val, Result);
}

static APSInt HandleFloatToIntCast(QualType DestType, QualType SrcType,
                                   APFloat &Value, const ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerOrEnumerationType();

  // FIXME: Warning for overflow.
  APSInt Result(DestWidth, !DestSigned);
  bool ignored;
  (void)Value.convertToInteger(Result, llvm::APFloat::rmTowardZero, &ignored);
  return Result;
}

static APFloat HandleFloatToFloatCast(QualType DestType, QualType SrcType,
                                      APFloat &Value, const ASTContext &Ctx) {
  bool ignored;
  APFloat Result = Value;
  Result.convert(Ctx.getFloatTypeSemantics(DestType),
                 APFloat::rmNearestTiesToEven, &ignored);
  return Result;
}

static APSInt HandleIntToIntCast(QualType DestType, QualType SrcType,
                                 APSInt &Value, const ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  APSInt Result = Value;
  // Figure out if this is a truncate, extend or noop cast.
  // If the input is signed, do a sign extend, noop, or truncate.
  Result = Result.extOrTrunc(DestWidth);
  Result.setIsUnsigned(DestType->isUnsignedIntegerOrEnumerationType());
  return Result;
}

static APFloat HandleIntToFloatCast(QualType DestType, QualType SrcType,
                                    APSInt &Value, const ASTContext &Ctx) {

  APFloat Result(Ctx.getFloatTypeSemantics(DestType), 1);
  Result.convertFromAPInt(Value, Value.isSigned(),
                          APFloat::rmNearestTiesToEven);
  return Result;
}

/// If the given LValue refers to a base subobject of some object, find the most
/// derived object and the corresponding complete record type. This is necessary
/// in order to find the offset of a virtual base class.
static bool ExtractMostDerivedObject(EvalInfo &Info, LValue &Result,
                                     const CXXRecordDecl *&MostDerivedType) {
  SubobjectDesignator &D = Result.Designator;
  if (D.Invalid || !Result.Base)
    return false;

  const Type *T = getType(Result.Base).getTypePtr();

  // Find path prefix which leads to the most-derived subobject.
  unsigned MostDerivedPathLength = 0;
  MostDerivedType = T->getAsCXXRecordDecl();
  bool MostDerivedIsArrayElement = false;

  for (unsigned I = 0, N = D.Entries.size(); I != N; ++I) {
    bool IsArray = T && T->isArrayType();
    if (IsArray)
      T = T->getBaseElementTypeUnsafe();
    else if (const FieldDecl *FD = getAsField(D.Entries[I]))
      T = FD->getType().getTypePtr();
    else
      T = 0;

    if (T) {
      MostDerivedType = T->getAsCXXRecordDecl();
      MostDerivedPathLength = I + 1;
      MostDerivedIsArrayElement = IsArray;
    }
  }

  if (!MostDerivedType)
    return false;

  // (B*)&d + 1 has no most-derived object.
  if (D.OnePastTheEnd && MostDerivedPathLength != D.Entries.size())
    return false;

  // Remove the trailing base class path entries and their offsets.
  const RecordDecl *RD = MostDerivedType;
  for (unsigned I = MostDerivedPathLength, N = D.Entries.size(); I != N; ++I) {
    const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);
    const CXXRecordDecl *Base = getAsBaseClass(D.Entries[I]);
    if (isVirtualBaseClass(D.Entries[I])) {
      assert(I == MostDerivedPathLength &&
             "virtual base class must be immediately after most-derived class");
      Result.Offset -= Layout.getVBaseClassOffset(Base);
    } else
      Result.Offset -= Layout.getBaseClassOffset(Base);
    RD = Base;
  }
  D.Entries.resize(MostDerivedPathLength);
  D.ArrayElement = MostDerivedIsArrayElement;
  return true;
}

static void HandleLValueDirectBase(EvalInfo &Info, LValue &Obj,
                                   const CXXRecordDecl *Derived,
                                   const CXXRecordDecl *Base,
                                   const ASTRecordLayout *RL = 0) {
  if (!RL) RL = &Info.Ctx.getASTRecordLayout(Derived);
  Obj.getLValueOffset() += RL->getBaseClassOffset(Base);
  Obj.Designator.addDecl(Base, /*Virtual*/ false);
}

static bool HandleLValueBase(EvalInfo &Info, LValue &Obj,
                             const CXXRecordDecl *DerivedDecl,
                             const CXXBaseSpecifier *Base) {
  const CXXRecordDecl *BaseDecl = Base->getType()->getAsCXXRecordDecl();

  if (!Base->isVirtual()) {
    HandleLValueDirectBase(Info, Obj, DerivedDecl, BaseDecl);
    return true;
  }

  // Extract most-derived object and corresponding type.
  if (!ExtractMostDerivedObject(Info, Obj, DerivedDecl))
    return false;

  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(DerivedDecl);
  Obj.getLValueOffset() += Layout.getVBaseClassOffset(BaseDecl);
  Obj.Designator.addDecl(BaseDecl, /*Virtual*/ true);
  return true;
}

/// Update LVal to refer to the given field, which must be a member of the type
/// currently described by LVal.
static void HandleLValueMember(EvalInfo &Info, LValue &LVal,
                               const FieldDecl *FD,
                               const ASTRecordLayout *RL = 0) {
  if (!RL)
    RL = &Info.Ctx.getASTRecordLayout(FD->getParent());

  unsigned I = FD->getFieldIndex();
  LVal.Offset += Info.Ctx.toCharUnitsFromBits(RL->getFieldOffset(I));
  LVal.Designator.addDecl(FD);
}

/// Get the size of the given type in char units.
static bool HandleSizeof(EvalInfo &Info, QualType Type, CharUnits &Size) {
  // sizeof(void), __alignof__(void), sizeof(function) = 1 as a gcc
  // extension.
  if (Type->isVoidType() || Type->isFunctionType()) {
    Size = CharUnits::One();
    return true;
  }

  if (!Type->isConstantSizeType()) {
    // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
    return false;
  }

  Size = Info.Ctx.getTypeSizeInChars(Type);
  return true;
}

/// Update a pointer value to model pointer arithmetic.
/// \param Info - Information about the ongoing evaluation.
/// \param LVal - The pointer value to be updated.
/// \param EltTy - The pointee type represented by LVal.
/// \param Adjustment - The adjustment, in objects of type EltTy, to add.
static bool HandleLValueArrayAdjustment(EvalInfo &Info, LValue &LVal,
                                        QualType EltTy, int64_t Adjustment) {
  CharUnits SizeOfPointee;
  if (!HandleSizeof(Info, EltTy, SizeOfPointee))
    return false;

  // Compute the new offset in the appropriate width.
  LVal.Offset += Adjustment * SizeOfPointee;
  LVal.Designator.adjustIndex(Adjustment);
  return true;
}

/// Try to evaluate the initializer for a variable declaration.
static bool EvaluateVarDeclInit(EvalInfo &Info, const VarDecl *VD,
                                CallStackFrame *Frame, CCValue &Result) {
  // If this is a parameter to an active constexpr function call, perform
  // argument substitution.
  if (const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(VD)) {
    if (!Frame || !Frame->Arguments)
      return false;
    Result = Frame->Arguments[PVD->getFunctionScopeIndex()];
    return true;
  }

  // If we're currently evaluating the initializer of this declaration, use that
  // in-flight value.
  if (Info.EvaluatingDecl == VD) {
    Result = CCValue(*Info.EvaluatingDeclValue, CCValue::GlobalValue());
    return !Result.isUninit();
  }

  // Never evaluate the initializer of a weak variable. We can't be sure that
  // this is the definition which will be used.
  if (IsWeakDecl(VD))
    return false;

  const Expr *Init = VD->getAnyInitializer();
  if (!Init || Init->isValueDependent())
    return false;

  if (APValue *V = VD->getEvaluatedValue()) {
    Result = CCValue(*V, CCValue::GlobalValue());
    return !Result.isUninit();
  }

  if (VD->isEvaluatingValue())
    return false;

  VD->setEvaluatingValue();

  Expr::EvalStatus EStatus;
  EvalInfo InitInfo(Info.Ctx, EStatus);
  APValue EvalResult;
  InitInfo.setEvaluatingDecl(VD, EvalResult);
  LValue LVal;
  LVal.set(VD);
  // FIXME: The caller will need to know whether the value was a constant
  // expression. If not, we should propagate up a diagnostic.
  if (!EvaluateConstantExpression(EvalResult, InitInfo, LVal, Init)) {
    // FIXME: If the evaluation failure was not permanent (for instance, if we
    // hit a variable with no declaration yet, or a constexpr function with no
    // definition yet), the standard is unclear as to how we should behave.
    //
    // Either the initializer should be evaluated when the variable is defined,
    // or a failed evaluation of the initializer should be reattempted each time
    // it is used.
    VD->setEvaluatedValue(APValue());
    return false;
  }

  VD->setEvaluatedValue(EvalResult);
  Result = CCValue(EvalResult, CCValue::GlobalValue());
  return true;
}

static bool IsConstNonVolatile(QualType T) {
  Qualifiers Quals = T.getQualifiers();
  return Quals.hasConst() && !Quals.hasVolatile();
}

/// Get the base index of the given base class within an APValue representing
/// the given derived class.
static unsigned getBaseIndex(const CXXRecordDecl *Derived,
                             const CXXRecordDecl *Base) {
  Base = Base->getCanonicalDecl();
  unsigned Index = 0;
  for (CXXRecordDecl::base_class_const_iterator I = Derived->bases_begin(),
         E = Derived->bases_end(); I != E; ++I, ++Index) {
    if (I->getType()->getAsCXXRecordDecl()->getCanonicalDecl() == Base)
      return Index;
  }

  llvm_unreachable("base class missing from derived class's bases list");
}

/// Extract the designated sub-object of an rvalue.
static bool ExtractSubobject(EvalInfo &Info, CCValue &Obj, QualType ObjType,
                             const SubobjectDesignator &Sub, QualType SubType) {
  if (Sub.Invalid || Sub.OnePastTheEnd)
    return false;
  if (Sub.Entries.empty())
    return true;

  assert(!Obj.isLValue() && "extracting subobject of lvalue");
  const APValue *O = &Obj;
  // Walk the designator's path to find the subobject.
  for (unsigned I = 0, N = Sub.Entries.size(); I != N; ++I) {
    if (ObjType->isArrayType()) {
      // Next subobject is an array element.
      const ConstantArrayType *CAT = Info.Ctx.getAsConstantArrayType(ObjType);
      if (!CAT)
        return false;
      uint64_t Index = Sub.Entries[I].ArrayIndex;
      if (CAT->getSize().ule(Index))
        return false;
      if (O->getArrayInitializedElts() > Index)
        O = &O->getArrayInitializedElt(Index);
      else
        O = &O->getArrayFiller();
      ObjType = CAT->getElementType();
    } else if (const FieldDecl *Field = getAsField(Sub.Entries[I])) {
      // Next subobject is a class, struct or union field.
      RecordDecl *RD = ObjType->castAs<RecordType>()->getDecl();
      if (RD->isUnion()) {
        const FieldDecl *UnionField = O->getUnionField();
        if (!UnionField ||
            UnionField->getCanonicalDecl() != Field->getCanonicalDecl())
          return false;
        O = &O->getUnionValue();
      } else
        O = &O->getStructField(Field->getFieldIndex());
      ObjType = Field->getType();
    } else {
      // Next subobject is a base class.
      const CXXRecordDecl *Derived = ObjType->getAsCXXRecordDecl();
      const CXXRecordDecl *Base = getAsBaseClass(Sub.Entries[I]);
      O = &O->getStructBase(getBaseIndex(Derived, Base));
      ObjType = Info.Ctx.getRecordType(Base);
    }

    if (O->isUninit())
      return false;
  }

  Obj = CCValue(*O, CCValue::GlobalValue());
  return true;
}

/// HandleLValueToRValueConversion - Perform an lvalue-to-rvalue conversion on
/// the given lvalue. This can also be used for 'lvalue-to-lvalue' conversions
/// for looking up the glvalue referred to by an entity of reference type.
///
/// \param Info - Information about the ongoing evaluation.
/// \param Type - The type we expect this conversion to produce.
/// \param LVal - The glvalue on which we are attempting to perform this action.
/// \param RVal - The produced value will be placed here.
static bool HandleLValueToRValueConversion(EvalInfo &Info, QualType Type,
                                           const LValue &LVal, CCValue &RVal) {
  const Expr *Base = LVal.Base.dyn_cast<const Expr*>();
  CallStackFrame *Frame = LVal.Frame;

  // FIXME: Indirection through a null pointer deserves a diagnostic.
  if (!LVal.Base)
    return false;

  if (const ValueDecl *D = LVal.Base.dyn_cast<const ValueDecl*>()) {
    // In C++98, const, non-volatile integers initialized with ICEs are ICEs.
    // In C++11, constexpr, non-volatile variables initialized with constant
    // expressions are constant expressions too. Inside constexpr functions,
    // parameters are constant expressions even if they're non-const.
    // In C, such things can also be folded, although they are not ICEs.
    //
    // FIXME: volatile-qualified ParmVarDecls need special handling. A literal
    // interpretation of C++11 suggests that volatile parameters are OK if
    // they're never read (there's no prohibition against constructing volatile
    // objects in constant expressions), but lvalue-to-rvalue conversions on
    // them are not permitted.
    const VarDecl *VD = dyn_cast<VarDecl>(D);
    if (!VD || VD->isInvalidDecl())
      return false;
    QualType VT = VD->getType();
    if (!isa<ParmVarDecl>(VD)) {
      if (!IsConstNonVolatile(VT))
        return false;
      // FIXME: Allow folding of values of any literal type in all languages.
      if (!VT->isIntegralOrEnumerationType() && !VT->isRealFloatingType() &&
          !VD->isConstexpr())
        return false;
    }
    if (!EvaluateVarDeclInit(Info, VD, Frame, RVal))
      return false;

    if (isa<ParmVarDecl>(VD) || !VD->getAnyInitializer()->isLValue())
      return ExtractSubobject(Info, RVal, VT, LVal.Designator, Type);

    // The declaration was initialized by an lvalue, with no lvalue-to-rvalue
    // conversion. This happens when the declaration and the lvalue should be
    // considered synonymous, for instance when initializing an array of char
    // from a string literal. Continue as if the initializer lvalue was the
    // value we were originally given.
    assert(RVal.getLValueOffset().isZero() &&
           "offset for lvalue init of non-reference");
    Base = RVal.getLValueBase().get<const Expr*>();
    Frame = RVal.getLValueFrame();
  }

  // FIXME: Support PredefinedExpr, ObjCEncodeExpr, MakeStringConstant
  if (const StringLiteral *S = dyn_cast<StringLiteral>(Base)) {
    const SubobjectDesignator &Designator = LVal.Designator;
    if (Designator.Invalid || Designator.Entries.size() != 1)
      return false;

    assert(Type->isIntegerType() && "string element not integer type");
    uint64_t Index = Designator.Entries[0].ArrayIndex;
    if (Index > S->getLength())
      return false;
    APSInt Value(S->getCharByteWidth() * Info.Ctx.getCharWidth(),
                 Type->isUnsignedIntegerType());
    if (Index < S->getLength())
      Value = S->getCodeUnit(Index);
    RVal = CCValue(Value);
    return true;
  }

  if (Frame) {
    // If this is a temporary expression with a nontrivial initializer, grab the
    // value from the relevant stack frame.
    RVal = Frame->Temporaries[Base];
  } else if (const CompoundLiteralExpr *CLE
             = dyn_cast<CompoundLiteralExpr>(Base)) {
    // In C99, a CompoundLiteralExpr is an lvalue, and we defer evaluating the
    // initializer until now for such expressions. Such an expression can't be
    // an ICE in C, so this only matters for fold.
    assert(!Info.getLangOpts().CPlusPlus && "lvalue compound literal in c++?");
    if (!Evaluate(RVal, Info, CLE->getInitializer()))
      return false;
  } else
    return false;

  return ExtractSubobject(Info, RVal, Base->getType(), LVal.Designator, Type);
}

/// Build an lvalue for the object argument of a member function call.
static bool EvaluateObjectArgument(EvalInfo &Info, const Expr *Object,
                                   LValue &This) {
  if (Object->getType()->isPointerType())
    return EvaluatePointer(Object, This, Info);

  if (Object->isGLValue())
    return EvaluateLValue(Object, This, Info);

  // Implicitly promote a prvalue *this object to a glvalue.
  This.set(Object, Info.CurrentCall);
  return EvaluateConstantExpression(Info.CurrentCall->Temporaries[Object], Info,
                                    This, Object);
}

namespace {
enum EvalStmtResult {
  /// Evaluation failed.
  ESR_Failed,
  /// Hit a 'return' statement.
  ESR_Returned,
  /// Evaluation succeeded.
  ESR_Succeeded
};
}

// Evaluate a statement.
static EvalStmtResult EvaluateStmt(CCValue &Result, EvalInfo &Info,
                                   const Stmt *S) {
  switch (S->getStmtClass()) {
  default:
    return ESR_Failed;

  case Stmt::NullStmtClass:
  case Stmt::DeclStmtClass:
    return ESR_Succeeded;

  case Stmt::ReturnStmtClass:
    if (Evaluate(Result, Info, cast<ReturnStmt>(S)->getRetValue()))
      return ESR_Returned;
    return ESR_Failed;

  case Stmt::CompoundStmtClass: {
    const CompoundStmt *CS = cast<CompoundStmt>(S);
    for (CompoundStmt::const_body_iterator BI = CS->body_begin(),
           BE = CS->body_end(); BI != BE; ++BI) {
      EvalStmtResult ESR = EvaluateStmt(Result, Info, *BI);
      if (ESR != ESR_Succeeded)
        return ESR;
    }
    return ESR_Succeeded;
  }
  }
}

namespace {
typedef SmallVector<CCValue, 8> ArgVector;
}

/// EvaluateArgs - Evaluate the arguments to a function call.
static bool EvaluateArgs(ArrayRef<const Expr*> Args, ArgVector &ArgValues,
                         EvalInfo &Info) {
  for (ArrayRef<const Expr*>::iterator I = Args.begin(), E = Args.end();
       I != E; ++I)
    if (!Evaluate(ArgValues[I - Args.begin()], Info, *I))
      return false;
  return true;
}

/// Evaluate a function call.
static bool HandleFunctionCall(const LValue *This, ArrayRef<const Expr*> Args,
                               const Stmt *Body, EvalInfo &Info,
                               CCValue &Result) {
  // FIXME: Implement a proper call limit, along with a command-line flag.
  if (Info.NumCalls >= 1000000 || Info.CallStackDepth >= 512)
    return false;

  ArgVector ArgValues(Args.size());
  if (!EvaluateArgs(Args, ArgValues, Info))
    return false;

  CallStackFrame Frame(Info, This, ArgValues.data());
  return EvaluateStmt(Result, Info, Body) == ESR_Returned;
}

/// Evaluate a constructor call.
static bool HandleConstructorCall(const LValue &This,
                                  ArrayRef<const Expr*> Args,
                                  const CXXConstructorDecl *Definition,
                                  EvalInfo &Info,
                                  APValue &Result) {
  if (Info.NumCalls >= 1000000 || Info.CallStackDepth >= 512)
    return false;

  ArgVector ArgValues(Args.size());
  if (!EvaluateArgs(Args, ArgValues, Info))
    return false;

  CallStackFrame Frame(Info, &This, ArgValues.data());

  // If it's a delegating constructor, just delegate.
  if (Definition->isDelegatingConstructor()) {
    CXXConstructorDecl::init_const_iterator I = Definition->init_begin();
    return EvaluateConstantExpression(Result, Info, This, (*I)->getInit());
  }

  // Reserve space for the struct members.
  const CXXRecordDecl *RD = Definition->getParent();
  if (!RD->isUnion())
    Result = APValue(APValue::UninitStruct(), RD->getNumBases(),
                     std::distance(RD->field_begin(), RD->field_end()));

  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);

  unsigned BasesSeen = 0;
#ifndef NDEBUG
  CXXRecordDecl::base_class_const_iterator BaseIt = RD->bases_begin();
#endif
  for (CXXConstructorDecl::init_const_iterator I = Definition->init_begin(),
       E = Definition->init_end(); I != E; ++I) {
    if ((*I)->isBaseInitializer()) {
      QualType BaseType((*I)->getBaseClass(), 0);
#ifndef NDEBUG
      // Non-virtual base classes are initialized in the order in the class
      // definition. We cannot have a virtual base class for a literal type.
      assert(!BaseIt->isVirtual() && "virtual base for literal type");
      assert(Info.Ctx.hasSameType(BaseIt->getType(), BaseType) &&
             "base class initializers not in expected order");
      ++BaseIt;
#endif
      LValue Subobject = This;
      HandleLValueDirectBase(Info, Subobject, RD,
                             BaseType->getAsCXXRecordDecl(), &Layout);
      if (!EvaluateConstantExpression(Result.getStructBase(BasesSeen++), Info,
                                      Subobject, (*I)->getInit()))
        return false;
    } else if (FieldDecl *FD = (*I)->getMember()) {
      LValue Subobject = This;
      HandleLValueMember(Info, Subobject, FD, &Layout);
      if (RD->isUnion()) {
        Result = APValue(FD);
        if (!EvaluateConstantExpression(Result.getUnionValue(), Info,
                                        Subobject, (*I)->getInit()))
          return false;
      } else if (!EvaluateConstantExpression(
                   Result.getStructField(FD->getFieldIndex()),
                   Info, Subobject, (*I)->getInit()))
        return false;
    } else {
      // FIXME: handle indirect field initializers
      return false;
    }
  }

  return true;
}

namespace {
class HasSideEffect
  : public ConstStmtVisitor<HasSideEffect, bool> {
  const ASTContext &Ctx;
public:

  HasSideEffect(const ASTContext &C) : Ctx(C) {}

  // Unhandled nodes conservatively default to having side effects.
  bool VisitStmt(const Stmt *S) {
    return true;
  }

  bool VisitParenExpr(const ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
    return Visit(E->getResultExpr());
  }
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }
  bool VisitObjCIvarRefExpr(const ObjCIvarRefExpr *E) {
    if (Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }
  bool VisitBlockDeclRefExpr (const BlockDeclRefExpr *E) {
    if (Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }

  // We don't want to evaluate BlockExprs multiple times, as they generate
  // a ton of code.
  bool VisitBlockExpr(const BlockExpr *E) { return true; }
  bool VisitPredefinedExpr(const PredefinedExpr *E) { return false; }
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E)
    { return Visit(E->getInitializer()); }
  bool VisitMemberExpr(const MemberExpr *E) { return Visit(E->getBase()); }
  bool VisitIntegerLiteral(const IntegerLiteral *E) { return false; }
  bool VisitFloatingLiteral(const FloatingLiteral *E) { return false; }
  bool VisitStringLiteral(const StringLiteral *E) { return false; }
  bool VisitCharacterLiteral(const CharacterLiteral *E) { return false; }
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E)
    { return false; }
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E)
    { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Ctx)); }
  bool VisitCastExpr(const CastExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitBinAssign(const BinaryOperator *E) { return true; }
  bool VisitCompoundAssignOperator(const BinaryOperator *E) { return true; }
  bool VisitBinaryOperator(const BinaryOperator *E)
  { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitUnaryPreInc(const UnaryOperator *E) { return true; }
  bool VisitUnaryPostInc(const UnaryOperator *E) { return true; }
  bool VisitUnaryPreDec(const UnaryOperator *E) { return true; }
  bool VisitUnaryPostDec(const UnaryOperator *E) { return true; }
  bool VisitUnaryDeref(const UnaryOperator *E) {
    if (Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return Visit(E->getSubExpr());
  }
  bool VisitUnaryOperator(const UnaryOperator *E) { return Visit(E->getSubExpr()); }
    
  // Has side effects if any element does.
  bool VisitInitListExpr(const InitListExpr *E) {
    for (unsigned i = 0, e = E->getNumInits(); i != e; ++i)
      if (Visit(E->getInit(i))) return true;
    if (const Expr *filler = E->getArrayFiller())
      return Visit(filler);
    return false;
  }
    
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *) { return false; }
};

class OpaqueValueEvaluation {
  EvalInfo &info;
  OpaqueValueExpr *opaqueValue;

public:
  OpaqueValueEvaluation(EvalInfo &info, OpaqueValueExpr *opaqueValue,
                        Expr *value)
    : info(info), opaqueValue(opaqueValue) {

    // If evaluation fails, fail immediately.
    if (!Evaluate(info.OpaqueValues[opaqueValue], info, value)) {
      this->opaqueValue = 0;
      return;
    }
  }

  bool hasError() const { return opaqueValue == 0; }

  ~OpaqueValueEvaluation() {
    // FIXME: This will not work for recursive constexpr functions using opaque
    // values. Restore the former value.
    if (opaqueValue) info.OpaqueValues.erase(opaqueValue);
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Generic Evaluation
//===----------------------------------------------------------------------===//
namespace {

template <class Derived, typename RetTy=void>
class ExprEvaluatorBase
  : public ConstStmtVisitor<Derived, RetTy> {
private:
  RetTy DerivedSuccess(const CCValue &V, const Expr *E) {
    return static_cast<Derived*>(this)->Success(V, E);
  }
  RetTy DerivedError(const Expr *E) {
    return static_cast<Derived*>(this)->Error(E);
  }
  RetTy DerivedValueInitialization(const Expr *E) {
    return static_cast<Derived*>(this)->ValueInitialization(E);
  }

protected:
  EvalInfo &Info;
  typedef ConstStmtVisitor<Derived, RetTy> StmtVisitorTy;
  typedef ExprEvaluatorBase ExprEvaluatorBaseTy;

  RetTy ValueInitialization(const Expr *E) { return DerivedError(E); }

public:
  ExprEvaluatorBase(EvalInfo &Info) : Info(Info) {}

  RetTy VisitStmt(const Stmt *) {
    llvm_unreachable("Expression evaluator should not be called on stmts");
  }
  RetTy VisitExpr(const Expr *E) {
    return DerivedError(E);
  }

  RetTy VisitParenExpr(const ParenExpr *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitUnaryExtension(const UnaryOperator *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitUnaryPlus(const UnaryOperator *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitChooseExpr(const ChooseExpr *E)
    { return StmtVisitorTy::Visit(E->getChosenSubExpr(Info.Ctx)); }
  RetTy VisitGenericSelectionExpr(const GenericSelectionExpr *E)
    { return StmtVisitorTy::Visit(E->getResultExpr()); }
  RetTy VisitSubstNonTypeTemplateParmExpr(const SubstNonTypeTemplateParmExpr *E)
    { return StmtVisitorTy::Visit(E->getReplacement()); }
  RetTy VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E)
    { return StmtVisitorTy::Visit(E->getExpr()); }

  RetTy VisitBinaryConditionalOperator(const BinaryConditionalOperator *E) {
    OpaqueValueEvaluation opaque(Info, E->getOpaqueValue(), E->getCommon());
    if (opaque.hasError())
      return DerivedError(E);

    bool cond;
    if (!EvaluateAsBooleanCondition(E->getCond(), cond, Info))
      return DerivedError(E);

    return StmtVisitorTy::Visit(cond ? E->getTrueExpr() : E->getFalseExpr());
  }

  RetTy VisitConditionalOperator(const ConditionalOperator *E) {
    bool BoolResult;
    if (!EvaluateAsBooleanCondition(E->getCond(), BoolResult, Info))
      return DerivedError(E);

    Expr *EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();
    return StmtVisitorTy::Visit(EvalExpr);
  }

  RetTy VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
    const CCValue *Value = Info.getOpaqueValue(E);
    if (!Value)
      return (E->getSourceExpr() ? StmtVisitorTy::Visit(E->getSourceExpr())
                                 : DerivedError(E));
    return DerivedSuccess(*Value, E);
  }

  RetTy VisitCallExpr(const CallExpr *E) {
    const Expr *Callee = E->getCallee();
    QualType CalleeType = Callee->getType();

    const FunctionDecl *FD = 0;
    LValue *This = 0, ThisVal;
    llvm::ArrayRef<const Expr*> Args(E->getArgs(), E->getNumArgs());

    // Extract function decl and 'this' pointer from the callee.
    if (CalleeType->isSpecificBuiltinType(BuiltinType::BoundMember)) {
      // Explicit bound member calls, such as x.f() or p->g();
      // FIXME: Handle a BinaryOperator callee ('.*' or '->*').
      const MemberExpr *ME = dyn_cast<MemberExpr>(Callee->IgnoreParens());
      if (!ME)
        return DerivedError(Callee);
      if (!EvaluateObjectArgument(Info, ME->getBase(), ThisVal))
        return DerivedError(ME->getBase());
      This = &ThisVal;
      FD = dyn_cast<FunctionDecl>(ME->getMemberDecl());
      if (!FD)
        return DerivedError(ME);
    } else if (CalleeType->isFunctionPointerType()) {
      CCValue Call;
      if (!Evaluate(Call, Info, Callee) || !Call.isLValue() ||
          !Call.getLValueOffset().isZero())
        return DerivedError(Callee);

      FD = dyn_cast_or_null<FunctionDecl>(
                             Call.getLValueBase().dyn_cast<const ValueDecl*>());
      if (!FD)
        return DerivedError(Callee);

      // Overloaded operator calls to member functions are represented as normal
      // calls with '*this' as the first argument.
      const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
      if (MD && !MD->isStatic()) {
        if (!EvaluateObjectArgument(Info, Args[0], ThisVal))
          return false;
        This = &ThisVal;
        Args = Args.slice(1);
      }

      // Don't call function pointers which have been cast to some other type.
      if (!Info.Ctx.hasSameType(CalleeType->getPointeeType(), FD->getType()))
        return DerivedError(E);
    } else
      return DerivedError(E);

    const FunctionDecl *Definition;
    Stmt *Body = FD->getBody(Definition);
    CCValue CCResult;
    APValue Result;

    if (Body && Definition->isConstexpr() && !Definition->isInvalidDecl() &&
        HandleFunctionCall(This, Args, Body, Info, CCResult) &&
        CheckConstantExpression(CCResult, Result))
      return DerivedSuccess(CCValue(Result, CCValue::GlobalValue()), E);

    return DerivedError(E);
  }

  RetTy VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
    return StmtVisitorTy::Visit(E->getInitializer());
  }
  RetTy VisitInitListExpr(const InitListExpr *E) {
    if (Info.getLangOpts().CPlusPlus0x) {
      if (E->getNumInits() == 0)
        return DerivedValueInitialization(E);
      if (E->getNumInits() == 1)
        return StmtVisitorTy::Visit(E->getInit(0));
    }
    return DerivedError(E);
  }
  RetTy VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return DerivedValueInitialization(E);
  }
  RetTy VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    return DerivedValueInitialization(E);
  }

  /// A member expression where the object is a prvalue is itself a prvalue.
  RetTy VisitMemberExpr(const MemberExpr *E) {
    assert(!E->isArrow() && "missing call to bound member function?");

    CCValue Val;
    if (!Evaluate(Val, Info, E->getBase()))
      return false;

    QualType BaseTy = E->getBase()->getType();

    const FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
    if (!FD) return false;
    assert(!FD->getType()->isReferenceType() && "prvalue reference?");
    assert(BaseTy->getAs<RecordType>()->getDecl()->getCanonicalDecl() ==
           FD->getParent()->getCanonicalDecl() && "record / field mismatch");

    SubobjectDesignator Designator;
    Designator.addDecl(FD);

    return ExtractSubobject(Info, Val, BaseTy, Designator, E->getType()) &&
           DerivedSuccess(Val, E);
  }

  RetTy VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      break;

    case CK_NoOp:
      return StmtVisitorTy::Visit(E->getSubExpr());

    case CK_LValueToRValue: {
      LValue LVal;
      if (EvaluateLValue(E->getSubExpr(), LVal, Info)) {
        CCValue RVal;
        if (HandleLValueToRValueConversion(Info, E->getType(), LVal, RVal))
          return DerivedSuccess(RVal, E);
      }
      break;
    }
    }

    return DerivedError(E);
  }

  /// Visit a value which is evaluated, but whose value is ignored.
  void VisitIgnoredValue(const Expr *E) {
    CCValue Scratch;
    if (!Evaluate(Scratch, Info, E))
      Info.EvalStatus.HasSideEffects = true;
  }
};

}

//===----------------------------------------------------------------------===//
// LValue Evaluation
//
// This is used for evaluating lvalues (in C and C++), xvalues (in C++11),
// function designators (in C), decl references to void objects (in C), and
// temporaries (if building with -Wno-address-of-temporary).
//
// LValue evaluation produces values comprising a base expression of one of the
// following types:
// - Declarations
//  * VarDecl
//  * FunctionDecl
// - Literals
//  * CompoundLiteralExpr in C
//  * StringLiteral
//  * PredefinedExpr
//  * ObjCStringLiteralExpr
//  * ObjCEncodeExpr
//  * AddrLabelExpr
//  * BlockExpr
//  * CallExpr for a MakeStringConstant builtin
// - Locals and temporaries
//  * Any Expr, with a Frame indicating the function in which the temporary was
//    evaluated.
// plus an offset in bytes.
//===----------------------------------------------------------------------===//
namespace {
class LValueExprEvaluator
  : public ExprEvaluatorBase<LValueExprEvaluator, bool> {
  LValue &Result;
  const Decl *PrevDecl;

  bool Success(APValue::LValueBase B) {
    Result.set(B);
    return true;
  }
public:

  LValueExprEvaluator(EvalInfo &info, LValue &Result) :
    ExprEvaluatorBaseTy(info), Result(Result), PrevDecl(0) {}

  bool Success(const CCValue &V, const Expr *E) {
    Result.setFrom(V);
    return true;
  }
  bool Error(const Expr *E) {
    return false;
  }
  
  bool VisitVarDecl(const Expr *E, const VarDecl *VD);

  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitPredefinedExpr(const PredefinedExpr *E) { return Success(E); }
  bool VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
  bool VisitMemberExpr(const MemberExpr *E);
  bool VisitStringLiteral(const StringLiteral *E) { return Success(E); }
  bool VisitObjCEncodeExpr(const ObjCEncodeExpr *E) { return Success(E); }
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E);
  bool VisitUnaryDeref(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return ExprEvaluatorBaseTy::VisitCastExpr(E);

    case CK_LValueBitCast:
      if (!Visit(E->getSubExpr()))
        return false;
      Result.Designator.setInvalid();
      return true;

    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase: {
      if (!Visit(E->getSubExpr()))
        return false;

      // Now figure out the necessary offset to add to the base LV to get from
      // the derived class to the base class.
      QualType Type = E->getSubExpr()->getType();

      for (CastExpr::path_const_iterator PathI = E->path_begin(),
           PathE = E->path_end(); PathI != PathE; ++PathI) {
        if (!HandleLValueBase(Info, Result, Type->getAsCXXRecordDecl(), *PathI))
          return false;
        Type = (*PathI)->getType();
      }

      return true;
    }
    }
  }

  // FIXME: Missing: __real__, __imag__

};
} // end anonymous namespace

/// Evaluate an expression as an lvalue. This can be legitimately called on
/// expressions which are not glvalues, in a few cases:
///  * function designators in C,
///  * "extern void" objects,
///  * temporaries, if building with -Wno-address-of-temporary.
static bool EvaluateLValue(const Expr* E, LValue& Result, EvalInfo &Info) {
  assert((E->isGLValue() || E->getType()->isFunctionType() ||
          E->getType()->isVoidType() || isa<CXXTemporaryObjectExpr>(E)) &&
         "can't evaluate expression as an lvalue");
  return LValueExprEvaluator(Info, Result).Visit(E);
}

bool LValueExprEvaluator::VisitDeclRefExpr(const DeclRefExpr *E) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(E->getDecl()))
    return Success(FD);
  if (const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl()))
    return VisitVarDecl(E, VD);
  return Error(E);
}

bool LValueExprEvaluator::VisitVarDecl(const Expr *E, const VarDecl *VD) {
  if (!VD->getType()->isReferenceType()) {
    if (isa<ParmVarDecl>(VD)) {
      Result.set(VD, Info.CurrentCall);
      return true;
    }
    return Success(VD);
  }

  CCValue V;
  if (EvaluateVarDeclInit(Info, VD, Info.CurrentCall, V))
    return Success(V, E);

  return Error(E);
}

bool LValueExprEvaluator::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *E) {
  Result.set(E, Info.CurrentCall);
  return EvaluateConstantExpression(Info.CurrentCall->Temporaries[E], Info,
                                    Result, E->GetTemporaryExpr());
}

bool
LValueExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  assert(!Info.getLangOpts().CPlusPlus && "lvalue compound literal in c++?");
  // Defer visiting the literal until the lvalue-to-rvalue conversion. We can
  // only see this when folding in C, so there's no standard to follow here.
  return Success(E);
}

bool LValueExprEvaluator::VisitMemberExpr(const MemberExpr *E) {
  // Handle static data members.
  if (const VarDecl *VD = dyn_cast<VarDecl>(E->getMemberDecl())) {
    VisitIgnoredValue(E->getBase());
    return VisitVarDecl(E, VD);
  }

  // Handle static member functions.
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(E->getMemberDecl())) {
    if (MD->isStatic()) {
      VisitIgnoredValue(E->getBase());
      return Success(MD);
    }
  }

  // Handle non-static data members.
  QualType BaseTy;
  if (E->isArrow()) {
    if (!EvaluatePointer(E->getBase(), Result, Info))
      return false;
    BaseTy = E->getBase()->getType()->getAs<PointerType>()->getPointeeType();
  } else {
    if (!Visit(E->getBase()))
      return false;
    BaseTy = E->getBase()->getType();
  }

  const FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
  if (!FD) return false;
  assert(BaseTy->getAs<RecordType>()->getDecl()->getCanonicalDecl() ==
         FD->getParent()->getCanonicalDecl() && "record / field mismatch");
  (void)BaseTy;

  HandleLValueMember(Info, Result, FD);

  if (FD->getType()->isReferenceType()) {
    CCValue RefValue;
    if (!HandleLValueToRValueConversion(Info, FD->getType(), Result, RefValue))
      return false;
    return Success(RefValue, E);
  }
  return true;
}

bool LValueExprEvaluator::VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // FIXME: Deal with vectors as array subscript bases.
  if (E->getBase()->getType()->isVectorType())
    return false;

  if (!EvaluatePointer(E->getBase(), Result, Info))
    return false;

  APSInt Index;
  if (!EvaluateInteger(E->getIdx(), Index, Info))
    return false;
  int64_t IndexValue
    = Index.isSigned() ? Index.getSExtValue()
                       : static_cast<int64_t>(Index.getZExtValue());

  return HandleLValueArrayAdjustment(Info, Result, E->getType(), IndexValue);
}

bool LValueExprEvaluator::VisitUnaryDeref(const UnaryOperator *E) {
  return EvaluatePointer(E->getSubExpr(), Result, Info);
}

//===----------------------------------------------------------------------===//
// Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class PointerExprEvaluator
  : public ExprEvaluatorBase<PointerExprEvaluator, bool> {
  LValue &Result;

  bool Success(const Expr *E) {
    Result.set(E);
    return true;
  }
public:

  PointerExprEvaluator(EvalInfo &info, LValue &Result)
    : ExprEvaluatorBaseTy(info), Result(Result) {}

  bool Success(const CCValue &V, const Expr *E) {
    Result.setFrom(V);
    return true;
  }
  bool Error(const Stmt *S) {
    return false;
  }
  bool ValueInitialization(const Expr *E) {
    return Success((Expr*)0);
  }

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitCastExpr(const CastExpr* E);
  bool VisitUnaryAddrOf(const UnaryOperator *E);
  bool VisitObjCStringLiteral(const ObjCStringLiteral *E)
      { return Success(E); }
  bool VisitAddrLabelExpr(const AddrLabelExpr *E)
      { return Success(E); }
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBlockExpr(const BlockExpr *E) {
    if (!E->getBlockDecl()->hasCaptures())
      return Success(E);
    return false;
  }
  bool VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E)
      { return ValueInitialization(E); }
  bool VisitCXXThisExpr(const CXXThisExpr *E) {
    if (!Info.CurrentCall->This)
      return false;
    Result = *Info.CurrentCall->This;
    return true;
  }

  // FIXME: Missing: @protocol, @selector
};
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, LValue& Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->hasPointerRepresentation());
  return PointerExprEvaluator(Info, Result).Visit(E);
}

bool PointerExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() != BO_Add &&
      E->getOpcode() != BO_Sub)
    return false;

  const Expr *PExp = E->getLHS();
  const Expr *IExp = E->getRHS();
  if (IExp->getType()->isPointerType())
    std::swap(PExp, IExp);

  if (!EvaluatePointer(PExp, Result, Info))
    return false;

  llvm::APSInt Offset;
  if (!EvaluateInteger(IExp, Offset, Info))
    return false;
  int64_t AdditionalOffset
    = Offset.isSigned() ? Offset.getSExtValue()
                        : static_cast<int64_t>(Offset.getZExtValue());
  if (E->getOpcode() == BO_Sub)
    AdditionalOffset = -AdditionalOffset;

  QualType Pointee = PExp->getType()->getAs<PointerType>()->getPointeeType();
  return HandleLValueArrayAdjustment(Info, Result, Pointee, AdditionalOffset);
}

bool PointerExprEvaluator::VisitUnaryAddrOf(const UnaryOperator *E) {
  return EvaluateLValue(E->getSubExpr(), Result, Info);
}

bool PointerExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    break;

  case CK_BitCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
    if (!Visit(SubExpr))
      return false;
    Result.Designator.setInvalid();
    return true;

  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase: {
    if (!EvaluatePointer(E->getSubExpr(), Result, Info))
      return false;

    // Now figure out the necessary offset to add to the base LV to get from
    // the derived class to the base class.
    QualType Type =
        E->getSubExpr()->getType()->castAs<PointerType>()->getPointeeType();

    for (CastExpr::path_const_iterator PathI = E->path_begin(),
         PathE = E->path_end(); PathI != PathE; ++PathI) {
      if (!HandleLValueBase(Info, Result, Type->getAsCXXRecordDecl(), *PathI))
        return false;
      Type = (*PathI)->getType();
    }

    return true;
  }

  case CK_NullToPointer:
    return ValueInitialization(E);

  case CK_IntegralToPointer: {
    CCValue Value;
    if (!EvaluateIntegerOrLValue(SubExpr, Value, Info))
      break;

    if (Value.isInt()) {
      unsigned Size = Info.Ctx.getTypeSize(E->getType());
      uint64_t N = Value.getInt().extOrTrunc(Size).getZExtValue();
      Result.Base = (Expr*)0;
      Result.Offset = CharUnits::fromQuantity(N);
      Result.Frame = 0;
      Result.Designator.setInvalid();
      return true;
    } else {
      // Cast is of an lvalue, no need to change value.
      Result.setFrom(Value);
      return true;
    }
  }
  case CK_ArrayToPointerDecay:
    // FIXME: Support array-to-pointer decay on array rvalues.
    if (!SubExpr->isGLValue())
      return Error(E);
    if (!EvaluateLValue(SubExpr, Result, Info))
      return false;
    // The result is a pointer to the first element of the array.
    Result.Designator.addIndex(0);
    return true;

  case CK_FunctionToPointerDecay:
    return EvaluateLValue(SubExpr, Result, Info);
  }

  return ExprEvaluatorBaseTy::VisitCastExpr(E);
}

bool PointerExprEvaluator::VisitCallExpr(const CallExpr *E) {
  if (IsStringLiteralCall(E))
    return Success(E);

  return ExprEvaluatorBaseTy::VisitCallExpr(E);
}

//===----------------------------------------------------------------------===//
// Record Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class RecordExprEvaluator
  : public ExprEvaluatorBase<RecordExprEvaluator, bool> {
    const LValue &This;
    APValue &Result;
  public:

    RecordExprEvaluator(EvalInfo &info, const LValue &This, APValue &Result)
      : ExprEvaluatorBaseTy(info), This(This), Result(Result) {}

    bool Success(const CCValue &V, const Expr *E) {
      return CheckConstantExpression(V, Result);
    }
    bool Error(const Expr *E) { return false; }

    bool VisitCastExpr(const CastExpr *E);
    bool VisitInitListExpr(const InitListExpr *E);
    bool VisitCXXConstructExpr(const CXXConstructExpr *E);
  };
}

bool RecordExprEvaluator::VisitCastExpr(const CastExpr *E) {
  switch (E->getCastKind()) {
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_ConstructorConversion:
    return Visit(E->getSubExpr());

  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase: {
    CCValue DerivedObject;
    if (!Evaluate(DerivedObject, Info, E->getSubExpr()) ||
        !DerivedObject.isStruct())
      return false;

    // Derived-to-base rvalue conversion: just slice off the derived part.
    APValue *Value = &DerivedObject;
    const CXXRecordDecl *RD = E->getSubExpr()->getType()->getAsCXXRecordDecl();
    for (CastExpr::path_const_iterator PathI = E->path_begin(),
         PathE = E->path_end(); PathI != PathE; ++PathI) {
      assert(!(*PathI)->isVirtual() && "record rvalue with virtual base");
      const CXXRecordDecl *Base = (*PathI)->getType()->getAsCXXRecordDecl();
      Value = &Value->getStructBase(getBaseIndex(RD, Base));
      RD = Base;
    }
    Result = *Value;
    return true;
  }
  }
}

bool RecordExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const RecordDecl *RD = E->getType()->castAs<RecordType>()->getDecl();
  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);

  if (RD->isUnion()) {
    Result = APValue(E->getInitializedFieldInUnion());
    if (!E->getNumInits())
      return true;
    LValue Subobject = This;
    HandleLValueMember(Info, Subobject, E->getInitializedFieldInUnion(),
                       &Layout);
    return EvaluateConstantExpression(Result.getUnionValue(), Info,
                                      Subobject, E->getInit(0));
  }

  assert((!isa<CXXRecordDecl>(RD) || !cast<CXXRecordDecl>(RD)->getNumBases()) &&
         "initializer list for class with base classes");
  Result = APValue(APValue::UninitStruct(), 0,
                   std::distance(RD->field_begin(), RD->field_end()));
  unsigned ElementNo = 0;
  for (RecordDecl::field_iterator Field = RD->field_begin(),
       FieldEnd = RD->field_end(); Field != FieldEnd; ++Field) {
    // Anonymous bit-fields are not considered members of the class for
    // purposes of aggregate initialization.
    if (Field->isUnnamedBitfield())
      continue;

    LValue Subobject = This;
    HandleLValueMember(Info, Subobject, *Field, &Layout);

    if (ElementNo < E->getNumInits()) {
      if (!EvaluateConstantExpression(
            Result.getStructField((*Field)->getFieldIndex()),
            Info, Subobject, E->getInit(ElementNo++)))
        return false;
    } else {
      // Perform an implicit value-initialization for members beyond the end of
      // the initializer list.
      ImplicitValueInitExpr VIE(Field->getType());
      if (!EvaluateConstantExpression(
            Result.getStructField((*Field)->getFieldIndex()),
            Info, Subobject, &VIE))
        return false;
    }
  }

  return true;
}

bool RecordExprEvaluator::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  const CXXConstructorDecl *FD = E->getConstructor();
  const FunctionDecl *Definition = 0;
  FD->getBody(Definition);

  if (!Definition || !Definition->isConstexpr() || Definition->isInvalidDecl())
    return false;

  // FIXME: Elide the copy/move construction wherever we can.
  if (E->isElidable())
    if (const MaterializeTemporaryExpr *ME
          = dyn_cast<MaterializeTemporaryExpr>(E->getArg(0)))
      return Visit(ME->GetTemporaryExpr());

  llvm::ArrayRef<const Expr*> Args(E->getArgs(), E->getNumArgs());
  return HandleConstructorCall(This, Args, cast<CXXConstructorDecl>(Definition),
                               Info, Result);
}

static bool EvaluateRecord(const Expr *E, const LValue &This,
                           APValue &Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isRecordType() &&
         E->getType()->isLiteralType() &&
         "can't evaluate expression as a record rvalue");
  return RecordExprEvaluator(Info, This, Result).Visit(E);
}

//===----------------------------------------------------------------------===//
// Vector Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class VectorExprEvaluator
  : public ExprEvaluatorBase<VectorExprEvaluator, bool> {
    APValue &Result;
  public:

    VectorExprEvaluator(EvalInfo &info, APValue &Result)
      : ExprEvaluatorBaseTy(info), Result(Result) {}

    bool Success(const ArrayRef<APValue> &V, const Expr *E) {
      assert(V.size() == E->getType()->castAs<VectorType>()->getNumElements());
      // FIXME: remove this APValue copy.
      Result = APValue(V.data(), V.size());
      return true;
    }
    bool Success(const CCValue &V, const Expr *E) {
      assert(V.isVector());
      Result = V;
      return true;
    }
    bool Error(const Expr *E) { return false; }
    bool ValueInitialization(const Expr *E);

    bool VisitUnaryReal(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    bool VisitCastExpr(const CastExpr* E);
    bool VisitInitListExpr(const InitListExpr *E);
    bool VisitUnaryImag(const UnaryOperator *E);
    // FIXME: Missing: unary -, unary ~, binary add/sub/mul/div,
    //                 binary comparisons, binary and/or/xor,
    //                 shufflevector, ExtVectorElementExpr
    //        (Note that these require implementing conversions
    //         between vector types.)
  };
} // end anonymous namespace

static bool EvaluateVector(const Expr* E, APValue& Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isVectorType() &&"not a vector rvalue");
  return VectorExprEvaluator(Info, Result).Visit(E);
}

bool VectorExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const VectorType *VTy = E->getType()->castAs<VectorType>();
  QualType EltTy = VTy->getElementType();
  unsigned NElts = VTy->getNumElements();
  unsigned EltWidth = Info.Ctx.getTypeSize(EltTy);

  const Expr* SE = E->getSubExpr();
  QualType SETy = SE->getType();

  switch (E->getCastKind()) {
  case CK_VectorSplat: {
    APValue Val = APValue();
    if (SETy->isIntegerType()) {
      APSInt IntResult;
      if (!EvaluateInteger(SE, IntResult, Info))
         return Error(E);
      Val = APValue(IntResult);
    } else if (SETy->isRealFloatingType()) {
       APFloat F(0.0);
       if (!EvaluateFloat(SE, F, Info))
         return Error(E);
       Val = APValue(F);
    } else {
      return Error(E);
    }

    // Splat and create vector APValue.
    SmallVector<APValue, 4> Elts(NElts, Val);
    return Success(Elts, E);
  }
  case CK_BitCast: {
    // FIXME: this is wrong for any cast other than a no-op cast.
    if (SETy->isVectorType())
      return Visit(SE);

    if (!SETy->isIntegerType())
      return Error(E);

    APSInt Init;
    if (!EvaluateInteger(SE, Init, Info))
      return Error(E);

    assert((EltTy->isIntegerType() || EltTy->isRealFloatingType()) &&
           "Vectors must be composed of ints or floats");

    SmallVector<APValue, 4> Elts;
    for (unsigned i = 0; i != NElts; ++i) {
      APSInt Tmp = Init.extOrTrunc(EltWidth);

      if (EltTy->isIntegerType())
        Elts.push_back(APValue(Tmp));
      else
        Elts.push_back(APValue(APFloat(Tmp)));

      Init >>= EltWidth;
    }
    return Success(Elts, E);
  }
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);
  }
}

bool
VectorExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const VectorType *VT = E->getType()->castAs<VectorType>();
  unsigned NumInits = E->getNumInits();
  unsigned NumElements = VT->getNumElements();

  QualType EltTy = VT->getElementType();
  SmallVector<APValue, 4> Elements;

  // If a vector is initialized with a single element, that value
  // becomes every element of the vector, not just the first.
  // This is the behavior described in the IBM AltiVec documentation.
  if (NumInits == 1) {

    // Handle the case where the vector is initialized by another
    // vector (OpenCL 6.1.6).
    if (E->getInit(0)->getType()->isVectorType())
      return Visit(E->getInit(0));

    APValue InitValue;
    if (EltTy->isIntegerType()) {
      llvm::APSInt sInt(32);
      if (!EvaluateInteger(E->getInit(0), sInt, Info))
        return Error(E);
      InitValue = APValue(sInt);
    } else {
      llvm::APFloat f(0.0);
      if (!EvaluateFloat(E->getInit(0), f, Info))
        return Error(E);
      InitValue = APValue(f);
    }
    for (unsigned i = 0; i < NumElements; i++) {
      Elements.push_back(InitValue);
    }
  } else {
    for (unsigned i = 0; i < NumElements; i++) {
      if (EltTy->isIntegerType()) {
        llvm::APSInt sInt(32);
        if (i < NumInits) {
          if (!EvaluateInteger(E->getInit(i), sInt, Info))
            return Error(E);
        } else {
          sInt = Info.Ctx.MakeIntValue(0, EltTy);
        }
        Elements.push_back(APValue(sInt));
      } else {
        llvm::APFloat f(0.0);
        if (i < NumInits) {
          if (!EvaluateFloat(E->getInit(i), f, Info))
            return Error(E);
        } else {
          f = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy));
        }
        Elements.push_back(APValue(f));
      }
    }
  }
  return Success(Elements, E);
}

bool
VectorExprEvaluator::ValueInitialization(const Expr *E) {
  const VectorType *VT = E->getType()->getAs<VectorType>();
  QualType EltTy = VT->getElementType();
  APValue ZeroElement;
  if (EltTy->isIntegerType())
    ZeroElement = APValue(Info.Ctx.MakeIntValue(0, EltTy));
  else
    ZeroElement =
        APValue(APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy)));

  SmallVector<APValue, 4> Elements(VT->getNumElements(), ZeroElement);
  return Success(Elements, E);
}

bool VectorExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  VisitIgnoredValue(E->getSubExpr());
  return ValueInitialization(E);
}

//===----------------------------------------------------------------------===//
// Array Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class ArrayExprEvaluator
  : public ExprEvaluatorBase<ArrayExprEvaluator, bool> {
    const LValue &This;
    APValue &Result;
  public:

    ArrayExprEvaluator(EvalInfo &Info, const LValue &This, APValue &Result)
      : ExprEvaluatorBaseTy(Info), This(This), Result(Result) {}

    bool Success(const APValue &V, const Expr *E) {
      assert(V.isArray() && "Expected array type");
      Result = V;
      return true;
    }
    bool Error(const Expr *E) { return false; }

    bool ValueInitialization(const Expr *E) {
      const ConstantArrayType *CAT =
          Info.Ctx.getAsConstantArrayType(E->getType());
      if (!CAT)
        return false;

      Result = APValue(APValue::UninitArray(), 0,
                       CAT->getSize().getZExtValue());
      if (!Result.hasArrayFiller()) return true;

      // Value-initialize all elements.
      LValue Subobject = This;
      Subobject.Designator.addIndex(0);
      ImplicitValueInitExpr VIE(CAT->getElementType());
      return EvaluateConstantExpression(Result.getArrayFiller(), Info,
                                        Subobject, &VIE);
    }

    // FIXME: We also get CXXConstructExpr, in cases like:
    //   struct S { constexpr S(); }; constexpr S s[10];
    bool VisitInitListExpr(const InitListExpr *E);
  };
} // end anonymous namespace

static bool EvaluateArray(const Expr *E, const LValue &This,
                          APValue &Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isArrayType() &&
         E->getType()->isLiteralType() && "not a literal array rvalue");
  return ArrayExprEvaluator(Info, This, Result).Visit(E);
}

bool ArrayExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const ConstantArrayType *CAT = Info.Ctx.getAsConstantArrayType(E->getType());
  if (!CAT)
    return false;

  Result = APValue(APValue::UninitArray(), E->getNumInits(),
                   CAT->getSize().getZExtValue());
  LValue Subobject = This;
  Subobject.Designator.addIndex(0);
  unsigned Index = 0;
  for (InitListExpr::const_iterator I = E->begin(), End = E->end();
       I != End; ++I, ++Index) {
    if (!EvaluateConstantExpression(Result.getArrayInitializedElt(Index),
                                    Info, Subobject, cast<Expr>(*I)))
      return false;
    if (!HandleLValueArrayAdjustment(Info, Subobject, CAT->getElementType(), 1))
      return false;
  }

  if (!Result.hasArrayFiller()) return true;
  assert(E->hasArrayFiller() && "no array filler for incomplete init list");
  // FIXME: The Subobject here isn't necessarily right. This rarely matters,
  // but sometimes does:
  //   struct S { constexpr S() : p(&p) {} void *p; };
  //   S s[10] = {};
  return EvaluateConstantExpression(Result.getArrayFiller(), Info,
                                    Subobject, E->getArrayFiller());
}

//===----------------------------------------------------------------------===//
// Integer Evaluation
//
// As a GNU extension, we support casting pointers to sufficiently-wide integer
// types and back in constant folding. Integer values are thus represented
// either as an integer-valued APValue, or as an lvalue-valued APValue.
//===----------------------------------------------------------------------===//

namespace {
class IntExprEvaluator
  : public ExprEvaluatorBase<IntExprEvaluator, bool> {
  CCValue &Result;
public:
  IntExprEvaluator(EvalInfo &info, CCValue &result)
    : ExprEvaluatorBaseTy(info), Result(result) {}

  bool Success(const llvm::APSInt &SI, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() &&
           "Invalid evaluation result.");
    assert(SI.isSigned() == E->getType()->isSignedIntegerOrEnumerationType() &&
           "Invalid evaluation result.");
    assert(SI.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = CCValue(SI);
    return true;
  }

  bool Success(const llvm::APInt &I, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    assert(I.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = CCValue(APSInt(I));
    Result.getInt().setIsUnsigned(
                            E->getType()->isUnsignedIntegerOrEnumerationType());
    return true;
  }

  bool Success(uint64_t Value, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    Result = CCValue(Info.Ctx.MakeIntValue(Value, E->getType()));
    return true;
  }

  bool Success(CharUnits Size, const Expr *E) {
    return Success(Size.getQuantity(), E);
  }


  bool Error(SourceLocation L, diag::kind D, const Expr *E) {
    // Take the first error.
    if (Info.EvalStatus.Diag == 0) {
      Info.EvalStatus.DiagLoc = L;
      Info.EvalStatus.Diag = D;
      Info.EvalStatus.DiagExpr = E;
    }
    return false;
  }

  bool Success(const CCValue &V, const Expr *E) {
    if (V.isLValue()) {
      Result = V;
      return true;
    }
    return Success(V.getInt(), E);
  }
  bool Error(const Expr *E) {
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  bool ValueInitialization(const Expr *E) { return Success(0, E); }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitCharacterLiteral(const CharacterLiteral *E) {
    return Success(E->getValue(), E);
  }

  bool CheckReferencedDecl(const Expr *E, const Decl *D);
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (CheckReferencedDecl(E, E->getDecl()))
      return true;

    return ExprEvaluatorBaseTy::VisitDeclRefExpr(E);
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (CheckReferencedDecl(E, E->getMemberDecl())) {
      VisitIgnoredValue(E->getBase());
      return true;
    }

    return ExprEvaluatorBaseTy::VisitMemberExpr(E);
  }

  bool VisitCallExpr(const CallExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitOffsetOfExpr(const OffsetOfExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr* E);
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);

  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return Success(E->getValue(), E);
  }

  // Note, GNU defines __null as an integer, not a pointer.
  bool VisitGNUNullExpr(const GNUNullExpr *E) {
    return ValueInitialization(E);
  }

  bool VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitBinaryTypeTraitExpr(const BinaryTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  bool VisitCXXNoexceptExpr(const CXXNoexceptExpr *E);
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *E);

private:
  CharUnits GetAlignOfExpr(const Expr *E);
  CharUnits GetAlignOfType(QualType T);
  static QualType GetObjectType(APValue::LValueBase B);
  bool TryEvaluateBuiltinObjectSize(const CallExpr *E);
  // FIXME: Missing: array subscript of vector, member of vector
};
} // end anonymous namespace

/// EvaluateIntegerOrLValue - Evaluate an rvalue integral-typed expression, and
/// produce either the integer value or a pointer.
///
/// GCC has a heinous extension which folds casts between pointer types and
/// pointer-sized integral types. We support this by allowing the evaluation of
/// an integer rvalue to produce a pointer (represented as an lvalue) instead.
/// Some simple arithmetic on such values is supported (they are treated much
/// like char*).
static bool EvaluateIntegerOrLValue(const Expr* E, CCValue &Result,
                                    EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isIntegralOrEnumerationType());
  return IntExprEvaluator(Info, Result).Visit(E);
}

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  CCValue Val;
  if (!EvaluateIntegerOrLValue(E, Val, Info) || !Val.isInt())
    return false;
  Result = Val.getInt();
  return true;
}

bool IntExprEvaluator::CheckReferencedDecl(const Expr* E, const Decl* D) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D)) {
    // Check for signedness/width mismatches between E type and ECD value.
    bool SameSign = (ECD->getInitVal().isSigned()
                     == E->getType()->isSignedIntegerOrEnumerationType());
    bool SameWidth = (ECD->getInitVal().getBitWidth()
                      == Info.Ctx.getIntWidth(E->getType()));
    if (SameSign && SameWidth)
      return Success(ECD->getInitVal(), E);
    else {
      // Get rid of mismatch (otherwise Success assertions will fail)
      // by computing a new value matching the type of E.
      llvm::APSInt Val = ECD->getInitVal();
      if (!SameSign)
        Val.setIsSigned(!ECD->getInitVal().isSigned());
      if (!SameWidth)
        Val = Val.extOrTrunc(Info.Ctx.getIntWidth(E->getType()));
      return Success(Val, E);
    }
  }
  return false;
}

/// EvaluateBuiltinClassifyType - Evaluate __builtin_classify_type the same way
/// as GCC.
static int EvaluateBuiltinClassifyType(const CallExpr *E) {
  // The following enum mimics the values returned by GCC.
  // FIXME: Does GCC differ between lvalue and rvalue references here?
  enum gcc_type_class {
    no_type_class = -1,
    void_type_class, integer_type_class, char_type_class,
    enumeral_type_class, boolean_type_class,
    pointer_type_class, reference_type_class, offset_type_class,
    real_type_class, complex_type_class,
    function_type_class, method_type_class,
    record_type_class, union_type_class,
    array_type_class, string_type_class,
    lang_type_class
  };

  // If no argument was supplied, default to "no_type_class". This isn't
  // ideal, however it is what gcc does.
  if (E->getNumArgs() == 0)
    return no_type_class;

  QualType ArgTy = E->getArg(0)->getType();
  if (ArgTy->isVoidType())
    return void_type_class;
  else if (ArgTy->isEnumeralType())
    return enumeral_type_class;
  else if (ArgTy->isBooleanType())
    return boolean_type_class;
  else if (ArgTy->isCharType())
    return string_type_class; // gcc doesn't appear to use char_type_class
  else if (ArgTy->isIntegerType())
    return integer_type_class;
  else if (ArgTy->isPointerType())
    return pointer_type_class;
  else if (ArgTy->isReferenceType())
    return reference_type_class;
  else if (ArgTy->isRealType())
    return real_type_class;
  else if (ArgTy->isComplexType())
    return complex_type_class;
  else if (ArgTy->isFunctionType())
    return function_type_class;
  else if (ArgTy->isStructureOrClassType())
    return record_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else if (ArgTy->isArrayType())
    return array_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else  // FIXME: offset_type_class, method_type_class, & lang_type_class?
    llvm_unreachable("CallExpr::isBuiltinClassifyType(): unimplemented type");
  return -1;
}

/// Retrieves the "underlying object type" of the given expression,
/// as used by __builtin_object_size.
QualType IntExprEvaluator::GetObjectType(APValue::LValueBase B) {
  if (const ValueDecl *D = B.dyn_cast<const ValueDecl*>()) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(D))
      return VD->getType();
  } else if (const Expr *E = B.get<const Expr*>()) {
    if (isa<CompoundLiteralExpr>(E))
      return E->getType();
  }

  return QualType();
}

bool IntExprEvaluator::TryEvaluateBuiltinObjectSize(const CallExpr *E) {
  // TODO: Perhaps we should let LLVM lower this?
  LValue Base;
  if (!EvaluatePointer(E->getArg(0), Base, Info))
    return false;

  // If we can prove the base is null, lower to zero now.
  if (!Base.getLValueBase()) return Success(0, E);

  QualType T = GetObjectType(Base.getLValueBase());
  if (T.isNull() ||
      T->isIncompleteType() ||
      T->isFunctionType() ||
      T->isVariablyModifiedType() ||
      T->isDependentType())
    return false;

  CharUnits Size = Info.Ctx.getTypeSizeInChars(T);
  CharUnits Offset = Base.getLValueOffset();

  if (!Offset.isNegative() && Offset <= Size)
    Size -= Offset;
  else
    Size = CharUnits::Zero();
  return Success(Size, E);
}

bool IntExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall()) {
  default:
    return ExprEvaluatorBaseTy::VisitCallExpr(E);

  case Builtin::BI__builtin_object_size: {
    if (TryEvaluateBuiltinObjectSize(E))
      return true;

    // If evaluating the argument has side-effects we can't determine
    // the size of the object and lower it to unknown now.
    if (E->getArg(0)->HasSideEffects(Info.Ctx)) {
      if (E->getArg(1)->EvaluateKnownConstInt(Info.Ctx).getZExtValue() <= 1)
        return Success(-1ULL, E);
      return Success(0, E);
    }

    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  case Builtin::BI__builtin_classify_type:
    return Success(EvaluateBuiltinClassifyType(E), E);

  case Builtin::BI__builtin_constant_p:
    // __builtin_constant_p always has one operand: it returns true if that
    // operand can be folded, false otherwise.
    return Success(E->getArg(0)->isEvaluatable(Info.Ctx), E);
      
  case Builtin::BI__builtin_eh_return_data_regno: {
    int Operand = E->getArg(0)->EvaluateKnownConstInt(Info.Ctx).getZExtValue();
    Operand = Info.Ctx.getTargetInfo().getEHDataRegisterNumber(Operand);
    return Success(Operand, E);
  }

  case Builtin::BI__builtin_expect:
    return Visit(E->getArg(0));
      
  case Builtin::BIstrlen:
  case Builtin::BI__builtin_strlen:
    // As an extension, we support strlen() and __builtin_strlen() as constant
    // expressions when the argument is a string literal.
    if (const StringLiteral *S
               = dyn_cast<StringLiteral>(E->getArg(0)->IgnoreParenImpCasts())) {
      // The string literal may have embedded null characters. Find the first
      // one and truncate there.
      StringRef Str = S->getString();
      StringRef::size_type Pos = Str.find(0);
      if (Pos != StringRef::npos)
        Str = Str.substr(0, Pos);
      
      return Success(Str.size(), E);
    }
      
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);

  case Builtin::BI__atomic_is_lock_free: {
    APSInt SizeVal;
    if (!EvaluateInteger(E->getArg(0), SizeVal, Info))
      return false;

    // For __atomic_is_lock_free(sizeof(_Atomic(T))), if the size is a power
    // of two less than the maximum inline atomic width, we know it is
    // lock-free.  If the size isn't a power of two, or greater than the
    // maximum alignment where we promote atomics, we know it is not lock-free
    // (at least not in the sense of atomic_is_lock_free).  Otherwise,
    // the answer can only be determined at runtime; for example, 16-byte
    // atomics have lock-free implementations on some, but not all,
    // x86-64 processors.

    // Check power-of-two.
    CharUnits Size = CharUnits::fromQuantity(SizeVal.getZExtValue());
    if (!Size.isPowerOfTwo())
#if 0
      // FIXME: Suppress this folding until the ABI for the promotion width
      // settles.
      return Success(0, E);
#else
      return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
#endif

#if 0
    // Check against promotion width.
    // FIXME: Suppress this folding until the ABI for the promotion width
    // settles.
    unsigned PromoteWidthBits =
        Info.Ctx.getTargetInfo().getMaxAtomicPromoteWidth();
    if (Size > Info.Ctx.toCharUnitsFromBits(PromoteWidthBits))
      return Success(0, E);
#endif

    // Check against inlining width.
    unsigned InlineWidthBits =
        Info.Ctx.getTargetInfo().getMaxAtomicInlineWidth();
    if (Size <= Info.Ctx.toCharUnitsFromBits(InlineWidthBits))
      return Success(1, E);

    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }
  }
}

static bool HasSameBase(const LValue &A, const LValue &B) {
  if (!A.getLValueBase())
    return !B.getLValueBase();
  if (!B.getLValueBase())
    return false;

  if (A.getLValueBase().getOpaqueValue() !=
      B.getLValueBase().getOpaqueValue()) {
    const Decl *ADecl = GetLValueBaseDecl(A);
    if (!ADecl)
      return false;
    const Decl *BDecl = GetLValueBaseDecl(B);
    if (!BDecl || ADecl->getCanonicalDecl() != BDecl->getCanonicalDecl())
      return false;
  }

  return IsGlobalLValue(A.getLValueBase()) ||
         A.getLValueFrame() == B.getLValueFrame();
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->isAssignmentOp())
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);

  if (E->getOpcode() == BO_Comma) {
    VisitIgnoredValue(E->getLHS());
    return Visit(E->getRHS());
  }

  if (E->isLogicalOp()) {
    // These need to be handled specially because the operands aren't
    // necessarily integral
    bool lhsResult, rhsResult;

    if (EvaluateAsBooleanCondition(E->getLHS(), lhsResult, Info)) {
      // We were able to evaluate the LHS, see if we can get away with not
      // evaluating the RHS: 0 && X -> 0, 1 || X -> 1
      if (lhsResult == (E->getOpcode() == BO_LOr))
        return Success(lhsResult, E);

      if (EvaluateAsBooleanCondition(E->getRHS(), rhsResult, Info)) {
        if (E->getOpcode() == BO_LOr)
          return Success(lhsResult || rhsResult, E);
        else
          return Success(lhsResult && rhsResult, E);
      }
    } else {
      if (EvaluateAsBooleanCondition(E->getRHS(), rhsResult, Info)) {
        // We can't evaluate the LHS; however, sometimes the result
        // is determined by the RHS: X && 0 -> 0, X || 1 -> 1.
        if (rhsResult == (E->getOpcode() == BO_LOr) ||
            !rhsResult == (E->getOpcode() == BO_LAnd)) {
          // Since we weren't able to evaluate the left hand side, it
          // must have had side effects.
          Info.EvalStatus.HasSideEffects = true;

          return Success(rhsResult, E);
        }
      }
    }

    return false;
  }

  QualType LHSTy = E->getLHS()->getType();
  QualType RHSTy = E->getRHS()->getType();

  if (LHSTy->isAnyComplexType()) {
    assert(RHSTy->isAnyComplexType() && "Invalid comparison");
    ComplexValue LHS, RHS;

    if (!EvaluateComplex(E->getLHS(), LHS, Info))
      return false;

    if (!EvaluateComplex(E->getRHS(), RHS, Info))
      return false;

    if (LHS.isComplexFloat()) {
      APFloat::cmpResult CR_r =
        LHS.getComplexFloatReal().compare(RHS.getComplexFloatReal());
      APFloat::cmpResult CR_i =
        LHS.getComplexFloatImag().compare(RHS.getComplexFloatImag());

      if (E->getOpcode() == BO_EQ)
        return Success((CR_r == APFloat::cmpEqual &&
                        CR_i == APFloat::cmpEqual), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid complex comparison.");
        return Success(((CR_r == APFloat::cmpGreaterThan ||
                         CR_r == APFloat::cmpLessThan ||
                         CR_r == APFloat::cmpUnordered) ||
                        (CR_i == APFloat::cmpGreaterThan ||
                         CR_i == APFloat::cmpLessThan ||
                         CR_i == APFloat::cmpUnordered)), E);
      }
    } else {
      if (E->getOpcode() == BO_EQ)
        return Success((LHS.getComplexIntReal() == RHS.getComplexIntReal() &&
                        LHS.getComplexIntImag() == RHS.getComplexIntImag()), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid compex comparison.");
        return Success((LHS.getComplexIntReal() != RHS.getComplexIntReal() ||
                        LHS.getComplexIntImag() != RHS.getComplexIntImag()), E);
      }
    }
  }

  if (LHSTy->isRealFloatingType() &&
      RHSTy->isRealFloatingType()) {
    APFloat RHS(0.0), LHS(0.0);

    if (!EvaluateFloat(E->getRHS(), RHS, Info))
      return false;

    if (!EvaluateFloat(E->getLHS(), LHS, Info))
      return false;

    APFloat::cmpResult CR = LHS.compare(RHS);

    switch (E->getOpcode()) {
    default:
      llvm_unreachable("Invalid binary operator!");
    case BO_LT:
      return Success(CR == APFloat::cmpLessThan, E);
    case BO_GT:
      return Success(CR == APFloat::cmpGreaterThan, E);
    case BO_LE:
      return Success(CR == APFloat::cmpLessThan || CR == APFloat::cmpEqual, E);
    case BO_GE:
      return Success(CR == APFloat::cmpGreaterThan || CR == APFloat::cmpEqual,
                     E);
    case BO_EQ:
      return Success(CR == APFloat::cmpEqual, E);
    case BO_NE:
      return Success(CR == APFloat::cmpGreaterThan
                     || CR == APFloat::cmpLessThan
                     || CR == APFloat::cmpUnordered, E);
    }
  }

  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    if (E->getOpcode() == BO_Sub || E->isComparisonOp()) {
      LValue LHSValue;
      if (!EvaluatePointer(E->getLHS(), LHSValue, Info))
        return false;

      LValue RHSValue;
      if (!EvaluatePointer(E->getRHS(), RHSValue, Info))
        return false;

      // Reject differing bases from the normal codepath; we special-case
      // comparisons to null.
      if (!HasSameBase(LHSValue, RHSValue)) {
        // Inequalities and subtractions between unrelated pointers have
        // unspecified or undefined behavior.
        if (!E->isEqualityOp())
          return false;
        // A constant address may compare equal to the address of a symbol.
        // The one exception is that address of an object cannot compare equal
        // to a null pointer constant.
        if ((!LHSValue.Base && !LHSValue.Offset.isZero()) ||
            (!RHSValue.Base && !RHSValue.Offset.isZero()))
          return false;
        // It's implementation-defined whether distinct literals will have
        // distinct addresses. In clang, we do not guarantee the addresses are
        // distinct. However, we do know that the address of a literal will be
        // non-null.
        if ((IsLiteralLValue(LHSValue) || IsLiteralLValue(RHSValue)) &&
            LHSValue.Base && RHSValue.Base)
          return false;
        // We can't tell whether weak symbols will end up pointing to the same
        // object.
        if (IsWeakLValue(LHSValue) || IsWeakLValue(RHSValue))
          return false;
        // Pointers with different bases cannot represent the same object.
        // (Note that clang defaults to -fmerge-all-constants, which can
        // lead to inconsistent results for comparisons involving the address
        // of a constant; this generally doesn't matter in practice.)
        return Success(E->getOpcode() == BO_NE, E);
      }

      // FIXME: Implement the C++11 restrictions:
      //  - Pointer subtractions must be on elements of the same array.
      //  - Pointer comparisons must be between members with the same access.

      if (E->getOpcode() == BO_Sub) {
        QualType Type = E->getLHS()->getType();
        QualType ElementType = Type->getAs<PointerType>()->getPointeeType();

        CharUnits ElementSize;
        if (!HandleSizeof(Info, ElementType, ElementSize))
          return false;

        CharUnits Diff = LHSValue.getLValueOffset() -
                             RHSValue.getLValueOffset();
        return Success(Diff / ElementSize, E);
      }

      const CharUnits &LHSOffset = LHSValue.getLValueOffset();
      const CharUnits &RHSOffset = RHSValue.getLValueOffset();
      switch (E->getOpcode()) {
      default: llvm_unreachable("missing comparison operator");
      case BO_LT: return Success(LHSOffset < RHSOffset, E);
      case BO_GT: return Success(LHSOffset > RHSOffset, E);
      case BO_LE: return Success(LHSOffset <= RHSOffset, E);
      case BO_GE: return Success(LHSOffset >= RHSOffset, E);
      case BO_EQ: return Success(LHSOffset == RHSOffset, E);
      case BO_NE: return Success(LHSOffset != RHSOffset, E);
      }
    }
  }
  if (!LHSTy->isIntegralOrEnumerationType() ||
      !RHSTy->isIntegralOrEnumerationType()) {
    // We can't continue from here for non-integral types, and they
    // could potentially confuse the following operations.
    return false;
  }

  // The LHS of a constant expr is always evaluated and needed.
  CCValue LHSVal;
  if (!EvaluateIntegerOrLValue(E->getLHS(), LHSVal, Info))
    return false; // error in subexpression.

  if (!Visit(E->getRHS()))
    return false;
  CCValue &RHSVal = Result;

  // Handle cases like (unsigned long)&a + 4.
  if (E->isAdditiveOp() && LHSVal.isLValue() && RHSVal.isInt()) {
    CharUnits AdditionalOffset = CharUnits::fromQuantity(
                                     RHSVal.getInt().getZExtValue());
    if (E->getOpcode() == BO_Add)
      LHSVal.getLValueOffset() += AdditionalOffset;
    else
      LHSVal.getLValueOffset() -= AdditionalOffset;
    Result = LHSVal;
    return true;
  }

  // Handle cases like 4 + (unsigned long)&a
  if (E->getOpcode() == BO_Add &&
        RHSVal.isLValue() && LHSVal.isInt()) {
    RHSVal.getLValueOffset() += CharUnits::fromQuantity(
                                    LHSVal.getInt().getZExtValue());
    // Note that RHSVal is Result.
    return true;
  }

  // All the following cases expect both operands to be an integer
  if (!LHSVal.isInt() || !RHSVal.isInt())
    return false;

  APSInt &LHS = LHSVal.getInt();
  APSInt &RHS = RHSVal.getInt();

  switch (E->getOpcode()) {
  default:
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case BO_Mul: return Success(LHS * RHS, E);
  case BO_Add: return Success(LHS + RHS, E);
  case BO_Sub: return Success(LHS - RHS, E);
  case BO_And: return Success(LHS & RHS, E);
  case BO_Xor: return Success(LHS ^ RHS, E);
  case BO_Or:  return Success(LHS | RHS, E);
  case BO_Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(LHS / RHS, E);
  case BO_Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(LHS % RHS, E);
  case BO_Shl: {
    // During constant-folding, a negative shift is an opposite shift.
    if (RHS.isSigned() && RHS.isNegative()) {
      RHS = -RHS;
      goto shift_right;
    }

  shift_left:
    unsigned SA
      = (unsigned) RHS.getLimitedValue(LHS.getBitWidth()-1);
    return Success(LHS << SA, E);
  }
  case BO_Shr: {
    // During constant-folding, a negative shift is an opposite shift.
    if (RHS.isSigned() && RHS.isNegative()) {
      RHS = -RHS;
      goto shift_left;
    }

  shift_right:
    unsigned SA =
      (unsigned) RHS.getLimitedValue(LHS.getBitWidth()-1);
    return Success(LHS >> SA, E);
  }

  case BO_LT: return Success(LHS < RHS, E);
  case BO_GT: return Success(LHS > RHS, E);
  case BO_LE: return Success(LHS <= RHS, E);
  case BO_GE: return Success(LHS >= RHS, E);
  case BO_EQ: return Success(LHS == RHS, E);
  case BO_NE: return Success(LHS != RHS, E);
  }
}

CharUnits IntExprEvaluator::GetAlignOfType(QualType T) {
  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = T->getAs<ReferenceType>())
    T = Ref->getPointeeType();

  // __alignof is defined to return the preferred alignment.
  return Info.Ctx.toCharUnitsFromBits(
    Info.Ctx.getPreferredTypeAlign(T.getTypePtr()));
}

CharUnits IntExprEvaluator::GetAlignOfExpr(const Expr *E) {
  E = E->IgnoreParens();

  // alignof decl is always accepted, even if it doesn't make sense: we default
  // to 1 in those cases.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return Info.Ctx.getDeclAlign(DRE->getDecl(), 
                                 /*RefAsPointee*/true);

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return Info.Ctx.getDeclAlign(ME->getMemberDecl(),
                                 /*RefAsPointee*/true);

  return GetAlignOfType(E->getType());
}


/// VisitUnaryExprOrTypeTraitExpr - Evaluate a sizeof, alignof or vec_step with
/// a result as the expression's type.
bool IntExprEvaluator::VisitUnaryExprOrTypeTraitExpr(
                                    const UnaryExprOrTypeTraitExpr *E) {
  switch(E->getKind()) {
  case UETT_AlignOf: {
    if (E->isArgumentType())
      return Success(GetAlignOfType(E->getArgumentType()), E);
    else
      return Success(GetAlignOfExpr(E->getArgumentExpr()), E);
  }

  case UETT_VecStep: {
    QualType Ty = E->getTypeOfArgument();

    if (Ty->isVectorType()) {
      unsigned n = Ty->getAs<VectorType>()->getNumElements();

      // The vec_step built-in functions that take a 3-component
      // vector return 4. (OpenCL 1.1 spec 6.11.12)
      if (n == 3)
        n = 4;

      return Success(n, E);
    } else
      return Success(1, E);
  }

  case UETT_SizeOf: {
    QualType SrcTy = E->getTypeOfArgument();
    // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
    //   the result is the size of the referenced type."
    // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
    //   result shall be the alignment of the referenced type."
    if (const ReferenceType *Ref = SrcTy->getAs<ReferenceType>())
      SrcTy = Ref->getPointeeType();

    CharUnits Sizeof;
    if (!HandleSizeof(Info, SrcTy, Sizeof))
      return false;
    return Success(Sizeof, E);
  }
  }

  llvm_unreachable("unknown expr/type trait");
  return false;
}

bool IntExprEvaluator::VisitOffsetOfExpr(const OffsetOfExpr *OOE) {
  CharUnits Result;
  unsigned n = OOE->getNumComponents();
  if (n == 0)
    return false;
  QualType CurrentType = OOE->getTypeSourceInfo()->getType();
  for (unsigned i = 0; i != n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = OOE->getComponent(i);
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array: {
      const Expr *Idx = OOE->getIndexExpr(ON.getArrayExprIndex());
      APSInt IdxResult;
      if (!EvaluateInteger(Idx, IdxResult, Info))
        return false;
      const ArrayType *AT = Info.Ctx.getAsArrayType(CurrentType);
      if (!AT)
        return false;
      CurrentType = AT->getElementType();
      CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(CurrentType);
      Result += IdxResult.getSExtValue() * ElementSize;
        break;
    }
        
    case OffsetOfExpr::OffsetOfNode::Field: {
      FieldDecl *MemberDecl = ON.getField();
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT) 
        return false;
      RecordDecl *RD = RT->getDecl();
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);
      unsigned i = MemberDecl->getFieldIndex();
      assert(i < RL.getFieldCount() && "offsetof field in wrong type");
      Result += Info.Ctx.toCharUnitsFromBits(RL.getFieldOffset(i));
      CurrentType = MemberDecl->getType().getNonReferenceType();
      break;
    }
        
    case OffsetOfExpr::OffsetOfNode::Identifier:
      llvm_unreachable("dependent __builtin_offsetof");
      return false;
        
    case OffsetOfExpr::OffsetOfNode::Base: {
      CXXBaseSpecifier *BaseSpec = ON.getBase();
      if (BaseSpec->isVirtual())
        return false;

      // Find the layout of the class whose base we are looking into.
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT) 
        return false;
      RecordDecl *RD = RT->getDecl();
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

      // Find the base class itself.
      CurrentType = BaseSpec->getType();
      const RecordType *BaseRT = CurrentType->getAs<RecordType>();
      if (!BaseRT)
        return false;
      
      // Add the offset to the base.
      Result += RL.getBaseClassOffset(cast<CXXRecordDecl>(BaseRT->getDecl()));
      break;
    }
    }
  }
  return Success(Result, OOE);
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->getOpcode() == UO_LNot) {
    // LNot's operand isn't necessarily an integer, so we handle it specially.
    bool bres;
    if (!EvaluateAsBooleanCondition(E->getSubExpr(), bres, Info))
      return false;
    return Success(!bres, E);
  }

  // Only handle integral operations...
  if (!E->getSubExpr()->getType()->isIntegralOrEnumerationType())
    return false;

  // Get the operand value.
  CCValue Val;
  if (!Evaluate(Val, Info, E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case UO_Extension:
    // FIXME: Should extension allow i-c-e extension expressions in its scope?
    // If so, we could clear the diagnostic ID.
    return Success(Val, E);
  case UO_Plus:
    // The result is just the value.
    return Success(Val, E);
  case UO_Minus:
    if (!Val.isInt()) return false;
    return Success(-Val.getInt(), E);
  case UO_Not:
    if (!Val.isInt()) return false;
    return Success(~Val.getInt(), E);
  }
}

/// HandleCast - This is used to evaluate implicit or explicit casts where the
/// result type is integer.
bool IntExprEvaluator::VisitCastExpr(const CastExpr *E) {
  const Expr *SubExpr = E->getSubExpr();
  QualType DestType = E->getType();
  QualType SrcType = SubExpr->getType();

  switch (E->getCastKind()) {
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralToFloating:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
    llvm_unreachable("invalid cast kind for integral value");

  case CK_BitCast:
  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_UserDefinedConversion:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
    return false;

  case CK_LValueToRValue:
  case CK_NoOp:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_MemberPointerToBoolean:
  case CK_PointerToBoolean:
  case CK_IntegralToBoolean:
  case CK_FloatingToBoolean:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    bool BoolResult;
    if (!EvaluateAsBooleanCondition(SubExpr, BoolResult, Info))
      return false;
    return Success(BoolResult, E);
  }

  case CK_IntegralCast: {
    if (!Visit(SubExpr))
      return false;

    if (!Result.isInt()) {
      // Only allow casts of lvalues if they are lossless.
      return Info.Ctx.getTypeSize(DestType) == Info.Ctx.getTypeSize(SrcType);
    }

    return Success(HandleIntToIntCast(DestType, SrcType,
                                      Result.getInt(), Info.Ctx), E);
  }

  case CK_PointerToIntegral: {
    LValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;

    if (LV.getLValueBase()) {
      // Only allow based lvalue casts if they are lossless.
      if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(SrcType))
        return false;

      LV.Designator.setInvalid();
      LV.moveInto(Result);
      return true;
    }

    APSInt AsInt = Info.Ctx.MakeIntValue(LV.getLValueOffset().getQuantity(), 
                                         SrcType);
    return Success(HandleIntToIntCast(DestType, SrcType, AsInt, Info.Ctx), E);
  }

  case CK_IntegralComplexToReal: {
    ComplexValue C;
    if (!EvaluateComplex(SubExpr, C, Info))
      return false;
    return Success(C.getComplexIntReal(), E);
  }

  case CK_FloatingToIntegral: {
    APFloat F(0.0);
    if (!EvaluateFloat(SubExpr, F, Info))
      return false;

    return Success(HandleFloatToIntCast(DestType, SrcType, F, Info.Ctx), E);
  }
  }

  llvm_unreachable("unknown cast resulting in integral value");
  return false;
}

bool IntExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntReal(), E);
  }

  return Visit(E->getSubExpr());
}

bool IntExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isComplexIntegerType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntImag(), E);
  }

  VisitIgnoredValue(E->getSubExpr());
  return Success(0, E);
}

bool IntExprEvaluator::VisitSizeOfPackExpr(const SizeOfPackExpr *E) {
  return Success(E->getPackLength(), E);
}

bool IntExprEvaluator::VisitCXXNoexceptExpr(const CXXNoexceptExpr *E) {
  return Success(E->getValue(), E);
}

//===----------------------------------------------------------------------===//
// Float Evaluation
//===----------------------------------------------------------------------===//

namespace {
class FloatExprEvaluator
  : public ExprEvaluatorBase<FloatExprEvaluator, bool> {
  APFloat &Result;
public:
  FloatExprEvaluator(EvalInfo &info, APFloat &result)
    : ExprEvaluatorBaseTy(info), Result(result) {}

  bool Success(const CCValue &V, const Expr *e) {
    Result = V.getFloat();
    return true;
  }
  bool Error(const Stmt *S) {
    return false;
  }

  bool ValueInitialization(const Expr *E) {
    Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
    return true;
  }

  bool VisitCallExpr(const CallExpr *E);

  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitCastExpr(const CastExpr *E);

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  // FIXME: Missing: array subscript of vector, member of vector,
  //                 ImplicitValueInitExpr
};
} // end anonymous namespace

static bool EvaluateFloat(const Expr* E, APFloat& Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isRealFloatingType());
  return FloatExprEvaluator(Info, Result).Visit(E);
}

static bool TryEvaluateBuiltinNaN(const ASTContext &Context,
                                  QualType ResultTy,
                                  const Expr *Arg,
                                  bool SNaN,
                                  llvm::APFloat &Result) {
  const StringLiteral *S = dyn_cast<StringLiteral>(Arg->IgnoreParenCasts());
  if (!S) return false;

  const llvm::fltSemantics &Sem = Context.getFloatTypeSemantics(ResultTy);

  llvm::APInt fill;

  // Treat empty strings as if they were zero.
  if (S->getString().empty())
    fill = llvm::APInt(32, 0);
  else if (S->getString().getAsInteger(0, fill))
    return false;

  if (SNaN)
    Result = llvm::APFloat::getSNaN(Sem, false, &fill);
  else
    Result = llvm::APFloat::getQNaN(Sem, false, &fill);
  return true;
}

bool FloatExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall()) {
  default:
    return ExprEvaluatorBaseTy::VisitCallExpr(E);

  case Builtin::BI__builtin_huge_val:
  case Builtin::BI__builtin_huge_valf:
  case Builtin::BI__builtin_huge_vall:
  case Builtin::BI__builtin_inf:
  case Builtin::BI__builtin_inff:
  case Builtin::BI__builtin_infl: {
    const llvm::fltSemantics &Sem =
      Info.Ctx.getFloatTypeSemantics(E->getType());
    Result = llvm::APFloat::getInf(Sem);
    return true;
  }

  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
    return TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                                 true, Result);

  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
    // If this is __builtin_nan() turn this into a nan, otherwise we
    // can't constant fold it.
    return TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                                 false, Result);

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
    if (!EvaluateFloat(E->getArg(0), Result, Info))
      return false;

    if (Result.isNegative())
      Result.changeSign();
    return true;

  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl: {
    APFloat RHS(0.);
    if (!EvaluateFloat(E->getArg(0), Result, Info) ||
        !EvaluateFloat(E->getArg(1), RHS, Info))
      return false;
    Result.copySign(RHS);
    return true;
  }
  }
}

bool FloatExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatReal;
    return true;
  }

  return Visit(E->getSubExpr());
}

bool FloatExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatImag;
    return true;
  }

  VisitIgnoredValue(E->getSubExpr());
  const llvm::fltSemantics &Sem = Info.Ctx.getFloatTypeSemantics(E->getType());
  Result = llvm::APFloat::getZero(Sem);
  return true;
}

bool FloatExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  switch (E->getOpcode()) {
  default: return false;
  case UO_Plus:
    return EvaluateFloat(E->getSubExpr(), Result, Info);
  case UO_Minus:
    if (!EvaluateFloat(E->getSubExpr(), Result, Info))
      return false;
    Result.changeSign();
    return true;
  }
}

bool FloatExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_Comma) {
    VisitIgnoredValue(E->getLHS());
    return Visit(E->getRHS());
  }

  // We can't evaluate pointer-to-member operations or assignments.
  if (E->isPtrMemOp() || E->isAssignmentOp())
    return false;

  // FIXME: Diagnostics?  I really don't understand how the warnings
  // and errors are supposed to work.
  APFloat RHS(0.0);
  if (!EvaluateFloat(E->getLHS(), Result, Info))
    return false;
  if (!EvaluateFloat(E->getRHS(), RHS, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case BO_Mul:
    Result.multiply(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Add:
    Result.add(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Sub:
    Result.subtract(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Div:
    Result.divide(RHS, APFloat::rmNearestTiesToEven);
    return true;
  }
}

bool FloatExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *E) {
  Result = E->getValue();
  return true;
}

bool FloatExprEvaluator::VisitCastExpr(const CastExpr *E) {
  const Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_IntegralToFloating: {
    APSInt IntResult;
    if (!EvaluateInteger(SubExpr, IntResult, Info))
      return false;
    Result = HandleIntToFloatCast(E->getType(), SubExpr->getType(),
                                  IntResult, Info.Ctx);
    return true;
  }

  case CK_FloatingCast: {
    if (!Visit(SubExpr))
      return false;
    Result = HandleFloatToFloatCast(E->getType(), SubExpr->getType(),
                                    Result, Info.Ctx);
    return true;
  }

  case CK_FloatingComplexToReal: {
    ComplexValue V;
    if (!EvaluateComplex(SubExpr, V, Info))
      return false;
    Result = V.getComplexFloatReal();
    return true;
  }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Complex Evaluation (for float and integer)
//===----------------------------------------------------------------------===//

namespace {
class ComplexExprEvaluator
  : public ExprEvaluatorBase<ComplexExprEvaluator, bool> {
  ComplexValue &Result;

public:
  ComplexExprEvaluator(EvalInfo &info, ComplexValue &Result)
    : ExprEvaluatorBaseTy(info), Result(Result) {}

  bool Success(const CCValue &V, const Expr *e) {
    Result.setFrom(V);
    return true;
  }
  bool Error(const Expr *E) {
    return false;
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitImaginaryLiteral(const ImaginaryLiteral *E);

  bool VisitCastExpr(const CastExpr *E);

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  // FIXME Missing: ImplicitValueInitExpr, InitListExpr
};
} // end anonymous namespace

static bool EvaluateComplex(const Expr *E, ComplexValue &Result,
                            EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isAnyComplexType());
  return ComplexExprEvaluator(Info, Result).Visit(E);
}

bool ComplexExprEvaluator::VisitImaginaryLiteral(const ImaginaryLiteral *E) {
  const Expr* SubExpr = E->getSubExpr();

  if (SubExpr->getType()->isRealFloatingType()) {
    Result.makeComplexFloat();
    APFloat &Imag = Result.FloatImag;
    if (!EvaluateFloat(SubExpr, Imag, Info))
      return false;

    Result.FloatReal = APFloat(Imag.getSemantics());
    return true;
  } else {
    assert(SubExpr->getType()->isIntegerType() &&
           "Unexpected imaginary literal.");

    Result.makeComplexInt();
    APSInt &Imag = Result.IntImag;
    if (!EvaluateInteger(SubExpr, Imag, Info))
      return false;

    Result.IntReal = APSInt(Imag.getBitWidth(), !Imag.isSigned());
    return true;
  }
}

bool ComplexExprEvaluator::VisitCastExpr(const CastExpr *E) {

  switch (E->getCastKind()) {
  case CK_BitCast:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_LValueToRValue:
  case CK_NoOp:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_UserDefinedConversion:
    return false;

  case CK_FloatingRealToComplex: {
    APFloat &Real = Result.FloatReal;
    if (!EvaluateFloat(E->getSubExpr(), Real, Info))
      return false;

    Result.makeComplexFloat();
    Result.FloatImag = APFloat(Real.getSemantics());
    return true;
  }

  case CK_FloatingComplexCast: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();

    Result.FloatReal
      = HandleFloatToFloatCast(To, From, Result.FloatReal, Info.Ctx);
    Result.FloatImag
      = HandleFloatToFloatCast(To, From, Result.FloatImag, Info.Ctx);
    return true;
  }

  case CK_FloatingComplexToIntegralComplex: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();
    Result.makeComplexInt();
    Result.IntReal = HandleFloatToIntCast(To, From, Result.FloatReal, Info.Ctx);
    Result.IntImag = HandleFloatToIntCast(To, From, Result.FloatImag, Info.Ctx);
    return true;
  }

  case CK_IntegralRealToComplex: {
    APSInt &Real = Result.IntReal;
    if (!EvaluateInteger(E->getSubExpr(), Real, Info))
      return false;

    Result.makeComplexInt();
    Result.IntImag = APSInt(Real.getBitWidth(), !Real.isSigned());
    return true;
  }

  case CK_IntegralComplexCast: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();

    Result.IntReal = HandleIntToIntCast(To, From, Result.IntReal, Info.Ctx);
    Result.IntImag = HandleIntToIntCast(To, From, Result.IntImag, Info.Ctx);
    return true;
  }

  case CK_IntegralComplexToFloatingComplex: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();
    Result.makeComplexFloat();
    Result.FloatReal = HandleIntToFloatCast(To, From, Result.IntReal, Info.Ctx);
    Result.FloatImag = HandleIntToFloatCast(To, From, Result.IntImag, Info.Ctx);
    return true;
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
  return false;
}

bool ComplexExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->isPtrMemOp() || E->isAssignmentOp())
    return ExprEvaluatorBaseTy::VisitBinaryOperator(E);

  if (E->getOpcode() == BO_Comma) {
    VisitIgnoredValue(E->getLHS());
    return Visit(E->getRHS());
  }

  if (!Visit(E->getLHS()))
    return false;

  ComplexValue RHS;
  if (!EvaluateComplex(E->getRHS(), RHS, Info))
    return false;

  assert(Result.isComplexFloat() == RHS.isComplexFloat() &&
         "Invalid operands to binary operator.");
  switch (E->getOpcode()) {
  default: return false;
  case BO_Add:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().add(RHS.getComplexFloatReal(),
                                       APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(RHS.getComplexFloatImag(),
                                       APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() += RHS.getComplexIntReal();
      Result.getComplexIntImag() += RHS.getComplexIntImag();
    }
    break;
  case BO_Sub:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().subtract(RHS.getComplexFloatReal(),
                                            APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().subtract(RHS.getComplexFloatImag(),
                                            APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() -= RHS.getComplexIntReal();
      Result.getComplexIntImag() -= RHS.getComplexIntImag();
    }
    break;
  case BO_Mul:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();

      APFloat Tmp = LHS_r;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal().subtract(Tmp, APFloat::rmNearestTiesToEven);

      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(Tmp, APFloat::rmNearestTiesToEven);
    } else {
      ComplexValue LHS = Result;
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() -
         LHS.getComplexIntImag() * RHS.getComplexIntImag());
      Result.getComplexIntImag() =
        (LHS.getComplexIntReal() * RHS.getComplexIntImag() +
         LHS.getComplexIntImag() * RHS.getComplexIntReal());
    }
    break;
  case BO_Div:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();
      APFloat &Res_r = Result.getComplexFloatReal();
      APFloat &Res_i = Result.getComplexFloatImag();

      APFloat Den = RHS_r;
      Den.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      APFloat Tmp = RHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Den.add(Tmp, APFloat::rmNearestTiesToEven);

      Res_r = LHS_r;
      Res_r.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Res_r.add(Tmp, APFloat::rmNearestTiesToEven);
      Res_r.divide(Den, APFloat::rmNearestTiesToEven);

      Res_i = LHS_i;
      Res_i.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Res_i.subtract(Tmp, APFloat::rmNearestTiesToEven);
      Res_i.divide(Den, APFloat::rmNearestTiesToEven);
    } else {
      if (RHS.getComplexIntReal() == 0 && RHS.getComplexIntImag() == 0) {
        // FIXME: what about diagnostics?
        return false;
      }
      ComplexValue LHS = Result;
      APSInt Den = RHS.getComplexIntReal() * RHS.getComplexIntReal() +
        RHS.getComplexIntImag() * RHS.getComplexIntImag();
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() +
         LHS.getComplexIntImag() * RHS.getComplexIntImag()) / Den;
      Result.getComplexIntImag() =
        (LHS.getComplexIntImag() * RHS.getComplexIntReal() -
         LHS.getComplexIntReal() * RHS.getComplexIntImag()) / Den;
    }
    break;
  }

  return true;
}

bool ComplexExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // FIXME: what about diagnostics?
    return false;
  case UO_Extension:
    return true;
  case UO_Plus:
    // The result is always just the subexpr.
    return true;
  case UO_Minus:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().changeSign();
      Result.getComplexFloatImag().changeSign();
    }
    else {
      Result.getComplexIntReal() = -Result.getComplexIntReal();
      Result.getComplexIntImag() = -Result.getComplexIntImag();
    }
    return true;
  case UO_Not:
    if (Result.isComplexFloat())
      Result.getComplexFloatImag().changeSign();
    else
      Result.getComplexIntImag() = -Result.getComplexIntImag();
    return true;
  }
}

//===----------------------------------------------------------------------===//
// Top level Expr::EvaluateAsRValue method.
//===----------------------------------------------------------------------===//

static bool Evaluate(CCValue &Result, EvalInfo &Info, const Expr *E) {
  // In C, function designators are not lvalues, but we evaluate them as if they
  // are.
  if (E->isGLValue() || E->getType()->isFunctionType()) {
    LValue LV;
    if (!EvaluateLValue(E, LV, Info))
      return false;
    LV.moveInto(Result);
  } else if (E->getType()->isVectorType()) {
    if (!EvaluateVector(E, Result, Info))
      return false;
  } else if (E->getType()->isIntegralOrEnumerationType()) {
    if (!IntExprEvaluator(Info, Result).Visit(E))
      return false;
  } else if (E->getType()->hasPointerRepresentation()) {
    LValue LV;
    if (!EvaluatePointer(E, LV, Info))
      return false;
    LV.moveInto(Result);
  } else if (E->getType()->isRealFloatingType()) {
    llvm::APFloat F(0.0);
    if (!EvaluateFloat(E, F, Info))
      return false;
    Result = CCValue(F);
  } else if (E->getType()->isAnyComplexType()) {
    ComplexValue C;
    if (!EvaluateComplex(E, C, Info))
      return false;
    C.moveInto(Result);
  } else if (E->getType()->isMemberPointerType()) {
    // FIXME: Implement evaluation of pointer-to-member types.
    return false;
  } else if (E->getType()->isArrayType() && E->getType()->isLiteralType()) {
    LValue LV;
    LV.set(E, Info.CurrentCall);
    if (!EvaluateArray(E, LV, Info.CurrentCall->Temporaries[E], Info))
      return false;
    Result = Info.CurrentCall->Temporaries[E];
  } else if (E->getType()->isRecordType() && E->getType()->isLiteralType()) {
    LValue LV;
    LV.set(E, Info.CurrentCall);
    if (!EvaluateRecord(E, LV, Info.CurrentCall->Temporaries[E], Info))
      return false;
    Result = Info.CurrentCall->Temporaries[E];
  } else
    return false;

  return true;
}

/// EvaluateConstantExpression - Evaluate an expression as a constant expression
/// in-place in an APValue. In some cases, the in-place evaluation is essential,
/// since later initializers for an object can indirectly refer to subobjects
/// which were initialized earlier.
static bool EvaluateConstantExpression(APValue &Result, EvalInfo &Info,
                                       const LValue &This, const Expr *E) {
  if (E->isRValue() && E->getType()->isLiteralType()) {
    // Evaluate arrays and record types in-place, so that later initializers can
    // refer to earlier-initialized members of the object.
    if (E->getType()->isArrayType())
      return EvaluateArray(E, This, Result, Info);
    else if (E->getType()->isRecordType())
      return EvaluateRecord(E, This, Result, Info);
  }

  // For any other type, in-place evaluation is unimportant.
  CCValue CoreConstResult;
  return Evaluate(CoreConstResult, Info, E) &&
         CheckConstantExpression(CoreConstResult, Result);
}


/// EvaluateAsRValue - Return true if this is a constant which we can fold using
/// any crazy technique (that has nothing to do with language standards) that
/// we want to.  If this function returns true, it returns the folded constant
/// in Result. If this expression is a glvalue, an lvalue-to-rvalue conversion
/// will be applied to the result.
bool Expr::EvaluateAsRValue(EvalResult &Result, const ASTContext &Ctx) const {
  // FIXME: Evaluating initializers for large arrays can cause performance
  // problems, and we don't use such values yet. Once we have a more efficient
  // array representation, this should be reinstated, and used by CodeGen.
  if (isRValue() && getType()->isArrayType())
    return false;

  EvalInfo Info(Ctx, Result);

  // FIXME: If this is the initializer for an lvalue, pass that in.
  CCValue Value;
  if (!::Evaluate(Value, Info, this))
    return false;

  if (isGLValue()) {
    LValue LV;
    LV.setFrom(Value);
    if (!HandleLValueToRValueConversion(Info, getType(), LV, Value))
      return false;
  }

  // Check this core constant expression is a constant expression, and if so,
  // convert it to one.
  return CheckConstantExpression(Value, Result.Val);
}

bool Expr::EvaluateAsBooleanCondition(bool &Result,
                                      const ASTContext &Ctx) const {
  EvalResult Scratch;
  return EvaluateAsRValue(Scratch, Ctx) &&
         HandleConversionToBool(CCValue(Scratch.Val, CCValue::GlobalValue()),
                                Result);
}

bool Expr::EvaluateAsInt(APSInt &Result, const ASTContext &Ctx) const {
  EvalResult ExprResult;
  if (!EvaluateAsRValue(ExprResult, Ctx) || ExprResult.HasSideEffects ||
      !ExprResult.Val.isInt()) {
    return false;
  }
  Result = ExprResult.Val.getInt();
  return true;
}

bool Expr::EvaluateAsLValue(EvalResult &Result, const ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  LValue LV;
  return EvaluateLValue(this, LV, Info) && !Result.HasSideEffects &&
         CheckLValueConstantExpression(LV, Result.Val);
}

/// isEvaluatable - Call EvaluateAsRValue to see if this expression can be
/// constant folded, but discard the result.
bool Expr::isEvaluatable(const ASTContext &Ctx) const {
  EvalResult Result;
  return EvaluateAsRValue(Result, Ctx) && !Result.HasSideEffects;
}

bool Expr::HasSideEffects(const ASTContext &Ctx) const {
  return HasSideEffect(Ctx).Visit(this);
}

APSInt Expr::EvaluateKnownConstInt(const ASTContext &Ctx) const {
  EvalResult EvalResult;
  bool Result = EvaluateAsRValue(EvalResult, Ctx);
  (void)Result;
  assert(Result && "Could not evaluate expression");
  assert(EvalResult.Val.isInt() && "Expression did not evaluate to integer");

  return EvalResult.Val.getInt();
}

 bool Expr::EvalResult::isGlobalLValue() const {
   assert(Val.isLValue());
   return IsGlobalLValue(Val.getLValueBase());
 }


/// isIntegerConstantExpr - this recursive routine will test if an expression is
/// an integer constant expression.

/// FIXME: Pass up a reason why! Invalid operation in i-c-e, division by zero,
/// comma, etc
///
/// FIXME: Handle offsetof.  Two things to do:  Handle GCC's __builtin_offsetof
/// to support gcc 4.0+  and handle the idiom GCC recognizes with a null pointer
/// cast+dereference.

// CheckICE - This function does the fundamental ICE checking: the returned
// ICEDiag contains a Val of 0, 1, or 2, and a possibly null SourceLocation.
// Note that to reduce code duplication, this helper does no evaluation
// itself; the caller checks whether the expression is evaluatable, and
// in the rare cases where CheckICE actually cares about the evaluated
// value, it calls into Evalute.
//
// Meanings of Val:
// 0: This expression is an ICE.
// 1: This expression is not an ICE, but if it isn't evaluated, it's
//    a legal subexpression for an ICE. This return value is used to handle
//    the comma operator in C99 mode.
// 2: This expression is not an ICE, and is not a legal subexpression for one.

namespace {

struct ICEDiag {
  unsigned Val;
  SourceLocation Loc;

  public:
  ICEDiag(unsigned v, SourceLocation l) : Val(v), Loc(l) {}
  ICEDiag() : Val(0) {}
};

}

static ICEDiag NoDiag() { return ICEDiag(); }

static ICEDiag CheckEvalInICE(const Expr* E, ASTContext &Ctx) {
  Expr::EvalResult EVResult;
  if (!E->EvaluateAsRValue(EVResult, Ctx) || EVResult.HasSideEffects ||
      !EVResult.Val.isInt()) {
    return ICEDiag(2, E->getLocStart());
  }
  return NoDiag();
}

static ICEDiag CheckICE(const Expr* E, ASTContext &Ctx) {
  assert(!E->isValueDependent() && "Should not see value dependent exprs!");
  if (!E->getType()->isIntegralOrEnumerationType()) {
    return ICEDiag(2, E->getLocStart());
  }

  switch (E->getStmtClass()) {
#define ABSTRACT_STMT(Node)
#define STMT(Node, Base) case Expr::Node##Class:
#define EXPR(Node, Base)
#include "clang/AST/StmtNodes.inc"
  case Expr::PredefinedExprClass:
  case Expr::FloatingLiteralClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::StringLiteralClass:
  case Expr::ArraySubscriptExprClass:
  case Expr::MemberExprClass:
  case Expr::CompoundAssignOperatorClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::ParenListExprClass:
  case Expr::VAArgExprClass:
  case Expr::AddrLabelExprClass:
  case Expr::StmtExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CUDAKernelCallExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXTypeidExprClass:
  case Expr::CXXUuidofExprClass:
  case Expr::CXXNullPtrLiteralExprClass:
  case Expr::CXXThisExprClass:
  case Expr::CXXThrowExprClass:
  case Expr::CXXNewExprClass:
  case Expr::CXXDeleteExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::UnresolvedLookupExprClass:
  case Expr::DependentScopeDeclRefExprClass:
  case Expr::CXXConstructExprClass:
  case Expr::CXXBindTemporaryExprClass:
  case Expr::ExprWithCleanupsClass:
  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXUnresolvedConstructExprClass:
  case Expr::CXXDependentScopeMemberExprClass:
  case Expr::UnresolvedMemberExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::BlockExprClass:
  case Expr::BlockDeclRefExprClass:
  case Expr::NoStmtClass:
  case Expr::OpaqueValueExprClass:
  case Expr::PackExpansionExprClass:
  case Expr::SubstNonTypeTemplateParmPackExprClass:
  case Expr::AsTypeExprClass:
  case Expr::ObjCIndirectCopyRestoreExprClass:
  case Expr::MaterializeTemporaryExprClass:
  case Expr::PseudoObjectExprClass:
  case Expr::AtomicExprClass:
    return ICEDiag(2, E->getLocStart());

  case Expr::InitListExprClass:
    if (Ctx.getLangOptions().CPlusPlus0x) {
      const InitListExpr *ILE = cast<InitListExpr>(E);
      if (ILE->getNumInits() == 0)
        return NoDiag();
      if (ILE->getNumInits() == 1)
        return CheckICE(ILE->getInit(0), Ctx);
      // Fall through for more than 1 expression.
    }
    return ICEDiag(2, E->getLocStart());

  case Expr::SizeOfPackExprClass:
  case Expr::GNUNullExprClass:
    // GCC considers the GNU __null value to be an integral constant expression.
    return NoDiag();

  case Expr::SubstNonTypeTemplateParmExprClass:
    return
      CheckICE(cast<SubstNonTypeTemplateParmExpr>(E)->getReplacement(), Ctx);

  case Expr::ParenExprClass:
    return CheckICE(cast<ParenExpr>(E)->getSubExpr(), Ctx);
  case Expr::GenericSelectionExprClass:
    return CheckICE(cast<GenericSelectionExpr>(E)->getResultExpr(), Ctx);
  case Expr::IntegerLiteralClass:
  case Expr::CharacterLiteralClass:
  case Expr::CXXBoolLiteralExprClass:
  case Expr::CXXScalarValueInitExprClass:
  case Expr::UnaryTypeTraitExprClass:
  case Expr::BinaryTypeTraitExprClass:
  case Expr::ArrayTypeTraitExprClass:
  case Expr::ExpressionTraitExprClass:
  case Expr::CXXNoexceptExprClass:
    return NoDiag();
  case Expr::CallExprClass:
  case Expr::CXXOperatorCallExprClass: {
    // C99 6.6/3 allows function calls within unevaluated subexpressions of
    // constant expressions, but they can never be ICEs because an ICE cannot
    // contain an operand of (pointer to) function type.
    const CallExpr *CE = cast<CallExpr>(E);
    if (CE->isBuiltinCall())
      return CheckEvalInICE(E, Ctx);
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::DeclRefExprClass:
    if (isa<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return NoDiag();
    if (Ctx.getLangOptions().CPlusPlus && IsConstNonVolatile(E->getType())) {
      const NamedDecl *D = cast<DeclRefExpr>(E)->getDecl();

      // Parameter variables are never constants.  Without this check,
      // getAnyInitializer() can find a default argument, which leads
      // to chaos.
      if (isa<ParmVarDecl>(D))
        return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());

      // C++ 7.1.5.1p2
      //   A variable of non-volatile const-qualified integral or enumeration
      //   type initialized by an ICE can be used in ICEs.
      if (const VarDecl *Dcl = dyn_cast<VarDecl>(D)) {
        if (!Dcl->getType()->isIntegralOrEnumerationType())
          return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());

        // Look for a declaration of this variable that has an initializer.
        const VarDecl *ID = 0;
        const Expr *Init = Dcl->getAnyInitializer(ID);
        if (Init) {
          if (ID->isInitKnownICE()) {
            // We have already checked whether this subexpression is an
            // integral constant expression.
            if (ID->isInitICE())
              return NoDiag();
            else
              return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          // It's an ICE whether or not the definition we found is
          // out-of-line.  See DR 721 and the discussion in Clang PR
          // 6206 for details.

          if (Dcl->isCheckingICE()) {
            return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          Dcl->setCheckingICE();
          ICEDiag Result = CheckICE(Init, Ctx);
          // Cache the result of the ICE test.
          Dcl->setInitKnownICE(Result.Val == 0);
          return Result;
        }
      }
    }
    return ICEDiag(2, E->getLocStart());
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);
    switch (Exp->getOpcode()) {
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
    case UO_AddrOf:
    case UO_Deref:
      // C99 6.6/3 allows increment and decrement within unevaluated
      // subexpressions of constant expressions, but they can never be ICEs
      // because an ICE cannot contain an lvalue operand.
      return ICEDiag(2, E->getLocStart());
    case UO_Extension:
    case UO_LNot:
    case UO_Plus:
    case UO_Minus:
    case UO_Not:
    case UO_Real:
    case UO_Imag:
      return CheckICE(Exp->getSubExpr(), Ctx);
    }
    
    // OffsetOf falls through here.
  }
  case Expr::OffsetOfExprClass: {
      // Note that per C99, offsetof must be an ICE. And AFAIK, using
      // EvaluateAsRValue matches the proposed gcc behavior for cases like
      // "offsetof(struct s{int x[4];}, x[1.0])".  This doesn't affect
      // compliance: we should warn earlier for offsetof expressions with
      // array subscripts that aren't ICEs, and if the array subscripts
      // are ICEs, the value of the offsetof must be an integer constant.
      return CheckEvalInICE(E, Ctx);
  }
  case Expr::UnaryExprOrTypeTraitExprClass: {
    const UnaryExprOrTypeTraitExpr *Exp = cast<UnaryExprOrTypeTraitExpr>(E);
    if ((Exp->getKind() ==  UETT_SizeOf) &&
        Exp->getTypeOfArgument()->isVariableArrayType())
      return ICEDiag(2, E->getLocStart());
    return NoDiag();
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(E);
    switch (Exp->getOpcode()) {
    case BO_PtrMemD:
    case BO_PtrMemI:
    case BO_Assign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      // C99 6.6/3 allows assignments within unevaluated subexpressions of
      // constant expressions, but they can never be ICEs because an ICE cannot
      // contain an lvalue operand.
      return ICEDiag(2, E->getLocStart());

    case BO_Mul:
    case BO_Div:
    case BO_Rem:
    case BO_Add:
    case BO_Sub:
    case BO_Shl:
    case BO_Shr:
    case BO_LT:
    case BO_GT:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
    case BO_And:
    case BO_Xor:
    case BO_Or:
    case BO_Comma: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (Exp->getOpcode() == BO_Div ||
          Exp->getOpcode() == BO_Rem) {
        // EvaluateAsRValue gives an error for undefined Div/Rem, so make sure
        // we don't evaluate one.
        if (LHSResult.Val == 0 && RHSResult.Val == 0) {
          llvm::APSInt REval = Exp->getRHS()->EvaluateKnownConstInt(Ctx);
          if (REval == 0)
            return ICEDiag(1, E->getLocStart());
          if (REval.isSigned() && REval.isAllOnesValue()) {
            llvm::APSInt LEval = Exp->getLHS()->EvaluateKnownConstInt(Ctx);
            if (LEval.isMinSignedValue())
              return ICEDiag(1, E->getLocStart());
          }
        }
      }
      if (Exp->getOpcode() == BO_Comma) {
        if (Ctx.getLangOptions().C99) {
          // C99 6.6p3 introduces a strange edge case: comma can be in an ICE
          // if it isn't evaluated.
          if (LHSResult.Val == 0 && RHSResult.Val == 0)
            return ICEDiag(1, E->getLocStart());
        } else {
          // In both C89 and C++, commas in ICEs are illegal.
          return ICEDiag(2, E->getLocStart());
        }
      }
      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    case BO_LAnd:
    case BO_LOr: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);

      // C++0x [expr.const]p2:
      //   [...] subexpressions of logical AND (5.14), logical OR
      //   (5.15), and condi- tional (5.16) operations that are not
      //   evaluated are not considered.
      if (Ctx.getLangOptions().CPlusPlus0x && LHSResult.Val == 0) {
        if (Exp->getOpcode() == BO_LAnd && 
            Exp->getLHS()->EvaluateKnownConstInt(Ctx) == 0)
          return LHSResult;

        if (Exp->getOpcode() == BO_LOr &&
            Exp->getLHS()->EvaluateKnownConstInt(Ctx) != 0)
          return LHSResult;
      }

      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (LHSResult.Val == 0 && RHSResult.Val == 1) {
        // Rare case where the RHS has a comma "side-effect"; we need
        // to actually check the condition to see whether the side
        // with the comma is evaluated.
        if ((Exp->getOpcode() == BO_LAnd) !=
            (Exp->getLHS()->EvaluateKnownConstInt(Ctx) == 0))
          return RHSResult;
        return NoDiag();
      }

      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    }
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::ObjCBridgedCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(E)->getSubExpr();
    if (isa<ExplicitCastExpr>(E) &&
        isa<FloatingLiteral>(SubExpr->IgnoreParenImpCasts()))
      return NoDiag();
    switch (cast<CastExpr>(E)->getCastKind()) {
    case CK_LValueToRValue:
    case CK_NoOp:
    case CK_IntegralToBoolean:
    case CK_IntegralCast:
      return CheckICE(SubExpr, Ctx);
    default:
      return ICEDiag(2, E->getLocStart());
    }
  }
  case Expr::BinaryConditionalOperatorClass: {
    const BinaryConditionalOperator *Exp = cast<BinaryConditionalOperator>(E);
    ICEDiag CommonResult = CheckICE(Exp->getCommon(), Ctx);
    if (CommonResult.Val == 2) return CommonResult;
    ICEDiag FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);
    if (FalseResult.Val == 2) return FalseResult;
    if (CommonResult.Val == 1) return CommonResult;
    if (FalseResult.Val == 1 &&
        Exp->getCommon()->EvaluateKnownConstInt(Ctx) == 0) return NoDiag();
    return FalseResult;
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(E);
    // If the condition (ignoring parens) is a __builtin_constant_p call,
    // then only the true side is actually considered in an integer constant
    // expression, and it is fully evaluated.  This is an important GNU
    // extension.  See GCC PR38377 for discussion.
    if (const CallExpr *CallCE
        = dyn_cast<CallExpr>(Exp->getCond()->IgnoreParenCasts()))
      if (CallCE->isBuiltinCall() == Builtin::BI__builtin_constant_p) {
        Expr::EvalResult EVResult;
        if (!E->EvaluateAsRValue(EVResult, Ctx) || EVResult.HasSideEffects ||
            !EVResult.Val.isInt()) {
          return ICEDiag(2, E->getLocStart());
        }
        return NoDiag();
      }
    ICEDiag CondResult = CheckICE(Exp->getCond(), Ctx);
    if (CondResult.Val == 2)
      return CondResult;

    // C++0x [expr.const]p2:
    //   subexpressions of [...] conditional (5.16) operations that
    //   are not evaluated are not considered
    bool TrueBranch = Ctx.getLangOptions().CPlusPlus0x
      ? Exp->getCond()->EvaluateKnownConstInt(Ctx) != 0
      : false;
    ICEDiag TrueResult = NoDiag();
    if (!Ctx.getLangOptions().CPlusPlus0x || TrueBranch)
      TrueResult = CheckICE(Exp->getTrueExpr(), Ctx);
    ICEDiag FalseResult = NoDiag();
    if (!Ctx.getLangOptions().CPlusPlus0x || !TrueBranch)
      FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);

    if (TrueResult.Val == 2)
      return TrueResult;
    if (FalseResult.Val == 2)
      return FalseResult;
    if (CondResult.Val == 1)
      return CondResult;
    if (TrueResult.Val == 0 && FalseResult.Val == 0)
      return NoDiag();
    // Rare case where the diagnostics depend on which side is evaluated
    // Note that if we get here, CondResult is 0, and at least one of
    // TrueResult and FalseResult is non-zero.
    if (Exp->getCond()->EvaluateKnownConstInt(Ctx) == 0) {
      return FalseResult;
    }
    return TrueResult;
  }
  case Expr::CXXDefaultArgExprClass:
    return CheckICE(cast<CXXDefaultArgExpr>(E)->getExpr(), Ctx);
  case Expr::ChooseExprClass: {
    return CheckICE(cast<ChooseExpr>(E)->getChosenSubExpr(Ctx), Ctx);
  }
  }

  // Silence a GCC warning
  return ICEDiag(2, E->getLocStart());
}

bool Expr::isIntegerConstantExpr(llvm::APSInt &Result, ASTContext &Ctx,
                                 SourceLocation *Loc, bool isEvaluated) const {
  ICEDiag d = CheckICE(this, Ctx);
  if (d.Val != 0) {
    if (Loc) *Loc = d.Loc;
    return false;
  }
  if (!EvaluateAsInt(Result, Ctx))
    llvm_unreachable("ICE cannot be evaluated!");
  return true;
}
