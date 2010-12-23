//== BasicObjCFoundationChecks.cpp - Simple Apple-Foundation checks -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicObjCFoundationChecks, a class that encapsulates
//  a set of simple checks to run on Objective-C code using Apple's Foundation
//  classes.
//
//===----------------------------------------------------------------------===//

#include "BasicObjCFoundationChecks.h"

#include "clang/StaticAnalyzer/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.h"
#include "clang/StaticAnalyzer/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/PathSensitive/GRState.h"
#include "clang/StaticAnalyzer/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.h"
#include "clang/StaticAnalyzer/Checkers/LocalCheckers.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ASTContext.h"

using namespace clang;
using namespace ento;

namespace {
class APIMisuse : public BugType {
public:
  APIMisuse(const char* name) : BugType(name, "API Misuse (Apple)") {}
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static const ObjCInterfaceType* GetReceiverType(const ObjCMessageExpr* ME) {
  QualType T;
  switch (ME->getReceiverKind()) {
    case ObjCMessageExpr::Instance:
      T = ME->getInstanceReceiver()->getType();
      break;
      
    case ObjCMessageExpr::SuperInstance:
      T = ME->getSuperType();
      break;
      
    case ObjCMessageExpr::Class:
    case ObjCMessageExpr::SuperClass:
      return 0;
  }
  
  if (const ObjCObjectPointerType *PT = T->getAs<ObjCObjectPointerType>())
    return PT->getInterfaceType();
  
  return NULL;
}

static const char* GetReceiverNameType(const ObjCMessageExpr* ME) {
  if (const ObjCInterfaceType *ReceiverType = GetReceiverType(ME))
    return ReceiverType->getDecl()->getIdentifier()->getNameStart();
  return NULL;
}

static bool isNSString(llvm::StringRef ClassName) {
  return ClassName == "NSString" || ClassName == "NSMutableString";
}

static inline bool isNil(SVal X) {
  return isa<loc::ConcreteInt>(X);
}

//===----------------------------------------------------------------------===//
// NilArgChecker - Check for prohibited nil arguments to ObjC method calls.
//===----------------------------------------------------------------------===//

namespace {
  class NilArgChecker : public CheckerVisitor<NilArgChecker> {
    APIMisuse *BT;
    void WarnNilArg(CheckerContext &C, const ObjCMessageExpr* ME, unsigned Arg);
  public:
    NilArgChecker() : BT(0) {}
    static void *getTag() { static int x = 0; return &x; }
    void PreVisitObjCMessageExpr(CheckerContext &C, const ObjCMessageExpr *ME);
  };
}

void NilArgChecker::WarnNilArg(CheckerContext &C,
                               const clang::ObjCMessageExpr *ME,
                               unsigned int Arg)
{
  if (!BT)
    BT = new APIMisuse("nil argument");
  
  if (ExplodedNode *N = C.generateSink()) {
    llvm::SmallString<128> sbuf;
    llvm::raw_svector_ostream os(sbuf);
    os << "Argument to '" << GetReceiverNameType(ME) << "' method '"
       << ME->getSelector().getAsString() << "' cannot be nil";

    RangedBugReport *R = new RangedBugReport(*BT, os.str(), N);
    R->addRange(ME->getArg(Arg)->getSourceRange());
    C.EmitReport(R);
  }
}

void NilArgChecker::PreVisitObjCMessageExpr(CheckerContext &C,
                                            const ObjCMessageExpr *ME)
{
  const ObjCInterfaceType *ReceiverType = GetReceiverType(ME);
  if (!ReceiverType)
    return;
  
  if (isNSString(ReceiverType->getDecl()->getIdentifier()->getName())) {
    Selector S = ME->getSelector();
    
    if (S.isUnarySelector())
      return;
    
    // FIXME: This is going to be really slow doing these checks with
    //  lexical comparisons.
    
    std::string NameStr = S.getAsString();
    llvm::StringRef Name(NameStr);
    assert(!Name.empty());
    
    // FIXME: Checking for initWithFormat: will not work in most cases
    //  yet because [NSString alloc] returns id, not NSString*.  We will
    //  need support for tracking expected-type information in the analyzer
    //  to find these errors.
    if (Name == "caseInsensitiveCompare:" ||
        Name == "compare:" ||
        Name == "compare:options:" ||
        Name == "compare:options:range:" ||
        Name == "compare:options:range:locale:" ||
        Name == "componentsSeparatedByCharactersInSet:" ||
        Name == "initWithFormat:") {
      if (isNil(C.getState()->getSVal(ME->getArg(0))))
        WarnNilArg(C, ME, 0);
    }
  }
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

namespace {
class CFNumberCreateChecker : public CheckerVisitor<CFNumberCreateChecker> {
  APIMisuse* BT;
  IdentifierInfo* II;
public:
  CFNumberCreateChecker() : BT(0), II(0) {}
  ~CFNumberCreateChecker() {}
  static void *getTag() { static int x = 0; return &x; }
  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
private:
  void EmitError(const TypedRegion* R, const Expr* Ex,
                uint64_t SourceSize, uint64_t TargetSize, uint64_t NumberKind);
};
} // end anonymous namespace

enum CFNumberType {
  kCFNumberSInt8Type = 1,
  kCFNumberSInt16Type = 2,
  kCFNumberSInt32Type = 3,
  kCFNumberSInt64Type = 4,
  kCFNumberFloat32Type = 5,
  kCFNumberFloat64Type = 6,
  kCFNumberCharType = 7,
  kCFNumberShortType = 8,
  kCFNumberIntType = 9,
  kCFNumberLongType = 10,
  kCFNumberLongLongType = 11,
  kCFNumberFloatType = 12,
  kCFNumberDoubleType = 13,
  kCFNumberCFIndexType = 14,
  kCFNumberNSIntegerType = 15,
  kCFNumberCGFloatType = 16
};

namespace {
  template<typename T>
  class Optional {
    bool IsKnown;
    T Val;
  public:
    Optional() : IsKnown(false), Val(0) {}
    Optional(const T& val) : IsKnown(true), Val(val) {}

    bool isKnown() const { return IsKnown; }

    const T& getValue() const {
      assert (isKnown());
      return Val;
    }

    operator const T&() const {
      return getValue();
    }
  };
}

static Optional<uint64_t> GetCFNumberSize(ASTContext& Ctx, uint64_t i) {
  static const unsigned char FixedSize[] = { 8, 16, 32, 64, 32, 64 };

  if (i < kCFNumberCharType)
    return FixedSize[i-1];

  QualType T;

  switch (i) {
    case kCFNumberCharType:     T = Ctx.CharTy;     break;
    case kCFNumberShortType:    T = Ctx.ShortTy;    break;
    case kCFNumberIntType:      T = Ctx.IntTy;      break;
    case kCFNumberLongType:     T = Ctx.LongTy;     break;
    case kCFNumberLongLongType: T = Ctx.LongLongTy; break;
    case kCFNumberFloatType:    T = Ctx.FloatTy;    break;
    case kCFNumberDoubleType:   T = Ctx.DoubleTy;   break;
    case kCFNumberCFIndexType:
    case kCFNumberNSIntegerType:
    case kCFNumberCGFloatType:
      // FIXME: We need a way to map from names to Type*.
    default:
      return Optional<uint64_t>();
  }

  return Ctx.getTypeSize(T);
}

#if 0
static const char* GetCFNumberTypeStr(uint64_t i) {
  static const char* Names[] = {
    "kCFNumberSInt8Type",
    "kCFNumberSInt16Type",
    "kCFNumberSInt32Type",
    "kCFNumberSInt64Type",
    "kCFNumberFloat32Type",
    "kCFNumberFloat64Type",
    "kCFNumberCharType",
    "kCFNumberShortType",
    "kCFNumberIntType",
    "kCFNumberLongType",
    "kCFNumberLongLongType",
    "kCFNumberFloatType",
    "kCFNumberDoubleType",
    "kCFNumberCFIndexType",
    "kCFNumberNSIntegerType",
    "kCFNumberCGFloatType"
  };

  return i <= kCFNumberCGFloatType ? Names[i-1] : "Invalid CFNumberType";
}
#endif

void CFNumberCreateChecker::PreVisitCallExpr(CheckerContext &C,
                                             const CallExpr *CE)
{
  const Expr* Callee = CE->getCallee();
  const GRState *state = C.getState();
  SVal CallV = state->getSVal(Callee);
  const FunctionDecl* FD = CallV.getAsFunctionDecl();

  if (!FD)
    return;
  
  ASTContext &Ctx = C.getASTContext();
  if (!II)
    II = &Ctx.Idents.get("CFNumberCreate");

  if (FD->getIdentifier() != II || CE->getNumArgs() != 3)
    return;

  // Get the value of the "theType" argument.
  SVal TheTypeVal = state->getSVal(CE->getArg(1));

  // FIXME: We really should allow ranges of valid theType values, and
  //   bifurcate the state appropriately.
  nonloc::ConcreteInt* V = dyn_cast<nonloc::ConcreteInt>(&TheTypeVal);
  if (!V)
    return;

  uint64_t NumberKind = V->getValue().getLimitedValue();
  Optional<uint64_t> TargetSize = GetCFNumberSize(Ctx, NumberKind);

  // FIXME: In some cases we can emit an error.
  if (!TargetSize.isKnown())
    return;

  // Look at the value of the integer being passed by reference.  Essentially
  // we want to catch cases where the value passed in is not equal to the
  // size of the type being created.
  SVal TheValueExpr = state->getSVal(CE->getArg(2));

  // FIXME: Eventually we should handle arbitrary locations.  We can do this
  //  by having an enhanced memory model that does low-level typing.
  loc::MemRegionVal* LV = dyn_cast<loc::MemRegionVal>(&TheValueExpr);
  if (!LV)
    return;

  const TypedRegion* R = dyn_cast<TypedRegion>(LV->StripCasts());
  if (!R)
    return;

  QualType T = Ctx.getCanonicalType(R->getValueType());

  // FIXME: If the pointee isn't an integer type, should we flag a warning?
  //  People can do weird stuff with pointers.

  if (!T->isIntegerType())
    return;

  uint64_t SourceSize = Ctx.getTypeSize(T);

  // CHECK: is SourceSize == TargetSize
  if (SourceSize == TargetSize)
    return;

  // Generate an error.  Only generate a sink if 'SourceSize < TargetSize';
  // otherwise generate a regular node.
  //
  // FIXME: We can actually create an abstract "CFNumber" object that has
  //  the bits initialized to the provided values.
  //
  if (ExplodedNode *N = SourceSize < TargetSize ? C.generateSink() 
                                                : C.generateNode()) {
    llvm::SmallString<128> sbuf;
    llvm::raw_svector_ostream os(sbuf);
    
    os << (SourceSize == 8 ? "An " : "A ")
       << SourceSize << " bit integer is used to initialize a CFNumber "
                        "object that represents "
       << (TargetSize == 8 ? "an " : "a ")
       << TargetSize << " bit integer. ";
    
    if (SourceSize < TargetSize)
      os << (TargetSize - SourceSize)
      << " bits of the CFNumber value will be garbage." ;
    else
      os << (SourceSize - TargetSize)
      << " bits of the input integer will be lost.";

    if (!BT)
      BT = new APIMisuse("Bad use of CFNumberCreate");
    
    RangedBugReport *report = new RangedBugReport(*BT, os.str(), N);
    report->addRange(CE->getArg(2)->getSourceRange());
    C.EmitReport(report);
  }
}

//===----------------------------------------------------------------------===//
// CFRetain/CFRelease checking for null arguments.
//===----------------------------------------------------------------------===//

namespace {
class CFRetainReleaseChecker : public CheckerVisitor<CFRetainReleaseChecker> {
  APIMisuse *BT;
  IdentifierInfo *Retain, *Release;
public:
  CFRetainReleaseChecker(): BT(0), Retain(0), Release(0) {}
  static void *getTag() { static int x = 0; return &x; }
  void PreVisitCallExpr(CheckerContext& C, const CallExpr* CE);
};
} // end anonymous namespace


void CFRetainReleaseChecker::PreVisitCallExpr(CheckerContext& C,
                                              const CallExpr* CE) {
  // If the CallExpr doesn't have exactly 1 argument just give up checking.
  if (CE->getNumArgs() != 1)
    return;

  // Get the function declaration of the callee.
  const GRState* state = C.getState();
  SVal X = state->getSVal(CE->getCallee());
  const FunctionDecl* FD = X.getAsFunctionDecl();

  if (!FD)
    return;
  
  if (!BT) {
    ASTContext &Ctx = C.getASTContext();
    Retain = &Ctx.Idents.get("CFRetain");
    Release = &Ctx.Idents.get("CFRelease");
    BT = new APIMisuse("null passed to CFRetain/CFRelease");
  }

  // Check if we called CFRetain/CFRelease.
  const IdentifierInfo *FuncII = FD->getIdentifier();
  if (!(FuncII == Retain || FuncII == Release))
    return;

  // FIXME: The rest of this just checks that the argument is non-null.
  // It should probably be refactored and combined with AttrNonNullChecker.

  // Get the argument's value.
  const Expr *Arg = CE->getArg(0);
  SVal ArgVal = state->getSVal(Arg);
  DefinedSVal *DefArgVal = dyn_cast<DefinedSVal>(&ArgVal);
  if (!DefArgVal)
    return;

  // Get a NULL value.
  SValBuilder &svalBuilder = C.getSValBuilder();
  DefinedSVal zero = cast<DefinedSVal>(svalBuilder.makeZeroVal(Arg->getType()));

  // Make an expression asserting that they're equal.
  DefinedOrUnknownSVal ArgIsNull = svalBuilder.evalEQ(state, zero, *DefArgVal);

  // Are they equal?
  const GRState *stateTrue, *stateFalse;
  llvm::tie(stateTrue, stateFalse) = state->assume(ArgIsNull);

  if (stateTrue && !stateFalse) {
    ExplodedNode *N = C.generateSink(stateTrue);
    if (!N)
      return;

    const char *description = (FuncII == Retain)
                            ? "Null pointer argument in call to CFRetain"
                            : "Null pointer argument in call to CFRelease";

    EnhancedBugReport *report = new EnhancedBugReport(*BT, description, N);
    report->addRange(Arg->getSourceRange());
    report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, Arg);
    C.EmitReport(report);
    return;
  }

  // From here on, we know the argument is non-null.
  C.addTransition(stateFalse);
}

//===----------------------------------------------------------------------===//
// Check for sending 'retain', 'release', or 'autorelease' directly to a Class.
//===----------------------------------------------------------------------===//

namespace {
class ClassReleaseChecker : public CheckerVisitor<ClassReleaseChecker> {
  Selector releaseS;
  Selector retainS;
  Selector autoreleaseS;
  Selector drainS;
  BugType *BT;
public:
  ClassReleaseChecker()
    : BT(0) {}

  static void *getTag() { static int x = 0; return &x; }
      
  void PreVisitObjCMessageExpr(CheckerContext &C, const ObjCMessageExpr *ME);    
};
}

void ClassReleaseChecker::PreVisitObjCMessageExpr(CheckerContext &C,
                                                  const ObjCMessageExpr *ME) {
  
  if (!BT) {
    BT = new APIMisuse("message incorrectly sent to class instead of class "
                       "instance");
  
    ASTContext &Ctx = C.getASTContext();
    releaseS = GetNullarySelector("release", Ctx);
    retainS = GetNullarySelector("retain", Ctx);
    autoreleaseS = GetNullarySelector("autorelease", Ctx);
    drainS = GetNullarySelector("drain", Ctx);
  }
  
  ObjCInterfaceDecl *Class = 0;

  switch (ME->getReceiverKind()) {
  case ObjCMessageExpr::Class:
    Class = ME->getClassReceiver()->getAs<ObjCObjectType>()->getInterface();
    break;
  case ObjCMessageExpr::SuperClass:
    Class = ME->getSuperType()->getAs<ObjCObjectType>()->getInterface();
    break;
  case ObjCMessageExpr::Instance:
  case ObjCMessageExpr::SuperInstance:
    return;
  }

  Selector S = ME->getSelector();
  if (!(S == releaseS || S == retainS || S == autoreleaseS || S == drainS))
    return;
  
  if (ExplodedNode *N = C.generateNode()) {
    llvm::SmallString<200> buf;
    llvm::raw_svector_ostream os(buf);

    os << "The '" << S.getAsString() << "' message should be sent to instances "
          "of class '" << Class->getName()
       << "' and not the class directly";
  
    RangedBugReport *report = new RangedBugReport(*BT, os.str(), N);
    report->addRange(ME->getSourceRange());
    C.EmitReport(report);
  }
}

//===----------------------------------------------------------------------===//
// Check registration.
//===----------------------------------------------------------------------===//
  
void ento::RegisterAppleChecks(ExprEngine& Eng, const Decl &D) {
  Eng.registerCheck(new NilArgChecker());
  Eng.registerCheck(new CFNumberCreateChecker());
  RegisterNSErrorChecks(Eng.getBugReporter(), Eng, D);
  RegisterNSAutoreleasePoolChecks(Eng);
  Eng.registerCheck(new CFRetainReleaseChecker());
  Eng.registerCheck(new ClassReleaseChecker());
}
