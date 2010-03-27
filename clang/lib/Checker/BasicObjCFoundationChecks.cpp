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

#include "clang/Checker/PathSensitive/ExplodedGraph.h"
#include "clang/Checker/PathSensitive/GRSimpleAPICheck.h"
#include "clang/Checker/PathSensitive/GRExprEngine.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/MemRegion.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/Checkers/LocalCheckers.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

static const ObjCInterfaceType* GetReceiverType(const ObjCMessageExpr* ME) {
  const Expr* Receiver = ME->getReceiver();

  if (!Receiver)
    return NULL;

  if (const ObjCObjectPointerType *PT =
      Receiver->getType()->getAs<ObjCObjectPointerType>())
    return PT->getInterfaceType();

  return NULL;
}

static const char* GetReceiverNameType(const ObjCMessageExpr* ME) {
  if (const ObjCInterfaceType *ReceiverType = GetReceiverType(ME))
    return ReceiverType->getDecl()->getIdentifier()->getNameStart();
  return NULL;
}

namespace {

class APIMisuse : public BugType {
public:
  APIMisuse(const char* name) : BugType(name, "API Misuse (Apple)") {}
};

class BasicObjCFoundationChecks : public GRSimpleAPICheck {
  APIMisuse *BT;
  BugReporter& BR;
  ASTContext &Ctx;

  bool isNSString(const ObjCInterfaceType *T, llvm::StringRef suffix);
  bool AuditNSString(ExplodedNode* N, const ObjCMessageExpr* ME);

  void Warn(ExplodedNode* N, const Expr* E, const std::string& s);
  void WarnNilArg(ExplodedNode* N, const Expr* E);

  bool CheckNilArg(ExplodedNode* N, unsigned Arg);

public:
  BasicObjCFoundationChecks(ASTContext& ctx, BugReporter& br)
    : BT(0), BR(br), Ctx(ctx) {}

  bool Audit(ExplodedNode* N, GRStateManager&);

private:
  void WarnNilArg(ExplodedNode* N, const ObjCMessageExpr* ME, unsigned Arg) {
    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
    os << "Argument to '" << GetReceiverNameType(ME) << "' method '"
       << ME->getSelector().getAsString() << "' cannot be nil.";

    // Lazily create the BugType object for NilArg.  This will be owned
    // by the BugReporter object 'BR' once we call BR.EmitWarning.
    if (!BT) BT = new APIMisuse("nil argument");

    RangedBugReport *R = new RangedBugReport(*BT, os.str(), N);
    R->addRange(ME->getArg(Arg)->getSourceRange());
    BR.EmitReport(R);
  }
};

} // end anonymous namespace


GRSimpleAPICheck*
clang::CreateBasicObjCFoundationChecks(ASTContext& Ctx, BugReporter& BR) {
  return new BasicObjCFoundationChecks(Ctx, BR);
}



bool BasicObjCFoundationChecks::Audit(ExplodedNode* N,
                                      GRStateManager&) {

  const ObjCMessageExpr* ME =
    cast<ObjCMessageExpr>(cast<PostStmt>(N->getLocation()).getStmt());

  const ObjCInterfaceType *ReceiverType = GetReceiverType(ME);

  if (!ReceiverType)
    return false;

  if (isNSString(ReceiverType,
                 ReceiverType->getDecl()->getIdentifier()->getName()))
    return AuditNSString(N, ME);

  return false;
}

static inline bool isNil(SVal X) {
  return isa<loc::ConcreteInt>(X);
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

bool BasicObjCFoundationChecks::CheckNilArg(ExplodedNode* N, unsigned Arg) {
  const ObjCMessageExpr* ME =
    cast<ObjCMessageExpr>(cast<PostStmt>(N->getLocation()).getStmt());

  const Expr * E = ME->getArg(Arg);

  if (isNil(N->getState()->getSVal(E))) {
    WarnNilArg(N, ME, Arg);
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// NSString checking.
//===----------------------------------------------------------------------===//

bool BasicObjCFoundationChecks::isNSString(const ObjCInterfaceType *T,
                                           llvm::StringRef ClassName) {
  return ClassName == "NSString" || ClassName == "NSMutableString";
}

bool BasicObjCFoundationChecks::AuditNSString(ExplodedNode* N,
                                              const ObjCMessageExpr* ME) {

  Selector S = ME->getSelector();

  if (S.isUnarySelector())
    return false;

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
      Name == "initWithFormat:")
    return CheckNilArg(N, 0);

  return false;
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

namespace {

class AuditCFNumberCreate : public GRSimpleAPICheck {
  APIMisuse* BT;

  // FIXME: Either this should be refactored into GRSimpleAPICheck, or
  //   it should always be passed with a call to Audit.  The latter
  //   approach makes this class more stateless.
  ASTContext& Ctx;
  IdentifierInfo* II;
  BugReporter& BR;

public:
  AuditCFNumberCreate(ASTContext& ctx, BugReporter& br)
  : BT(0), Ctx(ctx), II(&Ctx.Idents.get("CFNumberCreate")), BR(br){}

  ~AuditCFNumberCreate() {}

  bool Audit(ExplodedNode* N, GRStateManager&);

private:
  void AddError(const TypedRegion* R, const Expr* Ex, ExplodedNode *N,
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

bool AuditCFNumberCreate::Audit(ExplodedNode* N,GRStateManager&){
  const CallExpr* CE =
    cast<CallExpr>(cast<PostStmt>(N->getLocation()).getStmt());
  const Expr* Callee = CE->getCallee();
  SVal CallV = N->getState()->getSVal(Callee);
  const FunctionDecl* FD = CallV.getAsFunctionDecl();

  if (!FD || FD->getIdentifier() != II || CE->getNumArgs()!=3)
    return false;

  // Get the value of the "theType" argument.
  SVal TheTypeVal = N->getState()->getSVal(CE->getArg(1));

    // FIXME: We really should allow ranges of valid theType values, and
    //   bifurcate the state appropriately.
  nonloc::ConcreteInt* V = dyn_cast<nonloc::ConcreteInt>(&TheTypeVal);

  if (!V)
    return false;

  uint64_t NumberKind = V->getValue().getLimitedValue();
  Optional<uint64_t> TargetSize = GetCFNumberSize(Ctx, NumberKind);

  // FIXME: In some cases we can emit an error.
  if (!TargetSize.isKnown())
    return false;

  // Look at the value of the integer being passed by reference.  Essentially
  // we want to catch cases where the value passed in is not equal to the
  // size of the type being created.
  SVal TheValueExpr = N->getState()->getSVal(CE->getArg(2));

  // FIXME: Eventually we should handle arbitrary locations.  We can do this
  //  by having an enhanced memory model that does low-level typing.
  loc::MemRegionVal* LV = dyn_cast<loc::MemRegionVal>(&TheValueExpr);

  if (!LV)
    return false;

  const TypedRegion* R = dyn_cast<TypedRegion>(LV->StripCasts());

  if (!R)
    return false;

  QualType T = Ctx.getCanonicalType(R->getValueType(Ctx));

  // FIXME: If the pointee isn't an integer type, should we flag a warning?
  //  People can do weird stuff with pointers.

  if (!T->isIntegerType())
    return false;

  uint64_t SourceSize = Ctx.getTypeSize(T);

  // CHECK: is SourceSize == TargetSize

  if (SourceSize == TargetSize)
    return false;

  AddError(R, CE->getArg(2), N, SourceSize, TargetSize, NumberKind);

  // FIXME: We can actually create an abstract "CFNumber" object that has
  //  the bits initialized to the provided values.
  return SourceSize < TargetSize;
}

void AuditCFNumberCreate::AddError(const TypedRegion* R, const Expr* Ex,
                                   ExplodedNode *N,
                                   uint64_t SourceSize, uint64_t TargetSize,
                                   uint64_t NumberKind) {

  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

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

  // Lazily create the BugType object.  This will be owned
  // by the BugReporter object 'BR' once we call BR.EmitWarning.
  if (!BT) BT = new APIMisuse("Bad use of CFNumberCreate");
  RangedBugReport *report = new RangedBugReport(*BT, os.str(), N);
  report->addRange(Ex->getSourceRange());
  BR.EmitReport(report);
}

GRSimpleAPICheck*
clang::CreateAuditCFNumberCreate(ASTContext& Ctx, BugReporter& BR) {
  return new AuditCFNumberCreate(Ctx, BR);
}

//===----------------------------------------------------------------------===//
// CFRetain/CFRelease auditing for null arguments.
//===----------------------------------------------------------------------===//

namespace {
class AuditCFRetainRelease : public GRSimpleAPICheck {
  APIMisuse *BT;

  // FIXME: Either this should be refactored into GRSimpleAPICheck, or
  //   it should always be passed with a call to Audit.  The latter
  //   approach makes this class more stateless.
  ASTContext& Ctx;
  IdentifierInfo *Retain, *Release;
  BugReporter& BR;

public:
  AuditCFRetainRelease(ASTContext& ctx, BugReporter& br)
  : BT(0), Ctx(ctx),
    Retain(&Ctx.Idents.get("CFRetain")), Release(&Ctx.Idents.get("CFRelease")),
    BR(br){}

  ~AuditCFRetainRelease() {}

  bool Audit(ExplodedNode* N, GRStateManager&);
};
} // end anonymous namespace


bool AuditCFRetainRelease::Audit(ExplodedNode* N, GRStateManager&) {
  const CallExpr* CE = cast<CallExpr>(cast<PostStmt>(N->getLocation()).getStmt());

  // If the CallExpr doesn't have exactly 1 argument just give up checking.
  if (CE->getNumArgs() != 1)
    return false;

  // Check if we called CFRetain/CFRelease.
  const GRState* state = N->getState();
  SVal X = state->getSVal(CE->getCallee());
  const FunctionDecl* FD = X.getAsFunctionDecl();

  if (!FD)
    return false;

  const IdentifierInfo *FuncII = FD->getIdentifier();
  if (!(FuncII == Retain || FuncII == Release))
    return false;

  // Finally, check if the argument is NULL.
  // FIXME: We should be able to bifurcate the state here, as a successful
  // check will result in the value not being NULL afterwards.
  // FIXME: Need a way to register vistors for the BugReporter.  Would like
  // to benefit from the same diagnostics that regular null dereference
  // reporting has.
  if (state->getStateManager().isEqual(state, CE->getArg(0), 0)) {
    if (!BT)
      BT = new APIMisuse("null passed to CFRetain/CFRelease");

    const char *description = (FuncII == Retain)
                            ? "Null pointer argument in call to CFRetain"
                            : "Null pointer argument in call to CFRelease";

    RangedBugReport *report = new RangedBugReport(*BT, description, N);
    report->addRange(CE->getArg(0)->getSourceRange());
    BR.EmitReport(report);
    return true;
  }

  return false;
}


GRSimpleAPICheck*
clang::CreateAuditCFRetainRelease(ASTContext& Ctx, BugReporter& BR) {
  return new AuditCFRetainRelease(Ctx, BR);
}

//===----------------------------------------------------------------------===//
// Check for sending 'retain', 'release', or 'autorelease' directly to a Class.
//===----------------------------------------------------------------------===//

namespace {
class ClassReleaseChecker :
    public CheckerVisitor<ClassReleaseChecker> {
  Selector releaseS;
  Selector retainS;
  Selector autoreleaseS;
  Selector drainS;
  BugType *BT;
public:
  ClassReleaseChecker(ASTContext &Ctx)
    : releaseS(GetNullarySelector("release", Ctx)),
      retainS(GetNullarySelector("retain", Ctx)),
      autoreleaseS(GetNullarySelector("autorelease", Ctx)),
      drainS(GetNullarySelector("drain", Ctx)),
      BT(0) {}

  static void *getTag() { static int x = 0; return &x; }
      
  void PreVisitObjCMessageExpr(CheckerContext &C, const ObjCMessageExpr *ME);    
};
}

void ClassReleaseChecker::PreVisitObjCMessageExpr(CheckerContext &C,
                                                  const ObjCMessageExpr *ME) {
  
  const IdentifierInfo *ClsName = ME->getClassName();
  if (!ClsName)
    return;
  
  Selector S = ME->getSelector();
  if (!(S == releaseS || S == retainS || S == autoreleaseS || S == drainS))
    return;
  
  if (!BT)
    BT = new APIMisuse("message incorrectly sent to class instead of class "
                       "instance");
  
  ExplodedNode *N = C.GenerateNode();

  if (!N)
    return;
  
  llvm::SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);

  os << "The '" << S.getAsString() << "' message should be sent to instances "
        "of class '" << ClsName->getName()
     << "' and not the class directly";
  
  RangedBugReport *report = new RangedBugReport(*BT, os.str(), N);
  report->addRange(ME->getSourceRange());
  C.EmitReport(report);
}

//===----------------------------------------------------------------------===//
// Check registration.
//===----------------------------------------------------------------------===//

void clang::RegisterAppleChecks(GRExprEngine& Eng, const Decl &D) {
  ASTContext& Ctx = Eng.getContext();
  BugReporter &BR = Eng.getBugReporter();

  Eng.AddCheck(CreateBasicObjCFoundationChecks(Ctx, BR),
               Stmt::ObjCMessageExprClass);
  Eng.AddCheck(CreateAuditCFNumberCreate(Ctx, BR), Stmt::CallExprClass);
  Eng.AddCheck(CreateAuditCFRetainRelease(Ctx, BR), Stmt::CallExprClass);

  RegisterNSErrorChecks(BR, Eng, D);
  RegisterNSAutoreleasePoolChecks(Eng);
  Eng.registerCheck(new ClassReleaseChecker(Ctx));
}
