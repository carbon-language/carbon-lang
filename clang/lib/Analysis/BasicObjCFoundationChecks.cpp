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

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/Analysis/PathSensitive/GRSimpleAPICheck.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

static ObjCInterfaceType* GetReceiverType(ObjCMessageExpr* ME) {
  Expr* Receiver = ME->getReceiver();
  
  if (!Receiver)
    return NULL;
  
  QualType X = Receiver->getType();
  
  if (X->isPointerType()) {
    Type* TP = X.getTypePtr();
    const PointerType* T = TP->getAsPointerType();    
    return dyn_cast<ObjCInterfaceType>(T->getPointeeType().getTypePtr());
  }

  // FIXME: Support ObjCQualifiedIdType?
  return NULL;
}

static const char* GetReceiverNameType(ObjCMessageExpr* ME) {
  ObjCInterfaceType* ReceiverType = GetReceiverType(ME);
  return ReceiverType ? ReceiverType->getDecl()->getIdentifier()->getName()
                      : NULL;
}

namespace {

class VISIBILITY_HIDDEN APIMisuse : public BugType {
public:
  APIMisuse(const char* name) : BugType(name, "API Misuse (Apple)") {}
};
  
class VISIBILITY_HIDDEN BasicObjCFoundationChecks : public GRSimpleAPICheck {
  APIMisuse *BT;
  BugReporter& BR;
  ASTContext &Ctx;
  GRStateManager* VMgr;

  SVal GetSVal(const GRState* St, Expr* E) { return VMgr->GetSVal(St, E); }
      
  bool isNSString(ObjCInterfaceType* T, const char* suffix);
  bool AuditNSString(NodeTy* N, ObjCMessageExpr* ME);
      
  void Warn(NodeTy* N, Expr* E, const std::string& s);  
  void WarnNilArg(NodeTy* N, Expr* E);
  
  bool CheckNilArg(NodeTy* N, unsigned Arg);

public:
  BasicObjCFoundationChecks(ASTContext& ctx, GRStateManager* vmgr,
                            BugReporter& br) 
    : BT(0), BR(br), Ctx(ctx), VMgr(vmgr) {}
        
  bool Audit(ExplodedNode<GRState>* N, GRStateManager&);
  
private:  
  void WarnNilArg(NodeTy* N, ObjCMessageExpr* ME, unsigned Arg) {    
    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
    os << "Argument to '" << GetReceiverNameType(ME) << "' method '"
       << ME->getSelector().getAsString() << "' cannot be nil.";
    
    // Lazily create the BugType object for NilArg.  This will be owned
    // by the BugReporter object 'BR' once we call BR.EmitWarning.
    if (!BT) BT = new APIMisuse("nil argument");
    
    RangedBugReport *R = new RangedBugReport(*BT, os.str().c_str(), N);
    R->addRange(ME->getArg(Arg)->getSourceRange());
    BR.EmitReport(R);
  }
};
  
} // end anonymous namespace


GRSimpleAPICheck*
clang::CreateBasicObjCFoundationChecks(ASTContext& Ctx,
                                       GRStateManager* VMgr, BugReporter& BR) {
  
  return new BasicObjCFoundationChecks(Ctx, VMgr, BR);  
}



bool BasicObjCFoundationChecks::Audit(ExplodedNode<GRState>* N,
                                      GRStateManager&) {
  
  ObjCMessageExpr* ME =
    cast<ObjCMessageExpr>(cast<PostStmt>(N->getLocation()).getStmt());

  ObjCInterfaceType* ReceiverType = GetReceiverType(ME);
  
  if (!ReceiverType)
    return false;
  
  const char* name = ReceiverType->getDecl()->getIdentifier()->getName();
  
  if (!name)
    return false;

  if (name[0] != 'N' || name[1] != 'S')
    return false;
      
  name += 2;
  
  // FIXME: Make all of this faster.
  
  if (isNSString(ReceiverType, name))
    return AuditNSString(N, ME);

  return false;
}

static inline bool isNil(SVal X) {
  return isa<loc::ConcreteInt>(X);  
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

bool BasicObjCFoundationChecks::CheckNilArg(NodeTy* N, unsigned Arg) {
  ObjCMessageExpr* ME =
    cast<ObjCMessageExpr>(cast<PostStmt>(N->getLocation()).getStmt());
  
  Expr * E = ME->getArg(Arg);
  
  if (isNil(GetSVal(N->getState(), E))) {
    WarnNilArg(N, ME, Arg);
    return true;
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// NSString checking.
//===----------------------------------------------------------------------===//

bool BasicObjCFoundationChecks::isNSString(ObjCInterfaceType* T,
                                           const char* suffix) {
  
  return !strcmp("String", suffix) || !strcmp("MutableString", suffix);
}

bool BasicObjCFoundationChecks::AuditNSString(NodeTy* N, 
                                              ObjCMessageExpr* ME) {
  
  Selector S = ME->getSelector();
  
  if (S.isUnarySelector())
    return false;

  // FIXME: This is going to be really slow doing these checks with
  //  lexical comparisons.
  
  std::string name = S.getAsString();
  assert (!name.empty());
  const char* cstr = &name[0];
  unsigned len = name.size();
      
  switch (len) {
    default:
      break;
    case 8:      
      if (!strcmp(cstr, "compare:"))
        return CheckNilArg(N, 0);
              
      break;
      
    case 15:
      // FIXME: Checking for initWithFormat: will not work in most cases
      //  yet because [NSString alloc] returns id, not NSString*.  We will
      //  need support for tracking expected-type information in the analyzer
      //  to find these errors.
      if (!strcmp(cstr, "initWithFormat:"))
        return CheckNilArg(N, 0);
      
      break;
    
    case 16:
      if (!strcmp(cstr, "compare:options:"))
        return CheckNilArg(N, 0);
      
      break;
      
    case 22:
      if (!strcmp(cstr, "compare:options:range:"))
        return CheckNilArg(N, 0);
      
      break;
      
    case 23:
      
      if (!strcmp(cstr, "caseInsensitiveCompare:"))
        return CheckNilArg(N, 0);
      
      break;

    case 29:
      if (!strcmp(cstr, "compare:options:range:locale:"))
        return CheckNilArg(N, 0);
    
      break;    
      
    case 37:
    if (!strcmp(cstr, "componentsSeparatedByCharactersInSet:"))
      return CheckNilArg(N, 0);
    
    break;    
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN AuditCFNumberCreate : public GRSimpleAPICheck {
  APIMisuse* BT;
  
  // FIXME: Either this should be refactored into GRSimpleAPICheck, or
  //   it should always be passed with a call to Audit.  The latter
  //   approach makes this class more stateless.
  ASTContext& Ctx;
  IdentifierInfo* II;
  GRStateManager* VMgr;
  BugReporter& BR;
    
  SVal GetSVal(const GRState* St, Expr* E) { return VMgr->GetSVal(St, E); }
  
public:
  AuditCFNumberCreate(ASTContext& ctx, GRStateManager* vmgr, BugReporter& br) 
  : BT(0), Ctx(ctx), II(&Ctx.Idents.get("CFNumberCreate")), VMgr(vmgr), BR(br){}
  
  ~AuditCFNumberCreate() {}
  
  bool Audit(ExplodedNode<GRState>* N, GRStateManager&);
  
private:
  void AddError(const TypedRegion* R, Expr* Ex, ExplodedNode<GRState> *N,
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
  static unsigned char FixedSize[] = { 8, 16, 32, 64, 32, 64 };
  
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

bool AuditCFNumberCreate::Audit(ExplodedNode<GRState>* N,GRStateManager&){  
  CallExpr* CE = cast<CallExpr>(cast<PostStmt>(N->getLocation()).getStmt());
  Expr* Callee = CE->getCallee();  
  SVal CallV = GetSVal(N->getState(), Callee);  
  const FunctionDecl* FD = CallV.getAsFunctionDecl();

  if (!FD || FD->getIdentifier() != II || CE->getNumArgs()!=3)
    return false;
  
  // Get the value of the "theType" argument.
  SVal  TheTypeVal = GetSVal(N->getState(), CE->getArg(1));
  
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
  SVal TheValueExpr = GetSVal(N->getState(), CE->getArg(2));
  
  // FIXME: Eventually we should handle arbitrary locations.  We can do this
  //  by having an enhanced memory model that does low-level typing.
  loc::MemRegionVal* LV = dyn_cast<loc::MemRegionVal>(&TheValueExpr);

  if (!LV)
    return false;
  
  const TypedRegion* R = dyn_cast<TypedRegion>(LV->getRegion());
  if (!R) return false;
  
  while (const TypedViewRegion* ATR = dyn_cast<TypedViewRegion>(R)) {
    R = dyn_cast<TypedRegion>(ATR->getSuperRegion());
    if (!R) return false;
  }
  
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

void AuditCFNumberCreate::AddError(const TypedRegion* R, Expr* Ex,
                                   ExplodedNode<GRState> *N,
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
  RangedBugReport *report = new RangedBugReport(*BT, os.str().c_str(), N);
  report->addRange(Ex->getSourceRange());
  BR.EmitReport(report);
}

GRSimpleAPICheck*
clang::CreateAuditCFNumberCreate(ASTContext& Ctx,
                                 GRStateManager* VMgr, BugReporter& BR) {  
  return new AuditCFNumberCreate(Ctx, VMgr, BR);
}

//===----------------------------------------------------------------------===//
// Check registration.

void clang::RegisterAppleChecks(GRExprEngine& Eng) {
  ASTContext& Ctx = Eng.getContext();
  GRStateManager* VMgr = &Eng.getStateManager();
  BugReporter &BR = Eng.getBugReporter();

  Eng.AddCheck(CreateBasicObjCFoundationChecks(Ctx, VMgr, BR),
               Stmt::ObjCMessageExprClass);

  Eng.AddCheck(CreateAuditCFNumberCreate(Ctx, VMgr, BR),
               Stmt::CallExprClass);
  
  RegisterNSErrorChecks(BR, Eng);
}
