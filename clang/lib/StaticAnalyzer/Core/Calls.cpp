//===- Calls.cpp - Wrapper for all function and method calls ------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines CallEvent and its subclasses, which represent path-
/// sensitive instances of different kinds of function and method calls
/// (C, C++, and Objective-C).
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/Calls.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/AST/ParentMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace ento;

QualType CallEvent::getResultType() const {
  QualType ResultTy = getDeclaredResultType();

  if (ResultTy.isNull())
    ResultTy = getOriginExpr()->getType();

  return ResultTy;
}

static bool isCallbackArg(SVal V, QualType T) {
  // If the parameter is 0, it's harmless.
  if (V.isZeroConstant())
    return false;

  // If a parameter is a block or a callback, assume it can modify pointer.
  if (T->isBlockPointerType() ||
      T->isFunctionPointerType() ||
      T->isObjCSelType())
    return true;

  // Check if a callback is passed inside a struct (for both, struct passed by
  // reference and by value). Dig just one level into the struct for now.

  if (isa<PointerType>(T) || isa<ReferenceType>(T))
    T = T->getPointeeType();

  if (const RecordType *RT = T->getAsStructureType()) {
    const RecordDecl *RD = RT->getDecl();
    for (RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
         I != E; ++I) {
      QualType FieldT = I->getType();
      if (FieldT->isBlockPointerType() || FieldT->isFunctionPointerType())
        return true;
    }
  }

  return false;
}

bool CallEvent::hasNonZeroCallbackArg() const {
  unsigned NumOfArgs = getNumArgs();

  // If calling using a function pointer, assume the function does not
  // have a callback. TODO: We could check the types of the arguments here.
  if (!getDecl())
    return false;

  unsigned Idx = 0;
  for (CallEvent::param_type_iterator I = param_type_begin(),
                                       E = param_type_end();
       I != E && Idx < NumOfArgs; ++I, ++Idx) {
    if (NumOfArgs <= Idx)
      break;

    if (isCallbackArg(getArgSVal(Idx), *I))
      return true;
  }
  
  return false;
}

/// \brief Returns true if a type is a pointer-to-const or reference-to-const
/// with no further indirection.
static bool isPointerToConst(QualType Ty) {
  QualType PointeeTy = Ty->getPointeeType();
  if (PointeeTy == QualType())
    return false;
  if (!PointeeTy.isConstQualified())
    return false;
  if (PointeeTy->isAnyPointerType())
    return false;
  return true;
}

// Try to retrieve the function declaration and find the function parameter
// types which are pointers/references to a non-pointer const.
// We will not invalidate the corresponding argument regions.
static void findPtrToConstParams(llvm::SmallSet<unsigned, 1> &PreserveArgs,
                                 const CallEvent &Call) {
  unsigned Idx = 0;
  for (CallEvent::param_type_iterator I = Call.param_type_begin(),
                                      E = Call.param_type_end();
       I != E; ++I, ++Idx) {
    if (isPointerToConst(*I))
      PreserveArgs.insert(Idx);
  }
}

ProgramStateRef CallEvent::invalidateRegions(unsigned BlockCount,
                                              ProgramStateRef Orig) const {
  ProgramStateRef Result = (Orig ? Orig : getState());

  SmallVector<const MemRegion *, 8> RegionsToInvalidate;
  getExtraInvalidatedRegions(RegionsToInvalidate);

  // Indexes of arguments whose values will be preserved by the call.
  llvm::SmallSet<unsigned, 1> PreserveArgs;
  if (!argumentsMayEscape())
    findPtrToConstParams(PreserveArgs, *this);

  for (unsigned Idx = 0, Count = getNumArgs(); Idx != Count; ++Idx) {
    if (PreserveArgs.count(Idx))
      continue;

    SVal V = getArgSVal(Idx);

    // If we are passing a location wrapped as an integer, unwrap it and
    // invalidate the values referred by the location.
    if (nonloc::LocAsInteger *Wrapped = dyn_cast<nonloc::LocAsInteger>(&V))
      V = Wrapped->getLoc();
    else if (!isa<Loc>(V))
      continue;

    if (const MemRegion *R = V.getAsRegion()) {
      // Invalidate the value of the variable passed by reference.

      // Are we dealing with an ElementRegion?  If the element type is
      // a basic integer type (e.g., char, int) and the underlying region
      // is a variable region then strip off the ElementRegion.
      // FIXME: We really need to think about this for the general case
      //   as sometimes we are reasoning about arrays and other times
      //   about (char*), etc., is just a form of passing raw bytes.
      //   e.g., void *p = alloca(); foo((char*)p);
      if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
        // Checking for 'integral type' is probably too promiscuous, but
        // we'll leave it in for now until we have a systematic way of
        // handling all of these cases.  Eventually we need to come up
        // with an interface to StoreManager so that this logic can be
        // appropriately delegated to the respective StoreManagers while
        // still allowing us to do checker-specific logic (e.g.,
        // invalidating reference counts), probably via callbacks.
        if (ER->getElementType()->isIntegralOrEnumerationType()) {
          const MemRegion *superReg = ER->getSuperRegion();
          if (isa<VarRegion>(superReg) || isa<FieldRegion>(superReg) ||
              isa<ObjCIvarRegion>(superReg))
            R = cast<TypedRegion>(superReg);
        }
        // FIXME: What about layers of ElementRegions?
      }

      // Mark this region for invalidation.  We batch invalidate regions
      // below for efficiency.
      RegionsToInvalidate.push_back(R);
    }
  }

  // Invalidate designated regions using the batch invalidation API.
  // NOTE: Even if RegionsToInvalidate is empty, we may still invalidate
  //  global variables.
  return Result->invalidateRegions(RegionsToInvalidate, getOriginExpr(),
                                   BlockCount, getLocationContext(),
                                   /*Symbols=*/0, this);
}

ProgramPoint CallEvent::getProgramPoint(bool IsPreVisit,
                                        const ProgramPointTag *Tag) const {
  if (const Expr *E = getOriginExpr()) {
    if (IsPreVisit)
      return PreStmt(E, getLocationContext(), Tag);
    return PostStmt(E, getLocationContext(), Tag);
  }

  const Decl *D = getDecl();
  assert(D && "Cannot get a program point without a statement or decl");  

  SourceLocation Loc = getSourceRange().getBegin();
  if (IsPreVisit)
    return PreImplicitCall(D, Loc, getLocationContext(), Tag);
  return PostImplicitCall(D, Loc, getLocationContext(), Tag);
}


bool CallEvent::mayBeInlined(const Stmt *S) {
  return isa<CallExpr>(S);
}


CallEvent::param_iterator
AnyFunctionCall::param_begin(bool UseDefinitionParams) const {
  bool IgnoredDynamicDispatch;
  const Decl *D = UseDefinitionParams ? getDefinition(IgnoredDynamicDispatch)
                                      : getDecl();
  if (!D)
    return 0;

  return cast<FunctionDecl>(D)->param_begin();
}

CallEvent::param_iterator
AnyFunctionCall::param_end(bool UseDefinitionParams) const {
  bool IgnoredDynamicDispatch;
  const Decl *D = UseDefinitionParams ? getDefinition(IgnoredDynamicDispatch)
                                      : getDecl();
  if (!D)
    return 0;

  return cast<FunctionDecl>(D)->param_end();
}

QualType AnyFunctionCall::getDeclaredResultType() const {
  const FunctionDecl *D = getDecl();
  if (!D)
    return QualType();

  return D->getResultType();
}

bool AnyFunctionCall::argumentsMayEscape() const {
  if (hasNonZeroCallbackArg())
    return true;

  const FunctionDecl *D = getDecl();
  if (!D)
    return true;

  const IdentifierInfo *II = D->getIdentifier();
  if (!II)
    return true;

  // This set of "escaping" APIs is 

  // - 'int pthread_setspecific(ptheread_key k, const void *)' stores a
  //   value into thread local storage. The value can later be retrieved with
  //   'void *ptheread_getspecific(pthread_key)'. So even thought the
  //   parameter is 'const void *', the region escapes through the call.
  if (II->isStr("pthread_setspecific"))
    return true;

  // - xpc_connection_set_context stores a value which can be retrieved later
  //   with xpc_connection_get_context.
  if (II->isStr("xpc_connection_set_context"))
    return true;

  // - funopen - sets a buffer for future IO calls.
  if (II->isStr("funopen"))
    return true;

  StringRef FName = II->getName();

  // - CoreFoundation functions that end with "NoCopy" can free a passed-in
  //   buffer even if it is const.
  if (FName.endswith("NoCopy"))
    return true;

  // - NSXXInsertXX, for example NSMapInsertIfAbsent, since they can
  //   be deallocated by NSMapRemove.
  if (FName.startswith("NS") && (FName.find("Insert") != StringRef::npos))
    return true;

  // - Many CF containers allow objects to escape through custom
  //   allocators/deallocators upon container construction. (PR12101)
  if (FName.startswith("CF") || FName.startswith("CG")) {
    return StrInStrNoCase(FName, "InsertValue")  != StringRef::npos ||
           StrInStrNoCase(FName, "AddValue")     != StringRef::npos ||
           StrInStrNoCase(FName, "SetValue")     != StringRef::npos ||
           StrInStrNoCase(FName, "WithData")     != StringRef::npos ||
           StrInStrNoCase(FName, "AppendValue")  != StringRef::npos ||
           StrInStrNoCase(FName, "SetAttribute") != StringRef::npos;
  }

  return false;
}

SVal AnyFunctionCall::getArgSVal(unsigned Index) const {
  const Expr *ArgE = getArgExpr(Index);
  if (!ArgE)
    return UnknownVal();
  return getSVal(ArgE);
}

SourceRange AnyFunctionCall::getArgSourceRange(unsigned Index) const {
  const Expr *ArgE = getArgExpr(Index);
  if (!ArgE)
    return SourceRange();
  return ArgE->getSourceRange();
}


const FunctionDecl *SimpleCall::getDecl() const {
  const FunctionDecl *D = getOriginExpr()->getDirectCallee();
  if (D)
    return D;

  return getSVal(getOriginExpr()->getCallee()).getAsFunctionDecl();
}

void CallEvent::dump(raw_ostream &Out) const {
  ASTContext &Ctx = getState()->getStateManager().getContext();
  if (const Expr *E = getOriginExpr()) {
    E->printPretty(Out, Ctx, 0, Ctx.getLangOpts());
    Out << "\n";
    return;
  }

  if (const Decl *D = getDecl()) {
    Out << "Call to ";
    D->print(Out, Ctx.getLangOpts());
    return;
  }

  // FIXME: a string representation of the kind would be nice.
  Out << "Unknown call (type " << getKind() << ")";
}


void CXXInstanceCall::getExtraInvalidatedRegions(RegionList &Regions) const {
  if (const MemRegion *R = getCXXThisVal().getAsRegion())
    Regions.push_back(R);
}

static const CXXMethodDecl *devirtualize(const CXXMethodDecl *MD, SVal ThisVal){
  const MemRegion *R = ThisVal.getAsRegion();
  if (!R)
    return 0;

  const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R->StripCasts());
  if (!TR)
    return 0;

  const CXXRecordDecl *RD = TR->getValueType()->getAsCXXRecordDecl();
  if (!RD)
    return 0;

  const CXXMethodDecl *Result = MD->getCorrespondingMethodInClass(RD);
  const FunctionDecl *Definition;
  if (!Result->hasBody(Definition))
    return 0;

  return cast<CXXMethodDecl>(Definition);
}


const Decl *CXXInstanceCall::getDefinition(bool &IsDynamicDispatch) const {
  const Decl *D = SimpleCall::getDefinition(IsDynamicDispatch);
  if (!D)
    return 0;

  const CXXMethodDecl *MD = cast<CXXMethodDecl>(D);
  if (!MD->isVirtual())
    return MD;

  // If the method is virtual, see if we can find the actual implementation
  // based on context-sensitivity.
  if (const CXXMethodDecl *Devirtualized = devirtualize(MD, getCXXThisVal()))
    return Devirtualized;

  IsDynamicDispatch = true;
  return MD;
}


SVal CXXMemberCall::getCXXThisVal() const {
  const Expr *Base = getOriginExpr()->getImplicitObjectArgument();

  // FIXME: Will eventually need to cope with member pointers.  This is
  // a limitation in getImplicitObjectArgument().
  if (!Base)
    return UnknownVal();

  return getSVal(Base);
}


SVal CXXMemberOperatorCall::getCXXThisVal() const {
  const Expr *Base = getOriginExpr()->getArg(0);
  return getSVal(Base);
}


const BlockDataRegion *BlockCall::getBlockRegion() const {
  const Expr *Callee = getOriginExpr()->getCallee();
  const MemRegion *DataReg = getSVal(Callee).getAsRegion();

  return dyn_cast_or_null<BlockDataRegion>(DataReg);
}

CallEvent::param_iterator
BlockCall::param_begin(bool UseDefinitionParams) const {
  // Blocks don't have distinct declarations and definitions.
  (void)UseDefinitionParams;

  const BlockDecl *D = getBlockDecl();
  if (!D)
    return 0;
  return D->param_begin();
}

CallEvent::param_iterator
BlockCall::param_end(bool UseDefinitionParams) const {
  // Blocks don't have distinct declarations and definitions.
  (void)UseDefinitionParams;

  const BlockDecl *D = getBlockDecl();
  if (!D)
    return 0;
  return D->param_end();
}

void BlockCall::getExtraInvalidatedRegions(RegionList &Regions) const {
  // FIXME: This also needs to invalidate captured globals.
  if (const MemRegion *R = getBlockRegion())
    Regions.push_back(R);
}

QualType BlockCall::getDeclaredResultType() const {
  const BlockDataRegion *BR = getBlockRegion();
  if (!BR)
    return QualType();
  QualType BlockTy = BR->getCodeRegion()->getLocationType();
  return cast<FunctionType>(BlockTy->getPointeeType())->getResultType();
}


SVal CXXConstructorCall::getCXXThisVal() const {
  if (Data)
    return loc::MemRegionVal(static_cast<const MemRegion *>(Data));
  return UnknownVal();
}

void CXXConstructorCall::getExtraInvalidatedRegions(RegionList &Regions) const {
  if (Data)
    Regions.push_back(static_cast<const MemRegion *>(Data));
}


SVal CXXDestructorCall::getCXXThisVal() const {
  if (Data)
    return loc::MemRegionVal(static_cast<const MemRegion *>(Data));
  return UnknownVal();
}

void CXXDestructorCall::getExtraInvalidatedRegions(RegionList &Regions) const {
  if (Data)
    Regions.push_back(static_cast<const MemRegion *>(Data));
}

const Decl *CXXDestructorCall::getDefinition(bool &IsDynamicDispatch) const {
  const Decl *D = AnyFunctionCall::getDefinition(IsDynamicDispatch);
  if (!D)
    return 0;

  const CXXMethodDecl *MD = cast<CXXMethodDecl>(D);
  if (!MD->isVirtual())
    return MD;

  // If the method is virtual, see if we can find the actual implementation
  // based on context-sensitivity.
  if (const CXXMethodDecl *Devirtualized = devirtualize(MD, getCXXThisVal()))
    return Devirtualized;

  IsDynamicDispatch = true;
  return MD;
}


CallEvent::param_iterator
ObjCMethodCall::param_begin(bool UseDefinitionParams) const {
  bool IgnoredDynamicDispatch;
  const Decl *D = UseDefinitionParams ? getDefinition(IgnoredDynamicDispatch)
                                      : getDecl();
  if (!D)
    return 0;

  return cast<ObjCMethodDecl>(D)->param_begin();
}

CallEvent::param_iterator
ObjCMethodCall::param_end(bool UseDefinitionParams) const {
  bool IgnoredDynamicDispatch;
  const Decl *D = UseDefinitionParams ? getDefinition(IgnoredDynamicDispatch)
                                      : getDecl();
  if (!D)
    return 0;

  return cast<ObjCMethodDecl>(D)->param_end();
}

void
ObjCMethodCall::getExtraInvalidatedRegions(RegionList &Regions) const {
  if (const MemRegion *R = getReceiverSVal().getAsRegion())
    Regions.push_back(R);
}

QualType ObjCMethodCall::getDeclaredResultType() const {
  const ObjCMethodDecl *D = getDecl();
  if (!D)
    return QualType();

  return D->getResultType();
}

SVal ObjCMethodCall::getReceiverSVal() const {
  // FIXME: Is this the best way to handle class receivers?
  if (!isInstanceMessage())
    return UnknownVal();
    
  if (const Expr *Base = getOriginExpr()->getInstanceReceiver())
    return getSVal(Base);

  // An instance message with no expression means we are sending to super.
  // In this case the object reference is the same as 'self'.
  const LocationContext *LCtx = getLocationContext();
  const ImplicitParamDecl *SelfDecl = LCtx->getSelfDecl();
  assert(SelfDecl && "No message receiver Expr, but not in an ObjC method");
  return getState()->getSVal(getState()->getRegion(SelfDecl, LCtx));
}

SourceRange ObjCMethodCall::getSourceRange() const {
  switch (getMessageKind()) {
  case OCM_Message:
    return getOriginExpr()->getSourceRange();
  case OCM_PropertyAccess:
  case OCM_Subscript:
    return getContainingPseudoObjectExpr()->getSourceRange();
  }
  llvm_unreachable("unknown message kind");
}

typedef llvm::PointerIntPair<const PseudoObjectExpr *, 2> ObjCMessageDataTy;

const PseudoObjectExpr *ObjCMethodCall::getContainingPseudoObjectExpr() const {
  assert(Data != 0 && "Lazy lookup not yet performed.");
  assert(getMessageKind() != OCM_Message && "Explicit message send.");
  return ObjCMessageDataTy::getFromOpaqueValue(Data).getPointer();
}

ObjCMessageKind ObjCMethodCall::getMessageKind() const {
  if (Data == 0) {
    ParentMap &PM = getLocationContext()->getParentMap();
    const Stmt *S = PM.getParent(getOriginExpr());
    if (const PseudoObjectExpr *POE = dyn_cast_or_null<PseudoObjectExpr>(S)) {
      const Expr *Syntactic = POE->getSyntacticForm();

      // This handles the funny case of assigning to the result of a getter.
      // This can happen if the getter returns a non-const reference.
      if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(Syntactic))
        Syntactic = BO->getLHS();

      ObjCMessageKind K;
      switch (Syntactic->getStmtClass()) {
      case Stmt::ObjCPropertyRefExprClass:
        K = OCM_PropertyAccess;
        break;
      case Stmt::ObjCSubscriptRefExprClass:
        K = OCM_Subscript;
        break;
      default:
        // FIXME: Can this ever happen?
        K = OCM_Message;
        break;
      }

      if (K != OCM_Message) {
        const_cast<ObjCMethodCall *>(this)->Data
          = ObjCMessageDataTy(POE, K).getOpaqueValue();
        assert(getMessageKind() == K);
        return K;
      }
    }
    
    const_cast<ObjCMethodCall *>(this)->Data
      = ObjCMessageDataTy(0, 1).getOpaqueValue();
    assert(getMessageKind() == OCM_Message);
    return OCM_Message;
  }

  ObjCMessageDataTy Info = ObjCMessageDataTy::getFromOpaqueValue(Data);
  if (!Info.getPointer())
    return OCM_Message;
  return static_cast<ObjCMessageKind>(Info.getInt());
}
