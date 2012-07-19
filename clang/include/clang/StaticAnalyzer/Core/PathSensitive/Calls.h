//===- Calls.h - Wrapper for all function and method calls --------*- C++ -*--//
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

#ifndef LLVM_CLANG_STATICANALYZER_PATHSENSITIVE_CALL
#define LLVM_CLANG_STATICANALYZER_PATHSENSITIVE_CALL

#include "clang/Basic/SourceManager.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/PointerIntPair.h"

namespace clang {
class ProgramPoint;
class ProgramPointTag;

namespace ento {

enum CallEventKind {
  CE_Function,
  CE_CXXMember,
  CE_CXXMemberOperator,
  CE_BEG_CXX_INSTANCE_CALLS = CE_CXXMember,
  CE_END_CXX_INSTANCE_CALLS = CE_CXXMemberOperator,
  CE_Block,
  CE_BEG_SIMPLE_CALLS = CE_Function,
  CE_END_SIMPLE_CALLS = CE_Block,
  CE_CXXConstructor,
  CE_CXXDestructor,
  CE_CXXAllocator,
  CE_BEG_FUNCTION_CALLS = CE_Function,
  CE_END_FUNCTION_CALLS = CE_CXXAllocator,
  CE_ObjCMessage
};


/// \brief Represents an abstract call to a function or method along a
/// particular path.
class CallEvent {
public:
  typedef CallEventKind Kind;

private:
  // PointerIntPair doesn't respect IntrusiveRefCntPtr, so we have to manually
  // retain and release the state.
  llvm::PointerIntPair<const ProgramState *, 2> State;
  llvm::PointerIntPair<const LocationContext *, 2> LCtx;
  llvm::PointerUnion<const Expr *, const Decl *> Origin;

protected:
  // This is user data for subclasses.
  const void *Data;
  SourceLocation Location;

  CallEvent(const Expr *E, ProgramStateRef state, const LocationContext *lctx,
            Kind k)
    : State(state.getPtr(), (k & 0x3)),
      LCtx(lctx, ((k >> 2) & 0x3)),
      Origin(E) {
    IntrusiveRefCntPtrInfo<const ProgramState>::retain(getState());
    assert(k == getKind() && "More kinds than bits in the PointerIntPairs.");
  }

  CallEvent(const Decl *D, ProgramStateRef state, const LocationContext *lctx,
            Kind k)
    : State(state.getPtr(), (k & 0x3)),
      LCtx(lctx, ((k >> 2) & 0x3)),
      Origin(D) {
    IntrusiveRefCntPtrInfo<const ProgramState>::retain(getState());
    assert(k == getKind() && "More kinds than bits in the PointerIntPairs.");
  }

  const ProgramState *getState() const {
    return State.getPointer();
  }

  const LocationContext *getLocationContext() const {
    return LCtx.getPointer();
  }

  ~CallEvent() {
    IntrusiveRefCntPtrInfo<const ProgramState>::release(getState());
  }


  /// \brief Get the value of arbitrary expressions at this point in the path.
  SVal getSVal(const Stmt *S) const {
    return getState()->getSVal(S, getLocationContext());
  }

  typedef SmallVectorImpl<const MemRegion *> RegionList;

  /// \brief Used to specify non-argument regions that will be invalidated as a
  /// result of this call.
  void getExtraInvalidatedRegions(RegionList &Regions) const;

  QualType getDeclaredResultType() const;

public:
  /// \brief Returns the kind of call this is.
  Kind getKind() const {
    return static_cast<Kind>((State.getInt()) | (LCtx.getInt() << 2));
  }

  /// \brief Returns the declaration of the function or method that will be
  /// called. May be null.
  const Decl *getDecl() const;

  /// \brief Returns the definition of the function or method that will be
  /// called. May be null.
  ///
  /// This is used when deciding how to inline the call.
  ///
  /// \param IsDynamicDispatch True if the definition returned may not be the
  ///   definition that is actually invoked at runtime. Note that if we have
  ///   sufficient type information to devirtualize a dynamic method call,
  ///   we will (and \p IsDynamicDispatch will be set to \c false).
  const Decl *getDefinition(bool &IsDynamicDispatch) const;

  /// \brief Returns the expression whose value will be the result of this call.
  /// May be null.
  const Expr *getOriginExpr() const {
    return Origin.dyn_cast<const Expr *>();
  }

  /// \brief Returns the number of arguments (explicit and implicit).
  ///
  /// Note that this may be greater than the number of parameters in the
  /// callee's declaration, and that it may include arguments not written in
  /// the source.
  unsigned getNumArgs() const;

  /// \brief Returns true if the callee is known to be from a system header.
  bool isInSystemHeader() const {
    const Decl *D = getDecl();
    if (!D)
      return false;

    SourceLocation Loc = D->getLocation();
    if (Loc.isValid()) {
      const SourceManager &SM =
        getState()->getStateManager().getContext().getSourceManager();
      return SM.isInSystemHeader(D->getLocation());
    }

    // Special case for implicitly-declared global operator new/delete.
    // These should be considered system functions.
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      return FD->isOverloadedOperator() && FD->isImplicit() && FD->isGlobal();

    return false;
  }

  /// \brief Returns a source range for the entire call, suitable for
  /// outputting in diagnostics.
  SourceRange getSourceRange() const;

  /// \brief Returns the value of a given argument at the time of the call.
  SVal getArgSVal(unsigned Index) const;

  /// \brief Returns the expression associated with a given argument.
  /// May be null if this expression does not appear in the source.
  const Expr *getArgExpr(unsigned Index) const;

  /// \brief Returns the source range for errors associated with this argument.
  /// May be invalid if the argument is not written in the source.
  // FIXME: Is it better to return an invalid range or the range of the origin
  // expression?
  SourceRange getArgSourceRange(unsigned Index) const;

  /// \brief Returns the result type, adjusted for references.
  QualType getResultType() const;

  /// \brief Returns the value of the implicit 'this' object, or UndefinedVal if
  /// this is not a C++ member function call.
  SVal getCXXThisVal() const;

  /// \brief Returns true if any of the arguments appear to represent callbacks.
  bool hasNonZeroCallbackArg() const;

  /// \brief Returns true if any of the arguments are known to escape to long-
  /// term storage, even if this method will not modify them.
  // NOTE: The exact semantics of this are still being defined!
  // We don't really want a list of hardcoded exceptions in the long run,
  // but we don't want duplicated lists of known APIs in the short term either.
  bool argumentsMayEscape() const;

  /// \brief Returns an appropriate ProgramPoint for this call.
  ProgramPoint getProgramPoint(bool IsPreVisit = false,
                               const ProgramPointTag *Tag = 0) const;

  /// \brief Returns a new state with all argument regions invalidated.
  ///
  /// This accepts an alternate state in case some processing has already
  /// occurred.
  ProgramStateRef invalidateRegions(unsigned BlockCount,
                                    ProgramStateRef Orig = 0) const;

  /// \brief Returns true if this is a statement that can be considered for
  /// inlining.
  static bool mayBeInlined(const Stmt *S);

  // Iterator access to formal parameters and their types.
private:
  typedef std::const_mem_fun_t<QualType, ParmVarDecl> get_type_fun;
  
public:
  typedef const ParmVarDecl * const *param_iterator;

  /// Returns an iterator over the call's formal parameters.
  ///
  /// If UseDefinitionParams is set, this will return the parameter decls
  /// used in the callee's definition (suitable for inlining). Most of the
  /// time it is better to use the decl found by name lookup, which likely
  /// carries more annotations.
  ///
  /// Remember that the number of formal parameters may not match the number
  /// of arguments for all calls. However, the first parameter will always
  /// correspond with the argument value returned by \c getArgSVal(0).
  ///
  /// If the call has no accessible declaration (or definition, if
  /// \p UseDefinitionParams is set), \c param_begin() will be equal to
  /// \c param_end().
  param_iterator param_begin(bool UseDefinitionParams = false) const;
  /// \sa param_begin()
  param_iterator param_end(bool UseDefinitionParams = false) const;

  typedef llvm::mapped_iterator<param_iterator, get_type_fun>
    param_type_iterator;

  /// Returns an iterator over the types of the call's formal parameters.
  ///
  /// This uses the callee decl found by default name lookup rather than the
  /// definition because it represents a public interface, and probably has
  /// more annotations.
  param_type_iterator param_type_begin() const {
    return llvm::map_iterator(param_begin(),
                              get_type_fun(&ParmVarDecl::getType));
  }
  /// \sa param_type_begin()
  param_type_iterator param_type_end() const {
    return llvm::map_iterator(param_end(), get_type_fun(&ParmVarDecl::getType));
  }

  // For debugging purposes only
  void dump(raw_ostream &Out) const;
  LLVM_ATTRIBUTE_USED void dump() const { dump(llvm::errs()); }

  static bool classof(const CallEvent *) { return true; }
};


/// \brief Represents a call to any sort of function that might have a
/// FunctionDecl.
class AnyFunctionCall : public CallEvent {
  friend class CallEvent;

protected:
  AnyFunctionCall(const Expr *E, ProgramStateRef St,
                  const LocationContext *LCtx, Kind K)
    : CallEvent(E, St, LCtx, K) {}
  AnyFunctionCall(const Decl *D, ProgramStateRef St,
                  const LocationContext *LCtx, Kind K)
    : CallEvent(D, St, LCtx, K) {}

  // Most function calls have no extra invalidated regions.
  void getExtraInvalidatedRegions(RegionList &Regions) const {}

  QualType getDeclaredResultType() const;

public:
  // This function is overridden by subclasses, but they must return
  // a FunctionDecl.
  const FunctionDecl *getDecl() const {
    return cast_or_null<FunctionDecl>(CallEvent::getDecl());
  }

  const Decl *getDefinition(bool &IsDynamicDispatch) const {
    IsDynamicDispatch = false;
    const FunctionDecl *FD = getDecl();
    // Note that hasBody() will fill FD with the definition FunctionDecl.
    if (FD && FD->hasBody(FD))
      return FD;
    return 0;
  }

  bool argumentsMayEscape() const;

  SVal getArgSVal(unsigned Index) const;
  SourceRange getArgSourceRange(unsigned Index) const;

  param_iterator param_begin(bool UseDefinitionParams = false) const;
  param_iterator param_end(bool UseDefinitionParams = false) const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() >= CE_BEG_FUNCTION_CALLS &&
           CA->getKind() <= CE_END_FUNCTION_CALLS;
  }
};

/// \brief Represents a call to a written as a CallExpr.
class SimpleCall : public AnyFunctionCall {
protected:
  SimpleCall(const CallExpr *CE, ProgramStateRef St,
             const LocationContext *LCtx, Kind K)
    : AnyFunctionCall(CE, St, LCtx, K) {
  }

public:
  const CallExpr *getOriginExpr() const {
    return cast<CallExpr>(AnyFunctionCall::getOriginExpr());
  }

  const FunctionDecl *getDecl() const;

  unsigned getNumArgs() const { return getOriginExpr()->getNumArgs(); }
  SourceRange getSourceRange() const {
    return getOriginExpr()->getSourceRange();
  }
  
  const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() >= CE_BEG_SIMPLE_CALLS &&
           CA->getKind() <= CE_END_SIMPLE_CALLS;
  }
};

/// \brief Represents a C function or static C++ member function call.
///
/// Example: \c fun()
class FunctionCall : public SimpleCall {
public:
  FunctionCall(const CallExpr *CE, ProgramStateRef St,
               const LocationContext *LCtx)
    : SimpleCall(CE, St, LCtx, CE_Function) {}

  SVal getCXXThisVal() const { return UndefinedVal(); }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_Function;
  }
};

/// \brief Represents a non-static C++ member function call, no matter how
/// it is written.
class CXXInstanceCall : public SimpleCall {
  friend class CallEvent;

protected:
  void getExtraInvalidatedRegions(RegionList &Regions) const;

  CXXInstanceCall(const CallExpr *CE, ProgramStateRef St,
                  const LocationContext *LCtx, Kind K)
    : SimpleCall(CE, St, LCtx, K) {}

public:
  const Decl *getDefinition(bool &IsDynamicDispatch) const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() >= CE_BEG_CXX_INSTANCE_CALLS &&
           CA->getKind() <= CE_END_CXX_INSTANCE_CALLS;
  }
};

/// \brief Represents a non-static C++ member function call.
///
/// Example: \c obj.fun()
class CXXMemberCall : public CXXInstanceCall {
public:
  CXXMemberCall(const CXXMemberCallExpr *CE, ProgramStateRef St,
                const LocationContext *LCtx)
    : CXXInstanceCall(CE, St, LCtx, CE_CXXMember) {}

  const CXXMemberCallExpr *getOriginExpr() const {
    return cast<CXXMemberCallExpr>(SimpleCall::getOriginExpr());
  }

  SVal getCXXThisVal() const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXMember;
  }
};

/// \brief Represents a C++ overloaded operator call where the operator is
/// implemented as a non-static member function.
///
/// Example: <tt>iter + 1</tt>
class CXXMemberOperatorCall : public CXXInstanceCall {
public:
  CXXMemberOperatorCall(const CXXOperatorCallExpr *CE, ProgramStateRef St,
                        const LocationContext *LCtx)
    : CXXInstanceCall(CE, St, LCtx, CE_CXXMemberOperator) {}

  const CXXOperatorCallExpr *getOriginExpr() const {
    return cast<CXXOperatorCallExpr>(SimpleCall::getOriginExpr());
  }

  unsigned getNumArgs() const { return getOriginExpr()->getNumArgs() - 1; }
  const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index + 1);
  }

  SVal getCXXThisVal() const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXMemberOperator;
  }
};

/// \brief Represents a call to a block.
///
/// Example: <tt>^{ /* ... */ }()</tt>
class BlockCall : public SimpleCall {
  friend class CallEvent;

protected:
  void getExtraInvalidatedRegions(RegionList &Regions) const;

  QualType getDeclaredResultType() const;

public:
  BlockCall(const CallExpr *CE, ProgramStateRef St,
            const LocationContext *LCtx)
    : SimpleCall(CE, St, LCtx, CE_Block) {}

  /// \brief Returns the region associated with this instance of the block.
  ///
  /// This may be NULL if the block's origin is unknown.
  const BlockDataRegion *getBlockRegion() const;

  /// \brief Gets the declaration of the block.
  ///
  /// This is not an override of getDecl() because AnyFunctionCall has already
  /// assumed that it's a FunctionDecl.
  const BlockDecl *getBlockDecl() const {
    const BlockDataRegion *BR = getBlockRegion();
    if (!BR)
      return 0;
    return BR->getDecl();
  }

  const Decl *getDefinition(bool &IsDynamicDispatch) const {
    IsDynamicDispatch = false;
    return getBlockDecl();
  }

  param_iterator param_begin(bool UseDefinitionParams = false) const;
  param_iterator param_end(bool UseDefinitionParams = false) const;

  SVal getCXXThisVal() const { return UndefinedVal(); }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_Block;
  }
};

/// \brief Represents a call to a C++ constructor.
///
/// Example: \c T(1)
class CXXConstructorCall : public AnyFunctionCall {
  friend class CallEvent;

protected:
  void getExtraInvalidatedRegions(RegionList &Regions) const;

public:
  /// Represents a constructor call to a new or unknown region.
  CXXConstructorCall(const CXXConstructExpr *CE, ProgramStateRef St,
                     const LocationContext *LCtx)
    : AnyFunctionCall(CE, St, LCtx, CE_CXXConstructor) {
    Data = 0;
  }

  /// Represents a constructor call on an existing object region.
  CXXConstructorCall(const CXXConstructExpr *CE, const MemRegion *target,
                     ProgramStateRef St, const LocationContext *LCtx)
    : AnyFunctionCall(CE, St, LCtx, CE_CXXConstructor) {
    Data = target;
  }

  const CXXConstructExpr *getOriginExpr() const {
    return cast<CXXConstructExpr>(AnyFunctionCall::getOriginExpr());
  }

  SourceRange getSourceRange() const {
    return getOriginExpr()->getSourceRange();
  }

  const CXXConstructorDecl *getDecl() const {
    return getOriginExpr()->getConstructor();
  }

  unsigned getNumArgs() const { return getOriginExpr()->getNumArgs(); }

  const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  SVal getCXXThisVal() const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXConstructor;
  }
};

/// \brief Represents an implicit call to a C++ destructor.
///
/// This can occur at the end of a scope (for automatic objects), at the end
/// of a full-expression (for temporaries), or as part of a delete.
class CXXDestructorCall : public AnyFunctionCall {
  friend class CallEvent;

protected:
  void getExtraInvalidatedRegions(RegionList &Regions) const;

public:
  /// Creates an implicit destructor.
  ///
  /// \param DD The destructor that will be called.
  /// \param Trigger The statement whose completion causes this destructor call.
  /// \param Target The object region to be destructed.
  /// \param St The path-sensitive state at this point in the program.
  /// \param LCtx The location context at this point in the program.
  CXXDestructorCall(const CXXDestructorDecl *DD, const Stmt *Trigger,
                    const MemRegion *Target, ProgramStateRef St,
                    const LocationContext *LCtx)
    : AnyFunctionCall(DD, St, LCtx, CE_CXXDestructor) {
    Data = Target;
    Location = Trigger->getLocEnd();
  }

  SourceRange getSourceRange() const { return Location; }
  unsigned getNumArgs() const { return 0; }

  SVal getCXXThisVal() const;
  const Decl *getDefinition(bool &IsDynamicDispatch) const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXDestructor;
  }
};

/// \brief Represents the memory allocation call in a C++ new-expression.
///
/// This is a call to "operator new".
class CXXAllocatorCall : public AnyFunctionCall {
public:
  CXXAllocatorCall(const CXXNewExpr *E, ProgramStateRef St,
                   const LocationContext *LCtx)
    : AnyFunctionCall(E, St, LCtx, CE_CXXAllocator) {}

  const CXXNewExpr *getOriginExpr() const {
    return cast<CXXNewExpr>(AnyFunctionCall::getOriginExpr());
  }

  // FIXME: This isn't exactly the range of the allocator...
  SourceRange getSourceRange() const {
    return getOriginExpr()->getSourceRange();
  }

  const FunctionDecl *getDecl() const {
    return getOriginExpr()->getOperatorNew();
  }

  unsigned getNumArgs() const {
    return getOriginExpr()->getNumPlacementArgs() + 1;
  }

  const Expr *getArgExpr(unsigned Index) const {
    // The first argument of an allocator call is the size of the allocation.
    if (Index == 0)
      return 0;
    return getOriginExpr()->getPlacementArg(Index - 1);
  }

  SVal getCXXThisVal() const { return UndefinedVal(); }

  static bool classof(const CallEvent *CE) {
    return CE->getKind() == CE_CXXAllocator;
  }
};

/// \brief Represents the ways an Objective-C message send can occur.
//
// Note to maintainers: OCM_Message should always be last, since it does not
// need to fit in the Data field's low bits.
enum ObjCMessageKind {
  OCM_PropertyAccess,
  OCM_Subscript,
  OCM_Message
};

/// \brief Represents any expression that calls an Objective-C method.
///
/// This includes all of the kinds listed in ObjCMessageKind.
class ObjCMethodCall : public CallEvent {
  friend class CallEvent;

  const PseudoObjectExpr *getContainingPseudoObjectExpr() const;

protected:
  void getExtraInvalidatedRegions(RegionList &Regions) const;

  QualType getDeclaredResultType() const;

public:
  ObjCMethodCall(const ObjCMessageExpr *Msg, ProgramStateRef St,
                 const LocationContext *LCtx)
    : CallEvent(Msg, St, LCtx, CE_ObjCMessage) {
    Data = 0;
  }

  const ObjCMessageExpr *getOriginExpr() const {
    return cast<ObjCMessageExpr>(CallEvent::getOriginExpr());
  }
  const ObjCMethodDecl *getDecl() const {
    return getOriginExpr()->getMethodDecl();
  }
  unsigned getNumArgs() const {
    return getOriginExpr()->getNumArgs();
  }
  const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  bool isInstanceMessage() const {
    return getOriginExpr()->isInstanceMessage();
  }
  ObjCMethodFamily getMethodFamily() const {
    return getOriginExpr()->getMethodFamily();
  }
  Selector getSelector() const {
    return getOriginExpr()->getSelector();
  }

  SourceRange getSourceRange() const;

  /// \brief Returns the value of the receiver at the time of this call.
  SVal getReceiverSVal() const;

  /// \brief Get the interface for the receiver.
  ///
  /// This works whether this is an instance message or a class message.
  /// However, it currently just uses the static type of the receiver.
  const ObjCInterfaceDecl *getReceiverInterface() const {
    return getOriginExpr()->getReceiverInterface();
  }

  ObjCMessageKind getMessageKind() const;

  bool isSetter() const {
    switch (getMessageKind()) {
    case OCM_Message:
      llvm_unreachable("This is not a pseudo-object access!");
    case OCM_PropertyAccess:
      return getNumArgs() > 0;
    case OCM_Subscript:
      return getNumArgs() > 1;
    }
    llvm_unreachable("Unknown message kind");
  }

  const Decl *getDefinition(bool &IsDynamicDispatch) const {
    IsDynamicDispatch = true;
    
    const ObjCMethodDecl *MD = getDecl();
    if (!MD)
      return 0;

    for (Decl::redecl_iterator I = MD->redecls_begin(), E = MD->redecls_end();
         I != E; ++I) {
      if (cast<ObjCMethodDecl>(*I)->isThisDeclarationADefinition())
        return *I;
    }
    return 0;
  }

  SVal getCXXThisVal() const { return UndefinedVal(); }

  bool argumentsMayEscape() const {
    return hasNonZeroCallbackArg();
  }
  
  SVal getArgSVal(unsigned Index) const { return getSVal(getArgExpr(Index)); }
  SourceRange getArgSourceRange(unsigned Index) const {
    return getArgExpr(Index)->getSourceRange();
  }

  param_iterator param_begin(bool UseDefinitionParams = false) const;
  param_iterator param_end(bool UseDefinitionParams = false) const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_ObjCMessage;
  }
};


// FIXME: Use a .def or .td file for this.
#define DISPATCH(fn) \
  switch (getKind()) { \
  case CE_Function: \
    return cast<FunctionCall>(this)->fn(); \
  case CE_CXXMember: \
    return cast<CXXMemberCall>(this)->fn(); \
  case CE_CXXMemberOperator: \
    return cast<CXXMemberOperatorCall>(this)->fn(); \
  case CE_Block: \
    return cast<BlockCall>(this)->fn(); \
  case CE_CXXConstructor: \
    return cast<CXXConstructorCall>(this)->fn(); \
  case CE_CXXDestructor: \
    return cast<CXXDestructorCall>(this)->fn(); \
  case CE_CXXAllocator: \
    return cast<CXXAllocatorCall>(this)->fn(); \
  case CE_ObjCMessage: \
    return cast<ObjCMethodCall>(this)->fn(); \
  } \
  llvm_unreachable("unknown CallEvent kind");

#define DISPATCH_ARG(fn, arg) \
  switch (getKind()) { \
  case CE_Function: \
    return cast<FunctionCall>(this)->fn(arg); \
  case CE_CXXMember: \
    return cast<CXXMemberCall>(this)->fn(arg); \
  case CE_CXXMemberOperator: \
    return cast<CXXMemberOperatorCall>(this)->fn(arg); \
  case CE_Block: \
    return cast<BlockCall>(this)->fn(arg); \
  case CE_CXXConstructor: \
    return cast<CXXConstructorCall>(this)->fn(arg); \
  case CE_CXXDestructor: \
    return cast<CXXDestructorCall>(this)->fn(arg); \
  case CE_CXXAllocator: \
    return cast<CXXAllocatorCall>(this)->fn(arg); \
  case CE_ObjCMessage: \
    return cast<ObjCMethodCall>(this)->fn(arg); \
  } \
  llvm_unreachable("unknown CallEvent kind");

inline void CallEvent::getExtraInvalidatedRegions(RegionList &Regions) const {
  DISPATCH_ARG(getExtraInvalidatedRegions, Regions);
}

inline QualType CallEvent::getDeclaredResultType() const {
  DISPATCH(getDeclaredResultType);
}

inline const Decl *CallEvent::getDecl() const {
  if (const Decl *D = Origin.dyn_cast<const Decl *>())
    return D;
  DISPATCH(getDecl);
}

inline const Decl *CallEvent::getDefinition(bool &IsDynamicDispatch) const {
  DISPATCH_ARG(getDefinition, IsDynamicDispatch);
}

inline unsigned CallEvent::getNumArgs() const {
  DISPATCH(getNumArgs);
}

inline SourceRange CallEvent::getSourceRange() const {
  DISPATCH(getSourceRange);
}

inline SVal CallEvent::getArgSVal(unsigned Index) const {
  DISPATCH_ARG(getArgSVal, Index);
}

inline const Expr *CallEvent::getArgExpr(unsigned Index) const {
  DISPATCH_ARG(getArgExpr, Index);
}

inline SourceRange CallEvent::getArgSourceRange(unsigned Index) const {
  DISPATCH_ARG(getArgSourceRange, Index);
}

inline SVal CallEvent::getCXXThisVal() const {
  DISPATCH(getCXXThisVal);
}


inline bool CallEvent::argumentsMayEscape() const {
  DISPATCH(argumentsMayEscape);
}

inline CallEvent::param_iterator
CallEvent::param_begin(bool UseDefinitionParams) const {
  DISPATCH_ARG(param_begin, UseDefinitionParams);
}

inline CallEvent::param_iterator
CallEvent::param_end(bool UseDefinitionParams) const {
  DISPATCH_ARG(param_end, UseDefinitionParams);
}

#undef DISPATCH
#undef DISPATCH_ARG

} // end namespace ento
} // end namespace clang

#endif
