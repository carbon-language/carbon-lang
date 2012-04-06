//=-- ExprEngineObjC.cpp - ExprEngine support for Objective-C ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ExprEngine's support for Objective-C expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtObjC.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"

using namespace clang;
using namespace ento;

void ExprEngine::VisitLvalObjCIvarRefExpr(const ObjCIvarRefExpr *Ex, 
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();
  SVal baseVal = state->getSVal(Ex->getBase(), LCtx);
  SVal location = state->getLValue(Ex->getDecl(), baseVal);
  
  ExplodedNodeSet dstIvar;
  StmtNodeBuilder Bldr(Pred, dstIvar, *currentBuilderContext);
  Bldr.generateNode(Ex, Pred, state->BindExpr(Ex, LCtx, location));
  
  // Perform the post-condition check of the ObjCIvarRefExpr and store
  // the created nodes in 'Dst'.
  getCheckerManager().runCheckersForPostStmt(Dst, dstIvar, Ex, *this);
}

void ExprEngine::VisitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt *S,
                                             ExplodedNode *Pred,
                                             ExplodedNodeSet &Dst) {
  getCheckerManager().runCheckersForPreStmt(Dst, Pred, S, *this);
}

void ExprEngine::VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S,
                                            ExplodedNode *Pred,
                                            ExplodedNodeSet &Dst) {
  
  // ObjCForCollectionStmts are processed in two places.  This method
  // handles the case where an ObjCForCollectionStmt* occurs as one of the
  // statements within a basic block.  This transfer function does two things:
  //
  //  (1) binds the next container value to 'element'.  This creates a new
  //      node in the ExplodedGraph.
  //
  //  (2) binds the value 0/1 to the ObjCForCollectionStmt* itself, indicating
  //      whether or not the container has any more elements.  This value
  //      will be tested in ProcessBranch.  We need to explicitly bind
  //      this value because a container can contain nil elements.
  //
  // FIXME: Eventually this logic should actually do dispatches to
  //   'countByEnumeratingWithState:objects:count:' (NSFastEnumeration).
  //   This will require simulating a temporary NSFastEnumerationState, either
  //   through an SVal or through the use of MemRegions.  This value can
  //   be affixed to the ObjCForCollectionStmt* instead of 0/1; when the loop
  //   terminates we reclaim the temporary (it goes out of scope) and we
  //   we can test if the SVal is 0 or if the MemRegion is null (depending
  //   on what approach we take).
  //
  //  For now: simulate (1) by assigning either a symbol or nil if the
  //    container is empty.  Thus this transfer function will by default
  //    result in state splitting.

  const Stmt *elem = S->getElement();
  ProgramStateRef state = Pred->getState();
  SVal elementV;
  StmtNodeBuilder Bldr(Pred, Dst, *currentBuilderContext);
  
  if (const DeclStmt *DS = dyn_cast<DeclStmt>(elem)) {
    const VarDecl *elemD = cast<VarDecl>(DS->getSingleDecl());
    assert(elemD->getInit() == 0);
    elementV = state->getLValue(elemD, Pred->getLocationContext());
  }
  else {
    elementV = state->getSVal(elem, Pred->getLocationContext());
  }
  
  ExplodedNodeSet dstLocation;
  Bldr.takeNodes(Pred);
  evalLocation(dstLocation, S, elem, Pred, state, elementV, NULL, false);
  Bldr.addNodes(dstLocation);
  
  for (ExplodedNodeSet::iterator NI = dstLocation.begin(),
       NE = dstLocation.end(); NI!=NE; ++NI) {
    Pred = *NI;
    ProgramStateRef state = Pred->getState();
    const LocationContext *LCtx = Pred->getLocationContext();
    
    // Handle the case where the container still has elements.
    SVal TrueV = svalBuilder.makeTruthVal(1);
    ProgramStateRef hasElems = state->BindExpr(S, LCtx, TrueV);
    
    // Handle the case where the container has no elements.
    SVal FalseV = svalBuilder.makeTruthVal(0);
    ProgramStateRef noElems = state->BindExpr(S, LCtx, FalseV);
    
    if (loc::MemRegionVal *MV = dyn_cast<loc::MemRegionVal>(&elementV))
      if (const TypedValueRegion *R = 
          dyn_cast<TypedValueRegion>(MV->getRegion())) {
        // FIXME: The proper thing to do is to really iterate over the
        //  container.  We will do this with dispatch logic to the store.
        //  For now, just 'conjure' up a symbolic value.
        QualType T = R->getValueType();
        assert(Loc::isLocType(T));
        unsigned Count = currentBuilderContext->getCurrentBlockCount();
        SymbolRef Sym = SymMgr.getConjuredSymbol(elem, LCtx, T, Count);
        SVal V = svalBuilder.makeLoc(Sym);
        hasElems = hasElems->bindLoc(elementV, V);
        
        // Bind the location to 'nil' on the false branch.
        SVal nilV = svalBuilder.makeIntVal(0, T);
        noElems = noElems->bindLoc(elementV, nilV);
      }
    
    // Create the new nodes.
    Bldr.generateNode(S, Pred, hasElems);
    Bldr.generateNode(S, Pred, noElems);
  }
}

void ExprEngine::VisitObjCMessage(const ObjCMessage &msg,
                                  ExplodedNode *Pred,
                                  ExplodedNodeSet &Dst) {
  
  // Handle the previsits checks.
  ExplodedNodeSet dstPrevisit;
  getCheckerManager().runCheckersForPreObjCMessage(dstPrevisit, Pred, 
                                                   msg, *this);
  
  // Proceed with evaluate the message expression.
  ExplodedNodeSet dstEval;
  StmtNodeBuilder Bldr(dstPrevisit, dstEval, *currentBuilderContext);

  for (ExplodedNodeSet::iterator DI = dstPrevisit.begin(),
       DE = dstPrevisit.end(); DI != DE; ++DI) {
    
    ExplodedNode *Pred = *DI;
    bool RaisesException = false;
    
    if (const Expr *Receiver = msg.getInstanceReceiver()) {
      ProgramStateRef state = Pred->getState();
      SVal recVal = state->getSVal(Receiver, Pred->getLocationContext());
      if (!recVal.isUndef()) {
        // Bifurcate the state into nil and non-nil ones.
        DefinedOrUnknownSVal receiverVal = cast<DefinedOrUnknownSVal>(recVal);
        
        ProgramStateRef notNilState, nilState;
        llvm::tie(notNilState, nilState) = state->assume(receiverVal);
        
        // There are three cases: can be nil or non-nil, must be nil, must be
        // non-nil. We ignore must be nil, and merge the rest two into non-nil.
        if (nilState && !notNilState) {
          continue;
        }
        
        // Check if the "raise" message was sent.
        assert(notNilState);
        if (msg.getSelector() == RaiseSel)
          RaisesException = true;
        
        // If we raise an exception, for now treat it as a sink.
        // Eventually we will want to handle exceptions properly.
        // Dispatch to plug-in transfer function.
        evalObjCMessage(Bldr, msg, Pred, notNilState, RaisesException);
      }
    }
    else if (const ObjCInterfaceDecl *Iface = msg.getReceiverInterface()) {
      IdentifierInfo* ClsName = Iface->getIdentifier();
      Selector S = msg.getSelector();
      
      // Check for special instance methods.
      if (!NSExceptionII) {
        ASTContext &Ctx = getContext();
        NSExceptionII = &Ctx.Idents.get("NSException");
      }
      
      if (ClsName == NSExceptionII) {
        enum { NUM_RAISE_SELECTORS = 2 };
        
        // Lazily create a cache of the selectors.
        if (!NSExceptionInstanceRaiseSelectors) {
          ASTContext &Ctx = getContext();
          NSExceptionInstanceRaiseSelectors =
          new Selector[NUM_RAISE_SELECTORS];
          SmallVector<IdentifierInfo*, NUM_RAISE_SELECTORS> II;
          unsigned idx = 0;
          
          // raise:format:
          II.push_back(&Ctx.Idents.get("raise"));
          II.push_back(&Ctx.Idents.get("format"));
          NSExceptionInstanceRaiseSelectors[idx++] =
          Ctx.Selectors.getSelector(II.size(), &II[0]);
          
          // raise:format::arguments:
          II.push_back(&Ctx.Idents.get("arguments"));
          NSExceptionInstanceRaiseSelectors[idx++] =
          Ctx.Selectors.getSelector(II.size(), &II[0]);
        }
        
        for (unsigned i = 0; i < NUM_RAISE_SELECTORS; ++i)
          if (S == NSExceptionInstanceRaiseSelectors[i]) {
            RaisesException = true;
            break;
          }
      }
      
      // If we raise an exception, for now treat it as a sink.
      // Eventually we will want to handle exceptions properly.
      // Dispatch to plug-in transfer function.
      evalObjCMessage(Bldr, msg, Pred, Pred->getState(), RaisesException);
    }
  }
  
  // Finally, perform the post-condition check of the ObjCMessageExpr and store
  // the created nodes in 'Dst'.
  getCheckerManager().runCheckersForPostObjCMessage(Dst, dstEval, msg, *this);
}

void ExprEngine::evalObjCMessage(StmtNodeBuilder &Bldr,
                                 const ObjCMessage &msg,
                                 ExplodedNode *Pred,
                                 ProgramStateRef state,
                                 bool GenSink) {
  // First handle the return value.
  SVal ReturnValue = UnknownVal();

  // Some method families have known return values.
  switch (msg.getMethodFamily()) {
  default:
    break;
  case OMF_autorelease:
  case OMF_retain:
  case OMF_self: {
    // These methods return their receivers.
    const Expr *ReceiverE = msg.getInstanceReceiver();
    if (ReceiverE)
      ReturnValue = state->getSVal(ReceiverE, Pred->getLocationContext());
    break;
  }
  }

  // If we failed to figure out the return value, use a conjured value instead.
  if (ReturnValue.isUnknown()) {
    SValBuilder &SVB = getSValBuilder();
    QualType ResultTy = msg.getResultType(getContext());
    unsigned Count = currentBuilderContext->getCurrentBlockCount();
    const Expr *CurrentE = cast<Expr>(currentStmt);
    const LocationContext *LCtx = Pred->getLocationContext();
    ReturnValue = SVB.getConjuredSymbolVal(NULL, CurrentE, LCtx, ResultTy, Count);
  }

  // Bind the return value.
  const LocationContext *LCtx = Pred->getLocationContext();
  state = state->BindExpr(currentStmt, LCtx, ReturnValue);

  // Invalidate the arguments (and the receiver)
  state = invalidateArguments(state, CallOrObjCMessage(msg, state, LCtx), LCtx);

  // And create the new node.
  Bldr.generateNode(msg.getMessageExpr(), Pred, state, GenSink);
  assert(Bldr.hasGeneratedNodes());
}

