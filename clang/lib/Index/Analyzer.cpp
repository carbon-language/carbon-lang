//===--- Analyzer.cpp - Analysis for indexing information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Analyzer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Analyzer.h"
#include "clang/Index/Entity.h"
#include "clang/Index/TranslationUnit.h"
#include "clang/Index/Handlers.h"
#include "clang/Index/ASTLocation.h"
#include "clang/Index/GlobalSelector.h"
#include "clang/Index/DeclReferenceMap.h"
#include "clang/Index/SelectorMap.h"
#include "clang/Index/IndexProvider.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace idx;

namespace  {

//===----------------------------------------------------------------------===//
// DeclEntityAnalyzer Implementation
//===----------------------------------------------------------------------===//

class VISIBILITY_HIDDEN DeclEntityAnalyzer : public TranslationUnitHandler {
  Entity Ent;
  TULocationHandler &TULocHandler;

public:
  DeclEntityAnalyzer(Entity ent, TULocationHandler &handler)
    : Ent(ent), TULocHandler(handler) { }

  virtual void Handle(TranslationUnit *TU) {
    assert(TU && "Passed null translation unit");

    Decl *D = Ent.getDecl(TU->getASTContext());
    assert(D && "Couldn't resolve Entity");

    for (Decl::redecl_iterator I = D->redecls_begin(),
                               E = D->redecls_end(); I != E; ++I)
      TULocHandler.Handle(TULocation(TU, ASTLocation(*I)));
  }
};

//===----------------------------------------------------------------------===//
// RefEntityAnalyzer Implementation
//===----------------------------------------------------------------------===//

class VISIBILITY_HIDDEN RefEntityAnalyzer : public TranslationUnitHandler {
  Entity Ent;
  TULocationHandler &TULocHandler;

public:
  RefEntityAnalyzer(Entity ent, TULocationHandler &handler)
    : Ent(ent), TULocHandler(handler) { }

  virtual void Handle(TranslationUnit *TU) {
    assert(TU && "Passed null translation unit");

    Decl *D = Ent.getDecl(TU->getASTContext());
    assert(D && "Couldn't resolve Entity");
    NamedDecl *ND = dyn_cast<NamedDecl>(D);
    if (!ND)
      return;

    DeclReferenceMap &RefMap = TU->getDeclReferenceMap();
    for (DeclReferenceMap::astlocation_iterator
           I = RefMap.refs_begin(ND), E = RefMap.refs_end(ND); I != E; ++I)
      TULocHandler.Handle(TULocation(TU, *I));
  }
};

//===----------------------------------------------------------------------===//
// RefSelectorAnalyzer Implementation
//===----------------------------------------------------------------------===//

/// \brief Accepts an ObjC method and finds all message expressions that this
/// method may respond to.
class VISIBILITY_HIDDEN RefSelectorAnalyzer : public TranslationUnitHandler {
  Program &Prog;
  TULocationHandler &TULocHandler;

  // The original ObjCInterface associated with the method.
  Entity IFaceEnt;
  GlobalSelector GlobSel;
  bool IsInstanceMethod;

  /// \brief Super classes of the ObjCInterface.
  typedef llvm::SmallSet<Entity, 16> EntitiesSetTy;
  EntitiesSetTy HierarchyEntities;

public:
  RefSelectorAnalyzer(ObjCMethodDecl *MD,
                      Program &prog, TULocationHandler &handler)
    : Prog(prog), TULocHandler(handler) {
    assert(MD);

    // FIXME: Protocol methods.
    assert(!isa<ObjCProtocolDecl>(MD->getDeclContext()) &&
           "Protocol methods not supported yet");

    ObjCInterfaceDecl *IFD = MD->getClassInterface();
    assert(IFD);
    IFaceEnt = Entity::get(IFD, Prog);
    GlobSel = GlobalSelector::get(MD->getSelector(), Prog);
    IsInstanceMethod = MD->isInstanceMethod();

    for (ObjCInterfaceDecl *Cls = IFD->getSuperClass();
           Cls; Cls = Cls->getSuperClass())
      HierarchyEntities.insert(Entity::get(Cls, Prog));
  }

  virtual void Handle(TranslationUnit *TU) {
    assert(TU && "Passed null translation unit");

    ASTContext &Ctx = TU->getASTContext();
    // Null means it doesn't exist in this translation unit.
    ObjCInterfaceDecl *IFace =
        cast_or_null<ObjCInterfaceDecl>(IFaceEnt.getDecl(Ctx));
    Selector Sel = GlobSel.getSelector(Ctx);

    SelectorMap &SelMap = TU->getSelectorMap();
    for (SelectorMap::astlocation_iterator
           I = SelMap.refs_begin(Sel), E = SelMap.refs_end(Sel); I != E; ++I) {
      if (ValidReference(*I, IFace))
        TULocHandler.Handle(TULocation(TU, *I));
    }
  }

  /// \brief Determines whether the given message expression is likely to end
  /// up at the given interface decl.
  ///
  /// It returns true "eagerly", meaning it will return false only if it can
  /// "prove" statically that the interface cannot accept this message.
  bool ValidReference(ASTLocation ASTLoc, ObjCInterfaceDecl *IFace) {
    assert(ASTLoc.isValid());
    assert(ASTLoc.isStmt());

    // FIXME: Finding @selector references should be through another Analyzer
    // method, like FindSelectors.
    if (isa<ObjCSelectorExpr>(ASTLoc.getStmt()))
      return false;

    ObjCInterfaceDecl *MsgD = 0;
    ObjCMessageExpr *Msg = cast<ObjCMessageExpr>(ASTLoc.getStmt());

    if (Msg->getReceiver()) {
      const ObjCObjectPointerType *OPT =
          Msg->getReceiver()->getType()->getAsObjCInterfacePointerType();

      // Can be anything! Accept it as a possibility..
      if (!OPT || OPT->isObjCIdType() || OPT->isObjCQualifiedIdType())
        return true;

      // Expecting class method.
      if (OPT->isObjCClassType() || OPT->isObjCQualifiedClassType())
        return !IsInstanceMethod;

      MsgD = OPT->getInterfaceDecl();
      assert(MsgD);

      // Should be an instance method.
      if (!IsInstanceMethod)
        return false;

    } else {
      // Expecting class method.
      if (IsInstanceMethod)
        return false;

      MsgD = Msg->getClassInfo().first;
      // FIXME: Case when we only have an identifier.
      assert(MsgD && "Identifier only");
    }

    assert(MsgD);

    // Same interface ? We have a winner!
    if (MsgD == IFace)
      return true;

    // If the message interface is a superclass of the original interface,
    // accept this message as a possibility.
    if (HierarchyEntities.count(Entity::get(MsgD, Prog)))
      return true;

    // If the message interface is a subclass of the original interface, accept
    // the message unless there is a subclass in the hierarchy that will
    // "steal" the message (thus the message "will go" to the subclass and not
    /// the original interface).
    if (IFace) {
      Selector Sel = Msg->getSelector();
      for (ObjCInterfaceDecl *Cls = MsgD; Cls; Cls = Cls->getSuperClass()) {
        if (Cls == IFace)
          return true;
        if (Cls->getMethod(Sel, IsInstanceMethod))
          return false;
      }
    }

    // The interfaces are unrelated, don't accept the message.
    return false;
  }
};

//===----------------------------------------------------------------------===//
// MessageAnalyzer Implementation
//===----------------------------------------------------------------------===//

/// \brief Accepts an ObjC message expression and finds all methods that may
/// respond to it.
class VISIBILITY_HIDDEN MessageAnalyzer : public TranslationUnitHandler {
  Program &Prog;
  TULocationHandler &TULocHandler;

  // The ObjCInterface associated with the message. Can be null/invalid.
  Entity MsgIFaceEnt;
  GlobalSelector GlobSel;
  bool CanBeInstanceMethod;
  bool CanBeClassMethod;

  /// \brief Super classes of the ObjCInterface.
  typedef llvm::SmallSet<Entity, 16> EntitiesSetTy;
  EntitiesSetTy HierarchyEntities;

  /// \brief The interface in the message interface hierarchy that "intercepts"
  /// the selector.
  Entity ReceiverIFaceEnt;

public:
  MessageAnalyzer(ObjCMessageExpr *Msg,
                  Program &prog, TULocationHandler &handler)
    : Prog(prog), TULocHandler(handler),
      CanBeInstanceMethod(false),
      CanBeClassMethod(false) {

    assert(Msg);

    ObjCInterfaceDecl *MsgD = 0;

    while (true) {
      if (Msg->getReceiver() == 0) {
        CanBeClassMethod = true;
        MsgD = Msg->getClassInfo().first;
        // FIXME: Case when we only have an identifier.
        assert(MsgD && "Identifier only");
        break;
      }

      const ObjCObjectPointerType *OPT =
          Msg->getReceiver()->getType()->getAsObjCInterfacePointerType();

      if (!OPT || OPT->isObjCIdType() || OPT->isObjCQualifiedIdType()) {
        CanBeInstanceMethod = CanBeClassMethod = true;
        break;
      }

      if (OPT->isObjCClassType() || OPT->isObjCQualifiedClassType()) {
        CanBeClassMethod = true;
        break;
      }

      MsgD = OPT->getInterfaceDecl();
      assert(MsgD);
      CanBeInstanceMethod = true;
      break;
    }

    assert(CanBeInstanceMethod || CanBeClassMethod);

    Selector sel = Msg->getSelector();
    assert(!sel.isNull());

    MsgIFaceEnt = Entity::get(MsgD, Prog);
    GlobSel = GlobalSelector::get(sel, Prog);

    if (MsgD) {
      for (ObjCInterfaceDecl *Cls = MsgD->getSuperClass();
             Cls; Cls = Cls->getSuperClass())
        HierarchyEntities.insert(Entity::get(Cls, Prog));

      // Find the interface in the hierarchy that "receives" the message.
      for (ObjCInterfaceDecl *Cls = MsgD; Cls; Cls = Cls->getSuperClass()) {
        bool isReceiver = false;

        ObjCInterfaceDecl::lookup_const_iterator Meth, MethEnd;
        for (llvm::tie(Meth, MethEnd) = Cls->lookup(sel);
               Meth != MethEnd; ++Meth) {
          if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(*Meth))
            if ((MD->isInstanceMethod() && CanBeInstanceMethod) ||
                (MD->isClassMethod()    && CanBeClassMethod)) {
              isReceiver = true;
              break;
            }
        }

        if (isReceiver) {
          ReceiverIFaceEnt = Entity::get(Cls, Prog);
          break;
        }
      }
    }
  }

  virtual void Handle(TranslationUnit *TU) {
    assert(TU && "Passed null translation unit");
    ASTContext &Ctx = TU->getASTContext();

    // Null means it doesn't exist in this translation unit or there was no
    // interface that was determined to receive the original message.
    ObjCInterfaceDecl *ReceiverIFace =
        cast_or_null<ObjCInterfaceDecl>(ReceiverIFaceEnt.getDecl(Ctx));

    // No subclass for the original receiver interface, so it remains the
    // receiver.
    if (ReceiverIFaceEnt.isValid() && ReceiverIFace == 0)
      return;

    // Null means it doesn't exist in this translation unit or there was no
    // interface associated with the message in the first place.
    ObjCInterfaceDecl *MsgIFace =
        cast_or_null<ObjCInterfaceDecl>(MsgIFaceEnt.getDecl(Ctx));

    Selector Sel = GlobSel.getSelector(Ctx);
    SelectorMap &SelMap = TU->getSelectorMap();
    for (SelectorMap::method_iterator
           I = SelMap.methods_begin(Sel), E = SelMap.methods_end(Sel);
           I != E; ++I) {
      ObjCMethodDecl *D = *I;
      if (ValidMethod(D, MsgIFace, ReceiverIFace)) {
        for (ObjCMethodDecl::redecl_iterator
               RI = D->redecls_begin(), RE = D->redecls_end(); RI != RE; ++RI)
          TULocHandler.Handle(TULocation(TU, ASTLocation(*RI)));
      }
    }
  }

  /// \brief Determines whether the given method is likely to accept the
  /// original message.
  ///
  /// It returns true "eagerly", meaning it will return false only if it can
  /// "prove" statically that the method cannot accept the original message.
  bool ValidMethod(ObjCMethodDecl *D, ObjCInterfaceDecl *MsgIFace,
                   ObjCInterfaceDecl *ReceiverIFace) {
    assert(D);

    // FIXME: Protocol methods ?
    if (isa<ObjCProtocolDecl>(D->getDeclContext()))
      return false;

    // No specific interface associated with the message. Can be anything.
    if (MsgIFaceEnt.isInvalid())
      return true;

    if ((!CanBeInstanceMethod && D->isInstanceMethod()) ||
        (!CanBeClassMethod    && D->isClassMethod()))
      return false;

    ObjCInterfaceDecl *IFace = D->getClassInterface();
    assert(IFace);

    // If the original message interface is the same or a superclass of the
    // given interface, accept the method as a possibility.
    if (MsgIFace && MsgIFace->isSuperClassOf(IFace))
      return true;

    if (ReceiverIFace) {
      // The given interface, "overrides" the receiver.
      if (ReceiverIFace->isSuperClassOf(IFace))
        return true;
    } else {
      // No receiver was found for the original message.
      assert(ReceiverIFaceEnt.isInvalid());

      // If the original message interface is a subclass of the given interface,
      // accept the message.
      if (HierarchyEntities.count(Entity::get(IFace, Prog)))
        return true;
    }

    // The interfaces are unrelated, or the receiver interface wasn't
    // "overriden".
    return false;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Analyzer Implementation
//===----------------------------------------------------------------------===//

void Analyzer::FindDeclarations(Decl *D, TULocationHandler &Handler) {
  assert(D && "Passed null declaration");
  Entity Ent = Entity::get(D, Prog);
  if (Ent.isInvalid())
    return;

  DeclEntityAnalyzer DEA(Ent, Handler);
  Idxer.GetTranslationUnitsFor(Ent, DEA);
}

void Analyzer::FindReferences(Decl *D, TULocationHandler &Handler) {
  assert(D && "Passed null declaration");
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    RefSelectorAnalyzer RSA(MD, Prog, Handler);
    GlobalSelector Sel = GlobalSelector::get(MD->getSelector(), Prog);
    Idxer.GetTranslationUnitsFor(Sel, RSA);
    return;
  }

  Entity Ent = Entity::get(D, Prog);
  if (Ent.isInvalid())
    return;

  RefEntityAnalyzer REA(Ent, Handler);
  Idxer.GetTranslationUnitsFor(Ent, REA);
}

/// \brief Find methods that may respond to the given message and pass them
/// to Handler.
void Analyzer::FindObjCMethods(ObjCMessageExpr *Msg,
                               TULocationHandler &Handler) {
  assert(Msg);
  MessageAnalyzer MsgAnalyz(Msg, Prog, Handler);
  GlobalSelector GlobSel = GlobalSelector::get(Msg->getSelector(), Prog);
  Idxer.GetTranslationUnitsFor(GlobSel, MsgAnalyz);
}
