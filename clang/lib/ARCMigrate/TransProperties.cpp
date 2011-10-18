//===--- TransProperties.cpp - Tranformations to ARC mode -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// rewriteProperties:
//
// - Adds strong/weak/unsafe_unretained ownership specifier to properties that
//   are missing one.
// - Migrates properties from (retain) to (strong) and (assign) to
//   (unsafe_unretained/weak).
// - If a property is synthesized, adds the ownership specifier in the ivar
//   backing the property.
//
//  @interface Foo : NSObject {
//      NSObject *x;
//  }
//  @property (assign) id x;
//  @end
// ---->
//  @interface Foo : NSObject {
//      NSObject *__weak x;
//  }
//  @property (weak) id x;
//  @end
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <map>

using namespace clang;
using namespace arcmt;
using namespace trans;

namespace {

class PropertiesRewriter {
  MigrationPass &Pass;
  ObjCImplementationDecl *CurImplD;
  
  enum PropActionKind {
    PropAction_None,
    PropAction_RetainToStrong,
    PropAction_RetainRemoved,
    PropAction_AssignToStrong,
    PropAction_AssignRewritten,
    PropAction_MaybeAddStrong,
    PropAction_MaybeAddWeakOrUnsafe
  };

  struct PropData {
    ObjCPropertyDecl *PropD;
    ObjCIvarDecl *IvarD;
    ObjCPropertyImplDecl *ImplD;

    PropData(ObjCPropertyDecl *propD) : PropD(propD), IvarD(0), ImplD(0) { }
  };

  typedef SmallVector<PropData, 2> PropsTy;
  typedef std::map<unsigned, PropsTy> AtPropDeclsTy;
  AtPropDeclsTy AtProps;
  llvm::DenseMap<IdentifierInfo *, PropActionKind> ActionOnProp;

public:
  PropertiesRewriter(MigrationPass &pass) : Pass(pass) { }

  static void collectProperties(ObjCContainerDecl *D, AtPropDeclsTy &AtProps) {
    for (ObjCInterfaceDecl::prop_iterator
           propI = D->prop_begin(),
           propE = D->prop_end(); propI != propE; ++propI) {
      if (propI->getAtLoc().isInvalid())
        continue;
      PropsTy &props = AtProps[propI->getAtLoc().getRawEncoding()];
      props.push_back(*propI);
    }
  }

  void doTransform(ObjCImplementationDecl *D) {
    CurImplD = D;
    ObjCInterfaceDecl *iface = D->getClassInterface();
    if (!iface)
      return;

    collectProperties(iface, AtProps);

    typedef DeclContext::specific_decl_iterator<ObjCPropertyImplDecl>
        prop_impl_iterator;
    for (prop_impl_iterator
           I = prop_impl_iterator(D->decls_begin()),
           E = prop_impl_iterator(D->decls_end()); I != E; ++I) {
      ObjCPropertyImplDecl *implD = *I;
      if (implD->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
        continue;
      ObjCPropertyDecl *propD = implD->getPropertyDecl();
      if (!propD || propD->isInvalidDecl())
        continue;
      ObjCIvarDecl *ivarD = implD->getPropertyIvarDecl();
      if (!ivarD || ivarD->isInvalidDecl())
        continue;
      unsigned rawAtLoc = propD->getAtLoc().getRawEncoding();
      AtPropDeclsTy::iterator findAtLoc = AtProps.find(rawAtLoc);
      if (findAtLoc == AtProps.end())
        continue;
      
      PropsTy &props = findAtLoc->second;
      for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
        if (I->PropD == propD) {
          I->IvarD = ivarD;
          I->ImplD = implD;
          break;
        }
      }
    }

    for (AtPropDeclsTy::iterator
           I = AtProps.begin(), E = AtProps.end(); I != E; ++I) {
      SourceLocation atLoc = SourceLocation::getFromRawEncoding(I->first);
      PropsTy &props = I->second;
      QualType ty = getPropertyType(props);
      if (!ty->isObjCRetainableType())
        continue;
      if (hasIvarWithExplicitOwnership(props))
        continue;
      
      Transaction Trans(Pass.TA);
      rewriteProperty(props, atLoc);
    }

    AtPropDeclsTy AtExtProps;
    // Look through extensions.
    for (ObjCCategoryDecl *Cat = iface->getCategoryList();
           Cat; Cat = Cat->getNextClassCategory())
      if (Cat->IsClassExtension())
        collectProperties(Cat, AtExtProps);

    for (AtPropDeclsTy::iterator
           I = AtExtProps.begin(), E = AtExtProps.end(); I != E; ++I) {
      SourceLocation atLoc = SourceLocation::getFromRawEncoding(I->first);
      PropsTy &props = I->second;
      Transaction Trans(Pass.TA);
      doActionForExtensionProp(props, atLoc);
    }
  }

private:
  void doPropAction(PropActionKind kind,
                    PropsTy &props, SourceLocation atLoc,
                    bool markAction = true) {
    if (markAction)
      for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I)
        ActionOnProp[I->PropD->getIdentifier()] = kind;

    switch (kind) {
    case PropAction_None:
      return;
    case PropAction_RetainToStrong:
      rewriteAttribute("retain", "strong", atLoc);
      return;
    case PropAction_RetainRemoved:
      removeAttribute("retain", atLoc);
      return;
    case PropAction_AssignToStrong:
      rewriteAttribute("assign", "strong", atLoc);
      return;
    case PropAction_AssignRewritten:
      return rewriteAssign(props, atLoc);
    case PropAction_MaybeAddStrong:
      return maybeAddStrongAttr(props, atLoc);
    case PropAction_MaybeAddWeakOrUnsafe:
      return maybeAddWeakOrUnsafeUnretainedAttr(props, atLoc);
    }
  }

  void doActionForExtensionProp(PropsTy &props, SourceLocation atLoc) {
    llvm::DenseMap<IdentifierInfo *, PropActionKind>::iterator I;
    I = ActionOnProp.find(props[0].PropD->getIdentifier());
    if (I == ActionOnProp.end())
      return;

    doPropAction(I->second, props, atLoc, false);
  }

  void rewriteProperty(PropsTy &props, SourceLocation atLoc) {
    ObjCPropertyDecl::PropertyAttributeKind propAttrs = getPropertyAttrs(props);
    
    if (propAttrs & (ObjCPropertyDecl::OBJC_PR_copy |
                     ObjCPropertyDecl::OBJC_PR_unsafe_unretained |
                     ObjCPropertyDecl::OBJC_PR_strong |
                     ObjCPropertyDecl::OBJC_PR_weak))
      return;

    if (propAttrs & ObjCPropertyDecl::OBJC_PR_retain) {
      if (propAttrs & ObjCPropertyDecl::OBJC_PR_readonly)
        return doPropAction(PropAction_RetainToStrong, props, atLoc);
      else
        // strong is the default.
        return doPropAction(PropAction_RetainRemoved, props, atLoc);
    }

    if (propAttrs & ObjCPropertyDecl::OBJC_PR_assign) {
      if (hasIvarAssignedAPlusOneObject(props)) {
        return doPropAction(PropAction_AssignToStrong, props, atLoc);
      }
      return doPropAction(PropAction_AssignRewritten, props, atLoc);
    }

    if (hasIvarAssignedAPlusOneObject(props))
      return doPropAction(PropAction_MaybeAddStrong, props, atLoc);

    return doPropAction(PropAction_MaybeAddWeakOrUnsafe, props, atLoc);
  }

  void rewriteAssign(PropsTy &props, SourceLocation atLoc) const {
    bool canUseWeak = canApplyWeak(Pass.Ctx, getPropertyType(props));

    bool rewroteAttr = rewriteAttribute("assign",
                                     canUseWeak ? "weak" : "unsafe_unretained",
                                         atLoc);
    if (!rewroteAttr)
      canUseWeak = false;

    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
      if (isUserDeclared(I->IvarD))
        Pass.TA.insert(I->IvarD->getLocation(),
                       canUseWeak ? "__weak " : "__unsafe_unretained ");
      if (I->ImplD)
        Pass.TA.clearDiagnostic(diag::err_arc_assign_property_ownership,
                                I->ImplD->getLocation());
    }
  }

  void maybeAddWeakOrUnsafeUnretainedAttr(PropsTy &props,
                                          SourceLocation atLoc) const {
    ObjCPropertyDecl::PropertyAttributeKind propAttrs = getPropertyAttrs(props);

    bool canUseWeak = canApplyWeak(Pass.Ctx, getPropertyType(props));
    if (!(propAttrs & ObjCPropertyDecl::OBJC_PR_readonly) ||
        !hasAllIvarsBacked(props)) {
      bool addedAttr = addAttribute(canUseWeak ? "weak" : "unsafe_unretained",
                                    atLoc);
      if (!addedAttr)
        canUseWeak = false;
    }

    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
      if (isUserDeclared(I->IvarD))
        Pass.TA.insert(I->IvarD->getLocation(),
                       canUseWeak ? "__weak " : "__unsafe_unretained ");
      if (I->ImplD) {
        Pass.TA.clearDiagnostic(diag::err_arc_assign_property_ownership,
                                I->ImplD->getLocation());
        Pass.TA.clearDiagnostic(
                           diag::err_arc_objc_property_default_assign_on_object,
                           I->ImplD->getLocation());
      }
    }
  }

  void maybeAddStrongAttr(PropsTy &props, SourceLocation atLoc) const {
    ObjCPropertyDecl::PropertyAttributeKind propAttrs = getPropertyAttrs(props);

    if (!(propAttrs & ObjCPropertyDecl::OBJC_PR_readonly) ||
        !hasAllIvarsBacked(props)) {
      addAttribute("strong", atLoc);
    }

    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
      if (I->ImplD) {
        Pass.TA.clearDiagnostic(diag::err_arc_assign_property_ownership,
                                I->ImplD->getLocation());
        Pass.TA.clearDiagnostic(
                           diag::err_arc_objc_property_default_assign_on_object,
                           I->ImplD->getLocation());
      }
    }
  }

  bool removeAttribute(StringRef fromAttr, SourceLocation atLoc) const {
    return rewriteAttribute(fromAttr, StringRef(), atLoc);
  }

  bool rewriteAttribute(StringRef fromAttr, StringRef toAttr,
                        SourceLocation atLoc) const {
    if (atLoc.isMacroID())
      return false;

    SourceManager &SM = Pass.Ctx.getSourceManager();

    // Break down the source location.
    std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(atLoc);

    // Try to load the file buffer.
    bool invalidTemp = false;
    StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
    if (invalidTemp)
      return false;

    const char *tokenBegin = file.data() + locInfo.second;

    // Lex from the start of the given location.
    Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
                Pass.Ctx.getLangOptions(),
                file.begin(), tokenBegin, file.end());
    Token tok;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::at)) return false;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::raw_identifier)) return false;
    if (StringRef(tok.getRawIdentifierData(), tok.getLength())
          != "property")
      return false;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::l_paren)) return false;
    
    Token BeforeTok = tok;
    Token AfterTok;
    AfterTok.startToken();
    SourceLocation AttrLoc;
    
    lexer.LexFromRawLexer(tok);
    if (tok.is(tok::r_paren))
      return false;

    while (1) {
      if (tok.isNot(tok::raw_identifier)) return false;
      StringRef ident(tok.getRawIdentifierData(), tok.getLength());
      if (ident == fromAttr) {
        if (!toAttr.empty()) {
          Pass.TA.replaceText(tok.getLocation(), fromAttr, toAttr);
          return true;
        }
        // We want to remove the attribute.
        AttrLoc = tok.getLocation();
      }

      do {
        lexer.LexFromRawLexer(tok);
        if (AttrLoc.isValid() && AfterTok.is(tok::unknown))
          AfterTok = tok;
      } while (tok.isNot(tok::comma) && tok.isNot(tok::r_paren));
      if (tok.is(tok::r_paren))
        break;
      if (AttrLoc.isInvalid())
        BeforeTok = tok;
      lexer.LexFromRawLexer(tok);
    }

    if (toAttr.empty() && AttrLoc.isValid() && AfterTok.isNot(tok::unknown)) {
      // We want to remove the attribute.
      if (BeforeTok.is(tok::l_paren) && AfterTok.is(tok::r_paren)) {
        Pass.TA.remove(SourceRange(BeforeTok.getLocation(),
                                   AfterTok.getLocation()));
      } else if (BeforeTok.is(tok::l_paren) && AfterTok.is(tok::comma)) {
        Pass.TA.remove(SourceRange(AttrLoc, AfterTok.getLocation()));
      } else {
        Pass.TA.remove(SourceRange(BeforeTok.getLocation(), AttrLoc));
      }

      return true;
    }
    
    return false;
  }

  bool addAttribute(StringRef attr, SourceLocation atLoc) const {
    if (atLoc.isMacroID())
      return false;

    SourceManager &SM = Pass.Ctx.getSourceManager();

    // Break down the source location.
    std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(atLoc);

    // Try to load the file buffer.
    bool invalidTemp = false;
    StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
    if (invalidTemp)
      return false;

    const char *tokenBegin = file.data() + locInfo.second;

    // Lex from the start of the given location.
    Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
                Pass.Ctx.getLangOptions(),
                file.begin(), tokenBegin, file.end());
    Token tok;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::at)) return false;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::raw_identifier)) return false;
    if (StringRef(tok.getRawIdentifierData(), tok.getLength())
          != "property")
      return false;
    lexer.LexFromRawLexer(tok);

    if (tok.isNot(tok::l_paren)) {
      Pass.TA.insert(tok.getLocation(), std::string("(") + attr.str() + ") ");
      return true;
    }
    
    lexer.LexFromRawLexer(tok);
    if (tok.is(tok::r_paren)) {
      Pass.TA.insert(tok.getLocation(), attr);
      return true;
    }

    if (tok.isNot(tok::raw_identifier)) return false;

    Pass.TA.insert(tok.getLocation(), std::string(attr) + ", ");
    return true;
  }

  class PlusOneAssign : public RecursiveASTVisitor<PlusOneAssign> {
    ObjCIvarDecl *Ivar;
  public:
    PlusOneAssign(ObjCIvarDecl *D) : Ivar(D) {}

    bool VisitBinAssign(BinaryOperator *E) {
      Expr *lhs = E->getLHS()->IgnoreParenImpCasts();
      if (ObjCIvarRefExpr *RE = dyn_cast<ObjCIvarRefExpr>(lhs)) {
        if (RE->getDecl() != Ivar)
          return true;

      if (ObjCMessageExpr *
            ME = dyn_cast<ObjCMessageExpr>(E->getRHS()->IgnoreParenCasts()))
        if (ME->getMethodFamily() == OMF_retain)
          return false;

      ImplicitCastExpr *implCE = dyn_cast<ImplicitCastExpr>(E->getRHS());
      while (implCE && implCE->getCastKind() ==  CK_BitCast)
        implCE = dyn_cast<ImplicitCastExpr>(implCE->getSubExpr());

      if (implCE && implCE->getCastKind() == CK_ARCConsumeObject)
        return false;
      }

      return true;
    }
  };

  bool hasIvarAssignedAPlusOneObject(PropsTy &props) const {
    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
      PlusOneAssign oneAssign(I->IvarD);
      bool notFound = oneAssign.TraverseDecl(CurImplD);
      if (!notFound)
        return true;
    }

    return false;
  }

  bool hasIvarWithExplicitOwnership(PropsTy &props) const {
    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
      if (isUserDeclared(I->IvarD)) {
        if (isa<AttributedType>(I->IvarD->getType()))
          return true;
        if (I->IvarD->getType().getLocalQualifiers().getObjCLifetime()
              != Qualifiers::OCL_Strong)
          return true;
      }
    }

    return false;    
  }

  bool hasAllIvarsBacked(PropsTy &props) const {
    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I)
      if (!isUserDeclared(I->IvarD))
        return false;

    return true;
  }

  bool isUserDeclared(ObjCIvarDecl *ivarD) const {
    return ivarD && !ivarD->getSynthesize();
  }

  QualType getPropertyType(PropsTy &props) const {
    assert(!props.empty());
    QualType ty = props[0].PropD->getType();

#ifndef NDEBUG
    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I)
      assert(ty == I->PropD->getType());
#endif

    return ty;
  }

  ObjCPropertyDecl::PropertyAttributeKind
  getPropertyAttrs(PropsTy &props) const {
    assert(!props.empty());
    ObjCPropertyDecl::PropertyAttributeKind
      attrs = props[0].PropD->getPropertyAttributesAsWritten();

#ifndef NDEBUG
    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I)
      assert(attrs == I->PropD->getPropertyAttributesAsWritten());
#endif

    return attrs;
  }
};

class ImplementationChecker :
                             public RecursiveASTVisitor<ImplementationChecker> {
  MigrationPass &Pass;

public:
  ImplementationChecker(MigrationPass &pass) : Pass(pass) { }

  bool TraverseObjCImplementationDecl(ObjCImplementationDecl *D) {
    PropertiesRewriter(Pass).doTransform(D);
    return true;
  }
};

} // anonymous namespace

void trans::rewriteProperties(MigrationPass &pass) {
  ImplementationChecker(pass).TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
