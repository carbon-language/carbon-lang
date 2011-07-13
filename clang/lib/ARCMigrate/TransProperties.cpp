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
using llvm::StringRef;

namespace {

class PropertiesRewriter {
  MigrationPass &Pass;

  struct PropData {
    ObjCPropertyDecl *PropD;
    ObjCIvarDecl *IvarD;
    ObjCPropertyImplDecl *ImplD;

    PropData(ObjCPropertyDecl *propD) : PropD(propD), IvarD(0), ImplD(0) { }
  };

  typedef llvm::SmallVector<PropData, 2> PropsTy;
  typedef std::map<unsigned, PropsTy> AtPropDeclsTy;
  AtPropDeclsTy AtProps;

public:
  PropertiesRewriter(MigrationPass &pass) : Pass(pass) { }

  void doTransform(ObjCImplementationDecl *D) {
    ObjCInterfaceDecl *iface = D->getClassInterface();
    if (!iface)
      return;

    for (ObjCInterfaceDecl::prop_iterator
           propI = iface->prop_begin(),
           propE = iface->prop_end(); propI != propE; ++propI) {
      if (propI->getAtLoc().isInvalid())
        continue;
      PropsTy &props = AtProps[propI->getAtLoc().getRawEncoding()];
      props.push_back(*propI);
    }

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
  }

private:
  void rewriteProperty(PropsTy &props, SourceLocation atLoc) const {
    ObjCPropertyDecl::PropertyAttributeKind propAttrs = getPropertyAttrs(props);
    
    if (propAttrs & (ObjCPropertyDecl::OBJC_PR_copy |
                     ObjCPropertyDecl::OBJC_PR_unsafe_unretained |
                     ObjCPropertyDecl::OBJC_PR_strong |
                     ObjCPropertyDecl::OBJC_PR_weak))
      return;

    if (propAttrs & ObjCPropertyDecl::OBJC_PR_retain) {
      rewriteAttribute("retain", "strong", atLoc);
      return;
    }

    if (propAttrs & ObjCPropertyDecl::OBJC_PR_assign)
      return rewriteAssign(props, atLoc);

    return maybeAddWeakOrUnsafeUnretainedAttr(props, atLoc);
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
    if ((propAttrs & ObjCPropertyDecl::OBJC_PR_readonly) &&
        hasNoBackingIvars(props))
      return;

    bool canUseWeak = canApplyWeak(Pass.Ctx, getPropertyType(props));
    bool addedAttr = addAttribute(canUseWeak ? "weak" : "unsafe_unretained",
                                  atLoc);
    if (!addedAttr)
      canUseWeak = false;

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

  bool rewriteAttribute(llvm::StringRef fromAttr, llvm::StringRef toAttr,
                        SourceLocation atLoc) const {
    if (atLoc.isMacroID())
      return false;

    SourceManager &SM = Pass.Ctx.getSourceManager();

    // Break down the source location.
    std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(atLoc);

    // Try to load the file buffer.
    bool invalidTemp = false;
    llvm::StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
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
    if (llvm::StringRef(tok.getRawIdentifierData(), tok.getLength())
          != "property")
      return false;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::l_paren)) return false;
    
    lexer.LexFromRawLexer(tok);
    if (tok.is(tok::r_paren))
      return false;

    while (1) {
      if (tok.isNot(tok::raw_identifier)) return false;
      llvm::StringRef ident(tok.getRawIdentifierData(), tok.getLength());
      if (ident == fromAttr) {
        Pass.TA.replaceText(tok.getLocation(), fromAttr, toAttr);
        return true;
      }

      do {
        lexer.LexFromRawLexer(tok);
      } while (tok.isNot(tok::comma) && tok.isNot(tok::r_paren));
      if (tok.is(tok::r_paren))
        break;
      lexer.LexFromRawLexer(tok);
    }

    return false;
  }

  bool addAttribute(llvm::StringRef attr, SourceLocation atLoc) const {
    if (atLoc.isMacroID())
      return false;

    SourceManager &SM = Pass.Ctx.getSourceManager();

    // Break down the source location.
    std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(atLoc);

    // Try to load the file buffer.
    bool invalidTemp = false;
    llvm::StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
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
    if (llvm::StringRef(tok.getRawIdentifierData(), tok.getLength())
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

  bool hasNoBackingIvars(PropsTy &props) const {
    for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I)
      if (isUserDeclared(I->IvarD))
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
