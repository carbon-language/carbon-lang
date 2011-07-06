//===--- TransProperties.cpp - Tranformations to ARC mode -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// changeIvarsOfAssignProperties:
//
// If a property is synthesized with 'assign' attribute and the user didn't
// set a lifetime attribute, change the property to 'weak' or add
// __unsafe_unretained if the ARC runtime is not available.
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

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

namespace {

class AssignPropertiesTrans {
  MigrationPass &Pass;
  struct PropData {
    ObjCPropertyDecl *PropD;
    ObjCIvarDecl *IvarD;
    bool ShouldChangeToWeak;
    SourceLocation ArcPropAssignErrorLoc;
  };

  typedef llvm::SmallVector<PropData, 2> PropsTy; 
  typedef llvm::DenseMap<unsigned, PropsTy> PropsMapTy;
  PropsMapTy PropsMap;

public:
  AssignPropertiesTrans(MigrationPass &pass) : Pass(pass) { }

  void doTransform(ObjCImplementationDecl *D) {
    SourceManager &SM = Pass.Ctx.getSourceManager();

    ObjCInterfaceDecl *IFace = D->getClassInterface();
    for (ObjCInterfaceDecl::prop_iterator
           I = IFace->prop_begin(), E = IFace->prop_end(); I != E; ++I) {
      ObjCPropertyDecl *propD = *I;
      unsigned loc = SM.getInstantiationLoc(propD->getAtLoc()).getRawEncoding();
      PropsTy &props = PropsMap[loc];
      props.push_back(PropData());
      props.back().PropD = propD;
      props.back().IvarD = 0;
      props.back().ShouldChangeToWeak = false;
    }

    typedef DeclContext::specific_decl_iterator<ObjCPropertyImplDecl>
        prop_impl_iterator;
    for (prop_impl_iterator
           I = prop_impl_iterator(D->decls_begin()),
           E = prop_impl_iterator(D->decls_end()); I != E; ++I) {
      VisitObjCPropertyImplDecl(*I);
    }

    for (PropsMapTy::iterator
           I = PropsMap.begin(), E = PropsMap.end(); I != E; ++I) {
      SourceLocation atLoc = SourceLocation::getFromRawEncoding(I->first);
      PropsTy &props = I->second;
      if (shouldApplyWeakToAllProp(props)) {
        if (changeAssignToWeak(atLoc)) {
          // Couldn't add the 'weak' property attribute,
          // try adding __unsafe_unretained.
          applyUnsafeUnretained(props);
        } else {
          for (PropsTy::iterator
                 PI = props.begin(), PE = props.end(); PI != PE; ++PI) {
            applyWeak(*PI);
          }
        }
      } else {
        // We should not add 'weak' attribute since not all properties need it.
        // So just add __unsafe_unretained to the ivars.
        applyUnsafeUnretained(props);
      }
    }
  }

  bool shouldApplyWeakToAllProp(PropsTy &props) {
    for (PropsTy::iterator
           PI = props.begin(), PE = props.end(); PI != PE; ++PI) {
      if (!PI->ShouldChangeToWeak)
        return false;
    }
    return true;
  }

  void applyWeak(PropData &prop) {
    assert(Pass.Ctx.getLangOptions().ObjCRuntimeHasWeak);

    Transaction Trans(Pass.TA);
    Pass.TA.insert(prop.IvarD->getLocation(), "__weak "); 
    Pass.TA.clearDiagnostic(diag::err_arc_assign_property_ownership,
                            prop.ArcPropAssignErrorLoc);
  }

  void applyUnsafeUnretained(PropsTy &props) {
    for (PropsTy::iterator
           PI = props.begin(), PE = props.end(); PI != PE; ++PI) {
      if (PI->ShouldChangeToWeak) {
        Transaction Trans(Pass.TA);
        Pass.TA.insert(PI->IvarD->getLocation(), "__unsafe_unretained ");
        Pass.TA.clearDiagnostic(diag::err_arc_assign_property_ownership,
                                PI->ArcPropAssignErrorLoc);
      }
    }
  }

  bool VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
    SourceManager &SM = Pass.Ctx.getSourceManager();

    if (D->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      return true;
    ObjCPropertyDecl *propD = D->getPropertyDecl();
    if (!propD || propD->isInvalidDecl())
      return true;
    ObjCIvarDecl *ivarD = D->getPropertyIvarDecl();
    if (!ivarD || ivarD->isInvalidDecl())
      return true;
    if (!(propD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_assign))
      return true;
    if (isa<AttributedType>(ivarD->getType().getTypePtr()))
      return true;
    if (ivarD->getType().getLocalQualifiers().getObjCLifetime()
          != Qualifiers::OCL_Strong)
      return true;
    if (!Pass.TA.hasDiagnostic(
                      diag::err_arc_assign_property_ownership, D->getLocation()))
      return true;

    // There is a "error: existing ivar for assign property must be
    // __unsafe_unretained"; fix it.

    if (!Pass.Ctx.getLangOptions().ObjCRuntimeHasWeak) {
      // We will just add __unsafe_unretained to the ivar.
      Transaction Trans(Pass.TA);
      Pass.TA.insert(ivarD->getLocation(), "__unsafe_unretained ");
      Pass.TA.clearDiagnostic(
                      diag::err_arc_assign_property_ownership, D->getLocation());
    } else {
      // Mark that we want the ivar to become weak.
      unsigned loc = SM.getInstantiationLoc(propD->getAtLoc()).getRawEncoding();
      PropsTy &props = PropsMap[loc];
      for (PropsTy::iterator I = props.begin(), E = props.end(); I != E; ++I) {
        if (I->PropD == propD) {
          I->IvarD = ivarD;
          I->ShouldChangeToWeak = true;
          I->ArcPropAssignErrorLoc = D->getLocation();
        }
      }
    }

    return true;
  }

private:
  bool changeAssignToWeak(SourceLocation atLoc) {
    SourceManager &SM = Pass.Ctx.getSourceManager();

    // Break down the source location.
    std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(atLoc);

    // Try to load the file buffer.
    bool invalidTemp = false;
    llvm::StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
    if (invalidTemp)
      return true;

    const char *tokenBegin = file.data() + locInfo.second;

    // Lex from the start of the given location.
    Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
                Pass.Ctx.getLangOptions(),
                file.begin(), tokenBegin, file.end());
    Token tok;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::at)) return true;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::raw_identifier)) return true;
    if (llvm::StringRef(tok.getRawIdentifierData(), tok.getLength())
          != "property")
      return true;
    lexer.LexFromRawLexer(tok);
    if (tok.isNot(tok::l_paren)) return true;
    
    SourceLocation LParen = tok.getLocation();
    SourceLocation assignLoc;
    bool isEmpty = false;

    lexer.LexFromRawLexer(tok);
    if (tok.is(tok::r_paren)) {
      isEmpty = true;
    } else {
      while (1) {
        if (tok.isNot(tok::raw_identifier)) return true;
        llvm::StringRef ident(tok.getRawIdentifierData(), tok.getLength());
        if (ident == "assign")
          assignLoc = tok.getLocation();
  
        do {
          lexer.LexFromRawLexer(tok);
        } while (tok.isNot(tok::comma) && tok.isNot(tok::r_paren));
        if (tok.is(tok::r_paren))
          break;
        lexer.LexFromRawLexer(tok);
      }
    }

    Transaction Trans(Pass.TA);
    if (assignLoc.isValid())
      Pass.TA.replaceText(assignLoc, "assign", "weak");
    else 
      Pass.TA.insertAfterToken(LParen, isEmpty ? "weak" : "weak, ");
    return false;
  }
};

class PropertiesChecker : public RecursiveASTVisitor<PropertiesChecker> {
  MigrationPass &Pass;

public:
  PropertiesChecker(MigrationPass &pass) : Pass(pass) { }

  bool TraverseObjCImplementationDecl(ObjCImplementationDecl *D) {
    AssignPropertiesTrans(Pass).doTransform(D);
    return true;
  }
};

} // anonymous namespace

void trans::changeIvarsOfAssignProperties(MigrationPass &pass) {
  PropertiesChecker(pass).TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
