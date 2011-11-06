//===--- TransGCAttrs.cpp - Transformations to ARC mode --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/Support/SaveAndRestore.h"

using namespace clang;
using namespace arcmt;
using namespace trans;

namespace {

/// \brief Collects all the places where GC attributes __strong/__weak occur.
class GCAttrsCollector : public RecursiveASTVisitor<GCAttrsCollector> {
  MigrationContext &MigrateCtx;
  bool FullyMigratable;

  typedef RecursiveASTVisitor<GCAttrsCollector> base;
public:
  explicit GCAttrsCollector(MigrationContext &ctx)
    : MigrateCtx(ctx), FullyMigratable(false) { }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  bool VisitAttributedTypeLoc(AttributedTypeLoc TL) {
    handleAttr(TL);
    return true;
  }

  bool TraverseDecl(Decl *D) {
    if (!D || D->isImplicit())
      return true;

    bool migratable = isMigratable(D);
    SaveAndRestore<bool> Save(FullyMigratable, migratable);
    
    if (DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D))
      lookForAttribute(DD, DD->getTypeSourceInfo());
    else if (ObjCPropertyDecl *PropD = dyn_cast<ObjCPropertyDecl>(D))
      lookForAttribute(PropD, PropD->getTypeSourceInfo());
    return base::TraverseDecl(D);
  }

  void lookForAttribute(Decl *D, TypeSourceInfo *TInfo) {
    if (!TInfo)
      return;
    TypeLoc TL = TInfo->getTypeLoc();
    while (TL) {
      if (const AttributedTypeLoc *Attr = dyn_cast<AttributedTypeLoc>(&TL)) {
        if (handleAttr(*Attr, D))
          break;
        TL = Attr->getModifiedLoc();
      } if (const ArrayTypeLoc *Arr = dyn_cast<ArrayTypeLoc>(&TL)) {
        TL = Arr->getElementLoc();
      } else if (const PointerTypeLoc *PT = dyn_cast<PointerTypeLoc>(&TL)) {
        TL = PT->getPointeeLoc();
      } else if (const ReferenceTypeLoc *RT = dyn_cast<ReferenceTypeLoc>(&TL))
        TL = RT->getPointeeLoc();
      else
        break;
    }
  }

  bool handleAttr(AttributedTypeLoc TL, Decl *D = 0) {
    if (TL.getAttrKind() != AttributedType::attr_objc_ownership)
      return false;

    SourceLocation Loc = TL.getAttrNameLoc();
    unsigned RawLoc = Loc.getRawEncoding();
    if (MigrateCtx.AttrSet.count(RawLoc))
      return true;

    ASTContext &Ctx = MigrateCtx.Pass.Ctx;
    SourceManager &SM = Ctx.getSourceManager();
    llvm::SmallString<32> Buf;
    bool Invalid = false;
    StringRef Spell = Lexer::getSpelling(
                                  SM.getSpellingLoc(TL.getAttrEnumOperandLoc()),
                                  Buf, SM, Ctx.getLangOptions(), &Invalid);
    if (Invalid)
      return false;
    MigrationContext::GCAttrOccurrence::AttrKind Kind;
    if (Spell == "strong")
      Kind = MigrationContext::GCAttrOccurrence::Strong;
    else if (Spell == "weak")
      Kind = MigrationContext::GCAttrOccurrence::Weak;
    else
      return false;
 
    MigrateCtx.AttrSet.insert(RawLoc);
    MigrateCtx.GCAttrs.push_back(MigrationContext::GCAttrOccurrence());
    MigrationContext::GCAttrOccurrence &Attr = MigrateCtx.GCAttrs.back();

    Attr.Kind = Kind;
    Attr.Loc = Loc;
    Attr.ModifiedType = TL.getModifiedLoc().getType();
    Attr.Dcl = D;
    Attr.FullyMigratable = FullyMigratable;
    return true;
  }

  bool isMigratable(Decl *D) {
    if (isa<TranslationUnitDecl>(D))
      return false;

    if (isInMainFile(D))
      return true;

    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      return FD->hasBody();

    if (ObjCContainerDecl *ContD = dyn_cast<ObjCContainerDecl>(D)) {
      if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(ContD))
        return ID->getImplementation() != 0;
      if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(ContD))
        return CD->getImplementation() != 0;
      if (isa<ObjCImplDecl>(ContD))
        return true;
      return false;
    }

    if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
      for (CXXRecordDecl::method_iterator
             MI = RD->method_begin(), ME = RD->method_end(); MI != ME; ++MI) {
        if ((*MI)->isOutOfLine())
          return true;
      }
      return false;
    }

    return isMigratable(cast<Decl>(D->getDeclContext()));
  }

  bool isInMainFile(Decl *D) {
    if (!D)
      return false;

    for (Decl::redecl_iterator
           I = D->redecls_begin(), E = D->redecls_end(); I != E; ++I)
      if (!isInMainFile((*I)->getLocation()))
        return false;
    
    return true;
  }

  bool isInMainFile(SourceLocation Loc) {
    if (Loc.isInvalid())
      return false;

    SourceManager &SM = MigrateCtx.Pass.Ctx.getSourceManager();
    return SM.isInFileID(SM.getExpansionLoc(Loc), SM.getMainFileID());
  }
};

} // anonymous namespace

void GCAttrsTraverser::traverseTU(MigrationContext &MigrateCtx) {
  GCAttrsCollector(MigrateCtx).TraverseDecl(
                                  MigrateCtx.Pass.Ctx.getTranslationUnitDecl());
#if 0
  llvm::errs() << "\n################\n";
  for (unsigned i = 0, e = MigrateCtx.GCAttrs.size(); i != e; ++i) {
    MigrationContext::GCAttrOccurrence &Attr = MigrateCtx.GCAttrs[i];
    llvm::errs() << "KIND: "
        << (Attr.Kind == MigrationContext::GCAttrOccurrence::Strong ? "strong"
                                                                    : "weak");
    llvm::errs() << "\nLOC: ";
    Attr.Loc.dump(MigrateCtx.Pass.Ctx.getSourceManager());
    llvm::errs() << "\nTYPE: ";
    Attr.ModifiedType.dump();
    if (Attr.Dcl) {
      llvm::errs() << "DECL:\n";
      Attr.Dcl->dump();
    } else {
      llvm::errs() << "DECL: NONE";
    }
    llvm::errs() << "\nMIGRATABLE: " << Attr.FullyMigratable;
    llvm::errs() << "\n----------------\n";
  }
  llvm::errs() << "\n################\n";
#endif
}
