//===-- TargetAttributesSema.cpp - Encapsulate target attributes-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains semantic analysis implementation for target-specific
// attributes.
//
//===----------------------------------------------------------------------===//

#include "TargetAttributesSema.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/Triple.h"

using namespace clang;

TargetAttributesSema::~TargetAttributesSema() {}
bool TargetAttributesSema::ProcessDeclAttribute(Scope *scope, Decl *D,
                                    const AttributeList &Attr, Sema &S) const {
  return false;
}

static void HandleARMInterruptAttr(Decl *d,
                                   const AttributeList &Attr, Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments)
        << 1;
    return;
  }

  StringRef Str;
  SourceLocation ArgLoc;

  if (Attr.getNumArgs() == 0)
    Str = "";
  else if (!S.checkStringLiteralArgumentAttr(Attr, 0, Str, &ArgLoc))
    return;

  ARMInterruptAttr::InterruptType Kind;
  if (!ARMInterruptAttr::ConvertStrToInterruptType(Str, Kind)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
        << Attr.getName() << Str << ArgLoc;
    return;
  }

  unsigned Index = Attr.getAttributeSpellingListIndex();
  d->addAttr(::new (S.Context)
             ARMInterruptAttr(Attr.getLoc(), S.Context, Kind, Index));
}

namespace {
  class ARMAttributesSema : public TargetAttributesSema {
  public:
    ARMAttributesSema() { }
    bool ProcessDeclAttribute(Scope *scope, Decl *D,
                              const AttributeList &Attr, Sema &S) const {
      if (Attr.getName()->getName() == "interrupt") {
        HandleARMInterruptAttr(D, Attr, S);
        return true;
      }
      return false;
    }
  };
}

static void HandleMSP430InterruptAttr(Decl *d,
                                      const AttributeList &Attr, Sema &S) {
    // Check the attribute arguments.
    if (Attr.getNumArgs() != 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
        << Attr.getName() << 1;
      return;
    }

    // FIXME: Check for decl - it should be void ()(void).

    Expr *NumParamsExpr = static_cast<Expr *>(Attr.getArgAsExpr(0));
    llvm::APSInt NumParams(32);
    if (!NumParamsExpr->isIntegerConstantExpr(NumParams, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_type)
        << Attr.getName() << AANT_ArgumentIntegerConstant
        << NumParamsExpr->getSourceRange();
      return;
    }

    unsigned Num = NumParams.getLimitedValue(255);
    if ((Num & 1) || Num > 30) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << "interrupt" << (int)NumParams.getSExtValue()
        << NumParamsExpr->getSourceRange();
      return;
    }

    d->addAttr(::new (S.Context) MSP430InterruptAttr(Attr.getLoc(), S.Context, Num));
    d->addAttr(::new (S.Context) UsedAttr(Attr.getLoc(), S.Context));
  }

namespace {
  class MSP430AttributesSema : public TargetAttributesSema {
  public:
    MSP430AttributesSema() { }
    bool ProcessDeclAttribute(Scope *scope, Decl *D,
                              const AttributeList &Attr, Sema &S) const {
      if (Attr.getName()->getName() == "interrupt") {
        HandleMSP430InterruptAttr(D, Attr, S);
        return true;
      }
      return false;
    }
  };
}

static void HandleX86ForceAlignArgPointerAttr(Decl *D,
                                              const AttributeList& Attr,
                                              Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 0;
    return;
  }

  // If we try to apply it to a function pointer, don't warn, but don't
  // do anything, either. It doesn't matter anyway, because there's nothing
  // special about calling a force_align_arg_pointer function.
  ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (VD && VD->getType()->isFunctionPointerType())
    return;
  // Also don't warn on function pointer typedefs.
  TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD && (TD->getUnderlyingType()->isFunctionPointerType() ||
             TD->getUnderlyingType()->isFunctionType()))
    return;
  // Attribute can only be applied to function types.
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << /* function */0;
    return;
  }

  D->addAttr(::new (S.Context) X86ForceAlignArgPointerAttr(Attr.getRange(),
                                                           S.Context));
}

DLLImportAttr *Sema::mergeDLLImportAttr(Decl *D, SourceRange Range,
                                        unsigned AttrSpellingListIndex) {
  if (D->hasAttr<DLLExportAttr>()) {
    Diag(Range.getBegin(), diag::warn_attribute_ignored) << "dllimport";
    return NULL;
  }

  if (D->hasAttr<DLLImportAttr>())
    return NULL;

  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasDefinition()) {
      // dllimport cannot be applied to definitions.
      Diag(D->getLocation(), diag::warn_attribute_invalid_on_definition)
        << "dllimport";
      return NULL;
    }
  }

  return ::new (Context) DLLImportAttr(Range, Context,
                                       AttrSpellingListIndex);
}

static void HandleDLLImportAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 0;
    return;
  }

  // Attribute can be applied only to functions or variables.
  FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD && !isa<VarDecl>(D)) {
    // Apparently Visual C++ thinks it is okay to not emit a warning
    // in this case, so only emit a warning when -fms-extensions is not
    // specified.
    if (!S.getLangOpts().MicrosoftExt)
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  // Currently, the dllimport attribute is ignored for inlined functions.
  // Warning is emitted.
  if (FD && FD->isInlineSpecified()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "dllimport";
    return;
  }

  unsigned Index = Attr.getAttributeSpellingListIndex();
  DLLImportAttr *NewAttr = S.mergeDLLImportAttr(D, Attr.getRange(), Index);
  if (NewAttr)
    D->addAttr(NewAttr);
}

DLLExportAttr *Sema::mergeDLLExportAttr(Decl *D, SourceRange Range,
                                        unsigned AttrSpellingListIndex) {
  if (DLLImportAttr *Import = D->getAttr<DLLImportAttr>()) {
    Diag(Import->getLocation(), diag::warn_attribute_ignored) << "dllimport";
    D->dropAttr<DLLImportAttr>();
  }

  if (D->hasAttr<DLLExportAttr>())
    return NULL;

  return ::new (Context) DLLExportAttr(Range, Context,
                                       AttrSpellingListIndex);
}

static void HandleDLLExportAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 0;
    return;
  }

  // Attribute can be applied only to functions or variables.
  FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD && !isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  // Currently, the dllexport attribute is ignored for inlined functions, unless
  // the -fkeep-inline-functions flag has been used. Warning is emitted;
  if (FD && FD->isInlineSpecified()) {
    // FIXME: ... unless the -fkeep-inline-functions flag has been used.
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "dllexport";
    return;
  }

  unsigned Index = Attr.getAttributeSpellingListIndex();
  DLLExportAttr *NewAttr = S.mergeDLLExportAttr(D, Attr.getRange(), Index);
  if (NewAttr)
    D->addAttr(NewAttr);
}

namespace {
  class X86AttributesSema : public TargetAttributesSema {
  public:
    X86AttributesSema() { }
    bool ProcessDeclAttribute(Scope *scope, Decl *D,
                              const AttributeList &Attr, Sema &S) const {
      const llvm::Triple &Triple(S.Context.getTargetInfo().getTriple());
      if (Triple.getOS() == llvm::Triple::Win32 ||
          Triple.getOS() == llvm::Triple::MinGW32) {
        switch (Attr.getKind()) {
        case AttributeList::AT_DLLImport: HandleDLLImportAttr(D, Attr, S);
                                          return true;
        case AttributeList::AT_DLLExport: HandleDLLExportAttr(D, Attr, S);
                                          return true;
        default:                          break;
        }
      }
      if (Triple.getArch() != llvm::Triple::x86_64 &&
          (Attr.getName()->getName() == "force_align_arg_pointer" ||
           Attr.getName()->getName() == "__force_align_arg_pointer__")) {
        HandleX86ForceAlignArgPointerAttr(D, Attr, S);
        return true;
      }
      return false;
    }
  };
}

static void HandleMips16Attr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 0;
    return;
  }
  // Attribute can only be applied to function types.
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << /* function */0;
    return;
  }
  D->addAttr(::new (S.Context) Mips16Attr(Attr.getRange(), S.Context,
                                          Attr.getAttributeSpellingListIndex()));
}

static void HandleNoMips16Attr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << Attr.getName() << 0;
    return;
  }
  // Attribute can only be applied to function types.
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << /* function */0;
    return;
  }
  D->addAttr(::new (S.Context)
             NoMips16Attr(Attr.getRange(), S.Context,
                          Attr.getAttributeSpellingListIndex()));
}

namespace {
  class MipsAttributesSema : public TargetAttributesSema {
  public:
    MipsAttributesSema() { }
    bool ProcessDeclAttribute(Scope *scope, Decl *D, const AttributeList &Attr,
                              Sema &S) const {
      if (Attr.getName()->getName() == "mips16") {
        HandleMips16Attr(D, Attr, S);
        return true;
      } else if (Attr.getName()->getName() == "nomips16") {
        HandleNoMips16Attr(D, Attr, S);
        return true;
      }
      return false;
    }
  };
}

const TargetAttributesSema &Sema::getTargetAttributesSema() const {
  if (TheTargetAttributesSema)
    return *TheTargetAttributesSema;

  const llvm::Triple &Triple(Context.getTargetInfo().getTriple());
  switch (Triple.getArch()) {
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    return *(TheTargetAttributesSema = new ARMAttributesSema);
  case llvm::Triple::msp430:
    return *(TheTargetAttributesSema = new MSP430AttributesSema);
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return *(TheTargetAttributesSema = new X86AttributesSema);
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return *(TheTargetAttributesSema = new MipsAttributesSema);
  default:
    return *(TheTargetAttributesSema = new TargetAttributesSema);
  }
}
