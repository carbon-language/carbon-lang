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

#include "Sema.h"
#include "TargetAttributesSema.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/Triple.h"

using namespace clang;

TargetAttributesSema::~TargetAttributesSema() {}
bool TargetAttributesSema::ProcessDeclAttribute(Scope *scope, Decl *D,
                                    const AttributeList &Attr, Sema &S) const {
  return false;
}

static void HandleMSP430InterruptAttr(Decl *d,
                                      const AttributeList &Attr, Sema &S) {
    // Check the attribute arguments.
    if (Attr.getNumArgs() != 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
      return;
    }

    // FIXME: Check for decl - it should be void ()(void).

    Expr *NumParamsExpr = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt NumParams(32);
    if (!NumParamsExpr->isIntegerConstantExpr(NumParams, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
        << "interrupt" << NumParamsExpr->getSourceRange();
      return;
    }

    unsigned Num = NumParams.getLimitedValue(255);
    if ((Num & 1) || Num > 30) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
        << "interrupt" << (int)NumParams.getSExtValue()
        << NumParamsExpr->getSourceRange();
      return;
    }

    d->addAttr(::new (S.Context) MSP430InterruptAttr(Num));
    d->addAttr(::new (S.Context) UsedAttr());
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
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // If we try to apply it to a function pointer, warn. This is a special
  // instance of the warn_attribute_ignored warning that can be turned
  // off with -Wno-force-align-arg-pointer.
  ValueDecl* VD = dyn_cast<ValueDecl>(D);
  if (VD && VD->getType()->isFunctionPointerType()) {
    S.Diag(Attr.getLoc(), diag::warn_faap_attribute_ignored);
    return;
  }
  // Attribute can only be applied to function types.
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << /* function */0;
    return;
  }

  D->addAttr(::new (S.Context) X86ForceAlignArgPointerAttr());
}

namespace {
  class X86AttributesSema : public TargetAttributesSema {
  public:
    X86AttributesSema() { }
    bool ProcessDeclAttribute(Scope *scope, Decl *D,
                              const AttributeList &Attr, Sema &S) const {
      if (Attr.getName()->getName() == "force_align_arg_pointer") {
        HandleX86ForceAlignArgPointerAttr(D, Attr, S);
        return true;
      }
      return false;
    }
  };
}

const TargetAttributesSema &Sema::getTargetAttributesSema() const {
  if (TheTargetAttributesSema)
    return *TheTargetAttributesSema;

  const llvm::Triple &Triple(Context.Target.getTriple());
  switch (Triple.getArch()) {
  default:
    return *(TheTargetAttributesSema = new TargetAttributesSema);

  case llvm::Triple::msp430:
    return *(TheTargetAttributesSema = new MSP430AttributesSema);
  case llvm::Triple::x86:
    return *(TheTargetAttributesSema = new X86AttributesSema);
  }
}

