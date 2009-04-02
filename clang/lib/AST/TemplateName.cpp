//===--- TemplateName.h - C++ Template Name Representation-------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TemplateName interface and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TemplateName.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

TemplateDecl *TemplateName::getAsTemplateDecl() const {
  if (TemplateDecl *Template = Storage.dyn_cast<TemplateDecl *>())
    return Template;
  
  if (QualifiedTemplateName *QTN = getAsQualifiedTemplateName())
    return QTN->getTemplateDecl();

  return 0;
}

bool TemplateName::isDependent() const {
  if (TemplateDecl *Template = getAsTemplateDecl()) {
    // FIXME: We don't yet have a notion of dependent
    // declarations. When we do, check that. This hack won't last
    // long!.
    return isa<TemplateTemplateParmDecl>(Template);
  }

  return true;
}

void TemplateName::print(llvm::raw_ostream &OS, bool SuppressNNS) const {
  if (TemplateDecl *Template = Storage.dyn_cast<TemplateDecl *>())
    OS << Template->getIdentifier()->getName();
  else if (QualifiedTemplateName *QTN = getAsQualifiedTemplateName()) {
    if (!SuppressNNS)
      QTN->getQualifier()->print(OS);
    if (QTN->hasTemplateKeyword())
      OS << "template ";
    OS << QTN->getTemplateDecl()->getIdentifier()->getName();
  } else if (DependentTemplateName *DTN = getAsDependentTemplateName()) {
    if (!SuppressNNS)
      DTN->getQualifier()->print(OS);
    OS << "template ";
    OS << DTN->getName()->getName();
  }
}

void TemplateName::dump() const {
  print(llvm::errs());
}
