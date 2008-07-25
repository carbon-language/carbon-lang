//===--- ASTConsumer.cpp - Abstract interface for reading ASTs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTConsumer class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TranslationUnit.h"

using namespace clang;

ASTConsumer::~ASTConsumer() {}

void ASTConsumer::InitializeTU(TranslationUnit& TU) {
  Initialize(TU.getContext());
}
