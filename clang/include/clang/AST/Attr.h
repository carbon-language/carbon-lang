//===--- Attr.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Attribute class interfaces
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ATTR_H
#define LLVM_CLANG_AST_ATTR_H

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Type.h"

namespace llvm {
namespace clang {
class IdentifierInfo;
class Expr;

/// This will contain AST nodes for specific GCC attributes
/// "Raw" attribute list nodes (created during parsing) are declared
/// in "clang/Parse/AttributeList.h"

}  // end namespace clang
}  // end namespace llvm

#endif
