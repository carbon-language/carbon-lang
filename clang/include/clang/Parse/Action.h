//===--- Action.h - Parser Action Interface ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Action interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_ACTION_H
#define LLVM_CLANG_PARSE_ACTION_H

#include "clang/Basic/SourceLocation.h"

namespace llvm {
namespace clang {
  // Parse.
  class Scope;
  // Semantic.
  class Declarator;

/// Action - As the parser reads the input file and recognizes the productions
/// of the grammar, it invokes methods on this class to turn the parsed input
/// into something useful: e.g. a parse tree.
///
/// The callback methods that this class provides are phrased as actions that
/// the parser has just done or is about to do when the method is called.  They
/// are not requests that the actions module do the specified action.
///
/// All of the methods here are optional, but you must specify information about
/// whether something is a typedef or not in order for the parse to complete
/// accurately.  The EmptyAction class does this bare-minimum of tracking.
class Action {
public:
  /// Out-of-line virtual destructor to provide home for this class.
  virtual ~Action();
  
  // Types - Though these don't actually enforce strong typing, they document
  // what types are required to be identical for the actions.
  typedef void ExprTy;
  
  //===--------------------------------------------------------------------===//
  // Symbol table tracking callbacks.
  //===--------------------------------------------------------------------===//
  
  /// ParseDeclarator - This callback is invoked when a declarator is parsed and
  /// 'Init' specifies the initializer if any.  This is for things like:
  /// "int X = 4" or "typedef int foo".
  virtual void ParseDeclarator(SourceLocation Loc, Scope *S, Declarator &D,
                               ExprTy *Init) {}
  
  /// PopScope - This callback is called immediately before the specified scope
  /// is popped and deleted.
  virtual void PopScope(SourceLocation Loc, Scope *S) {}
  
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
