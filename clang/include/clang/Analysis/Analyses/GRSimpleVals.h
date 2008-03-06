//===-- GRSimpleVals.h- Simple, Path-Sens. Constant Prop. ---------*- C++ -*-==//
//   
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//               Constant Propagation via Graph Reachability
//
//  This file defines the interface to use the 'GRSimpleVals' path-sensitive
//  constant-propagation analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GRCONSTANTS
#define LLVM_CLANG_GRCONSTANTS

namespace clang {
  class Diagnostic;
  
  /// RunGRSimpleVals - This is a simple driver to run the GRSimpleVals analysis
  ///  on a provided CFG.  This interface will eventually be replaced with
  ///  something more elaborate as the requirements on the interface become
  ///  clearer.  The value returned is the number of nodes in the ExplodedGraph.
  unsigned RunGRSimpleVals(CFG& cfg, FunctionDecl& FD, ASTContext& Ctx,
                           Diagnostic& Diag, bool Visualize);
  
} // end clang namespace


#endif
