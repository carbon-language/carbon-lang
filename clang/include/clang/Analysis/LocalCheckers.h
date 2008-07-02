//==- LocalCheckers.h - Intra-Procedural+Flow-Sensitive Checkers -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface to call a set of intra-procedural (local)
//  checkers that use flow/path-sensitive analyses to find bugs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_LOCALCHECKERS_H
#define LLVM_CLANG_ANALYSIS_LOCALCHECKERS_H

namespace clang {

class CFG;
class Decl;
class Diagnostic;
class ASTContext;
class PathDiagnosticClient;
class GRTransferFuncs;
class BugType;
class LangOptions;
class ParentMap;
class LiveVariables;
  
void CheckDeadStores(CFG& cfg, ASTContext &Ctx, ParentMap& Parents,
                     Diagnostic &Diags); 
  
  void CheckDeadStores(CFG& cfg, ASTContext &Ctx, LiveVariables& L,
                       ParentMap& Parents, Diagnostic &Diags); 
  
void CheckUninitializedValues(CFG& cfg, ASTContext& Ctx, Diagnostic& Diags,
                              bool FullUninitTaint=false);
  
GRTransferFuncs* MakeGRSimpleValsTF();
GRTransferFuncs* MakeCFRefCountTF(ASTContext& Ctx, bool GCEnabled,
                                  bool StandardWarnings,
                                  const LangOptions& lopts); 
BugType* MakeDeadStoresChecker();
  
} // end namespace clang

#endif
