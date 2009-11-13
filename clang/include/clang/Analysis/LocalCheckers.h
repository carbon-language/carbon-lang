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
class BugReporter;
class ObjCImplementationDecl;
class LangOptions;
class GRExprEngine;

void CheckDeadStores(CFG &cfg, LiveVariables &L, ParentMap &map, 
                     BugReporter& BR);

void CheckUninitializedValues(CFG& cfg, ASTContext& Ctx, Diagnostic& Diags,
                              bool FullUninitTaint=false);

GRTransferFuncs* MakeCFRefCountTF(ASTContext& Ctx, bool GCEnabled,
                                  const LangOptions& lopts);

void CheckObjCDealloc(const ObjCImplementationDecl* D, const LangOptions& L,
                      BugReporter& BR);

void CheckObjCInstMethSignature(const ObjCImplementationDecl *ID,
                                BugReporter& BR);

void CheckObjCUnusedIvar(const ObjCImplementationDecl *D, BugReporter& BR);

void RegisterAppleChecks(GRExprEngine& Eng, const Decl &D);
void RegisterExperimentalChecks(GRExprEngine &Eng);
void RegisterExperimentalInternalChecks(GRExprEngine &Eng);

void CheckSecuritySyntaxOnly(const Decl *D, BugReporter &BR);

void CheckSizeofPointer(const Decl *D, BugReporter &BR);
} // end namespace clang

#endif
