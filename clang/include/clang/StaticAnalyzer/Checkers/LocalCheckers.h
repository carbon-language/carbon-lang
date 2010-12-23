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

#ifndef LLVM_CLANG_GR_LOCALCHECKERS_H
#define LLVM_CLANG_GR_LOCALCHECKERS_H

namespace clang {

class CFG;
class Decl;
class Diagnostic;
class ASTContext;
class LangOptions;
class ParentMap;
class LiveVariables;
class ObjCImplementationDecl;
class LangOptions;
class TranslationUnitDecl;

namespace ento {

class PathDiagnosticClient;
class TransferFuncs;
class BugType;
class BugReporter;
class ExprEngine;

void CheckDeadStores(CFG &cfg, LiveVariables &L, ParentMap &map, 
                     BugReporter& BR);

TransferFuncs* MakeCFRefCountTF(ASTContext& Ctx, bool GCEnabled,
                                  const LangOptions& lopts);

void CheckObjCDealloc(const ObjCImplementationDecl* D, const LangOptions& L,
                      BugReporter& BR);

void CheckObjCInstMethSignature(const ObjCImplementationDecl *ID,
                                BugReporter& BR);

void CheckObjCUnusedIvar(const ObjCImplementationDecl *D, BugReporter& BR);

void RegisterAppleChecks(ExprEngine& Eng, const Decl &D);
void RegisterExperimentalChecks(ExprEngine &Eng);
void RegisterExperimentalInternalChecks(ExprEngine &Eng);

void CheckLLVMConventions(TranslationUnitDecl &TU, BugReporter &BR);
void CheckSecuritySyntaxOnly(const Decl *D, BugReporter &BR);
void CheckSizeofPointer(const Decl *D, BugReporter &BR);

void RegisterCallInliner(ExprEngine &Eng);

} // end GR namespace

} // end namespace clang

#endif
