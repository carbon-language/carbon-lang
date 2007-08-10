//===--- SemaChecking.cpp - Extra Semantic Checking -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements extra semantic analysis beyond what is enforced 
//  by the C type system.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;

/// CheckFunctionCall - Check a direct function call for various correctness
/// and safety properties not strictly enforced by the C type system.
void
Sema::CheckFunctionCall(Expr *Fn, FunctionDecl *FDecl,
                        Expr** Args, unsigned NumArgsInCall) {
                        
  // Get the IdentifierInfo* for the called function.
  IdentifierInfo *FnInfo = FDecl->getIdentifier();
  
  // Search the KnownFunctionIDs for the identifier.
  unsigned i = 0, e = id_num_known_functions;
  for ( ; i != e; ++i) { if (KnownFunctionIDs[i] == FnInfo) break; }
  if( i == e ) return;
  
  // Printf checking.
  if (i <= id_vprintf) {
    // Retrieve the index of the format string parameter.
    unsigned format_idx = 0;
    switch (i) {
      default: assert(false && "No format string argument index.");
      case id_printf:    format_idx = 0; break;
      case id_fprintf:   format_idx = 1; break;
      case id_sprintf:   format_idx = 1; break;
      case id_snprintf:  format_idx = 2; break;
      case id_vsnprintf: format_idx = 2; break;
      case id_asprintf:  format_idx = 1; break;
      case id_vasprintf: format_idx = 1; break;
      case id_vfprintf:  format_idx = 1; break;
      case id_vsprintf:  format_idx = 1; break;
      case id_vprintf:   format_idx = 1; break;
    }    
    CheckPrintfArguments(Fn, i, FDecl, format_idx, Args, NumArgsInCall);
  }
}

/// CheckPrintfArguments - Check calls to printf (and similar functions) for
/// correct use of format strings.  Improper format strings to functions in
/// the printf family can be the source of bizarre bugs and very serious
/// security holes.  A good source of information is available in the following
/// paper (which includes additional references):
///
///  FormatGuard: Automatic Protection From printf Format String
///  Vulnerabilities, Proceedings of the 10th USENIX Security Symposium, 2001.
void
Sema::CheckPrintfArguments(Expr *Fn, unsigned id_idx, FunctionDecl *FDecl,
                           unsigned format_idx, Expr** Args, 
                           unsigned NumArgsInCall) {
                           
  assert( format_idx < NumArgsInCall );

  // CHECK: format string is not a string literal.
  // 
  // Dynamically generated format strings are difficult to automatically
  // vet at compile time.  Requiring that format strings are string literals
  // (1) permits the checking of format strings by the compiler and thereby
  // (2) can practically remove the source of many format string exploits.

  StringLiteral *FExpr = dyn_cast<StringLiteral>(Args[format_idx]);
  
  if ( FExpr == NULL )
    Diag( Args[format_idx]->getLocStart(), 
          diag::warn_printf_not_string_constant, Fn->getSourceRange() );
}