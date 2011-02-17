//== CheckerV2.h - Registration mechanism for checkers -----------*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerV2, used to create and register checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_CORE_CHECKERV2
#define LLVM_CLANG_SA_CORE_CHECKERV2

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace ento {
  class BugReporter;

namespace check {

struct _VoidCheck {
  static void _register(void *checker, CheckerManager &mgr) { }
};

template <typename DECL>
class ASTDecl {
  template <typename CHECKER>
  static void _checkDecl(void *checker, const Decl *D, AnalysisManager& mgr,
                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkASTDecl(llvm::cast<DECL>(D), mgr, BR);
  }

  static bool _handlesDecl(const Decl *D) {
    return llvm::isa<DECL>(D);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForDecl(checker, _checkDecl<CHECKER>, _handlesDecl);
  }
};

class ASTCodeBody {
  template <typename CHECKER>
  static void _checkBody(void *checker, const Decl *D, AnalysisManager& mgr,
                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkASTCodeBody(D, mgr, BR);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBody(checker, _checkBody<CHECKER>);
  }
};

} // end check namespace

template <typename CHECK1, typename CHECK2=check::_VoidCheck,
          typename CHECK3=check::_VoidCheck, typename CHECK4=check::_VoidCheck,
          typename CHECK5=check::_VoidCheck, typename CHECK6=check::_VoidCheck,
          typename CHECK7=check::_VoidCheck, typename CHECK8=check::_VoidCheck,
          typename CHECK9=check::_VoidCheck, typename CHECK10=check::_VoidCheck,
          typename CHECK11=check::_VoidCheck,typename CHECK12=check::_VoidCheck>
class CheckerV2 {
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    CHECK1::_register(checker, mgr);
    CHECK2::_register(checker, mgr);
    CHECK3::_register(checker, mgr);
    CHECK4::_register(checker, mgr);
    CHECK5::_register(checker, mgr);
    CHECK6::_register(checker, mgr);
    CHECK7::_register(checker, mgr);
    CHECK8::_register(checker, mgr);
    CHECK9::_register(checker, mgr);
    CHECK10::_register(checker, mgr);
    CHECK11::_register(checker, mgr);
    CHECK12::_register(checker, mgr);
  }
};

} // end ento namespace

} // end clang namespace

#endif
