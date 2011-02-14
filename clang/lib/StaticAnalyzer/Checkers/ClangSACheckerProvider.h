//===--- ClangSACheckerProvider.h - Clang SA Checkers Provider --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the entry point for creating the provider for the checkers defined
// in libclangStaticAnalyzerCheckers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_CHECKERS_CLANGSACHECKERPROVIDER_H
#define LLVM_CLANG_SA_CHECKERS_CLANGSACHECKERPROVIDER_H

namespace clang {

namespace ento {
  class CheckerProvider;

CheckerProvider *createClangSACheckerProvider();

} // end ento namespace

} // end clang namespace

#endif
