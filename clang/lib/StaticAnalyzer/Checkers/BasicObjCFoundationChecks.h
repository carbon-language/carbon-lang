//== BasicObjCFoundationChecks.h - Simple Apple-Foundation checks -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicObjCFoundationChecks, a class that encapsulates
//  a set of simple checks to run on Objective-C code using Apple's Foundation
//  classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_BASICOBJCFOUNDATIONCHECKS
#define LLVM_CLANG_GR_BASICOBJCFOUNDATIONCHECKS

namespace clang {

class ASTContext;
class Decl;

namespace ento {

class BugReporter;
class ExprEngine;

void RegisterNSErrorChecks(BugReporter& BR, ExprEngine &Eng, const Decl &D);

} // end GR namespace

} // end clang namespace

#endif
