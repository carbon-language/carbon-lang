//===--- Checkers.td - Static Analyzer Checkers -===-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "clang/StaticAnalyzer/Checkers/CheckerBase.td"

def Cocoa : Package<"cocoa">;

def : Checker<"ObjCSelfInitChecker">,
  InPackage<Cocoa>,
  Named<"SelfInit">,
  HelpText<"Check that 'self' is propely initialized inside an initializer method">,
  DescFile<"ObjCSelfInitChecker.cpp">;
