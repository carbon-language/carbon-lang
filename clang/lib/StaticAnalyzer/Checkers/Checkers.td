//===--- Checkers.td - Static Analyzer Checkers -===-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "clang/StaticAnalyzer/Checkers/CheckerBase.td"

def Core : Package<"core">;
def Cocoa : Package<"cocoa">;
def Unix : Package<"unix">;
def MacOSX : Package<"macosx">;

let ParentPackage = Cocoa in {

def ObjCSelfInitChecker : Checker<"SelfInit">,
  HelpText<"Check that 'self' is propely initialized inside an initializer method">,
  DescFile<"ObjCSelfInitChecker.cpp">;

def ObjCAtSyncChecker : Checker<"AtSync">,
  HelpText<"Check for null pointers used as mutexes for @synchronized">,
  DescFile<"ObjCAtSyncChecker.cpp">;

}

def StackAddrLeakChecker : Checker<"StackAddrLeak">,
  InPackage<Core>,
  HelpText<"Check that addresses to stack memory are not leaked outside the function">,
  DescFile<"StackAddrLeakChecker.cpp">;

def UnixAPIChecker : Checker<"API">,
  InPackage<Unix>,
  HelpText<"Check calls to various UNIX/Posix functions">,
  DescFile<"UnixAPIChecker.cpp">;

def MacOSXAPIChecker : Checker<"API">,
  InPackage<MacOSX>,
  HelpText<"Check calls to various MacOSXAPIChecker">,
  DescFile<"MacOSXAPIChecker.cpp">;
