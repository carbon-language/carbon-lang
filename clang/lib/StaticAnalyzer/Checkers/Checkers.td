//===--- Checkers.td - Static Analyzer Checkers -===-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "clang/StaticAnalyzer/Checkers/CheckerBase.td"

//===----------------------------------------------------------------------===//
// Packages.
//===----------------------------------------------------------------------===//

def Core : Package<"core">;
def Cocoa : Package<"cocoa">;
def Unix : Package<"unix">;
def MacOSX : Package<"macosx">;

def CoreExperimental : Package<"experimental">,
  InPackage<Core>, Hidden;

def UnixExperimental : Package<"experimental">,
  InPackage<Unix>, Hidden;

//===----------------------------------------------------------------------===//
// Groups.
//===----------------------------------------------------------------------===//

def AllExperimental : CheckerGroup<"all-experimental">,
  Hidden;

//===----------------------------------------------------------------------===//
// Checkers.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Hidden experimental checkers.
//===----------------------------------------------------------------------===//

let Group = AllExperimental in {

def CStringChecker : Checker<"CString">,
  InPackage<CoreExperimental>,
  HelpText<"Check calls to functions in <string.h>">,
  DescFile<"CStringChecker.cpp">;

def UnreachableCodeChecker : Checker<"UnreachableCode">,
  InPackage<CoreExperimental>,
  HelpText<"Check unreachable code">,
  DescFile<"UnreachableCodeChecker.cpp">;

def IdempotentOperationChecker : Checker<"IdempotentOps">,
  InPackage<CoreExperimental>,
  HelpText<"Warn about idempotent operations">,
  DescFile<"IdempotentOperationChecker.cpp">;

def ChrootChecker : Checker<"Chroot">,
  InPackage<UnixExperimental>,
  HelpText<"Check improper use of chroot">,
  DescFile<"ChrootChecker.cpp">;

def PthreadLockChecker : Checker<"PthreadLock">,
  InPackage<UnixExperimental>,
  HelpText<"Simple lock -> unlock checker">,
  DescFile<"PthreadLockChecker.cpp">;

def StreamChecker : Checker<"Stream">,
  InPackage<UnixExperimental>,
  HelpText<"Check stream handling functions">,
  DescFile<"StreamChecker.cpp">;

}
