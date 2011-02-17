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

def CocoaExperimental : Package<"experimental">,
  InPackage<Cocoa>, Hidden;

def UnixExperimental : Package<"experimental">,
  InPackage<Unix>, Hidden;

def LLVM : Package<"llvm">;
def Debug : Package<"debug">;

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

def NilArgChecker : Checker<"NilArg">,
  HelpText<"Check for prohibited nil arguments to ObjC method calls">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def ClassReleaseChecker : Checker<"ClassRelease">,
  HelpText<"Check for sending 'retain', 'release', or 'autorelease' directly to a Class">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def NSAutoreleasePoolChecker : Checker<"NSAutoreleasePool">,
  HelpText<"Warn for subpar uses of NSAutoreleasePool">,
  DescFile<"NSAutoreleasePoolChecker.cpp">;

def ObjCMethSigsChecker : Checker<"MethodSigs">,
  HelpText<"Warn about Objective-C method signatures with type incompatibilities">,
  DescFile<"CheckObjCInstMethSignature.cpp">;

def ObjCUnusedIvarsChecker : Checker<"UnusedIvars">,
  HelpText<"Warn about private ivars that are never used">,
  DescFile<"ObjCUnusedIVarsChecker.cpp">;

}

def StackAddrLeakChecker : Checker<"StackAddrLeak">,
  InPackage<Core>,
  HelpText<"Check that addresses to stack memory are not leaked outside the function">,
  DescFile<"StackAddrLeakChecker.cpp">;

def DeadStoresChecker : Checker<"DeadStores">,
  InPackage<Core>,
  HelpText<"Check for stores to dead variables">,
  DescFile<"DeadStoresChecker.cpp">;

def UnixAPIChecker : Checker<"API">,
  InPackage<Unix>,
  HelpText<"Check calls to various UNIX/Posix functions">,
  DescFile<"UnixAPIChecker.cpp">;

def MacOSXAPIChecker : Checker<"API">,
  InPackage<MacOSX>,
  HelpText<"Check calls to various MacOSXAPIChecker">,
  DescFile<"MacOSXAPIChecker.cpp">;

def CFNumberCreateChecker : Checker<"CFNumber">,
  InPackage<MacOSX>,
  HelpText<"Check for CFNumberCreate">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def CFRetainReleaseChecker : Checker<"CFRetainRelease">,
  InPackage<MacOSX>,
  HelpText<"Check for null arguments to CFRetain/CFRelease">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def LLVMConventionsChecker : Checker<"Conventions">,
  InPackage<LLVM>,
  HelpText<"Check code for LLVM codebase conventions">,
  DescFile<"LLVMConventionsChecker.cpp">;

def LiveVariablesDumper : Checker<"DumpLiveVars">,
  InPackage<Debug>,
  HelpText<"Print results of live variable analysis">,
  DescFile<"DebugCheckers.cpp">;

def CFGViewer : Checker<"ViewCFG">,
  InPackage<Debug>,
  HelpText<"View Control-Flow Graphs using GraphViz">,
  DescFile<"DebugCheckers.cpp">;

def CFGDumper : Checker<"DumpCFG">,
  InPackage<Debug>,
  HelpText<"Display Control-Flow Graphs">,
  DescFile<"DebugCheckers.cpp">;

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

def CastToStructChecker : Checker<"CastToStruct">,
  InPackage<CoreExperimental>,
  HelpText<"Check for cast from non-struct pointer to struct pointer">,
  DescFile<"CastToStructChecker.cpp">;

def FixedAddressChecker : Checker<"FixedAddr">,
  InPackage<CoreExperimental>,
  HelpText<"Check for assignment of a fixed address to a pointer">,
  DescFile<"FixedAddressChecker.cpp">;

def PointerArithChecker : Checker<"PointerArithm">,
  InPackage<CoreExperimental>,
  HelpText<"Check for pointer arithmetic on locations other than array elements">,
  DescFile<"PointerArithChecker">;

def PointerSubChecker : Checker<"PointerSub">,
  InPackage<CoreExperimental>,
  HelpText<"Check for pointer subtractions on two pointers pointing to different memory chunks">,
  DescFile<"PointerSubChecker">;

def SizeofPointerChecker : Checker<"SizeofPtr">,
  InPackage<CoreExperimental>,
  HelpText<"Warn about unintended use of sizeof() on pointer expressions">,
  DescFile<"CheckSizeofPointer.cpp">;

def SecuritySyntaxChecker : Checker<"SecuritySyntactic">,
  InPackage<CoreExperimental>,
  HelpText<"Perform quick security checks that require no data flow">,
  DescFile<"CheckSecuritySyntaxOnly.cpp">;

def ObjCDeallocChecker : Checker<"Dealloc">,
  InPackage<CocoaExperimental>,
  HelpText<"Warn about Objective-C classes that lack a correct implementation of -dealloc">,
  DescFile<"CheckObjCDealloc.cpp">;

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
