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
def DeadCode : Package<"deadcode">;

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
 
def NSErrorChecker : Checker<"NSError">,
  HelpText<"Check usage of NSError** parameters">,
  DescFile<"NSErrorChecker.cpp">;

} // end "cocoa"

let ParentPackage = Core in {

def DereferenceChecker : Checker<"Deref">,
  HelpText<"Check for null pointers at loads and stores">,
  DescFile<"DereferenceChecker.cpp">;

def CallAndMessageChecker : Checker<"CallAndMsg">,
  HelpText<"Check for errors of call and objc message expressions">,
  DescFile<"CallAndMessageChecker.cpp">;

def AdjustedReturnValueChecker : Checker<"AdjustRet">,
  HelpText<"Check to see if the return value of a function call is different than the caller expects">,
  DescFile<"AdjustedReturnValueChecker.cpp">;

def AttrNonNullChecker : Checker<"AttrNonNull">,
  HelpText<"Check for arguments declared to have nonnull attribute">,
  DescFile<"AttrNonNullChecker.cpp">;

def VLASizeChecker : Checker<"VLASize">,
  HelpText<"Check for declarations of VLA of undefined or zero size">,
  DescFile<"VLASizeChecker.cpp">;

def DivZeroChecker : Checker<"DivZero">,
  HelpText<"Check for division by zeros">,
  DescFile<"DivZeroChecker.cpp">;

def ReturnUndefChecker : Checker<"UndefReturn">,
  HelpText<"Check for undefined or garbage values being returned to the caller">,
  DescFile<"ReturnUndefChecker.cpp">;

def UndefinedArraySubscriptChecker : Checker<"UndefArraySubscript">,
  HelpText<"Check for undefined array subscripts">,
  DescFile<"UndefinedArraySubscriptChecker.cpp">;

def UndefinedAssignmentChecker : Checker<"UndefAssign">,
  HelpText<"Check for assigning undefined values">,
  DescFile<"UndefinedAssignmentChecker.cpp">;

def UndefBranchChecker : Checker<"UndefBranch">,
  HelpText<"Check for undefined branch conditions">,
  DescFile<"UndefBranchChecker.cpp">;

def UndefCapturedBlockVarChecker : Checker<"UndefBlockVar">,
  HelpText<"Check for blocks that capture uninitialized values">,
  DescFile<"UndefCapturedBlockVarChecker.cpp">;

def UndefResultChecker : Checker<"UndefBinOpResult">,
  HelpText<"Check for undefined results of non-assignment binary operators">,
  DescFile<"UndefResultChecker.cpp">;

def NoReturnFunctionChecker : Checker<"NoReturnFunc">,
  HelpText<"Evaluate functions that do not return to the caller">,
  DescFile<"NoReturnFunctionChecker.cpp">;

def BuiltinFunctionChecker : Checker<"BuiltinFunc">,
  HelpText<"Evaluate clang builtin functions">,
  DescFile<"BuiltinFunctionChecker.cpp">;

def StackAddrEscapeChecker : Checker<"StackAddrEscape">,
  HelpText<"Check that addresses to stack memory do not escape the function">,
  DescFile<"StackAddrEscapeChecker.cpp">;

} // end "core"

let ParentPackage = DeadCode in {

def DeadStoresChecker : Checker<"DeadStores">,
  HelpText<"Check for values stored to a variables that are never read afterwards">,
  DescFile<"DeadStoresChecker.cpp">;

def IdempotentOperationChecker : Checker<"IdempotentOperations">,
  HelpText<"Warn about idempotent operations">,
  DescFile<"IdempotentOperationChecker.cpp">;

} // end DeadCode

def UnixAPIChecker : Checker<"API">,
  InPackage<Unix>,
  HelpText<"Check calls to various UNIX/Posix functions">,
  DescFile<"UnixAPIChecker.cpp">;

def MacOSXAPIChecker : Checker<"API">,
  InPackage<MacOSX>,
  HelpText<"Check for proper uses of various Mac OS X APIs">,
  DescFile<"MacOSXAPIChecker.cpp">;

def OSAtomicChecker : Checker<"AtomicCAS">,
  InPackage<MacOSX>,
  HelpText<"Evaluate calls to OSAtomic functions">,
  DescFile<"OSAtomicChecker.cpp">;

def CFNumberCreateChecker : Checker<"CFNumber">,
  InPackage<MacOSX>,
  HelpText<"Check for proper uses of CFNumberCreate">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def CFRetainReleaseChecker : Checker<"CFRetainRelease">,
  InPackage<MacOSX>,
  HelpText<"Check for null arguments to CFRetain/CFRelease">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def CFErrorChecker : Checker<"CFError">,
  InPackage<MacOSX>,
  HelpText<"Check usage of CFErrorRef* parameters">,
  DescFile<"NSErrorChecker.cpp">;

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

def AnalyzerStatsChecker : Checker<"Stats">,
  InPackage<Debug>,
  HelpText<"Emit warnings with analyzer statistics">,
  DescFile<"AnalyzerStatsChecker.cpp">;

//===----------------------------------------------------------------------===//
// Hidden experimental checkers.
//===----------------------------------------------------------------------===//

let Group = AllExperimental in {

def ArrayBoundCheckerV2 : Checker<"Overflow">,
  InPackage<CoreExperimental>,
  HelpText<"Warn about buffer overflows">,
  DescFile<"ArrayBoundCheckerV2.cpp">,
  Hidden; // Must be specified explicitly in order to run.

def MallocChecker : Checker<"Malloc">,
  InPackage<CoreExperimental>,
  HelpText<"Check for potential memory leaks, double free, and use-after-free problems">,
  DescFile<"MallocChecker.cpp">;

def CStringChecker : Checker<"CString">,
  InPackage<CoreExperimental>,
  HelpText<"Check calls to functions in <string.h>">,
  DescFile<"CStringChecker.cpp">;

def UnreachableCodeChecker : Checker<"UnreachableCode">,
  InPackage<CoreExperimental>,
  HelpText<"Check unreachable code">,
  DescFile<"UnreachableCodeChecker.cpp">,
  Hidden; // Must be specified explicitly in order to run.

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

def ReturnPointerRangeChecker : Checker<"ReturnPtrRange">,
  InPackage<CoreExperimental>,
  HelpText<"Check for an out-of-bound pointer being returned to callers">,
  DescFile<"ReturnPointerRangeChecker.cpp">;

def ArrayBoundChecker : Checker<"ArrayBound">,
  InPackage<CoreExperimental>,
  HelpText<"Check for an out-of-bound pointer being returned to callers">,
  DescFile<"ArrayBoundChecker.cpp">;

def CastSizeChecker : Checker<"CastSize">,
  InPackage<CoreExperimental>,
  HelpText<"Check when casting a malloc'ed type T, whether the size is a multiple of the size of T">,
  DescFile<"CastSizeChecker.cpp">;

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

def IteratorsChecker : Checker<"Iterators">,
  InPackage<CoreExperimental>,
  HelpText<"Check improper uses of STL vector iterators">,
  DescFile<"IteratorsChecker.cpp">;

}
