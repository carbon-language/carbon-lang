//===-- lldb-forward.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_forward_h_
#define LLDB_lldb_forward_h_

#if defined(__cplusplus)

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb_private {

class   ABI;
class   Address;
class   AddressRange;
class   AddressResolver;
class   ArchSpec;
class   ArchDefaultUnwindPlan;
class   ArchVolatileRegs;
class   Args;
class   Baton;
class   Block;
class   Breakpoint;
class   BreakpointID;
class   BreakpointIDList;
class   BreakpointList;
class   BreakpointLocation;
class   BreakpointLocationCollection;
class   BreakpointLocationList;
class   BreakpointOptions;
class   BreakpointResolver;
class   BreakpointSite;
class   BreakpointSiteList;
class   Broadcaster;
class   CPPLanguageRuntime;
class   ClangASTContext;
class   ClangASTType;
class   ClangNamespaceDecl;
class   ClangExpression;
class   ClangExpressionDeclMap;
class   ClangExpressionParser;
class   ClangExpressionVariable;
class   ClangExpressionVariableList;
class   ClangExpressionVariableList;
class   ClangExpressionVariables;
class   ClangPersistentVariables;
class   ClangUserExpression;
class   CommandInterpreter;
class   CommandObject;
class   CommandReturnObject;
class   Communication;
class   CompileUnit;
class   Condition;
class   Connection;
class   ConnectionFileDescriptor;
class   ConstString;
class   DWARFCallFrameInfo;
class   DWARFExpression;
class   DataBuffer;
class   DataExtractor;
class   Debugger;
class   Declaration;
class   Disassembler;
class   DynamicLoader;
class   EmulateInstruction;
class   Error;
class   Event;
class   EventData;
class   ExecutionContext;
class   ExecutionContextScope;
class   FileSpec;
class   FileSpecList;
class   Flags;
class   FuncUnwinders;
class   Function;
class   FunctionInfo;
class   InlineFunctionInfo;
class   InputReader;
class   InstanceSettings;
class   Instruction;
class   LanguageRuntime;
class   LineTable;
class   Listener;
class   Log;
class   LogChannel;
class   Mangled;
class   Module;
class   ModuleList;
class   Mutex;
class   NameSearchContext;
class   ObjCLanguageRuntime;
class   ObjectContainer;
class   ObjectFile;
class   Options;
class   OptionValue;
class   NamedOption;
class   PathMappingList;
class   Platform;
class   Process;
class   ProcessInfo;
class   ProcessInstanceInfo;
class   ProcessInstanceInfoList;
class   ProcessInstanceInfoMatch;
class   ProcessLaunchInfo;
class   RegisterContext;
class   RegisterLocation;
class   RegisterLocationList;
class   RegularExpression;
class   Scalar;
class   ScriptInterpreter;
class   ScriptInterpreterPython;
class   SearchFilter;
class   Section;
class   SectionList;
class   SourceManager;
class   StackFrame;
class   StackFrameList;
class   StackID;
class   StopInfo;
class   Stoppoint;
class   StoppointCallbackContext;
class   StoppointLocation;
class   Stream;
class   StreamFile;
class   StreamString;
class   StringList;
class   Symbol;
class   SymbolContext;
class   SymbolContextList;
class   SymbolContextScope;
class   SymbolContextSpecifier;
class   SymbolFile;
class   SymbolVendor;
class   Symtab;
class   Target;
class   TargetList;
class   Thread;
class   ThreadList;
class   ThreadPlan;
class   ThreadPlanBase;
class   ThreadPlanRunToAddress;
class   ThreadPlanStepInstruction;
class   ThreadPlanStepOut;
class   ThreadPlanStepOverBreakpoint;
class   ThreadPlanStepRange;
class   ThreadPlanStepThrough;
class   ThreadPlanTracer;
class   ThreadSpec;
class   TimeValue;
class   Type;
class   TypeList;
class   UUID;
class   Unwind;
class   UnwindAssemblyProfiler;
class   UnwindPlan;
class   UnwindTable;
class   UserSettingsController;
class   VMRange;
class   Value;
class   ValueList;
class   ValueObject;
class   ValueObjectList;
class   Variable;
class   VariableList;
class   WatchpointLocation;
struct  LineEntry;

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // LLDB_lldb_forward_h_
