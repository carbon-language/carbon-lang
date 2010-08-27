//===-- lldb-forward.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_forward_h_
#define LLDB_forward_h_

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
class   Args;
class   Baton;
class   Block;
class   Breakpoint;
class   BreakpointID;
class   BreakpointIDList;
class   BreakpointSite;
class   BreakpointSiteList;
class   BreakpointList;
class   BreakpointLocation;
class   BreakpointLocationCollection;
class   BreakpointLocationList;
class   BreakpointOptions;
class   BreakpointResolver;
class   Broadcaster;
class   ClangASTContext;
class   ClangExpression;
class   ClangExpressionDeclMap;
class   ClangExpressionVariableList;
class   ClangExpressionVariableStore;
class   Debugger;
class   CommandInterpreter;
class   CommandObject;
class   CommandReturnObject;
class   Communication;
class   Condition;
class   CompileUnit;
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
class   Error;
class   Event;
class   EventData;
class   ExecutionContext;
class   ExecutionContextScope;
class   FileSpec;
class   FileSpecList;
class   Flags;
class   Function;
class   FunctionInfo;
class   InlineFunctionInfo;
class   InputReader;
struct  LineEntry;
class   LineTable;
class   Listener;
class   Log;
class   LogChannel;
class   Mangled;
class   Module;
class   ModuleList;
class   Mutex;
class   ObjCObjectPrinter;
class   ObjectContainer;
class   ObjectFile;
class   Options;
class   PathMappingList;
class   Process;
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
class   StateVariable;
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
class   SymbolFile;
class   SymbolVendor;
class   Symtab;
class   Target;
class   TargetList;
class   Thread;
class   ThreadList;
class   ThreadPlan;
class   ThreadPlanBase;
class   ThreadPlanStepInstruction;
class   ThreadPlanStepOut;
class   ThreadPlanStepOverBreakpoint;
class   ThreadPlanStepThrough;
class   ThreadPlanStepRange;
class   ThreadPlanRunToAddress;
class   ThreadSpec;
class   TimeValue;
class   Type;
class   TypeList;
class   Unwind;
class   UUID;
class   VMRange;
class   Value;
class   ValueList;
class   ValueObject;
class   ValueObjectList;
class   Variable;
class   VariableList;
class   WatchpointLocation;

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // LLDB_forward_h_
