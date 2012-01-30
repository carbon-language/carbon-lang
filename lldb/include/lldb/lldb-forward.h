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

#include <tr1/memory> // for std::tr1::shared_ptr, std::tr1::weak_ptr
#include "lldb/Utility/SharingPtr.h"

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb_private {

class   ABI;
class   Address;
class   AddressImpl;
class   AddressRange;
class   AddressResolver;
class   ArchSpec;
class   Args;
class   ASTResultSynthesizer;
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
class   ClangASTImporter;
class   ClangASTSource;
class   ClangASTType;
class   ClangNamespaceDecl;
class   ClangExpression;
class   ClangExpressionDeclMap;
class   ClangExpressionParser;
class   ClangExpressionVariable;
class   ClangExpressionVariableList;
class   ClangExpressionVariableList;
class   ClangExpressionVariables;
class   ClangFunction;
class   ClangPersistentVariables;
class   ClangUserExpression;
class   ClangUtilityFunction;
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
class   DataEncoder;
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
class   FormatCategory;
class   FormatManager;
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
struct  NameSearchContext;
class   ObjCLanguageRuntime;
class   ObjectContainer;
class   OptionGroup;
class   OptionGroupPlatform;
class   ObjectFile;
class   OperatingSystem;
class   Options;
class   OptionValue;
class   NamedOption;
class   PathMappingList;
class   Platform;
class   Process;
class   ProcessAttachInfo;
class   ProcessModID;
class   ProcessInfo;
class   ProcessInstanceInfo;
class   ProcessInstanceInfoList;
class   ProcessInstanceInfoMatch;
class   ProcessLaunchInfo;
class   RegisterContext;
class   RegisterLocation;
class   RegisterLocationList;
class   RegisterValue;
class   RegularExpression;
class   Scalar;
class   ScriptInterpreter;
#ifndef LLDB_DISABLE_PYTHON
class   ScriptInterpreterPython;
struct  ScriptSummaryFormat;
#endif
class   SearchFilter;
class   Section;
class   SectionImpl;
class   SectionList;
class   SourceManager;
class   SourceManagerImpl;
class   StackFrame;
class   StackFrameImpl;
class   StackFrameList;
class   StackID;
class   StopInfo;
class   Stoppoint;
class   StoppointCallbackContext;
class   StoppointLocation;
class   Stream;
template <unsigned N> class StreamBuffer;
class   StreamFile;
class   StreamString;
class   StringList;
struct  StringSummaryFormat;
struct  SummaryFormat;
class   Symbol;
class   SymbolContext;
class   SymbolContextList;
class   SymbolContextScope;
class   SymbolContextSpecifier;
class   SymbolFile;
class   SymbolFileType;
class   SymbolVendor;
class   Symtab;
class   SyntheticChildren;
class   SyntheticChildrenFrontEnd;
#ifndef LLDB_DISABLE_PYTHON
class   SyntheticScriptProvider;
#endif
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
class   TypeImpl;
class   TypeAndOrName;
class   TypeList;
class   TypeListImpl;
class   TypeMemberImpl;    
class   UUID;
class   Unwind;
class   UnwindAssembly;
class   UnwindPlan;
class   UnwindTable;
class   UserSettingsController;
class   VMRange;
class   Value;
struct  ValueFormat;
class   ValueList;
class   ValueObject;
class   ValueObjectChild;
class   ValueObjectConstResult;
class   ValueObjectConstResultChild;
class   ValueObjectConstResultImpl;
class   ValueObjectList;
class   Variable;
class   VariableList;
class   Watchpoint;
class   WatchpointList;
struct  LineEntry;

} // namespace lldb_private

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb {
    
    typedef std::tr1::shared_ptr<lldb_private::ABI> ABISP;
    typedef std::tr1::shared_ptr<lldb_private::Baton> BatonSP;
    typedef std::tr1::shared_ptr<lldb_private::Block> BlockSP;
    typedef std::tr1::shared_ptr<lldb_private::Breakpoint> BreakpointSP;
    typedef std::tr1::weak_ptr<lldb_private::Breakpoint> BreakpointWP;
    typedef std::tr1::shared_ptr<lldb_private::BreakpointSite> BreakpointSiteSP;
    typedef std::tr1::weak_ptr<lldb_private::BreakpointSite> BreakpointSiteWP;
    typedef std::tr1::shared_ptr<lldb_private::BreakpointLocation> BreakpointLocationSP;
    typedef std::tr1::weak_ptr<lldb_private::BreakpointLocation> BreakpointLocationWP;
    typedef std::tr1::shared_ptr<lldb_private::BreakpointResolver> BreakpointResolverSP;
    typedef std::tr1::shared_ptr<lldb_private::Broadcaster> BroadcasterSP;
    typedef std::tr1::shared_ptr<lldb_private::ClangExpressionVariable> ClangExpressionVariableSP;
    typedef std::tr1::shared_ptr<lldb_private::CommandObject> CommandObjectSP;
    typedef std::tr1::shared_ptr<lldb_private::Communication> CommunicationSP;
    typedef std::tr1::shared_ptr<lldb_private::Connection> ConnectionSP;
    typedef std::tr1::shared_ptr<lldb_private::CompileUnit> CompUnitSP;
    typedef std::tr1::shared_ptr<lldb_private::DataBuffer> DataBufferSP;
    typedef std::tr1::shared_ptr<lldb_private::DataExtractor> DataExtractorSP;
    typedef std::tr1::shared_ptr<lldb_private::Debugger> DebuggerSP;
    typedef std::tr1::weak_ptr<lldb_private::Debugger> DebuggerWP;
    typedef std::tr1::shared_ptr<lldb_private::Disassembler> DisassemblerSP;
    typedef std::tr1::shared_ptr<lldb_private::DynamicLoader> DynamicLoaderSP;
    typedef std::tr1::shared_ptr<lldb_private::Event> EventSP;
    typedef std::tr1::shared_ptr<lldb_private::FormatCategory> FormatCategorySP;
    typedef std::tr1::shared_ptr<lldb_private::Function> FunctionSP;
    typedef std::tr1::shared_ptr<lldb_private::InlineFunctionInfo> InlineFunctionInfoSP;
    typedef std::tr1::shared_ptr<lldb_private::InputReader> InputReaderSP;
    typedef std::tr1::shared_ptr<lldb_private::InstanceSettings> InstanceSettingsSP;
    typedef std::tr1::shared_ptr<lldb_private::Instruction> InstructionSP;
    typedef std::tr1::shared_ptr<lldb_private::LanguageRuntime> LanguageRuntimeSP;
    typedef std::tr1::shared_ptr<lldb_private::LineTable> LineTableSP;
    typedef std::tr1::shared_ptr<lldb_private::Listener> ListenerSP;
    typedef std::tr1::shared_ptr<lldb_private::Log> LogSP;
    typedef std::tr1::shared_ptr<lldb_private::LogChannel> LogChannelSP;
    typedef std::tr1::shared_ptr<lldb_private::Module> ModuleSP;
    typedef std::tr1::weak_ptr<lldb_private::Module> ModuleWP;
    typedef std::tr1::shared_ptr<lldb_private::ObjectFile> ObjectFileSP;
    typedef std::tr1::weak_ptr<lldb_private::ObjectFile> ObjectFileWP;
    typedef std::tr1::shared_ptr<lldb_private::OptionValue> OptionValueSP;
    typedef std::tr1::shared_ptr<lldb_private::Platform> PlatformSP;
    typedef std::tr1::shared_ptr<lldb_private::Process> ProcessSP;
    typedef std::tr1::weak_ptr<lldb_private::Process> ProcessWP;
    typedef std::tr1::shared_ptr<lldb_private::RegisterContext> RegisterContextSP;
    typedef std::tr1::shared_ptr<lldb_private::RegularExpression> RegularExpressionSP;
    typedef std::tr1::shared_ptr<lldb_private::Section> SectionSP;
    typedef std::tr1::shared_ptr<lldb_private::SearchFilter> SearchFilterSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef std::tr1::shared_ptr<lldb_private::ScriptSummaryFormat> ScriptFormatSP;
#endif // #ifndef LLDB_DISABLE_PYTHON
    typedef std::tr1::shared_ptr<lldb_private::StackFrame> StackFrameSP;
    typedef std::tr1::weak_ptr<lldb_private::StackFrame> StackFrameWP;
    typedef std::tr1::shared_ptr<lldb_private::StackFrameList> StackFrameListSP;
    typedef std::tr1::shared_ptr<lldb_private::StopInfo> StopInfoSP;
    typedef std::tr1::shared_ptr<lldb_private::StoppointLocation> StoppointLocationSP;
    typedef std::tr1::shared_ptr<lldb_private::Stream> StreamSP;
    typedef std::tr1::shared_ptr<lldb_private::StringSummaryFormat> StringSummaryFormatSP;
    typedef std::tr1::shared_ptr<lldb_private::SummaryFormat> SummaryFormatSP;
    typedef std::tr1::shared_ptr<lldb_private::SymbolFile> SymbolFileSP;
    typedef std::tr1::shared_ptr<lldb_private::SymbolFileType> SymbolFileTypeSP;
    typedef std::tr1::weak_ptr<lldb_private::SymbolFileType> SymbolFileTypeWP;
    typedef std::tr1::shared_ptr<lldb_private::SymbolContextSpecifier> SymbolContextSpecifierSP;
    typedef std::tr1::shared_ptr<lldb_private::SyntheticChildren> SyntheticChildrenSP;
    typedef std::tr1::shared_ptr<lldb_private::SyntheticChildrenFrontEnd> SyntheticChildrenFrontEndSP;
    typedef std::tr1::shared_ptr<lldb_private::Target> TargetSP;
    typedef std::tr1::weak_ptr<lldb_private::Target> TargetWP;
    typedef std::tr1::shared_ptr<lldb_private::Thread> ThreadSP;
    typedef std::tr1::weak_ptr<lldb_private::Thread> ThreadWP;
    typedef std::tr1::shared_ptr<lldb_private::ThreadPlan> ThreadPlanSP;
    typedef std::tr1::shared_ptr<lldb_private::ThreadPlanTracer> ThreadPlanTracerSP;
    typedef std::tr1::shared_ptr<lldb_private::Type> TypeSP;
    typedef std::tr1::weak_ptr<lldb_private::Type> TypeWP;
    typedef std::tr1::shared_ptr<lldb_private::TypeImpl> TypeImplSP;
    typedef std::tr1::shared_ptr<lldb_private::FuncUnwinders> FuncUnwindersSP;
    typedef std::tr1::shared_ptr<lldb_private::UserSettingsController> UserSettingsControllerSP;
    typedef std::tr1::weak_ptr<lldb_private::UserSettingsController> UserSettingsControllerWP;
    typedef std::tr1::shared_ptr<lldb_private::UnwindPlan> UnwindPlanSP;
    typedef lldb_private::SharingPtr<lldb_private::ValueObject> ValueObjectSP;
    typedef std::tr1::shared_ptr<lldb_private::Value> ValueSP;
    typedef std::tr1::shared_ptr<lldb_private::ValueFormat> ValueFormatSP;
    typedef std::tr1::shared_ptr<lldb_private::ValueList> ValueListSP;
    typedef std::tr1::shared_ptr<lldb_private::Variable> VariableSP;
    typedef std::tr1::shared_ptr<lldb_private::VariableList> VariableListSP;
    typedef std::tr1::shared_ptr<lldb_private::ValueObjectList> ValueObjectListSP;
    typedef std::tr1::shared_ptr<lldb_private::Watchpoint> WatchpointSP;
    
} // namespace lldb


#endif  // #if defined(__cplusplus)
#endif  // LLDB_lldb_forward_h_
