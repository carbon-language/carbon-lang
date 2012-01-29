//===-- lldb-forward-rtti.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_forward_rtti_h_
#define LLDB_lldb_forward_rtti_h_

#if defined(__cplusplus)

#include "lldb/lldb-types.h"
#include <tr1/memory> // for std::tr1::shared_ptr

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb {

    typedef std::tr1::shared_ptr<lldb_private::ABI> ABISP;
    typedef std::tr1::shared_ptr<lldb_private::Baton> BatonSP;
    typedef std::tr1::shared_ptr<lldb_private::Block> BlockSP;
    typedef std::tr1::shared_ptr<lldb_private::Breakpoint> BreakpointSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::BreakpointSite> BreakpointSiteSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::BreakpointLocation> BreakpointLocationSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::BreakpointResolver> BreakpointResolverSP;
    typedef std::tr1::shared_ptr<lldb_private::Broadcaster> BroadcasterSP;
    typedef std::tr1::shared_ptr<lldb_private::ClangExpressionVariable> ClangExpressionVariableSP;
    typedef std::tr1::shared_ptr<lldb_private::CommandObject> CommandObjectSP;
    typedef std::tr1::shared_ptr<lldb_private::Communication> CommunicationSP;
    typedef std::tr1::shared_ptr<lldb_private::Connection> ConnectionSP;
    typedef std::tr1::shared_ptr<lldb_private::CompileUnit> CompUnitSP;
    typedef std::tr1::shared_ptr<lldb_private::DataBuffer> DataBufferSP;
    typedef std::tr1::shared_ptr<lldb_private::DataExtractor> DataExtractorSP;
    typedef std::tr1::shared_ptr<lldb_private::Debugger> DebuggerSP; // make_shared_from_this
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
    typedef std::tr1::shared_ptr<lldb_private::Module> ModuleSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::ObjectFile> ObjectFileSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::OptionValue> OptionValueSP;
    typedef std::tr1::shared_ptr<lldb_private::Platform> PlatformSP;
    typedef std::tr1::shared_ptr<lldb_private::Process> ProcessSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::RegisterContext> RegisterContextSP;
    typedef std::tr1::shared_ptr<lldb_private::RegularExpression> RegularExpressionSP;
    typedef std::tr1::shared_ptr<lldb_private::Section> SectionSP;
    typedef std::tr1::shared_ptr<lldb_private::SearchFilter> SearchFilterSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef std::tr1::shared_ptr<lldb_private::ScriptSummaryFormat> ScriptFormatSP;
#endif // #ifndef LLDB_DISABLE_PYTHON
    typedef std::tr1::shared_ptr<lldb_private::StackFrame> StackFrameSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::StackFrameList> StackFrameListSP;
    typedef std::tr1::shared_ptr<lldb_private::StopInfo> StopInfoSP;
    typedef std::tr1::shared_ptr<lldb_private::StoppointLocation> StoppointLocationSP;
    typedef std::tr1::shared_ptr<lldb_private::Stream> StreamSP;
    typedef std::tr1::shared_ptr<lldb_private::StringSummaryFormat> StringSummaryFormatSP;
    typedef std::tr1::shared_ptr<lldb_private::SummaryFormat> SummaryFormatSP;
    typedef std::tr1::shared_ptr<lldb_private::SymbolFile> SymbolFileSP;
    typedef std::tr1::shared_ptr<lldb_private::SymbolFileType> SymbolFileTypeSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::SymbolContextSpecifier> SymbolContextSpecifierSP;
    typedef std::tr1::shared_ptr<lldb_private::SyntheticChildren> SyntheticChildrenSP;
    typedef std::tr1::shared_ptr<lldb_private::SyntheticChildrenFrontEnd> SyntheticChildrenFrontEndSP;
    typedef std::tr1::shared_ptr<lldb_private::Target> TargetSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::Thread> ThreadSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::ThreadPlan> ThreadPlanSP;
    typedef std::tr1::shared_ptr<lldb_private::ThreadPlanTracer> ThreadPlanTracerSP;
    typedef std::tr1::shared_ptr<lldb_private::Type> TypeSP; // make_shared_from_this
    typedef std::tr1::shared_ptr<lldb_private::TypeImpl> TypeImplSP;
    typedef std::tr1::shared_ptr<lldb_private::FuncUnwinders> FuncUnwindersSP;
    typedef std::tr1::shared_ptr<lldb_private::UserSettingsController> UserSettingsControllerSP;
    typedef std::tr1::shared_ptr<lldb_private::UnwindPlan> UnwindPlanSP;
    typedef SharedPtr<lldb_private::ValueObject>::Type ValueObjectSP;
    typedef std::tr1::shared_ptr<lldb_private::Value> ValueSP;
    typedef std::tr1::shared_ptr<lldb_private::ValueFormat> ValueFormatSP;
    typedef std::tr1::shared_ptr<lldb_private::ValueList> ValueListSP;
    typedef std::tr1::shared_ptr<lldb_private::Variable> VariableSP;
    typedef std::tr1::shared_ptr<lldb_private::VariableList> VariableListSP;
    typedef std::tr1::shared_ptr<lldb_private::ValueObjectList> ValueObjectListSP;
    typedef std::tr1::shared_ptr<lldb_private::Watchpoint> WatchpointSP;

} // namespace lldb

#endif  // #if defined(__cplusplus)

#endif  // LLDB_lldb_forward_rtti_h_
