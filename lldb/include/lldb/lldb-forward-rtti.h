//===-- lldb-forward-rtti.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_forward_rtti_h_
#define LLDB_forward_rtti_h_

#if defined(__cplusplus)

#include "lldb/lldb-types.h"
#include "lldb/lldb-forward.h"

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb {

    typedef SharedPtr<lldb_private::ABI>::Type ABISP;
    typedef SharedPtr<lldb_private::AddressResolver>::Type AddressResolverSP;
    typedef SharedPtr<lldb_private::Baton>::Type BatonSP;
    typedef SharedPtr<lldb_private::Block>::Type BlockSP;
    typedef SharedPtr<lldb_private::Breakpoint>::Type BreakpointSP;
    typedef SharedPtr<lldb_private::BreakpointSite>::Type BreakpointSiteSP;
    typedef SharedPtr<lldb_private::BreakpointLocation>::Type BreakpointLocationSP;
    typedef SharedPtr<lldb_private::BreakpointResolver>::Type BreakpointResolverSP;
    typedef SharedPtr<lldb_private::Broadcaster>::Type BroadcasterSP;
    typedef SharedPtr<lldb_private::CommandObject>::Type CommandObjectSP;
    typedef SharedPtr<lldb_private::Communication>::Type CommunicationSP;
    typedef SharedPtr<lldb_private::CompileUnit>::Type CompUnitSP;
    typedef SharedPtr<lldb_private::DataBuffer>::Type DataBufferSP;
    typedef SharedPtr<lldb_private::Debugger>::Type DebuggerSP;
    typedef SharedPtr<lldb_private::DynamicLoader>::Type DynamicLoaderSP;
    typedef SharedPtr<lldb_private::Event>::Type EventSP;
    typedef SharedPtr<lldb_private::Function>::Type FunctionSP;
    typedef SharedPtr<lldb_private::InlineFunctionInfo>::Type InlineFunctionInfoSP;
    typedef SharedPtr<lldb_private::InputReader>::Type InputReaderSP;
    typedef SharedPtr<lldb_private::LineTable>::Type LineTableSP;
    typedef SharedPtr<lldb_private::Listener>::Type ListenerSP;
    typedef SharedPtr<lldb_private::Log>::Type LogSP;
    typedef SharedPtr<lldb_private::LogChannel>::Type LogChannelSP;
    typedef SharedPtr<lldb_private::Module>::Type ModuleSP;
    typedef SharedPtr<lldb_private::Process>::Type ProcessSP;
    typedef SharedPtr<lldb_private::RegisterContext>::Type RegisterContextSP;
    typedef SharedPtr<lldb_private::Section>::Type SectionSP;
    typedef SharedPtr<lldb_private::SearchFilter>::Type SearchFilterSP;
    typedef SharedPtr<lldb_private::StackFrame>::Type StackFrameSP;
    typedef SharedPtr<lldb_private::StateVariable>::Type StateVariableSP;
    typedef SharedPtr<lldb_private::StopInfo>::Type StopInfoSP;
    typedef SharedPtr<lldb_private::StoppointLocation>::Type StoppointLocationSP;
    typedef SharedPtr<lldb_private::Stream>::Type StreamSP;
    typedef SharedPtr<lldb_private::SymbolFile>::Type SymbolFileSP;
    typedef SharedPtr<lldb_private::Target>::Type TargetSP;
    typedef SharedPtr<lldb_private::Thread>::Type ThreadSP;
    typedef SharedPtr<lldb_private::ThreadPlan>::Type ThreadPlanSP;
    typedef SharedPtr<lldb_private::Type>::Type TypeSP;
    typedef SharedPtr<lldb_private::ValueObject>::Type ValueObjectSP;
    typedef SharedPtr<lldb_private::Variable>::Type VariableSP;
    typedef SharedPtr<lldb_private::VariableList>::Type VariableListSP;

} // namespace lldb

#endif  // #if defined(__cplusplus)

#endif  // LLDB_forward_rtti_h_
