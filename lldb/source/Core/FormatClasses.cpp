//===-- FormatClasses.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes

// C++ Includes
#include <ostream>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/FormatClasses.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

std::string
ValueFormat::FormatObject(lldb::ValueObjectSP object)
{
    if (!object.get())
        return "NULL";
    
    StreamString sstr;
    
    if (ClangASTType::DumpTypeValue (object->GetClangAST(),            // The clang AST
                                     object->GetClangType(),           // The clang type to display
                                     &sstr,
                                     m_format,                          // Format to display this type with
                                     object->GetDataExtractor(),       // Data to extract from
                                     0,                                // Byte offset into "data"
                                     object->GetByteSize(),            // Byte size of item in "data"
                                     object->GetBitfieldBitSize(),     // Bitfield bit size
                                     object->GetBitfieldBitOffset()))  // Bitfield bit offset
        return (sstr.GetString());
    else
    {
        return ("unsufficient data for value");
    }
}

std::string
StringSummaryFormat::FormatObject(lldb::ValueObjectSP object)
{
    if (!object.get())
        return "NULL";
    
    StreamString s;
    ExecutionContext exe_ctx;
    object->GetExecutionContextScope()->CalculateExecutionContext(exe_ctx);
    SymbolContext sc;
    if (exe_ctx.frame)
        sc = exe_ctx.frame->GetSymbolContext(lldb::eSymbolContextEverything);
    
    if (m_show_members_oneliner)
    {
        ValueObjectSP synth_vobj = object->GetSyntheticValue(lldb::eUseSyntheticFilter);
        const uint32_t num_children = synth_vobj->GetNumChildren();
        if (num_children)
        {
            s.PutChar('(');
            
            for (uint32_t idx=0; idx<num_children; ++idx)
            {
                lldb::ValueObjectSP child_sp(synth_vobj->GetChildAtIndex(idx, true));
                if (child_sp.get())
                {
                    if (idx)
                        s.PutCString(", ");
                    s.PutCString(child_sp.get()->GetName().AsCString());
                    s.PutChar('=');
                    s.PutCString(child_sp.get()->GetPrintableRepresentation());
                }
            }
            
            s.PutChar(')');
            
            return s.GetString();
        }
        else
            return "";
        
    }
    else
    {
        if (Debugger::FormatPrompt(m_format.c_str(), &sc, &exe_ctx, &sc.line_entry.range.GetBaseAddress(), s, NULL, object.get()))
            return s.GetString();
        else
            return "";
    }
}

std::string
StringSummaryFormat::GetDescription()
{
    StreamString sstr;
    sstr.Printf ("`%s`%s%s%s%s%s%s",      m_format.c_str(),
                 m_cascades ? "" : " (not cascading)",
                 m_dont_show_children ? "" : " (show children)",
                 m_dont_show_value ? " (hide value)" : "",
                 m_show_members_oneliner ? " (one-line printout)" : "",
                 m_skip_pointers ? " (skip pointers)" : "",
                 m_skip_references ? " (skip references)" : "");
    return sstr.GetString();
}

std::string
ScriptSummaryFormat::FormatObject(lldb::ValueObjectSP object)
{
    return std::string(ScriptInterpreterPython::CallPythonScriptFunction(m_function_name.c_str(),
                                                                         object).c_str());
}

std::string
ScriptSummaryFormat::GetDescription()
{
    StreamString sstr;
    sstr.Printf ("%s%s%s%s%s%s\n%s",       m_cascades ? "" : " (not cascading)",
                 m_dont_show_children ? "" : " (show children)",
                 m_dont_show_value ? " (hide value)" : "",
                 m_show_members_oneliner ? " (one-line printout)" : "",
                 m_skip_pointers ? " (skip pointers)" : "",
                 m_skip_references ? " (skip references)" : "",
                 m_python_script.c_str());
    return sstr.GetString();
    
}

std::string
SyntheticFilter::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s {\n",
                m_cascades ? "" : " (not cascading)",
                m_skip_pointers ? " (skip pointers)" : "",
                m_skip_references ? " (skip references)" : "");
    
    for (int i = 0; i < GetCount(); i++)
    {
        sstr.Printf("    %s\n",
                    GetExpressionPathAtIndex(i).c_str());
    }
                    
    sstr.Printf("}");
    return sstr.GetString();
}

SyntheticScriptProvider::FrontEnd::FrontEnd(std::string pclass,
                                            lldb::ValueObjectSP be) :
SyntheticChildrenFrontEnd(be),
m_python_class(pclass)
{
    m_interpreter = be->GetUpdatePoint().GetTarget()->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    m_wrapper = (PyObject*)m_interpreter->CreateSyntheticScriptedProvider(m_python_class, m_backend);
}

std::string
SyntheticScriptProvider::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s Python class: %s",
                m_cascades ? "" : " (not cascading)",
                m_skip_pointers ? " (skip pointers)" : "",
                m_skip_references ? " (skip references)" : "",
                m_python_class.c_str());
    
    return sstr.GetString();
}