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
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

ValueFormat::ValueFormat (lldb::Format f,
                          bool casc,
                          bool skipptr,
                          bool skipref) : 
    m_cascades(casc),
    m_skip_pointers(skipptr),
    m_skip_references(skipref),
    m_format (f)
{
}

SummaryFormat::SummaryFormat(bool casc,
                             bool skipptr,
                             bool skipref,
                             bool nochildren,
                             bool novalue,
                             bool oneliner) :
    m_cascades(casc),
    m_skip_pointers(skipptr),
    m_skip_references(skipref),
    m_dont_show_children(nochildren),
    m_dont_show_value(novalue),
    m_show_members_oneliner(oneliner)
{
}

StringSummaryFormat::StringSummaryFormat(bool casc,
                                         bool skipptr,
                                         bool skipref,
                                         bool nochildren,
                                         bool novalue,
                                         bool oneliner,
                                         std::string f) :
    SummaryFormat(casc,
                  skipptr,
                  skipref,
                  nochildren,
                  novalue,
                  oneliner),
    m_format(f)
{
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
        ValueObjectSP synth_valobj = object->GetSyntheticValue(lldb::eUseSyntheticFilter);
        const uint32_t num_children = synth_valobj->GetNumChildren();
        if (num_children)
        {
            s.PutChar('(');
            
            for (uint32_t idx=0; idx<num_children; ++idx)
            {
                lldb::ValueObjectSP child_sp(synth_valobj->GetChildAtIndex(idx, true));
                if (child_sp.get())
                {
                    if (idx)
                        s.PutCString(", ");
                    s.PutCString(child_sp.get()->GetName().AsCString());
                    s.PutChar('=');
                    child_sp.get()->GetPrintableRepresentation(s);
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

ScriptSummaryFormat::ScriptSummaryFormat(bool casc,
                                         bool skipptr,
                                         bool skipref,
                                         bool nochildren,
                                         bool novalue,
                                         bool oneliner,
                                         std::string fname,
                                         std::string pscri) :
    SummaryFormat(casc,
                  skipptr,
                  skipref,
                  nochildren,
                  novalue,
                  oneliner),
    m_function_name(fname),
    m_python_script(pscri)
{
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

std::string
SyntheticArrayView::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s {\n",
                m_cascades ? "" : " (not cascading)",
                m_skip_pointers ? " (skip pointers)" : "",
                m_skip_references ? " (skip references)" : "");
    SyntheticArrayRange* ptr = &m_head;
    while (ptr && ptr != m_tail)
    {
        if (ptr->GetLow() == ptr->GetHigh())
            sstr.Printf("    [%d]\n",
                        ptr->GetLow());
        else
            sstr.Printf("    [%d-%d]\n",
                        ptr->GetLow(),
                        ptr->GetHigh());
        ptr = ptr->GetNext();
    }
    
    sstr.Printf("}");
    return sstr.GetString();
}

SyntheticScriptProvider::FrontEnd::FrontEnd(std::string pclass,
                                            lldb::ValueObjectSP be) :
    SyntheticChildrenFrontEnd(be),
    m_python_class(pclass)
{
    if (be.get() == NULL)
    {
        m_interpreter = NULL;
        m_wrapper = NULL;
        return;
    }
    
    m_interpreter = m_backend->GetUpdatePoint().GetTargetSP()->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    
    if (m_interpreter == NULL)
        m_wrapper = NULL;
    else
        m_wrapper = (PyObject*)m_interpreter->CreateSyntheticScriptedProvider(m_python_class, m_backend);
}

lldb::ValueObjectSP
SyntheticScriptProvider::FrontEnd::GetChildAtIndex (uint32_t idx, bool can_create)
{
    if (m_wrapper == NULL || m_interpreter == NULL)
        return lldb::ValueObjectSP();
    
    return m_interpreter->GetChildAtIndex(m_wrapper, idx);
}

std::string
SyntheticScriptProvider::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s Python class %s",
                m_cascades ? "" : " (not cascading)",
                m_skip_pointers ? " (skip pointers)" : "",
                m_skip_references ? " (skip references)" : "",
                m_python_class.c_str());
    
    return sstr.GetString();
}

const int
SyntheticArrayView::GetRealIndexForIndex(int i)
{
    if (i >= GetCount())
        return -1;
    
    SyntheticArrayRange* ptr = &m_head;
    
    int residual = i;
    
    while(ptr && ptr != m_tail)
    {
        if (residual >= ptr->GetSelfCount())
        {
            residual -= ptr->GetSelfCount();
            ptr = ptr->GetNext();
        }
        
        return ptr->GetLow() + residual;
    }
    
    return -1;
}

uint32_t
SyntheticArrayView::FrontEnd::GetIndexOfChildWithName (const ConstString &name_cs)
{
    const char* name_cstr = name_cs.GetCString();
    if (*name_cstr != '[')
        return UINT32_MAX;
    std::string name(name_cstr+1);
    if (name[name.size()-1] != ']')
        return UINT32_MAX;
    name = name.erase(name.size()-1,1);
    int index = Args::StringToSInt32 (name.c_str(), -1);
    if (index < 0)
        return UINT32_MAX;
    return index;
}