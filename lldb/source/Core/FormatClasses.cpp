//===-- FormatClasses.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes

#ifdef LLDB_DISABLE_PYTHON

struct PyObject;

#else   // #ifdef LLDB_DISABLE_PYTHON

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#endif  // #ifdef LLDB_DISABLE_PYTHON

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

TypeFormatImpl::TypeFormatImpl (lldb::Format f,
                          const Flags& flags) : 
    m_flags(flags),
    m_format (f)
{
}

std::string
TypeFormatImpl::GetDescription()
{
    StreamString sstr;
    sstr.Printf ("%s%s%s%s\n", 
                 FormatManager::GetFormatAsCString (GetFormat()),
                 Cascades() ? "" : " (not cascading)",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "");
    return sstr.GetString();
}

TypeSummaryImpl::TypeSummaryImpl(const TypeSummaryImpl::Flags& flags) :
    m_flags(flags)
{
}


StringSummaryFormat::StringSummaryFormat(const TypeSummaryImpl::Flags& flags,
                                         const char *format_cstr) :
    TypeSummaryImpl(flags),
    m_format()
{
  if (format_cstr)
    m_format.assign(format_cstr);
}

std::string
StringSummaryFormat::FormatObject(lldb::ValueObjectSP object)
{
    if (!object.get())
        return "NULL";
    
    StreamString s;
    ExecutionContext exe_ctx (object->GetExecutionContextRef());
    SymbolContext sc;
    StackFrame *frame = exe_ctx.GetFramePtr();
    if (frame)
        sc = frame->GetSymbolContext(lldb::eSymbolContextEverything);
    
    if (IsOneliner())
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
                    if (!HideNames())
                    {
                        s.PutCString(child_sp.get()->GetName().AsCString());
                        s.PutChar('=');
                    }
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
    
    sstr.Printf ("`%s`%s%s%s%s%s%s%s",      m_format.c_str(),
                 Cascades() ? "" : " (not cascading)",
                 !DoesPrintChildren() ? "" : " (show children)",
                 !DoesPrintValue() ? " (hide value)" : "",
                 IsOneliner() ? " (one-line printout)" : "",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "",
                 HideNames() ? " (hide member names)" : "");
    return sstr.GetString();
}

#ifndef LLDB_DISABLE_PYTHON


ScriptSummaryFormat::ScriptSummaryFormat(const TypeSummaryImpl::Flags& flags,
                                         const char * function_name,
                                         const char * python_script) :
    TypeSummaryImpl(flags),
    m_function_name(),
    m_python_script()
{
   if (function_name)
     m_function_name.assign(function_name);
   if (python_script)
     m_python_script.assign(python_script);
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
    sstr.Printf ("%s%s%s%s%s%s%s\n%s",       Cascades() ? "" : " (not cascading)",
                 !DoesPrintChildren() ? "" : " (show children)",
                 !DoesPrintValue() ? " (hide value)" : "",
                 IsOneliner() ? " (one-line printout)" : "",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "",
                 HideNames() ? " (hide member names)" : "",
                 m_python_script.c_str());
    return sstr.GetString();
    
}

#endif // #ifndef LLDB_DISABLE_PYTHON

std::string
TypeFilterImpl::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s {\n",
                Cascades() ? "" : " (not cascading)",
                SkipsPointers() ? " (skip pointers)" : "",
                SkipsReferences() ? " (skip references)" : "");
    
    for (int i = 0; i < GetCount(); i++)
    {
        sstr.Printf("    %s\n",
                    GetExpressionPathAtIndex(i));
    }
                    
    sstr.Printf("}");
    return sstr.GetString();
}

std::string
SyntheticArrayView::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s {\n",
                Cascades() ? "" : " (not cascading)",
                SkipsPointers() ? " (skip pointers)" : "",
                SkipsReferences() ? " (skip references)" : "");
    
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

#ifndef LLDB_DISABLE_PYTHON

TypeSyntheticImpl::FrontEnd::FrontEnd(std::string pclass, lldb::ValueObjectSP be) :
    SyntheticChildrenFrontEnd(be),
    m_python_class(pclass)
{
    if (be.get() == NULL)
    {
        m_interpreter = NULL;
        m_wrapper = NULL;
        return;
    }
    
    m_interpreter = m_backend->GetTargetSP()->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
    
    if (m_interpreter == NULL)
        m_wrapper = NULL;
    else
        m_wrapper = m_interpreter->CreateSyntheticScriptedProvider(m_python_class, m_backend);
}

TypeSyntheticImpl::FrontEnd::~FrontEnd()
{
    Py_XDECREF((PyObject*)m_wrapper);
}

lldb::ValueObjectSP
TypeSyntheticImpl::FrontEnd::GetChildAtIndex (uint32_t idx, bool can_create)
{
    if (m_wrapper == NULL || m_interpreter == NULL)
        return lldb::ValueObjectSP();
    
    return m_interpreter->GetChildAtIndex(m_wrapper, idx);
}

std::string
TypeSyntheticImpl::GetDescription()
{
    StreamString sstr;
    sstr.Printf("%s%s%s Python class %s",
                Cascades() ? "" : " (not cascading)",
                SkipsPointers() ? " (skip pointers)" : "",
                SkipsReferences() ? " (skip references)" : "",
                m_python_class.c_str());
    
    return sstr.GetString();
}

#endif // #ifndef LLDB_DISABLE_PYTHON

int
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
