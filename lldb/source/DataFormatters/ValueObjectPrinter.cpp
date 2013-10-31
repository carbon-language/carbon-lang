//===-- ValueObjectPrinter.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/ValueObjectPrinter.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

ValueObjectPrinter::ValueObjectPrinter (ValueObject* valobj,
                                        Stream* s,
                                        const DumpValueObjectOptions& options)
{
    Init(valobj,s,options,options.m_max_ptr_depth,0);
}

ValueObjectPrinter::ValueObjectPrinter (ValueObject* valobj,
                                        Stream* s,
                                        const DumpValueObjectOptions& options,
                                        uint32_t ptr_depth,
                                        uint32_t curr_depth)
{
    Init(valobj,s,options,ptr_depth,curr_depth);
}

void
ValueObjectPrinter::Init (ValueObject* valobj,
                          Stream* s,
                          const DumpValueObjectOptions& options,
                          uint32_t ptr_depth,
                          uint32_t curr_depth)
{
    m_orig_valobj = valobj;
    m_valobj = nullptr;
    m_stream = s;
    this->options = options;
    m_ptr_depth = ptr_depth;
    m_curr_depth = curr_depth;
    assert (m_orig_valobj && "cannot print a NULL ValueObject");
    assert (m_stream && "cannot print to a NULL Stream");
    m_should_print = eLazyBoolCalculate;
    m_is_nil = eLazyBoolCalculate;
    m_is_ptr = eLazyBoolCalculate;
    m_is_ref = eLazyBoolCalculate;
    m_is_aggregate = eLazyBoolCalculate;
    m_summary_formatter = {nullptr,false};
    m_value.assign("");
    m_summary.assign("");
    m_error.assign("");
}

bool
ValueObjectPrinter::PrintValueObject ()
{
    if (!GetDynamicValueIfNeeded () || m_valobj == nullptr)
        return false;
    
    if (ShouldPrintValueObject())
    {
        PrintLocationIfNeeded();
        m_stream->Indent();
        
        bool show_type = PrintTypeIfNeeded();
        
        PrintNameIfNeeded(show_type);
    }

    bool value_printed = false;
    bool summary_printed = false;
    
    bool val_summary_ok = PrintValueAndSummaryIfNeeded (value_printed,summary_printed);

    if (val_summary_ok)
        PrintChildrenIfNeeded (value_printed, summary_printed);
    else
        m_stream->EOL();
    
    return true;
}

bool
ValueObjectPrinter::GetDynamicValueIfNeeded ()
{
    if (m_valobj)
        return true;
    bool update_success = m_orig_valobj->UpdateValueIfNeeded (true);
    if (!update_success)
    {
        m_valobj = m_orig_valobj;
    }
    else
    {
        if (m_orig_valobj->IsDynamic())
        {
            if (options.m_use_dynamic == eNoDynamicValues)
            {
                ValueObject *static_value = m_orig_valobj->GetStaticValue().get();
                if (static_value)
                    m_valobj = static_value;
                else
                    m_valobj = m_orig_valobj;
            }
            else
                m_valobj = m_orig_valobj;
        }
        else
        {
            if (options.m_use_dynamic != eNoDynamicValues)
            {
                ValueObject *dynamic_value = m_orig_valobj->GetDynamicValue(options.m_use_dynamic).get();
                if (dynamic_value)
                    m_valobj = dynamic_value;
                else
                    m_valobj = m_orig_valobj;
            }
            else
                m_valobj = m_orig_valobj;
        }
    }
    m_clang_type = m_valobj->GetClangType();
    m_type_flags = m_clang_type.GetTypeInfo ();
    return true;
}

const char*
ValueObjectPrinter::GetDescriptionForDisplay ()
{
    const char* str = m_valobj->GetObjectDescription();
    if (!str)
        str = m_valobj->GetSummaryAsCString();
    if (!str)
        str = m_valobj->GetValueAsCString();
    return str;
}

const char*
ValueObjectPrinter::GetRootNameForDisplay (const char* if_fail)
{
    const char *root_valobj_name = options.m_root_valobj_name.empty() ?
        m_valobj->GetName().AsCString() :
        options.m_root_valobj_name.c_str();
    return root_valobj_name ? root_valobj_name : if_fail;
}

bool
ValueObjectPrinter::ShouldPrintValueObject ()
{
    if (m_should_print == eLazyBoolCalculate)
        m_should_print = (options.m_flat_output == false || m_type_flags.Test (ClangASTType::eTypeHasValue)) ? eLazyBoolYes : eLazyBoolNo;
    return m_should_print == eLazyBoolYes;
}

bool
ValueObjectPrinter::IsNil ()
{
    if (m_is_nil == eLazyBoolCalculate)
        m_is_nil = m_valobj->IsObjCNil() ? eLazyBoolYes : eLazyBoolNo;
    return m_is_nil == eLazyBoolYes;
}

bool
ValueObjectPrinter::IsPtr ()
{
    if (m_is_ptr == eLazyBoolCalculate)
        m_is_ptr = m_type_flags.Test (ClangASTType::eTypeIsPointer) ? eLazyBoolYes : eLazyBoolNo;
    return m_is_ptr == eLazyBoolYes;
}

bool
ValueObjectPrinter::IsRef ()
{
    if (m_is_ref == eLazyBoolCalculate)
        m_is_ref = m_type_flags.Test (ClangASTType::eTypeIsReference) ? eLazyBoolYes : eLazyBoolNo;
    return m_is_ref == eLazyBoolYes;
}

bool
ValueObjectPrinter::IsAggregate ()
{
    if (m_is_aggregate == eLazyBoolCalculate)
        m_is_aggregate = m_type_flags.Test (ClangASTType::eTypeHasChildren) ? eLazyBoolYes : eLazyBoolNo;
    return m_is_aggregate == eLazyBoolYes;
}

bool
ValueObjectPrinter::PrintLocationIfNeeded ()
{
    if (options.m_show_location)
    {
        m_stream->Printf("%s: ", m_valobj->GetLocationAsCString());
        return true;
    }
    return false;
}

bool
ValueObjectPrinter::PrintTypeIfNeeded ()
{
    bool show_type = true;
    // if we are at the root-level and been asked to hide the root's type, then hide it
    if (m_curr_depth == 0 && options.m_hide_root_type)
        show_type = false;
    else
        // otherwise decide according to the usual rules (asked to show types - always at the root level)
        show_type = options.m_show_types || (m_curr_depth == 0 && !options.m_flat_output);
    
    if (show_type)
    {
        // Some ValueObjects don't have types (like registers sets). Only print
        // the type if there is one to print
        ConstString qualified_type_name(m_valobj->GetQualifiedTypeName());
        if (qualified_type_name)
            m_stream->Printf("(%s) ", qualified_type_name.GetCString());
        else
            show_type = false;
    }
    return show_type;
}

bool
ValueObjectPrinter::PrintNameIfNeeded (bool show_type)
{
    if (options.m_flat_output)
    {
        // If we are showing types, also qualify the C++ base classes
        const bool qualify_cxx_base_classes = show_type;
        if (!options.m_hide_name)
        {
            m_valobj->GetExpressionPath(*m_stream, qualify_cxx_base_classes);
            m_stream->PutCString(" =");
            return true;
        }
    }
    else if (!options.m_hide_name)
    {
        const char *name_cstr = GetRootNameForDisplay("");
        m_stream->Printf ("%s =", name_cstr);
        return true;
    }
    return false;
}

bool
ValueObjectPrinter::CheckScopeIfNeeded ()
{
    if (options.m_scope_already_checked)
        return true;
    return m_valobj->IsInScope();
}

TypeSummaryImpl*
ValueObjectPrinter::GetSummaryFormatter ()
{
    if (m_summary_formatter.second == false)
    {
        TypeSummaryImpl* entry = options.m_summary_sp ? options.m_summary_sp.get() : m_valobj->GetSummaryFormat().get();
        
        if (options.m_omit_summary_depth > 0)
            entry = NULL;
        m_summary_formatter.first = entry;
        m_summary_formatter.second = true;
    }
    return m_summary_formatter.first;
}

void
ValueObjectPrinter::GetValueSummaryError (std::string& value,
                                          std::string& summary,
                                          std::string& error)
{
    if (options.m_format != eFormatDefault && options.m_format != m_valobj->GetFormat())
    {
        m_valobj->GetValueAsCString(options.m_format,
                                    value);
    }
    else
    {
        const char* val_cstr = m_valobj->GetValueAsCString();
        if (val_cstr)
            value.assign(val_cstr);
    }
    const char* err_cstr = m_valobj->GetError().AsCString();
    if (err_cstr)
        error.assign(err_cstr);
    
    if (ShouldPrintValueObject())
    {
        if (IsNil())
            summary.assign("nil");
        else if (options.m_omit_summary_depth == 0)
        {
            TypeSummaryImpl* entry = GetSummaryFormatter();
            if (entry)
                m_valobj->GetSummaryAsCString(entry, summary);
            else
            {
                const char* sum_cstr = m_valobj->GetSummaryAsCString();
                if (sum_cstr)
                    summary.assign(sum_cstr);
            }
        }
    }
}

bool
ValueObjectPrinter::PrintValueAndSummaryIfNeeded (bool& value_printed,
                                                  bool& summary_printed)
{
    bool error_printed = false;
    if (ShouldPrintValueObject())
    {
        if (!CheckScopeIfNeeded())
            m_error.assign("out of scope");
        if (m_error.empty())
        {
            GetValueSummaryError(m_value, m_summary, m_error);
        }
        if (m_error.size())
        {
            error_printed = true;
            m_stream->Printf (" <%s>\n", m_error.c_str());
        }
        else
        {
            // Make sure we have a value and make sure the summary didn't
            // specify that the value should not be printed - and do not print
            // the value if this thing is nil
            // (but show the value if the user passes a format explicitly)
            TypeSummaryImpl* entry = GetSummaryFormatter();
            if (!IsNil() && !m_value.empty() && (entry == NULL || (entry->DoesPrintValue() || options.m_format != eFormatDefault) || m_summary.empty()) && !options.m_hide_value)
            {
                m_stream->Printf(" %s", m_value.c_str());
                value_printed = true;
            }
            
            if (m_summary.size())
            {
                m_stream->Printf(" %s", m_summary.c_str());
                summary_printed = true;
            }
        }
    }
    return !error_printed;
}

bool
ValueObjectPrinter::PrintObjectDescriptionIfNeeded (bool value_printed,
                                                    bool summary_printed)
{
    if (ShouldPrintValueObject())
    {
        // let's avoid the overly verbose no description error for a nil thing
        if (options.m_use_objc && !IsNil())
        {
            if (!options.m_hide_value || !options.m_hide_name)
                m_stream->Printf(" ");
            const char *object_desc = nullptr;
            if (value_printed || summary_printed)
                object_desc = m_valobj->GetObjectDescription();
            else
                object_desc = GetDescriptionForDisplay();
            if (object_desc && *object_desc)
            {
                m_stream->Printf("%s\n", object_desc);
                return true;
            }
            else if (value_printed == false && summary_printed == false)
                return true;
            else
                return false;
        }
    }
    return true;
}

bool
ValueObjectPrinter::ShouldPrintChildren (bool is_failed_description,
                                         uint32_t& curr_ptr_depth)
{
    const bool is_ref = IsRef ();
    const bool is_ptr = IsPtr ();

    if (is_failed_description || m_curr_depth < options.m_max_depth)
    {
        // We will show children for all concrete types. We won't show
        // pointer contents unless a pointer depth has been specified.
        // We won't reference contents unless the reference is the
        // root object (depth of zero).
        
        // Use a new temporary pointer depth in case we override the
        // current pointer depth below...
        uint32_t curr_ptr_depth = m_ptr_depth;
        
        if (is_ptr || is_ref)
        {
            // We have a pointer or reference whose value is an address.
            // Make sure that address is not NULL
            AddressType ptr_address_type;
            if (m_valobj->GetPointerValue (&ptr_address_type) == 0)
                return false;
            
            else if (is_ref && m_curr_depth == 0)
            {
                // If this is the root object (depth is zero) that we are showing
                // and it is a reference, and no pointer depth has been supplied
                // print out what it references. Don't do this at deeper depths
                // otherwise we can end up with infinite recursion...
                curr_ptr_depth = 1;
            }
            
            return (curr_ptr_depth > 0);
        }
        
        TypeSummaryImpl* entry = GetSummaryFormatter();

        return (!entry || entry->DoesPrintChildren() || m_summary.empty());
    }
    return false;
}

ValueObject*
ValueObjectPrinter::GetValueObjectForChildrenGeneration ()
{
    ValueObjectSP synth_valobj_sp = m_valobj->GetSyntheticValue (options.m_use_synthetic);
    return (synth_valobj_sp ? synth_valobj_sp.get() : m_valobj);
}

void
ValueObjectPrinter::PrintChildrenPreamble ()
{
    if (options.m_flat_output)
    {
        if (ShouldPrintValueObject())
            m_stream->EOL();
    }
    else
    {
        if (ShouldPrintValueObject())
            m_stream->PutCString(IsRef () ? ": {\n" : " {\n");
        m_stream->IndentMore();
    }
}

void
ValueObjectPrinter::PrintChild (ValueObjectSP child_sp,
                                uint32_t curr_ptr_depth)
{
    DumpValueObjectOptions child_options(options);
    child_options.SetFormat(options.m_format).SetSummary().SetRootValueObjectName();
    child_options.SetScopeChecked(true).SetHideName(options.m_hide_name).SetHideValue(options.m_hide_value)
    .SetOmitSummaryDepth(child_options.m_omit_summary_depth > 1 ? child_options.m_omit_summary_depth - 1 : 0);
    if (child_sp.get())
    {
        ValueObjectPrinter child_printer(child_sp.get(),
                                         m_stream,
                                         child_options,
                                         (IsPtr() || IsRef()) ? curr_ptr_depth - 1 : curr_ptr_depth,
                                         m_curr_depth + 1);
        child_printer.PrintValueObject();
    }

}

uint32_t
ValueObjectPrinter::GetMaxNumChildrenToPrint (bool& print_dotdotdot)
{
    ValueObject* synth_m_valobj = GetValueObjectForChildrenGeneration();
    
    size_t num_children = synth_m_valobj->GetNumChildren();
    print_dotdotdot = false;
    if (num_children)
    {
        const size_t max_num_children = m_valobj->GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
        
        if (num_children > max_num_children && !options.m_ignore_cap)
        {
            print_dotdotdot = true;
            return max_num_children;
        }
    }
    return num_children;
}

void
ValueObjectPrinter::PrintChildrenPostamble (bool print_dotdotdot)
{
    if (!options.m_flat_output)
    {
        if (print_dotdotdot)
        {
            m_valobj->GetTargetSP()->GetDebugger().GetCommandInterpreter().ChildrenTruncated();
            m_stream->Indent("...\n");
        }
        m_stream->IndentLess();
        m_stream->Indent("}\n");
    }
}

void
ValueObjectPrinter::PrintChildren (uint32_t curr_ptr_depth)
{
    ValueObject* synth_m_valobj = GetValueObjectForChildrenGeneration();
    
    bool print_dotdotdot = false;
    size_t num_children = GetMaxNumChildrenToPrint(print_dotdotdot);
    if (num_children)
    {
        PrintChildrenPreamble ();
        
        for (size_t idx=0; idx<num_children; ++idx)
        {
            ValueObjectSP child_sp(synth_m_valobj->GetChildAtIndex(idx, true));
            PrintChild (child_sp, curr_ptr_depth);
        }
        
        PrintChildrenPostamble (print_dotdotdot);
    }
    else if (IsAggregate())
    {
        // Aggregate, no children...
        if (ShouldPrintValueObject())
            m_stream->PutCString(" {}\n");
    }
    else
    {
        if (ShouldPrintValueObject())
            m_stream->EOL();
    }
}

bool
ValueObjectPrinter::PrintChildrenOneLiner (bool hide_names)
{
    if (!GetDynamicValueIfNeeded () || m_valobj == nullptr)
        return false;
    
    ValueObject* synth_m_valobj = GetValueObjectForChildrenGeneration();
    
    bool print_dotdotdot = false;
    size_t num_children = GetMaxNumChildrenToPrint(print_dotdotdot);
    
    if (num_children)
    {
        m_stream->PutChar('(');
        
        for (uint32_t idx=0; idx<num_children; ++idx)
        {
            lldb::ValueObjectSP child_sp(synth_m_valobj->GetChildAtIndex(idx, true));
            lldb::ValueObjectSP child_dyn_sp = child_sp.get() ? child_sp->GetDynamicValue(options.m_use_dynamic) : child_sp;
            if (child_dyn_sp)
                child_sp = child_dyn_sp;
            if (child_sp)
            {
                if (idx)
                    m_stream->PutCString(", ");
                if (!hide_names)
                {
                    const char* name = child_sp.get()->GetName().AsCString();
                    if (name && *name)
                    {
                        m_stream->PutCString(name);
                        m_stream->PutCString(" = ");
                    }
                }
                child_sp->DumpPrintableRepresentation(*m_stream,
                                                      ValueObject::eValueObjectRepresentationStyleSummary,
                                                      lldb::eFormatInvalid,
                                                      ValueObject::ePrintableRepresentationSpecialCasesDisable);
            }
        }
        
        if (print_dotdotdot)
            m_stream->PutCString(", ...)");
        else
            m_stream->PutChar(')');
    }
    return true;
}

void
ValueObjectPrinter::PrintChildrenIfNeeded (bool value_printed,
                                           bool summary_printed)
{
    // this flag controls whether we tried to display a description for this object and failed
    // if that happens, we want to display the children, if any
    bool is_failed_description = !PrintObjectDescriptionIfNeeded(value_printed, summary_printed);
    
    uint32_t curr_ptr_depth = m_ptr_depth;
    bool print_children = ShouldPrintChildren (is_failed_description,curr_ptr_depth);
    bool print_oneline = (curr_ptr_depth > 0 || options.m_show_types || options.m_be_raw) ? false : DataVisualization::ShouldPrintAsOneLiner(*m_valobj);
    
    if (print_children)
    {
        if (print_oneline)
        {
            m_stream->PutChar(' ');
            PrintChildrenOneLiner (false);
            m_stream->EOL();
        }
        else
            PrintChildren (curr_ptr_depth);
    }
    else if (m_curr_depth >= options.m_max_depth && IsAggregate() && ShouldPrintValueObject())
    {
            m_stream->PutCString("{...}\n");
    }
    else
        m_stream->EOL();
}
