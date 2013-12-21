//===-- ValueObjectPrinter.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ValueObjectPrinter_h_
#define lldb_ValueObjectPrinter_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/lldb-public.h"

#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"

namespace lldb_private {

struct DumpValueObjectOptions
{
    uint32_t m_max_ptr_depth;
    uint32_t m_max_depth;
    bool m_show_types;
    bool m_show_location;
    bool m_use_objc;
    lldb::DynamicValueType m_use_dynamic;
    bool m_use_synthetic;
    bool m_scope_already_checked;
    bool m_flat_output;
    uint32_t m_omit_summary_depth;
    bool m_ignore_cap;
    lldb::Format m_format;
    lldb::TypeSummaryImplSP m_summary_sp;
    std::string m_root_valobj_name;
    bool m_hide_root_type;
    bool m_hide_name;
    bool m_hide_value;
    bool m_be_raw;
    
    DumpValueObjectOptions() :
    m_max_ptr_depth(0),
    m_max_depth(UINT32_MAX),
    m_show_types(false),
    m_show_location(false),
    m_use_objc(false),
    m_use_dynamic(lldb::eNoDynamicValues),
    m_use_synthetic(true),
    m_scope_already_checked(false),
    m_flat_output(false),
    m_omit_summary_depth(0),
    m_ignore_cap(false),
    m_format (lldb::eFormatDefault),
    m_summary_sp(),
    m_root_valobj_name(),
    m_hide_root_type(false),  // provide a special compact display for "po"
    m_hide_name(false), // provide a special compact display for "po"
    m_hide_value(false), // provide a special compact display for "po"
    m_be_raw(false)
    {}
    
    static const DumpValueObjectOptions
    DefaultOptions()
    {
        static DumpValueObjectOptions g_default_options;
        
        return g_default_options;
    }
    
    DumpValueObjectOptions (const DumpValueObjectOptions& rhs) :
    m_max_ptr_depth(rhs.m_max_ptr_depth),
    m_max_depth(rhs.m_max_depth),
    m_show_types(rhs.m_show_types),
    m_show_location(rhs.m_show_location),
    m_use_objc(rhs.m_use_objc),
    m_use_dynamic(rhs.m_use_dynamic),
    m_use_synthetic(rhs.m_use_synthetic),
    m_scope_already_checked(rhs.m_scope_already_checked),
    m_flat_output(rhs.m_flat_output),
    m_omit_summary_depth(rhs.m_omit_summary_depth),
    m_ignore_cap(rhs.m_ignore_cap),
    m_format(rhs.m_format),
    m_summary_sp(rhs.m_summary_sp),
    m_root_valobj_name(rhs.m_root_valobj_name),
    m_hide_root_type(rhs.m_hide_root_type),
    m_hide_name(rhs.m_hide_name),
    m_hide_value(rhs.m_hide_value),
    m_be_raw(rhs.m_be_raw)
    {}
    
    DumpValueObjectOptions&
    SetMaximumPointerDepth(uint32_t depth = 0)
    {
        m_max_ptr_depth = depth;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetMaximumDepth(uint32_t depth = 0)
    {
        m_max_depth = depth;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetShowTypes(bool show = false)
    {
        m_show_types = show;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetShowLocation(bool show = false)
    {
        m_show_location = show;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetUseObjectiveC(bool use = false)
    {
        m_use_objc = use;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetShowSummary(bool show = true)
    {
        if (show == false)
            SetOmitSummaryDepth(UINT32_MAX);
        else
            SetOmitSummaryDepth(0);
        return *this;
    }
    
    DumpValueObjectOptions&
    SetUseDynamicType(lldb::DynamicValueType dyn = lldb::eNoDynamicValues)
    {
        m_use_dynamic = dyn;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetUseSyntheticValue(bool use_synthetic = true)
    {
        m_use_synthetic = use_synthetic;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetScopeChecked(bool check = true)
    {
        m_scope_already_checked = check;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetFlatOutput(bool flat = false)
    {
        m_flat_output = flat;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetOmitSummaryDepth(uint32_t depth = 0)
    {
        m_omit_summary_depth = depth;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetIgnoreCap(bool ignore = false)
    {
        m_ignore_cap = ignore;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetRawDisplay(bool raw = false)
    {
        if (raw)
        {
            SetUseSyntheticValue(false);
            SetOmitSummaryDepth(UINT32_MAX);
            SetIgnoreCap(true);
            SetHideName(false);
            SetHideValue(false);
            m_be_raw = true;
        }
        else
        {
            SetUseSyntheticValue(true);
            SetOmitSummaryDepth(0);
            SetIgnoreCap(false);
            SetHideName(false);
            SetHideValue(false);
            m_be_raw = false;
        }
        return *this;
    }
    
    DumpValueObjectOptions&
    SetFormat (lldb::Format format = lldb::eFormatDefault)
    {
        m_format = format;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetSummary (lldb::TypeSummaryImplSP summary = lldb::TypeSummaryImplSP())
    {
        m_summary_sp = summary;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetRootValueObjectName (const char* name = NULL)
    {
        if (name)
            m_root_valobj_name.assign(name);
        else
            m_root_valobj_name.clear();
        return *this;
    }
    
    DumpValueObjectOptions&
    SetHideRootType (bool hide_root_type = false)
    {
        m_hide_root_type = hide_root_type;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetHideName (bool hide_name = false)
    {
        m_hide_name = hide_name;
        return *this;
    }
    
    DumpValueObjectOptions&
    SetHideValue (bool hide_value = false)
    {
        m_hide_value = hide_value;
        return *this;
    }
};
    
class ValueObjectPrinter
{
public:
    
    ValueObjectPrinter (ValueObject* valobj,
                        Stream* s,
                        const DumpValueObjectOptions& options);
    
    ~ValueObjectPrinter () {}
    
    bool
    PrintValueObject ();
    
protected:
    
    // only this class (and subclasses, if any) should ever be concerned with
    // the depth mechanism
    ValueObjectPrinter (ValueObject* valobj,
                        Stream* s,
                        const DumpValueObjectOptions& options,
                        uint32_t ptr_depth,
                        uint32_t curr_depth);
    
    // we should actually be using delegating constructors here
    // but some versions of GCC still have trouble with those
    void
    Init (ValueObject* valobj,
          Stream* s,
          const DumpValueObjectOptions& options,
          uint32_t ptr_depth,
          uint32_t curr_depth);
    
    bool
    GetDynamicValueIfNeeded ();
    
    const char*
    GetDescriptionForDisplay ();
    
    const char*
    GetRootNameForDisplay (const char* if_fail = nullptr);
    
    bool
    ShouldPrintValueObject ();
    
    bool
    IsNil ();
    
    bool
    IsPtr ();
    
    bool
    IsRef ();
    
    bool
    IsAggregate ();
    
    bool
    PrintLocationIfNeeded ();
    
    bool
    PrintTypeIfNeeded ();
    
    bool
    PrintNameIfNeeded (bool show_type);
    
    bool
    CheckScopeIfNeeded ();
    
    TypeSummaryImpl*
    GetSummaryFormatter ();
    
    void
    GetValueSummaryError (std::string& value,
                          std::string& summary,
                          std::string& error);
    
    bool
    PrintValueAndSummaryIfNeeded (bool& value_printed,
                                  bool& summary_printed);
    
    bool
    PrintObjectDescriptionIfNeeded (bool value_printed,
                                    bool summary_printed);
    
    bool
    ShouldPrintChildren (bool is_failed_description,
                         uint32_t& curr_ptr_depth);
    
    ValueObject*
    GetValueObjectForChildrenGeneration ();
    
    void
    PrintChildrenPreamble ();
    
    void
    PrintChildrenPostamble (bool print_dotdotdot);
    
    void
    PrintChild (lldb::ValueObjectSP child_sp,
                uint32_t curr_ptr_depth);
    
    uint32_t
    GetMaxNumChildrenToPrint (bool& print_dotdotdot);
    
    void
    PrintChildren (uint32_t curr_ptr_depth);
    
    void
    PrintChildrenIfNeeded (bool value_printed,
                           bool summary_printed);
    
    bool
    PrintChildrenOneLiner (bool hide_names);
    
private:
    
    ValueObject *m_orig_valobj;
    ValueObject *m_valobj;
    Stream *m_stream;
    DumpValueObjectOptions options;
    Flags m_type_flags;
    ClangASTType m_clang_type;
    uint32_t m_ptr_depth;
    uint32_t m_curr_depth;
    LazyBool m_should_print;
    LazyBool m_is_nil;
    LazyBool m_is_ptr;
    LazyBool m_is_ref;
    LazyBool m_is_aggregate;
    std::pair<TypeSummaryImpl*,bool> m_summary_formatter;
    std::string m_value;
    std::string m_summary;
    std::string m_error;
    
    friend class StringSummaryFormat;
    
    DISALLOW_COPY_AND_ASSIGN(ValueObjectPrinter);
};
    
} // namespace lldb_private

#endif	// lldb_ValueObjectPrinter_h_
