//===-- DataVisualization.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_DataVisualization_h_
#define lldb_DataVisualization_h_

// C Includes
// C++ Includes

// <locale> is not strictly-speaking a requirement for DataVisualization.h
// but including it ensures a smooth compilation of STLUtils.h. if <locale>
// is not included, a macro definition of isspace() and other cctype functions occurs
// which prevents <ostream> from getting included correctly. at least, this is what
// happens on OSX Lion. If other OSs don't have this side effect, you may want to
// #if defined (__APPLE__) this include directive
#include <locale>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward-rtti.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/FormatClasses.h"
#include "lldb/Core/FormatManager.h"

namespace lldb_private {

// this class is the high-level front-end of LLDB Data Visualization
// code in FormatManager.h/cpp is the low-level implementation of this feature
// clients should refer to this class as the entry-point into the data formatters
// unless they have a good reason to bypass this and go to the backend
class DataVisualization
{
public:
    
    // use this call to force the FM to consider itself updated even when there is no apparent reason for that
    static void
    ForceUpdate();
    
    static uint32_t
    GetCurrentRevision ();
    
    class ValueFormats
    {
    public:
        static lldb::ValueFormatSP
        GetFormat (ValueObject& valobj, lldb::DynamicValueType use_dynamic);
        
        static void
        Add (const ConstString &type, const lldb::ValueFormatSP &entry);
        
        static bool
        Delete (const ConstString &type);
        
        static void
        Clear ();
        
        static void
        LoopThrough (ValueFormat::ValueCallback callback, void* callback_baton);
        
        static uint32_t
        GetCount ();
    };
    
    static lldb::SummaryFormatSP
    GetSummaryFormat(ValueObject& valobj,
                     lldb::DynamicValueType use_dynamic);
    
    static lldb::SyntheticChildrenSP
    GetSyntheticChildren(ValueObject& valobj,
                         lldb::DynamicValueType use_dynamic);
    
    static bool
    AnyMatches(ConstString type_name,
               FormatCategory::FormatCategoryItems items = FormatCategory::ALL_ITEM_TYPES,
               bool only_enabled = true,
               const char** matching_category = NULL,
               FormatCategory::FormatCategoryItems* matching_type = NULL);
    
    class NamedSummaryFormats
    {
    public:
        static bool
        GetSummaryFormat (const ConstString &type, lldb::SummaryFormatSP &entry);
        
        static void
        Add (const ConstString &type, const lldb::SummaryFormatSP &entry);
        
        static bool
        Delete (const ConstString &type);
        
        static void
        Clear ();
        
        static void
        LoopThrough (SummaryFormat::SummaryCallback callback, void* callback_baton);
        
        static uint32_t
        GetCount ();
    };
    
    class Categories
    {
    public:
        
        static bool
        GetCategory (const ConstString &category, lldb::FormatCategorySP &entry);
        
        static void
        Add (const ConstString &category);
        
        static bool
        Delete (const ConstString &category);
        
        static void
        Clear ();
        
        static void
        Clear (ConstString &category);
        
        static void
        Enable (ConstString& category);
        
        static void
        Disable (ConstString& category);
        
        static void
        LoopThrough (FormatManager::CategoryCallback callback, void* callback_baton);
        
        static uint32_t
        GetCount ();
    };
};

    
} // namespace lldb_private

#endif	// lldb_DataVisualization_h_
