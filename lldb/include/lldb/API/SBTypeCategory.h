//===-- SBTypeCategory.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeCategory_h_
#define LLDB_SBTypeCategory_h_

#include "lldb/API/SBDefines.h"

namespace lldb {
    
    class SBTypeCategory
    {
    public:
        
        SBTypeCategory();
        
        SBTypeCategory (const lldb::SBTypeCategory &rhs);
        
        ~SBTypeCategory ();
        
        bool
        IsValid() const;
        
        bool
        GetEnabled ();
        
        void
        SetEnabled (bool);
        
        const char*
        GetName();
        
        bool
        GetDescription (lldb::SBStream &description, 
                        lldb::DescriptionLevel description_level);
        
        uint32_t
        GetNumFormats ();
        
        uint32_t
        GetNumSummaries ();
        
        uint32_t
        GetNumFilters ();
        
        uint32_t
        GetNumSynthetics ();
        
        SBTypeNameSpecifier
        GetTypeNameSpecifierForFilterAtIndex (uint32_t);
        
        SBTypeNameSpecifier
        GetTypeNameSpecifierForFormatAtIndex (uint32_t);
        
        SBTypeNameSpecifier
        GetTypeNameSpecifierForSummaryAtIndex (uint32_t);

        SBTypeNameSpecifier
        GetTypeNameSpecifierForSyntheticAtIndex (uint32_t);
        
        SBTypeFilter
        GetFilterForType (SBTypeNameSpecifier);

        SBTypeFormat
        GetFormatForType (SBTypeNameSpecifier);
        
        SBTypeSummary
        GetSummaryForType (SBTypeNameSpecifier);

        SBTypeSynthetic
        GetSyntheticForType (SBTypeNameSpecifier);
        
        SBTypeFilter
        GetFilterAtIndex (uint32_t);
        
        SBTypeFormat
        GetFormatAtIndex (uint32_t);
        
        SBTypeSummary
        GetSummaryAtIndex (uint32_t);
        
        SBTypeSynthetic
        GetSyntheticAtIndex (uint32_t);
        
        bool
        AddTypeFormat (SBTypeNameSpecifier,
                       SBTypeFormat);
        
        bool
        DeleteTypeFormat (SBTypeNameSpecifier);
        
        bool
        AddTypeSummary (SBTypeNameSpecifier,
                        SBTypeSummary);
        
        bool
        DeleteTypeSummary (SBTypeNameSpecifier);
        
        bool
        AddTypeFilter (SBTypeNameSpecifier,
                       SBTypeFilter);
        
        bool
        DeleteTypeFilter (SBTypeNameSpecifier);
        
        bool
        AddTypeSynthetic (SBTypeNameSpecifier,
                          SBTypeSynthetic);
        
        bool
        DeleteTypeSynthetic (SBTypeNameSpecifier);
        
        lldb::SBTypeCategory &
        operator = (const lldb::SBTypeCategory &rhs);
        
        bool
        operator == (lldb::SBTypeCategory &rhs);
        
        bool
        operator != (lldb::SBTypeCategory &rhs);
        
    protected:
        friend class SBDebugger;
        
        lldb::TypeCategoryImplSP
        GetSP ();
        
        void
        SetSP (const lldb::TypeCategoryImplSP &typecategory_impl_sp);    
        
        TypeCategoryImplSP m_opaque_sp;
        
        SBTypeCategory (const lldb::TypeCategoryImplSP &);
        
        SBTypeCategory (const char*);
        
        bool
        IsDefaultCategory();
        
    };
    
} // namespace lldb

#endif // LLDB_SBTypeCategory_h_
