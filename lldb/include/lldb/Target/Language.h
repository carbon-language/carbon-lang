//===-- Language.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Language_h_
#define liblldb_Language_h_

// C Includes
// C++ Includes
#include <functional>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/DataFormatters/StringPrinter.h"

namespace lldb_private {
    
class Language :
public PluginInterface
{
public:
    ~Language() override;
    
    static Language*
    FindPlugin (lldb::LanguageType language);
    
    // return false from callback to stop iterating
    static void
    ForEach (std::function<bool(Language*)> callback);
    
    virtual lldb::LanguageType
    GetLanguageType () const = 0;
    
    virtual lldb::TypeCategoryImplSP
    GetFormatters ();

    virtual std::vector<ConstString>
    GetPossibleFormattersMatches (ValueObject& valobj, lldb::DynamicValueType use_dynamic);

    virtual lldb_private::formatters::StringPrinter::EscapingHelper
    GetStringPrinterEscapingHelper (lldb_private::formatters::StringPrinter::GetPrintableElementType);
    
    // These are accessors for general information about the Languages lldb knows about:
    
    static lldb::LanguageType
    GetLanguageTypeFromString (const char *string);
    
    static const char *
    GetNameForLanguageType (lldb::LanguageType language);

    static void
    PrintAllLanguages (Stream &s, const char *prefix, const char *suffix);
        
    // return false from callback to stop iterating
    static void
    ForAllLanguages (std::function<bool(lldb::LanguageType)> callback);

    static bool
    LanguageIsCPlusPlus (lldb::LanguageType language);
    
    static bool
    LanguageIsObjC (lldb::LanguageType language);
    
    static bool
    LanguageIsC (lldb::LanguageType language);
    
    static bool
    LanguageIsPascal (lldb::LanguageType language);
    

protected:
    //------------------------------------------------------------------
    // Classes that inherit from Language can see and modify these
    //------------------------------------------------------------------
    
    Language();
private:
    
    DISALLOW_COPY_AND_ASSIGN (Language);
};
    
} // namespace lldb_private

#endif // liblldb_Language_h_
