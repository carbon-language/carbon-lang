//===-- ExpressionSourceCode.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExpressionSourceCode.h"

#include "lldb/Core/StreamString.h"

using namespace lldb_private;

bool ExpressionSourceCode::GetText (std::string &text, lldb::LanguageType wrapping_language, bool const_object) const
{
    if (m_wrap)
    {
        switch (wrapping_language) 
        {
        default:
            return false;
        case lldb::eLanguageTypeC:
        case lldb::eLanguageTypeC_plus_plus:
        case lldb::eLanguageTypeObjC:
            break;
        }
        
        StreamString wrap_stream;
        
        switch (wrapping_language) 
        {
        default:
            break;
        case lldb::eLanguageTypeC:
            wrap_stream.Printf("%s                             \n"
                               "typedef unsigned short unichar;\n"
                               "void                           \n"
                               "%s(void *$__lldb_arg)          \n"
                               "{                              \n"
                               "    %s;                        \n" 
                               "}                              \n",
                               m_prefix.c_str(),
                               m_name.c_str(),
                               m_body.c_str());
            break;
        case lldb::eLanguageTypeC_plus_plus:
            wrap_stream.Printf("%s                                     \n"
                               "typedef unsigned short unichar;        \n"
                               "void                                   \n"
                               "$__lldb_class::%s(void *$__lldb_arg) %s\n"
                               "{                                      \n"
                               "    %s;                                \n" 
                               "}                                      \n",
                               m_prefix.c_str(),
                               m_name.c_str(),
                               (const_object ? "const" : ""),
                               m_body.c_str());
            break;
        case lldb::eLanguageTypeObjC:
            wrap_stream.Printf("%s                                                      \n"
                                "typedef unsigned short unichar;                        \n"
                                "@interface $__lldb_objc_class ($__lldb_category)       \n"
                                "-(void)%s:(void *)$__lldb_arg;                         \n"
                                "@end                                                   \n"
                                "@implementation $__lldb_objc_class ($__lldb_category)  \n"
                                "-(void)%s:(void *)$__lldb_arg                          \n"
                                "{                                                      \n"
                                "    %s;                                                \n"
                                "}                                                      \n"
                                "@end                                                   \n",
                                m_prefix.c_str(),
                                m_name.c_str(),
                                m_name.c_str(),
                                m_body.c_str());
            break;
        }
        
        text = wrap_stream.GetString();
    }
    else
    {
        text.append(m_body);
    }
    
    return true;
}
