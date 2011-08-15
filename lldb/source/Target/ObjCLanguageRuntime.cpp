//===-- CPPLanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Type.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ObjCLanguageRuntime::~ObjCLanguageRuntime()
{
}

ObjCLanguageRuntime::ObjCLanguageRuntime (Process *process) :
    LanguageRuntime (process)
{

}

void
ObjCLanguageRuntime::AddToMethodCache (lldb::addr_t class_addr, lldb::addr_t selector, lldb::addr_t impl_addr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf ("Caching: class 0x%llx selector 0x%llx implementation 0x%llx.", class_addr, selector, impl_addr);
    }
    m_impl_cache.insert (std::pair<ClassAndSel,lldb::addr_t> (ClassAndSel(class_addr, selector), impl_addr));
}

lldb::addr_t
ObjCLanguageRuntime::LookupInMethodCache (lldb::addr_t class_addr, lldb::addr_t selector)
{
    MsgImplMap::iterator pos, end = m_impl_cache.end();
    pos = m_impl_cache.find (ClassAndSel(class_addr, selector));
    if (pos != end)
        return (*pos).second;
    return LLDB_INVALID_ADDRESS;
}

void
ObjCLanguageRuntime::AddToClassNameCache (lldb::addr_t class_addr, const char *name, lldb::TypeSP type_sp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf ("Caching: class 0x%llx name: %s.", class_addr, name);
    }
    
    TypeAndOrName class_type_or_name;
    
    if (type_sp != NULL)
        class_type_or_name.SetTypeSP (type_sp);
    else if (name && *name != '\0')
        class_type_or_name.SetName (name);
    else 
        return;
    m_class_name_cache.insert (std::pair<lldb::addr_t,TypeAndOrName> (class_addr, class_type_or_name));
}

void
ObjCLanguageRuntime::AddToClassNameCache (lldb::addr_t class_addr, const TypeAndOrName &class_type_or_name)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf ("Caching: class 0x%llx name: %s.", class_addr, class_type_or_name.GetName().AsCString());
    }
    
    m_class_name_cache.insert (std::pair<lldb::addr_t,TypeAndOrName> (class_addr, class_type_or_name));
}

TypeAndOrName
ObjCLanguageRuntime::LookupInClassNameCache (lldb::addr_t class_addr)
{
    ClassNameMap::iterator pos, end = m_class_name_cache.end();
    pos = m_class_name_cache.find (class_addr);
    if (pos != end)
        return (*pos).second;
    return TypeAndOrName ();
}

size_t
ObjCLanguageRuntime::GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name)
{
    return LLDB_INVALID_IVAR_OFFSET;
}


bool
ObjCLanguageRuntime::ParseMethodName (const char *name, 
                                      ConstString *class_name, 
                                      ConstString *method_name, 
                                      ConstString *base_name)
{
    if (class_name) { class_name->Clear(); }
    if (method_name) { method_name->Clear(); }
    if (base_name) { base_name->Clear(); }
    
    if (name && (name[0] == '-' || name[0] == '+') && name[1] == '[')
    {
        int name_len = strlen (name);
        // Objective C methods must have at least:
        //      "-[" or "+[" prefix
        //      One character for a class name
        //      One character for the space between the class name
        //      One character for the method name
        //      "]" suffix
        if (name_len >= 6 && name[name_len - 1] == ']')
        {
            const char *method_name_ptr;
            method_name_ptr = strchr (name, ' ');
            if (method_name_ptr)
            {
                if (class_name)
                    class_name->SetCStringWithLength (name + 2, method_name_ptr - name - 2);
                
                // Skip the space
                ++method_name_ptr;
                // Extract the objective C basename and add it to the
                // accelerator tables
                size_t method_name_len = name_len - (method_name_ptr - name) - 1;
                if (method_name)
                    method_name->SetCStringWithLength (method_name_ptr, method_name_len);                                
                
                // Also see if this is a "category" on our class.  If so strip off the category name,
                // and add the class name without it to the basename table. 
                
                if (base_name)
                {
                    const char *first_paren = (char *) memchr (name, '(', method_name_ptr - name);
                    if (first_paren)
                    {
                        const char *second_paren = (char *) memchr (first_paren, ')', method_name_ptr - first_paren);
                        if (second_paren)
                        {
                            std::string buffer (name, first_paren - name);
                            buffer.append (second_paren + 1);
                            base_name->SetCString (buffer.c_str());

                        }
                    }
                }
            }
            return true;
        }
        return false;
    }
    else
        return false;
}
