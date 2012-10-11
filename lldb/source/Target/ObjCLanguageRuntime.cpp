//===-- ObjCLanguageRuntime.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Type.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ObjCLanguageRuntime::~ObjCLanguageRuntime()
{
}

ObjCLanguageRuntime::ObjCLanguageRuntime (Process *process) :
    LanguageRuntime (process),
    m_has_new_literals_and_indexing (eLazyBoolCalculate),
    m_isa_to_descriptor_cache(),
    m_isa_to_descriptor_cache_is_up_to_date (false)
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
    
    if (type_sp)
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

lldb::TypeSP
ObjCLanguageRuntime::LookupInCompleteClassCache (ConstString &name)
{
    CompleteClassMap::iterator complete_class_iter = m_complete_class_cache.find(name);
    
    if (complete_class_iter != m_complete_class_cache.end())
    {
        TypeSP ret(complete_class_iter->second);
        
        if (!ret)
            m_complete_class_cache.erase(name);
        else
            return TypeSP(complete_class_iter->second);
    }
    
    ModuleList &modules = m_process->GetTarget().GetImages();
    
    SymbolContextList sc_list;
    
    modules.FindSymbolsWithNameAndType(name, eSymbolTypeObjCClass, sc_list);
    
    if (sc_list.GetSize() == 0)
        return TypeSP();
    
    SymbolContext sc;
    
    sc_list.GetContextAtIndex(0, sc);
    
    ModuleSP module_sp(sc.module_sp);
    
    if (!module_sp)
        return TypeSP();
    
    const SymbolContext null_sc;
    const bool exact_match = true;
    const uint32_t max_matches = UINT32_MAX;
    TypeList types;
    
    module_sp->FindTypes (null_sc,
                          name,
                          exact_match,
                          max_matches,
                          types);
    
    if (types.GetSize() == 1)
    {
        TypeSP candidate_type = types.GetTypeAtIndex(0);
        
        if (ClangASTContext::IsObjCClassType(candidate_type->GetClangForwardType()))
        {
            m_complete_class_cache[name] = TypeWP(candidate_type);
            return candidate_type;
        }
        else
        {
            return TypeSP();
        }
    }
    
    for (uint32_t ti = 0, te = types.GetSize();
         ti < te;
         ++ti)
    {
        TypeSP candidate_type = types.GetTypeAtIndex(ti);
        
        if (candidate_type->IsCompleteObjCClass() &&
            ClangASTContext::IsObjCClassType(candidate_type->GetClangForwardType()))
        {
            m_complete_class_cache[name] = TypeWP(candidate_type);
            return candidate_type;                                       
        }
    }
    
    return TypeSP();
}

size_t
ObjCLanguageRuntime::GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name)
{
    return LLDB_INVALID_IVAR_OFFSET;
}


uint32_t
ObjCLanguageRuntime::ParseMethodName (const char *name, 
                                      ConstString *class_name,              // Class name (with category if any)
                                      ConstString *selector_name,           // selector on its own
                                      ConstString *name_sans_category,      // Full function prototype with no category
                                      ConstString *class_name_sans_category)// Class name with no category (or empty if no category as answer will be in "class_name"
{
    if (class_name)
        class_name->Clear();
    if (selector_name)
        selector_name->Clear();
    if (name_sans_category)
        name_sans_category->Clear();
    if (class_name_sans_category)
        class_name_sans_category->Clear();
    
    uint32_t result = 0;

    if (IsPossibleObjCMethodName (name))
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
            const char *selector_name_ptr = strchr (name, ' ');
            if (selector_name_ptr)
            {
                if (class_name)
                {
                    class_name->SetCStringWithLength (name + 2, selector_name_ptr - name - 2);
                    ++result;
                }    
                
                // Skip the space
                ++selector_name_ptr;
                // Extract the objective C basename and add it to the
                // accelerator tables
                size_t selector_name_len = name_len - (selector_name_ptr - name) - 1;
                if (selector_name)
                {
                    selector_name->SetCStringWithLength (selector_name_ptr, selector_name_len);                                
                    ++result;
                }
                
                // Also see if this is a "category" on our class.  If so strip off the category name,
                // and add the class name without it to the basename table. 
                
                if (name_sans_category || class_name_sans_category)
                {
                    const char *open_paren = strchr (name, '(');
                    if (open_paren)
                    {
                        if (class_name_sans_category)
                        {
                            class_name_sans_category->SetCStringWithLength (name + 2, open_paren - name - 2);
                            ++result;
                        }
                        
                        if (name_sans_category)
                        {
                            const char *close_paren = strchr (open_paren, ')');
                            if (open_paren < close_paren)
                            {
                                std::string buffer (name, open_paren - name);
                                buffer.append (close_paren + 1);
                                name_sans_category->SetCString (buffer.c_str());
                                ++result;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

bool
ObjCLanguageRuntime::ClassDescriptor::IsPointerValid (lldb::addr_t value,
                                                      uint32_t ptr_size,
                                                      bool allow_NULLs,
                                                      bool allow_tagged,
                                                      bool check_version_specific) const
{
    if (!value)
        return allow_NULLs;
    if ( (value % 2) == 1  && allow_tagged)
        return true;
    if ((value % ptr_size) == 0)
        return (check_version_specific ? CheckPointer(value,ptr_size) : true);
    else
        return false;
}

ObjCLanguageRuntime::ObjCISA
ObjCLanguageRuntime::GetISA(const ConstString &name)
{
    UpdateISAToDescriptorMap();
    for (const ISAToDescriptorMap::value_type &val : m_isa_to_descriptor_cache)
        if (val.second && val.second->GetClassName() == name)
            return val.first;
    return 0;
}

ObjCLanguageRuntime::ObjCISA
ObjCLanguageRuntime::GetParentClass(ObjCLanguageRuntime::ObjCISA isa)
{
    ClassDescriptorSP objc_class_sp (GetClassDescriptor(isa));
    if (objc_class_sp)
    {
        ClassDescriptorSP objc_super_class_sp (objc_class_sp->GetSuperclass());
        if (objc_super_class_sp)
            return objc_super_class_sp->GetISA();
    }
    return 0;
}

ConstString
ObjCLanguageRuntime::GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa)
{
    ClassDescriptorSP objc_class_sp (GetNonKVOClassDescriptor(isa));
    if (objc_class_sp)
        return objc_class_sp->GetClassName();
    return ConstString();
}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetClassDescriptor (const ConstString &class_name)
{
    UpdateISAToDescriptorMap();
    for (const ISAToDescriptorMap::value_type &val : m_isa_to_descriptor_cache)
        if (val.second && val.second->GetClassName() == class_name)
            return val.second;
    return ClassDescriptorSP();

}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetClassDescriptor (ValueObject& valobj)
{
    ClassDescriptorSP objc_class_sp;
    // if we get an invalid VO (which might still happen when playing around
    // with pointers returned by the expression parser, don't consider this
    // a valid ObjC object)
    if (valobj.GetValue().GetContextType() != Value::eContextTypeInvalid)
    {
        addr_t isa_pointer = valobj.GetPointerValue();
        if (isa_pointer != LLDB_INVALID_ADDRESS)
        {
            ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
            
            Process *process = exe_ctx.GetProcessPtr();
            if (process)
            {
                Error error;
                ObjCISA isa = process->ReadPointerFromMemory(isa_pointer, error);
                if (isa != LLDB_INVALID_ADDRESS)
                    objc_class_sp = GetClassDescriptor (isa);
            }
        }
    }
    return objc_class_sp;
}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetNonKVOClassDescriptor (ValueObject& valobj)
{
    ObjCLanguageRuntime::ClassDescriptorSP objc_class_sp (GetClassDescriptor (valobj));
    if (objc_class_sp)
    {
        if (!objc_class_sp->IsKVO())
            return objc_class_sp;
        
        ClassDescriptorSP non_kvo_objc_class_sp(objc_class_sp->GetSuperclass());
        if (non_kvo_objc_class_sp && non_kvo_objc_class_sp->IsValid())
            return non_kvo_objc_class_sp;
    }
    return ClassDescriptorSP();
}


ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetClassDescriptor (ObjCISA isa)
{
    if (isa)
    {
        UpdateISAToDescriptorMap();
        ObjCLanguageRuntime::ISAToDescriptorIterator pos = m_isa_to_descriptor_cache.find(isa);    
        if (pos != m_isa_to_descriptor_cache.end())
            return pos->second;
    }
    return ClassDescriptorSP();
}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetNonKVOClassDescriptor (ObjCISA isa)
{
    if (isa)
    {
        ClassDescriptorSP objc_class_sp = GetClassDescriptor (isa);
        if (objc_class_sp && objc_class_sp->IsValid())
        {
            if (!objc_class_sp->IsKVO())
                return objc_class_sp;

            ClassDescriptorSP non_kvo_objc_class_sp(objc_class_sp->GetSuperclass());
            if (non_kvo_objc_class_sp && non_kvo_objc_class_sp->IsValid())
                return non_kvo_objc_class_sp;
        }
    }
    return ClassDescriptorSP();
}



