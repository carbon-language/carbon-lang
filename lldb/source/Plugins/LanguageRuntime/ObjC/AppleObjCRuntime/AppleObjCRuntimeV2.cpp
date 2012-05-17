//===-- AppleObjCRuntimeV2.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <string>
#include <vector>
#include <memory>
#include <stdint.h>

#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/ClangASTType.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "AppleObjCRuntimeV2.h"
#include "AppleObjCSymbolVendor.h"
#include "AppleObjCTrampolineHandler.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;
    

static const char *pluginName = "AppleObjCRuntimeV2";
static const char *pluginDesc = "Apple Objective C Language Runtime - Version 2";
static const char *pluginShort = "language.apple.objc.v2";


const char *AppleObjCRuntimeV2::g_find_class_name_function_name = "__lldb_apple_objc_v2_find_class_name";
const char *AppleObjCRuntimeV2::g_find_class_name_function_body = "                               \n\
extern \"C\"                                                                                      \n\
{                                                                                                 \n\
    extern void *gdb_class_getClass (void *objc_class);                                           \n\
    extern void *class_getName(void *objc_class);                                                 \n\
    extern int printf(const char *format, ...);                                                   \n\
    extern unsigned char class_isMetaClass (void *objc_class);                                    \n\
}                                                                                                 \n\
                                                                                                  \n\
struct __lldb_objc_object {                                                                       \n\
    void *isa;                                                                                    \n\
};                                                                                                \n\
                                                                                                  \n\
extern \"C\" void *__lldb_apple_objc_v2_find_class_name (                                         \n\
                                                          __lldb_objc_object *object_ptr,         \n\
                                                          int debug)                              \n\
{                                                                                                 \n\
    void *name = 0;                                                                               \n\
    if (debug)                                                                                    \n\
        printf (\"\\n*** Called in v2_find_class_name with object: 0x%p\\n\", object_ptr);        \n\
    // Call gdb_class_getClass so we can tell if the class is good.                               \n\
    void *objc_class = gdb_class_getClass (object_ptr->isa);                                      \n\
    if (objc_class)                                                                               \n\
    {                                                                                             \n\
        void *actual_class = (void *) [(id) object_ptr class];                                    \n\
        if (actual_class != 0)                                                                    \n\
        {                                                                                         \n\
            if (class_isMetaClass(actual_class) == 1)                                             \n\
            {                                                                                     \n\
                if (debug)                                                                        \n\
                    printf (\"\\n*** Found metaclass.\\n\");                                      \n\
            }                                                                                     \n\
            else                                                                                  \n\
            {                                                                                     \n\
                name = class_getName((void *) actual_class);                                      \n\
            }                                                                                     \n\
        }                                                                                         \n\
        if (debug)                                                                                \n\
            printf (\"\\n*** Found name: %s\\n\", name ? name : \"<NOT FOUND>\");                 \n\
    }                                                                                             \n\
    else if (debug)                                                                               \n\
        printf (\"\\n*** gdb_class_getClass returned NULL\\n\");                                  \n\
    return name;                                                                                  \n\
}                                                                                                 \n\
";

AppleObjCRuntimeV2::AppleObjCRuntimeV2 (Process *process, 
                                        const ModuleSP &objc_module_sp) : 
    AppleObjCRuntime (process),
    m_get_class_name_args(LLDB_INVALID_ADDRESS),
    m_get_class_name_args_mutex(Mutex::eMutexTypeNormal),
    m_isa_to_name_cache(),
    m_isa_to_parent_cache()
{
    static const ConstString g_gdb_object_getClass("gdb_object_getClass");
    m_has_object_getClass = (objc_module_sp->FindFirstSymbolWithNameAndType(g_gdb_object_getClass, eSymbolTypeCode) != NULL);
}

bool
AppleObjCRuntimeV2::RunFunctionToFindClassName(addr_t object_addr, Thread *thread, char *name_dst, size_t max_name_len)
{
    // Since we are going to run code we have to make sure only one thread at a time gets to try this.
    Mutex::Locker (m_get_class_name_args_mutex);
    
    StreamString errors;
    
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));  // FIXME - a more appropriate log channel?
    
    int32_t debug;
    if (log)
        debug = 1;
    else
        debug = 0;

    ValueList dispatch_values;
    
    Value void_ptr_value;
    ClangASTContext *clang_ast_context = m_process->GetTarget().GetScratchClangASTContext();
    
    clang_type_t clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
    void_ptr_value.SetValueType (Value::eValueTypeScalar);
    void_ptr_value.SetContext (Value::eContextTypeClangType, clang_void_ptr_type);
    void_ptr_value.GetScalar() = object_addr;
        
    dispatch_values.PushValue (void_ptr_value);
    
    Value int_value;
    clang_type_t clang_int_type = clang_ast_context->GetBuiltinTypeForEncodingAndBitSize(eEncodingSint, 32);
    int_value.SetValueType (Value::eValueTypeScalar);
    int_value.SetContext (Value::eContextTypeClangType, clang_int_type);
    int_value.GetScalar() = debug;
    
    dispatch_values.PushValue (int_value);
    
    ExecutionContext exe_ctx;
    thread->CalculateExecutionContext(exe_ctx);
    
    Address find_class_name_address;
    
    if (!m_get_class_name_code.get())
    {
        m_get_class_name_code.reset (new ClangUtilityFunction (g_find_class_name_function_body,
                                                               g_find_class_name_function_name));
                                                               
        if (!m_get_class_name_code->Install(errors, exe_ctx))
        {
            if (log)
                log->Printf ("Failed to install implementation lookup: %s.", errors.GetData());
            m_get_class_name_code.reset();
            return false;
        }
        find_class_name_address.Clear();
        find_class_name_address.SetOffset(m_get_class_name_code->StartAddress());
    }
    else
    {
        find_class_name_address.Clear();
        find_class_name_address.SetOffset(m_get_class_name_code->StartAddress());
    }

    // Next make the runner function for our implementation utility function.
    if (!m_get_class_name_function.get())
    {
         m_get_class_name_function.reset(new ClangFunction (*m_process,
                                                  clang_ast_context, 
                                                  clang_void_ptr_type, 
                                                  find_class_name_address, 
                                                  dispatch_values));
        
        errors.Clear();        
        unsigned num_errors = m_get_class_name_function->CompileFunction(errors);
        if (num_errors)
        {
            if (log)
                log->Printf ("Error compiling function: \"%s\".", errors.GetData());
            return false;
        }
        
        errors.Clear();
        if (!m_get_class_name_function->WriteFunctionWrapper(exe_ctx, errors))
        {
            if (log)
                log->Printf ("Error Inserting function: \"%s\".", errors.GetData());
            return false;
        }
    }

    if (m_get_class_name_code.get() == NULL || m_get_class_name_function.get() == NULL)
        return false;

    // Finally, write down the arguments, and call the function.  Note that we will re-use the same space in the target
    // for the args.  We're locking this to ensure that only one thread at a time gets to call this function, so we don't
    // have to worry about overwriting the arguments.
    
    if (!m_get_class_name_function->WriteFunctionArguments (exe_ctx, m_get_class_name_args, find_class_name_address, dispatch_values, errors))
        return false;
    
    bool stop_others = true;
    bool try_all_threads = true;
    bool unwind_on_error = true;
    
    ExecutionResults results = m_get_class_name_function->ExecuteFunction (exe_ctx, 
                                                     &m_get_class_name_args, 
                                                     errors, 
                                                     stop_others, 
                                                     100000, 
                                                     try_all_threads, 
                                                     unwind_on_error, 
                                                     void_ptr_value);
                                                     
    if (results != eExecutionCompleted)
    {
        if (log)
            log->Printf("Error evaluating our find class name function: %d.\n", results);
        return false;
    }
    
    addr_t result_ptr = void_ptr_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
    Error error;
    size_t chars_read = m_process->ReadCStringFromMemory (result_ptr, name_dst, max_name_len, error);
    
    // If we exhausted our buffer before finding a NULL we're probably off in the weeds somewhere...
    if (error.Fail() || chars_read == max_name_len)
        return false;
    else
        return true;
       
}

bool
AppleObjCRuntimeV2::GetDynamicTypeAndAddress (ValueObject &in_value, 
                                              DynamicValueType use_dynamic, 
                                              TypeAndOrName &class_type_or_name, 
                                              Address &address)
{
    // The Runtime is attached to a particular process, you shouldn't pass in a value from another process.
    assert (in_value.GetProcessSP().get() == m_process);
    assert (m_process != NULL);

    // Make sure we can have a dynamic value before starting...
    if (CouldHaveDynamicValue (in_value))
    {
        // First job, pull out the address at 0 offset from the object  That will be the ISA pointer.
        Error error;
        const addr_t object_ptr = in_value.GetPointerValue();
        const addr_t isa_addr = m_process->ReadPointerFromMemory (object_ptr, error);

        if (error.Fail())
            return false;

        address.SetRawAddress(object_ptr);

        // First check the cache...
        SymbolContext sc;
        class_type_or_name = LookupInClassNameCache (isa_addr);
        
        if (!class_type_or_name.IsEmpty())
        {
            if (class_type_or_name.GetTypeSP())
                return true;
            else
                return false;
        }

        // We don't have the object cached, so make sure the class
        // address is readable, otherwise this is not a good object:
        m_process->ReadPointerFromMemory (isa_addr, error);
        
        if (error.Fail())
            return false;

        const char *class_name = NULL;
        Address isa_address;
        Target &target = m_process->GetTarget();
        target.GetSectionLoadList().ResolveLoadAddress (isa_addr, isa_address);
        
        if (isa_address.IsValid())
        {
            // If the ISA pointer points to one of the sections in the binary, then see if we can
            // get the class name from the symbols.
        
            SectionSP section_sp (isa_address.GetSection());

            if (section_sp)
            {
                // If this points to a section that we know about, then this is
                // some static class or nothing.  See if it is in the right section 
                // and if its name is the right form.
                ConstString section_name = section_sp->GetName();
                static ConstString g_objc_class_section_name ("__objc_data");
                if (section_name == g_objc_class_section_name)
                {
                    isa_address.CalculateSymbolContext(&sc);
                    if (sc.symbol)
                    {
                        if (sc.symbol->GetType() == eSymbolTypeObjCClass)
                            class_name = sc.symbol->GetName().GetCString();
                        else if (sc.symbol->GetType() == eSymbolTypeObjCMetaClass)
                        {
                            // FIXME: Meta-classes can't have dynamic types...
                            return false;
                        }
                    }
                }
            }
        }
        
        char class_buffer[1024];
        if (class_name == NULL && use_dynamic == eDynamicCanRunTarget)
        {
            // If the class address didn't point into the binary, or
            // it points into the right section but there wasn't a symbol
            // there, try to look it up by calling the class method in the target.
            
            ExecutionContext exe_ctx (in_value.GetExecutionContextRef());
            
            Thread *thread_to_use = exe_ctx.GetThreadPtr();
            
            if (thread_to_use == NULL)
                thread_to_use = m_process->GetThreadList().GetSelectedThread().get();
                
            if (thread_to_use == NULL)
                return false;
                
            if (!RunFunctionToFindClassName (object_ptr, thread_to_use, class_buffer, 1024))
                return false;
                
             class_name = class_buffer;   
            
        }
        
        if (class_name && class_name[0])
        {
            class_type_or_name.SetName (class_name);
            
            TypeList class_types;
            SymbolContext sc;
            const bool exact_match = true;
            uint32_t num_matches = target.GetImages().FindTypes (sc,
                                                                 class_type_or_name.GetName(),
                                                                 exact_match,
                                                                 UINT32_MAX,
                                                                 class_types);
            if (num_matches == 1)
            {
                class_type_or_name.SetTypeSP (class_types.GetTypeAtIndex(0));
                return true;
            }
            else
            {
                for (size_t i  = 0; i < num_matches; i++)
                {
                    TypeSP this_type(class_types.GetTypeAtIndex(i));
                    if (this_type)
                    {
                        // Only consider "real" ObjC classes.  For now this means avoiding
                        // the Type objects that are made up from the OBJC_CLASS_$_<NAME> symbols.
                        // we don't want to use them since they are empty and useless.
                        if (this_type->IsRealObjCClass())
                        {
                            // There can only be one type with a given name,
                            // so we've just found duplicate definitions, and this
                            // one will do as well as any other.
                            // We don't consider something to have a dynamic type if
                            // it is the same as the static type.  So compare against
                            // the value we were handed:
                            
                            clang::ASTContext *in_ast_ctx = in_value.GetClangAST ();
                            clang::ASTContext *this_ast_ctx = this_type->GetClangAST ();
                            if (in_ast_ctx != this_ast_ctx
                                || !ClangASTContext::AreTypesSame (in_ast_ctx, 
                                                                   in_value.GetClangType(),
                                                                   this_type->GetClangFullType()))
                            {
                                class_type_or_name.SetTypeSP (this_type);
                            }
                            break;
                        }
                    }
                }
            }
            
            AddToClassNameCache (isa_addr, class_type_or_name);
            if (class_type_or_name.GetTypeSP())
                return true;
            else
                return false;
        }
    }
    
    return false;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
AppleObjCRuntimeV2::CreateInstance (Process *process, LanguageType language)
{
    // FIXME: This should be a MacOS or iOS process, and we need to look for the OBJC section to make
    // sure we aren't using the V1 runtime.
    if (language == eLanguageTypeObjC)
    {
        ModuleSP objc_module_sp;
        
        if (AppleObjCRuntime::GetObjCVersion (process, objc_module_sp) == eAppleObjC_V2)
            return new AppleObjCRuntimeV2 (process, objc_module_sp);
        else
            return NULL;
    }
    else
        return NULL;
}

void
AppleObjCRuntimeV2::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
AppleObjCRuntimeV2::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
AppleObjCRuntimeV2::GetPluginName()
{
    return pluginName;
}

const char *
AppleObjCRuntimeV2::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
AppleObjCRuntimeV2::GetPluginVersion()
{
    return 1;
}

BreakpointResolverSP
AppleObjCRuntimeV2::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    
    if (throw_bp)
        resolver_sp.reset (new BreakpointResolverName (bkpt,
                                                       "objc_exception_throw",
                                                       eFunctionNameTypeBase,
                                                       Breakpoint::Exact,
                                                       eLazyBoolNo));
    // FIXME: We don't do catch breakpoints for ObjC yet.
    // Should there be some way for the runtime to specify what it can do in this regard?
    return resolver_sp;
}

ClangUtilityFunction *
AppleObjCRuntimeV2::CreateObjectChecker(const char *name)
{
    char check_function_code[2048];
    
    int len = 0;
    if (m_has_object_getClass)
    {
        len = ::snprintf (check_function_code, 
                          sizeof(check_function_code),
                          "extern \"C\" void *gdb_object_getClass(void *);                                          \n"
                          "extern \"C\"  int printf(const char *format, ...);                                       \n"
                          "extern \"C\" void                                                                        \n"
                          "%s(void *$__lldb_arg_obj, void *$__lldb_arg_selector)                                    \n"
                          "{                                                                                        \n"
                          "   if ($__lldb_arg_obj == (void *)0)                                                     \n"
                          "       return; // nil is ok                                                              \n" 
                          "   if (!gdb_object_getClass($__lldb_arg_obj))                                            \n"
                          "       *((volatile int *)0) = 'ocgc';                                                    \n"
                          "   else if ($__lldb_arg_selector != (void *)0)                                           \n"
                          "   {                                                                                     \n"
                          "        signed char responds = (signed char) [(id) $__lldb_arg_obj                       \n"
                          "                                                respondsToSelector:                      \n"
                          "                                       (struct objc_selector *) $__lldb_arg_selector];   \n"
                          "       if (responds == (signed char) 0)                                                  \n"
                          "           *((volatile int *)0) = 'ocgc';                                                \n"
                          "   }                                                                                     \n"
                          "}                                                                                        \n",
                          name);
    }
    else
    {
        len = ::snprintf (check_function_code, 
                          sizeof(check_function_code), 
                          "extern \"C\" void *gdb_class_getClass(void *);                                           \n"
                          "extern \"C\"  int printf(const char *format, ...);                                       \n"
                          "extern \"C\"  void                                                                       \n"
                          "%s(void *$__lldb_arg_obj, void *$__lldb_arg_selector)                                    \n"
                          "{                                                                                        \n"
                          "   if ($__lldb_arg_obj == (void *)0)                                                     \n"
                          "       return; // nil is ok                                                              \n" 
                          "    void **$isa_ptr = (void **)$__lldb_arg_obj;                                          \n"
                          "    if (*$isa_ptr == (void *)0 || !gdb_class_getClass(*$isa_ptr))                        \n"
                          "       *((volatile int *)0) = 'ocgc';                                                    \n"
                          "   else if ($__lldb_arg_selector != (void *)0)                                           \n"
                          "   {                                                                                     \n"
                          "        signed char responds = (signed char) [(id) $__lldb_arg_obj                       \n"
                          "                                                respondsToSelector:                      \n"
                          "                                        (struct objc_selector *) $__lldb_arg_selector];  \n"
                          "       if (responds == (signed char) 0)                                                  \n"
                          "           *((volatile int *)0) = 'ocgc';                                                \n"
                          "   }                                                                                     \n"
                          "}                                                                                        \n", 
                          name);
    }
    
    assert (len < sizeof(check_function_code));

    return new ClangUtilityFunction(check_function_code, name);
}

size_t
AppleObjCRuntimeV2::GetByteOffsetForIvar (ClangASTType &parent_ast_type, const char *ivar_name)
{
    const char *class_name = parent_ast_type.GetConstTypeName().AsCString();

    if (!class_name || *class_name == '\0' || !ivar_name || *ivar_name == '\0')
        return LLDB_INVALID_IVAR_OFFSET;
    
    std::string buffer("OBJC_IVAR_$_");
    buffer.append (class_name);
    buffer.push_back ('.');
    buffer.append (ivar_name);
    ConstString ivar_const_str (buffer.c_str());
    
    SymbolContextList sc_list;
    Target &target = m_process->GetTarget();
    
    target.GetImages().FindSymbolsWithNameAndType(ivar_const_str, eSymbolTypeObjCIVar, sc_list);

    SymbolContext ivar_offset_symbol;
    if (sc_list.GetSize() != 1 
        || !sc_list.GetContextAtIndex(0, ivar_offset_symbol) 
        || ivar_offset_symbol.symbol == NULL)
        return LLDB_INVALID_IVAR_OFFSET;
    
    addr_t ivar_offset_address = ivar_offset_symbol.symbol->GetAddress().GetLoadAddress (&target);
    
    Error error;
    
    uint32_t ivar_offset = m_process->ReadUnsignedIntegerFromMemory (ivar_offset_address, 
                                                                     4, 
                                                                     LLDB_INVALID_IVAR_OFFSET, 
                                                                     error);
    return ivar_offset;
}

// tagged pointers are marked by having their least-significant bit
// set. this makes them "invalid" as pointers because they violate
// the alignment requirements. of course, this detection algorithm
// is not accurate (it might become better by incorporating further
// knowledge about the internals of tagged pointers)
bool
AppleObjCRuntimeV2::IsTaggedPointer(addr_t ptr)
{
    return (ptr & 0x01);
}


// this code relies on the assumption that an Objective-C object always starts
// with an ISA at offset 0. an ISA is effectively a pointer to an instance of
// struct class_t in the ObjCv2 runtime
ObjCLanguageRuntime::ObjCISA
AppleObjCRuntimeV2::GetISA(ValueObject& valobj)
{
    if (ClangASTType::GetMinimumLanguage(valobj.GetClangAST(),valobj.GetClangType()) != eLanguageTypeObjC)
        return 0;
    
    // if we get an invalid VO (which might still happen when playing around
    // with pointers returned by the expression parser, don't consider this
    // a valid ObjC object)
    if (valobj.GetValue().GetContextType() == Value::eContextTypeInvalid)
        return 0;
    
    addr_t isa_pointer = valobj.GetPointerValue();
    
    // tagged pointer
    if (IsTaggedPointer(isa_pointer))
        return g_objc_Tagged_ISA;

    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());

    Process *process = exe_ctx.GetProcessPtr();
    if (process)
    {
        uint8_t pointer_size = process->GetAddressByteSize();
    
        Error error;
        return process->ReadUnsignedIntegerFromMemory (isa_pointer,
                                                       pointer_size,
                                                       0,
                                                       error);
    }
    return 0;
}

// TODO: should we have a transparent_kvo parameter here to say if we 
// want to replace the KVO swizzled class with the actual user-level type?
ConstString
AppleObjCRuntimeV2::GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa)
{
    static const ConstString g_unknown ("unknown");

    if (!IsValidISA(isa))
        return ConstString();
     
    if (isa == g_objc_Tagged_ISA)
    {
        static const ConstString g_objc_tagged_isa_name ("_lldb_Tagged_ObjC_ISA");
        return g_objc_tagged_isa_name;
    }
    
    ISAToNameIterator found = m_isa_to_name_cache.find(isa);
    ISAToNameIterator end = m_isa_to_name_cache.end();
    
    if (found != end)
        return found->second;
    
    uint8_t pointer_size = m_process->GetAddressByteSize();
    Error error;
    
    /*
     struct class_t *isa;
     struct class_t *superclass;
     Cache cache;
     IMP *vtable;
-->     class_rw_t data;
     */
    
    addr_t rw_pointer = isa + (4 * pointer_size);
    //printf("rw_pointer: %llx\n", rw_pointer);
    uint64_t data_pointer =  m_process->ReadUnsignedIntegerFromMemory(rw_pointer,
                                                                      pointer_size,
                                                                      0,
                                                                      error);
    if (error.Fail())
    {
        return g_unknown;

    }
    
    /*
     uint32_t flags;
     uint32_t version;
     
-->     const class_ro_t *ro;
     */
    data_pointer += 8;
    //printf("data_pointer: %llx\n", data_pointer);
    uint64_t ro_pointer = m_process->ReadUnsignedIntegerFromMemory(data_pointer,
                                                                   pointer_size,
                                                                   0,
                                                                   error);
    if (error.Fail())
        return g_unknown;
    
    /*
     uint32_t flags;
     uint32_t instanceStart;
     uint32_t instanceSize;
     #ifdef __LP64__
     uint32_t reserved;
     #endif
     
     const uint8_t * ivarLayout;
     
-->     const char * name;
     */
    ro_pointer += 12;
    if (pointer_size == 8)
        ro_pointer += 4;
    ro_pointer += pointer_size;
    //printf("ro_pointer: %llx\n", ro_pointer);
    uint64_t name_pointer = m_process->ReadUnsignedIntegerFromMemory(ro_pointer,
                                                                     pointer_size,
                                                                     0,
                                                                     error);
    if (error.Fail())
        return g_unknown;
    
    //printf("name_pointer: %llx\n", name_pointer);
    char cstr[512];
    if (m_process->ReadCStringFromMemory(name_pointer, cstr, sizeof(cstr), error) > 0)
    {
        if (::strstr(cstr, "NSKVONotify") == cstr)
        {
            // the ObjC runtime implements KVO by replacing the isa with a special
            // NSKVONotifying_className that overrides the relevant methods
            // the side effect on us is that getting the typename for a KVO-ed object
            // will return the swizzled class instead of the actual one
            // this swizzled class is a descendant of the real class, so just
            // return the parent type and all should be fine
            ConstString class_name = GetActualTypeName(GetParentClass(isa));
            m_isa_to_name_cache[isa] = class_name;
            return class_name;
        }
        else
        {
            ConstString class_name = ConstString(cstr);
            m_isa_to_name_cache[isa] = class_name;
            return class_name;
        }
    }
    else
        return g_unknown;
}

ObjCLanguageRuntime::ObjCISA
AppleObjCRuntimeV2::GetParentClass(ObjCLanguageRuntime::ObjCISA isa)
{
    if (!IsValidISA(isa))
        return 0;
    
    if (isa == g_objc_Tagged_ISA)
        return 0;
    
    ISAToParentIterator found = m_isa_to_parent_cache.find(isa);
    ISAToParentIterator end = m_isa_to_parent_cache.end();
    
    if (found != end)
        return found->second;
    
    uint8_t pointer_size = m_process->GetAddressByteSize();
    Error error;
    /*
     struct class_t *isa;
-->     struct class_t *superclass;
     */
    addr_t parent_pointer = isa + pointer_size;
    //printf("rw_pointer: %llx\n", rw_pointer);
    
    uint64_t parent_isa =  m_process->ReadUnsignedIntegerFromMemory(parent_pointer,
                                                                    pointer_size,
                                                                    0,
                                                                    error);
    if (error.Fail())
        return 0;
    
    m_isa_to_parent_cache[isa] = parent_isa;
    
    return parent_isa;
}

SymbolVendor *
AppleObjCRuntimeV2::GetSymbolVendor()
{
    if (!m_symbol_vendor_ap.get())
        m_symbol_vendor_ap.reset(new AppleObjCSymbolVendor(m_process));
    
    return m_symbol_vendor_ap.get();
}
