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
#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "AppleObjCRuntimeV2.h"
#include "AppleObjCTypeVendor.h"
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
    m_process_wp (process->shared_from_this())
{
    static const ConstString g_gdb_object_getClass("gdb_object_getClass");
    m_has_object_getClass = (objc_module_sp->FindFirstSymbolWithNameAndType(g_gdb_object_getClass, eSymbolTypeCode) != NULL);
}

bool
AppleObjCRuntimeV2::RunFunctionToFindClassName(addr_t object_addr, Thread *thread, char *name_dst, size_t max_name_len)
{
    // Since we are going to run code we have to make sure only one thread at a time gets to try this.
    Mutex::Locker locker(m_get_class_name_args_mutex);
    
    StreamString errors;
    
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));  // FIXME - a more appropriate log channel?
    
    int32_t debug;
    if (log && log->GetVerbose())
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
                    isa_address.CalculateSymbolContext(&sc, eSymbolContextModule | eSymbolContextSymbol);
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

class RemoteNXMapTable
{
public:
    RemoteNXMapTable (lldb::ProcessSP process_sp,
                      lldb::addr_t load_addr) :
        m_process_sp(process_sp),
        m_end_iterator(*this, -1),
        m_load_addr(load_addr),
        m_map_pair_size(m_process_sp->GetAddressByteSize() * 2),
        m_NXMAPNOTAKEY(m_process_sp->GetAddressByteSize() == 8 ? 0xffffffffffffffffull : 0xffffffffull)
    {
        lldb::addr_t cursor = load_addr;
     
        Error err;
        
        // const struct +NXMapTablePrototype *prototype;
        m_prototype_ptr = m_process_sp->ReadPointerFromMemory(cursor, err);
        cursor += m_process_sp->GetAddressByteSize();
                
        // unsigned count;
        m_count = m_process_sp->ReadUnsignedIntegerFromMemory(cursor, sizeof(unsigned), 0, err);
        cursor += sizeof(unsigned);
        
        // unsigned nbBucketsMinusOne;
        m_nbBucketsMinusOne = m_process_sp->ReadUnsignedIntegerFromMemory(cursor, sizeof(unsigned), 0, err);
        cursor += sizeof(unsigned);
        
        // void *buckets;
        m_buckets_ptr = m_process_sp->ReadPointerFromMemory(cursor, err);
    }
    
    // const_iterator mimics NXMapState and its code comes from NXInitMapState and NXNextMapState.
    typedef std::pair<ConstString, ObjCLanguageRuntime::ObjCISA> element;

    friend class const_iterator;
    class const_iterator
    {
    public:
        const_iterator (RemoteNXMapTable &parent, int index) : m_parent(parent), m_index(index)
        {
            AdvanceToValidIndex();
        }
        
        const_iterator (const const_iterator &rhs) : m_parent(rhs.m_parent), m_index(rhs.m_index)
        {
            // AdvanceToValidIndex() has been called by rhs already.
        }
        
        const_iterator &operator=(const const_iterator &rhs)
        {
            // AdvanceToValidIndex() has been called by rhs already.
            assert (&m_parent == &rhs.m_parent);
            m_index = rhs.m_index;
            return *this;
        }
        
        bool operator==(const const_iterator &rhs) const
        {
            if (&m_parent != &rhs.m_parent)
                return false;
            if (m_index != rhs.m_index)
                return false;
            
            return true;
        }
        
        bool operator!=(const const_iterator &rhs) const
        {
            return !(operator==(rhs));
        }
        
        const_iterator &operator++()
        {
            AdvanceToValidIndex();
            return *this;
        }
        
        const element operator*() const
        {
            if (m_index == -1)
            {
                // TODO find a way to make this an error, but not an assert
                return element();
            }
         
            lldb::addr_t    pairs_ptr        = m_parent.m_buckets_ptr;
            size_t          map_pair_size   = m_parent.m_map_pair_size;
            lldb::addr_t    pair_ptr         = pairs_ptr + (m_index * map_pair_size);
            
            Error           err;
            
            lldb::addr_t    key     = m_parent.m_process_sp->ReadPointerFromMemory(pair_ptr, err);
            if (!err.Success())
                return element();
            lldb::addr_t    value   = m_parent.m_process_sp->ReadPointerFromMemory(pair_ptr + m_parent.m_process_sp->GetAddressByteSize(), err);
            if (!err.Success())
                return element();
            
            std::string key_string;
            
            m_parent.m_process_sp->ReadCStringFromMemory(key, key_string, err);
            if (!err.Success())
                return element();
            
            return element(ConstString(key_string.c_str()), (ObjCLanguageRuntime::ObjCISA)value);
        }
    private:
        void AdvanceToValidIndex ()
        {
            if (m_index == -1)
                return;
            
            lldb::addr_t    pairs_ptr        = m_parent.m_buckets_ptr;
            size_t          map_pair_size   = m_parent.m_map_pair_size;
            lldb::addr_t    NXMAPNOTAKEY    = m_parent.m_NXMAPNOTAKEY;
            Error           err;

            while (m_index--)
            {
                lldb::addr_t pair_ptr = pairs_ptr + (m_index * map_pair_size);
                lldb::addr_t key = m_parent.m_process_sp->ReadPointerFromMemory(pair_ptr, err);
                
                if (!err.Success())
                {
                    m_index = -1;
                    return;
                }
                
                if (key != NXMAPNOTAKEY)
                    return;
            }
        }
        RemoteNXMapTable   &m_parent;
        int                 m_index;
    };
    
    const_iterator begin ()
    {
        return const_iterator(*this, m_nbBucketsMinusOne + 1);
    }
    
    const_iterator end ()
    {
        return m_end_iterator;
    }
    
private:
    // contents of _NXMapTable struct
    lldb::addr_t                        m_prototype_ptr;
    uint32_t                            m_count;
    uint32_t                            m_nbBucketsMinusOne;
    lldb::addr_t                        m_buckets_ptr;
    
    lldb::ProcessSP                     m_process_sp;
    const_iterator                      m_end_iterator;
    lldb::addr_t                        m_load_addr;
    size_t                              m_map_pair_size;
    lldb::addr_t                        m_NXMAPNOTAKEY;
};

class RemoteObjCOpt
{
public:
    RemoteObjCOpt (lldb::ProcessSP process_sp,
                   lldb::addr_t load_addr) :
        m_process_sp(process_sp),
        m_end_iterator(*this, -1ll),
        m_load_addr(load_addr)
    {
        lldb::addr_t cursor = load_addr;
        
        Error err;
        
        // uint32_t version;
        m_version = m_process_sp->ReadUnsignedIntegerFromMemory(cursor, sizeof(uint32_t), 0, err);
        cursor += sizeof(uint32_t);
        
        // int32_t selopt_offset;
        cursor += sizeof(int32_t);
        
        // int32_t headeropt_offset;
        cursor += sizeof(int32_t);
        
        // int32_t clsopt_offset;
        {
            Scalar clsopt_offset;
            m_process_sp->ReadScalarIntegerFromMemory(cursor, sizeof(int32_t), /*is_signed*/ true, clsopt_offset, err);
            m_clsopt_offset = clsopt_offset.SInt();
            cursor += sizeof(int32_t);
        }
        
        if (m_version != 12)
            return;
        
        m_clsopt_ptr = load_addr + m_clsopt_offset;
        
        cursor = m_clsopt_ptr;
        
        // uint32_t capacity;
        m_capacity = m_process_sp->ReadUnsignedIntegerFromMemory(cursor, sizeof(uint32_t), 0, err);
        cursor += sizeof(uint32_t);
        
        // uint32_t occupied;
        cursor += sizeof(uint32_t);
        
        // uint32_t shift;
        cursor += sizeof(uint32_t);
        
        // uint32_t mask;
        m_mask = m_process_sp->ReadUnsignedIntegerFromMemory(cursor, sizeof(uint32_t), 0, err);
        cursor += sizeof(uint32_t);

        // uint32_t zero;
        m_zero_offset = cursor - m_clsopt_ptr;
        cursor += sizeof(uint32_t);
        
        // uint32_t unused;
        cursor += sizeof(uint32_t);
        
        // uint64_t salt;
        cursor += sizeof(uint64_t);
        
        // uint32_t scramble[256];
        cursor += sizeof(uint32_t) * 256;
        
        // uint8_t tab[mask+1];
        cursor += sizeof(uint8_t) * (m_mask + 1);
        
        // uint8_t checkbytes[capacity];
        cursor += sizeof(uint8_t) * m_capacity;
        
        // int32_t offset[capacity];
        cursor += sizeof(int32_t) * m_capacity;
        
        // objc_classheader_t clsOffsets[capacity];
        m_clsOffsets_ptr = cursor;
        cursor += (m_classheader_size * m_capacity);
        
        // uint32_t duplicateCount;
        m_duplicateCount = m_process_sp->ReadUnsignedIntegerFromMemory(cursor, sizeof(uint32_t), 0, err);
        cursor += sizeof(uint32_t);
        
        // objc_classheader_t duplicateOffsets[duplicateCount];
        m_duplicateOffsets_ptr = cursor;
    }
    
    friend class const_iterator;
    class const_iterator
    {
    public:
        const_iterator (RemoteObjCOpt &parent, int64_t index) : m_parent(parent), m_index(index)
        {
            AdvanceToValidIndex();
        }
        
        const_iterator (const const_iterator &rhs) : m_parent(rhs.m_parent), m_index(rhs.m_index)
        {
            // AdvanceToValidIndex() has been called by rhs already
        }
        
        const_iterator &operator=(const const_iterator &rhs)
        {
            assert (&m_parent == &rhs.m_parent);
            m_index = rhs.m_index;
            return *this;
        }
        
        bool operator==(const const_iterator &rhs) const
        {
            if (&m_parent != &rhs.m_parent)
                return false;
            if (m_index != rhs.m_index)
                return false;
            return true;
        }
        
        bool operator!=(const const_iterator &rhs) const
        {
            return !(operator==(rhs));
        }
        
        const_iterator &operator++()
        {
            AdvanceToValidIndex();
            return *this;
        }
        
        const ObjCLanguageRuntime::ObjCISA operator*() const
        {
            if (m_index == -1)
                return 0;
            
            Error err;
            return isaForIndex(err);
        }
    private:
        ObjCLanguageRuntime::ObjCISA isaForIndex(Error &err) const
        {
            if (m_index >= m_parent.m_capacity + m_parent.m_duplicateCount)
                return 0; // index out of range
            
            lldb::addr_t classheader_ptr;
            
            if (m_index >= m_parent.m_capacity)
            {
                // index in the duplicate offsets
                uint32_t index = (uint32_t)((uint64_t)m_index - (uint64_t)m_parent.m_capacity);
                classheader_ptr = m_parent.m_duplicateOffsets_ptr + (index * m_parent.m_classheader_size);
            }
            else
            {
                // index in the offsets
                uint32_t index = (uint32_t)m_index;
                classheader_ptr = m_parent.m_clsOffsets_ptr + (index * m_parent.m_classheader_size);
            }
            
            Scalar clsOffset;
            m_parent.m_process_sp->ReadScalarIntegerFromMemory(classheader_ptr, sizeof(int32_t), /*is_signed*/ true, clsOffset, err);
            if (!err.Success())
                return 0;
            
            int32_t clsOffset_int = clsOffset.SInt();
            if (clsOffset_int & 0x1)
                return 0; // not even

            if (clsOffset_int == m_parent.m_zero_offset)
                return 0; // == offsetof(objc_clsopt_t, zero)
            
            return m_parent.m_clsopt_ptr + (int64_t)clsOffset_int;
        }
        
        void AdvanceToValidIndex ()
        {
            if (m_index == -1)
                return;
            
            Error err;
            
            m_index--;
            
            while (m_index >= 0)
            {
                ObjCLanguageRuntime::ObjCISA objc_isa = isaForIndex(err);
                if (objc_isa)
                    return;
                m_index--;
            }
        }
        RemoteObjCOpt  &m_parent;
        int64_t         m_index;
    };
    
    const_iterator begin ()
    {
        return const_iterator(*this, (int64_t)m_capacity + (int64_t)m_duplicateCount);
    }
    
    const_iterator end ()
    {
        return m_end_iterator;
    }
    
private:
    // contents of objc_opt struct
    uint32_t                            m_version;
    int32_t                             m_clsopt_offset;
    
    lldb::addr_t                        m_clsopt_ptr;
    
    // contents of objc_clsopt struct
    uint32_t                            m_capacity;
    uint32_t                            m_mask;
    uint32_t                            m_duplicateCount;
    lldb::addr_t                        m_clsOffsets_ptr;
    lldb::addr_t                        m_duplicateOffsets_ptr;
    int32_t                             m_zero_offset;
    
    lldb::ProcessSP                     m_process_sp;
    const_iterator                      m_end_iterator;
    lldb::addr_t                        m_load_addr;
    const size_t                        m_classheader_size = (sizeof(int32_t) * 2);
};

class ClassDescriptorV2 : public ObjCLanguageRuntime::ClassDescriptor
{
public:
    friend class lldb_private::AppleObjCRuntimeV2;
    
private:
    // The constructor should only be invoked by the runtime as it builds its caches
    // or populates them.  A ClassDescriptorV2 should only ever exist in a cache.
    ClassDescriptorV2 (AppleObjCRuntimeV2 &runtime, ObjCLanguageRuntime::ObjCISA isa) :
        m_runtime (runtime),
        m_objc_class_ptr (isa),
        m_name ()
    {
    }
    
public:
    virtual ConstString
    GetClassName ()
    {
        if (!m_name)
        {
            lldb::ProcessSP process_sp = m_runtime.GetProcessSP();

            if (process_sp)
            {
                std::auto_ptr<objc_class_t> objc_class;
                std::auto_ptr<class_ro_t> class_ro;
                std::auto_ptr<class_rw_t> class_rw;
                
                if (!Read_objc_class(process_sp, objc_class))
                    return m_name;
                if (!Read_class_row(process_sp, *objc_class, class_ro, class_rw))
                    return m_name;
                
                m_name = ConstString(class_ro->m_name.c_str());
            }
        }
        return m_name;
    }
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetSuperclass ()
    {
        lldb::ProcessSP process_sp = m_runtime.GetProcessSP();

        if (!process_sp)
            return ObjCLanguageRuntime::ClassDescriptorSP();
        
        std::auto_ptr<objc_class_t> objc_class;

        if (!Read_objc_class(process_sp, objc_class))
            return ObjCLanguageRuntime::ClassDescriptorSP();

        return m_runtime.ObjCLanguageRuntime::GetClassDescriptor(objc_class->m_superclass);
    }
    
    virtual bool
    IsValid ()
    {
        return true;    // any Objective-C v2 runtime class descriptor we vend is valid
    }
    
    virtual bool
    IsTagged ()
    {
        return false;   // we use a special class for tagged descriptors
    }
    
    virtual uint64_t
    GetInstanceSize ()
    {
        lldb::ProcessSP process_sp = m_runtime.GetProcessSP();
        
        if (process_sp)
        {
            std::auto_ptr<objc_class_t> objc_class;
            std::auto_ptr<class_ro_t> class_ro;
            std::auto_ptr<class_rw_t> class_rw;
            
            if (!Read_objc_class(process_sp, objc_class))
                return 0;
            if (!Read_class_row(process_sp, *objc_class, class_ro, class_rw))
                return 0;
            
            return class_ro->m_instanceSize;
        }
        
        return 0;
    }
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA ()
    {        
        return m_objc_class_ptr;
    }
    
    virtual bool
    Describe (std::function <void (ObjCLanguageRuntime::ObjCISA)> const &superclass_func,
              std::function <void (const char *, const char *)> const &instance_method_func,
              std::function <void (const char *, const char *)> const &class_method_func)
    {
        lldb::ProcessSP process_sp = m_runtime.GetProcessSP();

        std::auto_ptr<objc_class_t> objc_class;
        std::auto_ptr<class_ro_t> class_ro;
        std::auto_ptr<class_rw_t> class_rw;
        
        if (!Read_objc_class(process_sp, objc_class))
            return 0;
        if (!Read_class_row(process_sp, *objc_class, class_ro, class_rw))
            return 0;
    
        static ConstString NSObject_name("NSObject");
        
        if (m_name != NSObject_name && superclass_func)
            superclass_func(objc_class->m_superclass);
        
        if (instance_method_func)
        {
            std::auto_ptr <method_list_t> base_method_list;
            
            base_method_list.reset(new method_list_t);
            if (!base_method_list->Read(process_sp, class_ro->m_baseMethods_ptr))
                return false;
            
            if (base_method_list->m_entsize != method_t::GetSize(process_sp))
                return false;
            
            std::auto_ptr <method_t> method;
            method.reset(new method_t);
            
            for (uint32_t i = 0, e = base_method_list->m_count; i < e; ++i)
            {
                method->Read(process_sp, base_method_list->m_first_ptr + (i * base_method_list->m_entsize));
                
                instance_method_func(method->m_name.c_str(), method->m_types.c_str());
            }
        }
        
        if (class_method_func)
        {
            ObjCLanguageRuntime::ClassDescriptorSP metaclass = m_runtime.ObjCLanguageRuntime::GetClassDescriptor(objc_class->m_isa);
            
            // We don't care about the metaclass's superclass, or its class methods.  Its instance methods are
            // our class methods.
            
            metaclass->Describe(std::function <void (ObjCLanguageRuntime::ObjCISA)> (nullptr),
                                class_method_func,
                                std::function <void (const char *, const char *)> (nullptr));
        }
        while (0);
            
        return true;
    }
    
    virtual
    ~ClassDescriptorV2 ()
    {
    }
        
private:
    static const uint32_t RW_REALIZED = (1 << 31);
    
    struct objc_class_t {
        ObjCLanguageRuntime::ObjCISA    m_isa;              // The class's metaclass.
        ObjCLanguageRuntime::ObjCISA    m_superclass;
        lldb::addr_t                    m_cache_ptr;
        lldb::addr_t                    m_vtable_ptr;
        lldb::addr_t                    m_data_ptr;
        uint8_t                         m_flags;
        
        objc_class_t () :
            m_isa (0),
            m_superclass (0),
            m_cache_ptr (0),
            m_vtable_ptr (0),
            m_data_ptr (0),
            m_flags (0)
        {
        }
        
        void
        Clear()
        {
            m_isa = 0;
            m_superclass = 0;
            m_cache_ptr = 0;
            m_vtable_ptr = 0;
            m_data_ptr = 0;
            m_flags = 0;
        }

        bool Read(ProcessSP &process_sp, lldb::addr_t addr)
        {
            size_t ptr_size = process_sp->GetAddressByteSize();
            
            size_t objc_class_size = ptr_size   // uintptr_t isa;
            + ptr_size   // Class superclass;
            + ptr_size   // void *cache;
            + ptr_size   // IMP *vtable;
            + ptr_size;  // uintptr_t data_NEVER_USE;
            
            DataBufferHeap objc_class_buf (objc_class_size, '\0');
            Error error;
            
            process_sp->ReadMemory(addr, objc_class_buf.GetBytes(), objc_class_size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(objc_class_buf.GetBytes(), objc_class_size, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
            
            uint32_t cursor = 0;
            
            m_isa           = extractor.GetAddress_unchecked(&cursor);   // uintptr_t isa;
            m_superclass    = extractor.GetAddress_unchecked(&cursor);   // Class superclass;
            m_cache_ptr     = extractor.GetAddress_unchecked(&cursor);   // void *cache;
            m_vtable_ptr    = extractor.GetAddress_unchecked(&cursor);   // IMP *vtable;
            lldb::addr_t data_NEVER_USE = extractor.GetAddress_unchecked(&cursor);   // uintptr_t data_NEVER_USE;
            
            m_flags         = (uint8_t)(data_NEVER_USE & (lldb::addr_t)3);
            m_data_ptr      = data_NEVER_USE & ~(lldb::addr_t)3;
            
            return true;
        }
    };
    
    struct class_ro_t {
        uint32_t                        m_flags;
        uint32_t                        m_instanceStart;
        uint32_t                        m_instanceSize;
        uint32_t                        m_reserved;
        
        lldb::addr_t                    m_ivarLayout_ptr;
        lldb::addr_t                    m_name_ptr;
        lldb::addr_t                    m_baseMethods_ptr;
        lldb::addr_t                    m_baseProtocols_ptr;
        lldb::addr_t                    m_ivars_ptr;
        
        lldb::addr_t                    m_weakIvarLayout_ptr;
        lldb::addr_t                    m_baseProperties_ptr;
        
        std::string                     m_name;
        
        bool Read(ProcessSP &process_sp, lldb::addr_t addr)
        {
            size_t ptr_size = process_sp->GetAddressByteSize();
            
            size_t size = sizeof(uint32_t)             // uint32_t flags;
            + sizeof(uint32_t)                         // uint32_t instanceStart;
            + sizeof(uint32_t)                         // uint32_t instanceSize;
            + (ptr_size == 8 ? sizeof(uint32_t) : 0)   // uint32_t reserved; // __LP64__ only
            + ptr_size                                 // const uint8_t *ivarLayout;
            + ptr_size                                 // const char *name;
            + ptr_size                                 // const method_list_t *baseMethods;
            + ptr_size                                 // const protocol_list_t *baseProtocols;
            + ptr_size                                 // const ivar_list_t *ivars;
            + ptr_size                                 // const uint8_t *weakIvarLayout;
            + ptr_size;                                // const property_list_t *baseProperties;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process_sp->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
            
            uint32_t cursor = 0;
            
            m_flags             = extractor.GetU32_unchecked(&cursor);
            m_instanceStart     = extractor.GetU32_unchecked(&cursor);
            m_instanceSize      = extractor.GetU32_unchecked(&cursor);
            if (ptr_size == 8)
                m_reserved      = extractor.GetU32_unchecked(&cursor);
            else
                m_reserved      = 0;
            m_ivarLayout_ptr     = extractor.GetAddress_unchecked(&cursor);
            m_name_ptr           = extractor.GetAddress_unchecked(&cursor);
            m_baseMethods_ptr    = extractor.GetAddress_unchecked(&cursor);
            m_baseProtocols_ptr  = extractor.GetAddress_unchecked(&cursor);
            m_ivars_ptr          = extractor.GetAddress_unchecked(&cursor);
            m_weakIvarLayout_ptr = extractor.GetAddress_unchecked(&cursor);
            m_baseProperties_ptr = extractor.GetAddress_unchecked(&cursor);
            
            DataBufferHeap name_buf(1024, '\0');
            
            process_sp->ReadCStringFromMemory(m_name_ptr, (char*)name_buf.GetBytes(), name_buf.GetByteSize(), error);
            
            if (error.Fail())
            {
                return false;
            }
            
            m_name.assign((char*)name_buf.GetBytes());
                
            return true;
        }
    };
    
    struct class_rw_t {
        uint32_t                        m_flags;
        uint32_t                        m_version;
        
        lldb::addr_t                    m_ro_ptr;
        union {
            lldb::addr_t                m_method_list_ptr;
            lldb::addr_t                m_method_lists_ptr;
        };
        lldb::addr_t                    m_properties_ptr;
        lldb::addr_t                    m_protocols_ptr;
        
        ObjCLanguageRuntime::ObjCISA    m_firstSubclass;
        ObjCLanguageRuntime::ObjCISA    m_nextSiblingClass;
        
        bool Read(ProcessSP &process_sp, lldb::addr_t addr)
        {
            size_t ptr_size = process_sp->GetAddressByteSize();
            
            size_t size = sizeof(uint32_t)  // uint32_t flags;
            + sizeof(uint32_t)  // uint32_t version;
            + ptr_size          // const class_ro_t *ro;
            + ptr_size          // union { method_list_t **method_lists; method_list_t *method_list; };
            + ptr_size          // struct chained_property_list *properties;
            + ptr_size          // const protocol_list_t **protocols;
            + ptr_size          // Class firstSubclass;
            + ptr_size;         // Class nextSiblingClass;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process_sp->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
            
            uint32_t cursor = 0;
            
            m_flags             = extractor.GetU32_unchecked(&cursor);
            m_version           = extractor.GetU32_unchecked(&cursor);
            m_ro_ptr             = extractor.GetAddress_unchecked(&cursor);
            m_method_list_ptr    = extractor.GetAddress_unchecked(&cursor);
            m_properties_ptr     = extractor.GetAddress_unchecked(&cursor);
            m_firstSubclass     = extractor.GetAddress_unchecked(&cursor);
            m_nextSiblingClass  = extractor.GetAddress_unchecked(&cursor);
            
            return true;
        }
    };
    
    struct method_list_t
    {
        uint32_t        m_entsize;
        uint32_t        m_count;
        lldb::addr_t    m_first_ptr;
        
        bool Read(ProcessSP &process_sp, lldb::addr_t addr)
        {
            size_t size = sizeof(uint32_t)  // uint32_t entsize_NEVER_USE;
            + sizeof(uint32_t); // uint32_t count;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process_sp->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
            
            uint32_t cursor = 0;
            
            m_entsize   = extractor.GetU32_unchecked(&cursor) & ~(uint32_t)3;
            m_count     = extractor.GetU32_unchecked(&cursor);
            m_first_ptr  = addr + cursor;
            
            return true;
        }
    };
    
    struct method_t
    {
        lldb::addr_t    m_name_ptr;
        lldb::addr_t    m_types_ptr;
        lldb::addr_t    m_imp_ptr;
        
        std::string     m_name;
        std::string     m_types;
        
        static size_t GetSize(ProcessSP &process_sp)
        {
            size_t ptr_size = process_sp->GetAddressByteSize();
            
            return ptr_size     // SEL name;
            + ptr_size   // const char *types;
            + ptr_size;  // IMP imp;
        }
        
        bool Read(ProcessSP &process_sp, lldb::addr_t addr)
        {
            size_t size = GetSize(process_sp);
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process_sp->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process_sp->GetByteOrder(), process_sp->GetAddressByteSize());
            
            uint32_t cursor = 0;
            
            m_name_ptr   = extractor.GetAddress_unchecked(&cursor);
            m_types_ptr  = extractor.GetAddress_unchecked(&cursor);
            m_imp_ptr    = extractor.GetAddress_unchecked(&cursor);
            
            const size_t buffer_size = 1024;
            size_t count;
            
            DataBufferHeap string_buf(buffer_size, 0);
            
            count = process_sp->ReadCStringFromMemory(m_name_ptr, (char*)string_buf.GetBytes(), buffer_size, error);
            m_name.assign((char*)string_buf.GetBytes(), count);
            
            count = process_sp->ReadCStringFromMemory(m_types_ptr, (char*)string_buf.GetBytes(), buffer_size, error);
            m_types.assign((char*)string_buf.GetBytes(), count);
            
            return true;
        }
    };
    
    bool Read_objc_class (lldb::ProcessSP process_sp, std::auto_ptr<objc_class_t> &objc_class)
    {
        objc_class.reset(new objc_class_t);
        
        bool ret = objc_class->Read (process_sp, m_objc_class_ptr);
        
        if (!ret)
            objc_class.reset();
        
        return ret;
    }
    
    bool Read_class_row (lldb::ProcessSP process_sp, const objc_class_t &objc_class, std::auto_ptr<class_ro_t> &class_ro, std::auto_ptr<class_rw_t> &class_rw)
    {
        class_ro.reset();
        class_rw.reset();
        
        Error error;
        uint32_t class_row_t_flags = process_sp->ReadUnsignedIntegerFromMemory(objc_class.m_data_ptr, sizeof(uint32_t), 0, error);
        if (!error.Success())
            return false;

        if (class_row_t_flags & RW_REALIZED)
        {
            class_rw.reset(new class_rw_t);
            
            if (!class_rw->Read(process_sp, objc_class.m_data_ptr))
            {
                class_rw.reset();
                return false;
            }
            
            class_ro.reset(new class_ro_t);
            
            if (!class_ro->Read(process_sp, class_rw->m_ro_ptr))
            {
                class_rw.reset();
                class_ro.reset();
                return false;
            }
        }
        else
        {
            class_ro.reset(new class_ro_t);
            
            if (!class_ro->Read(process_sp, objc_class.m_data_ptr))
            {
                class_ro.reset();
                return false;
            }
        }
        
        return true;
    }

    AppleObjCRuntimeV2 &m_runtime;          // The runtime, so we can read information lazily.
    lldb::addr_t        m_objc_class_ptr;   // The address of the objc_class_t.  (I.e., objects of this class type have this as their ISA)
    ConstString         m_name;             // May be NULL
};

class ClassDescriptorV2Tagged : public ObjCLanguageRuntime::ClassDescriptor
{
public:
    ClassDescriptorV2Tagged (ValueObject &isa_pointer)
    {
        m_valid = false;
        uint64_t value = isa_pointer.GetValueAsUnsigned(0);
        lldb::ProcessSP process_sp = isa_pointer.GetProcessSP();
        if (process_sp)
            m_pointer_size = process_sp->GetAddressByteSize();
        else
        {
            m_name = ConstString("");
            m_pointer_size = 0;
            return;
        }
        
        m_valid = true;
        m_class_bits = (value & 0xE) >> 1;
        lldb::TargetSP target_sp = isa_pointer.GetTargetSP();
        
        LazyBool is_lion = IsLion(target_sp);
        
        // TODO: check for OSX version - for now assume Mtn Lion
        if (is_lion == eLazyBoolCalculate)
        {
            // if we can't determine the matching table (e.g. we have no Foundation),
            // assume this is not a valid tagged pointer
            m_valid = false;
        }
        else if (is_lion == eLazyBoolNo)
        {
            switch (m_class_bits)
            {
                case 0:
                    m_name = ConstString("NSAtom");
                    break;
                case 3:
                    m_name = ConstString("NSNumber");
                    break;
                case 4:
                    m_name = ConstString("NSDateTS");
                    break;
                case 5:
                    m_name = ConstString("NSManagedObject");
                    break;
                case 6:
                    m_name = ConstString("NSDate");
                    break;
                default:
                    m_valid = false;
                    break;
            }
        }
        else
        {
            switch (m_class_bits)
            {
                case 1:
                    m_name = ConstString("NSNumber");
                    break;
                case 5:
                    m_name = ConstString("NSManagedObject");
                    break;
                case 6:
                    m_name = ConstString("NSDate");
                    break;
                case 7:
                    m_name = ConstString("NSDateTS");
                    break;
                default:
                    m_valid = false;
                    break;
            }
        }
        if (!m_valid)
            m_name = ConstString("");
        else
        {
            m_info_bits = (value & 0xF0ULL) >> 4;
            m_value_bits = (value & ~0x0000000000000000FFULL) >> 8;
        }
    }
    
    virtual ConstString
    GetClassName ()
    {
        return m_name;
    }
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetSuperclass ()
    {
        // tagged pointers can represent a class that has a superclass, but since that information is not
        // stored in the object itself, we would have to query the runtime to discover the hierarchy
        // for the time being, we skip this step in the interest of static discovery
        return ObjCLanguageRuntime::ClassDescriptorSP(new ObjCLanguageRuntime::ClassDescriptor_Invalid());
    }
    
    virtual bool
    IsValid ()
    {
        return m_valid;
    }
    
    virtual bool
    IsKVO ()
    {
        return false; // tagged pointers are not KVO'ed
    }
    
    virtual bool
    IsCFType ()
    {
        return false; // tagged pointers are not CF objects
    }
    
    virtual bool
    IsTagged ()
    {
        return true;   // we use this class to describe tagged pointers
    }
    
    virtual uint64_t
    GetInstanceSize ()
    {
        return (IsValid() ? m_pointer_size : 0);
    }
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA ()
    {
        return 0; // tagged pointers have no ISA
    }
    
    virtual uint64_t
    GetClassBits ()
    {
        return (IsValid() ? m_class_bits : 0);
    }
    
    // these calls are not part of any formal tagged pointers specification
    virtual uint64_t
    GetValueBits ()
    {
        return (IsValid() ? m_value_bits : 0);
    }
    
    virtual uint64_t
    GetInfoBits ()
    {
        return (IsValid() ? m_info_bits : 0);
    }
    
    virtual
    ~ClassDescriptorV2Tagged ()
    {}
    
protected:
    // TODO make this into a smarter OS version detector
    LazyBool
    IsLion (lldb::TargetSP &target_sp)
    {
        if (!target_sp)
            return eLazyBoolCalculate;
        ModuleList& modules = target_sp->GetImages();
        for (uint32_t idx = 0; idx < modules.GetSize(); idx++)
        {
            lldb::ModuleSP module_sp = modules.GetModuleAtIndex(idx);
            if (!module_sp)
                continue;
            if (strcmp(module_sp->GetFileSpec().GetFilename().AsCString(""),"Foundation") == 0)
            {
                uint32_t major = UINT32_MAX;
                module_sp->GetVersion(&major,1);
                if (major == UINT32_MAX)
                    return eLazyBoolCalculate;
                
                return (major > 900 ? eLazyBoolNo : eLazyBoolYes);
            }
        }
        return eLazyBoolCalculate;
    }
    
private:
    ConstString m_name;
    uint8_t m_pointer_size;
    bool m_valid;
    uint64_t m_class_bits;
    uint64_t m_info_bits;
    uint64_t m_value_bits;
};

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::CreateClassDescriptor (ObjCISA isa)
{
    ClassDescriptorSP objc_class_sp;
    if (isa != 0)
        objc_class_sp.reset (new ClassDescriptorV2(*this, isa));
    return objc_class_sp;
}

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::GetClassDescriptor (ValueObject& in_value)
{
    ClassDescriptorSP objc_class_sp;
    uint64_t ptr_value = in_value.GetValueAsUnsigned(0);
    if (ptr_value)
    {
        if (ptr_value & 1)
            objc_class_sp = ClassDescriptorSP(new ClassDescriptorV2Tagged(in_value));
        else
            objc_class_sp = ObjCLanguageRuntime::GetClassDescriptor (in_value);
    }
    return objc_class_sp;
}

ModuleSP FindLibobjc (Target &target)
{
    ModuleList& modules = target.GetImages();
    for (uint32_t idx = 0; idx < modules.GetSize(); idx++)
    {
        lldb::ModuleSP module_sp = modules.GetModuleAtIndex(idx);
        if (!module_sp)
            continue;
        if (strncmp(module_sp->GetFileSpec().GetFilename().AsCString(""), "libobjc.", sizeof("libobjc.") - 1) == 0)
            return module_sp;
    }
    
    return ModuleSP();
}

void
AppleObjCRuntimeV2::UpdateISAToDescriptorMap_Impl()
{
    lldb::LogSP log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    Process *process_ptr = GetProcess();
    
    if (!process_ptr)
        return;
    
    ProcessSP process_sp = process_ptr->shared_from_this();
    
    Target &target(process_sp->GetTarget());
    
    ModuleSP objc_module_sp(FindLibobjc(target));
    
    if (!objc_module_sp)
        return;
    
    do
    {
        SymbolContextList sc_list;
    
        size_t num_symbols = objc_module_sp->FindSymbolsWithNameAndType(ConstString("gdb_objc_realized_classes"),
                                                                        lldb::eSymbolTypeData,
                                                                        sc_list);
    
        if (!num_symbols)
            break;
        
        SymbolContext gdb_objc_realized_classes_sc;
        
        if (!sc_list.GetContextAtIndex(0, gdb_objc_realized_classes_sc))
             break;
        
        AddressRange gdb_objc_realized_classes_addr_range;
        
        const uint32_t scope = eSymbolContextSymbol;
        const uint32_t range_idx = 0;
        bool use_inline_block_range = false;

        if (!gdb_objc_realized_classes_sc.GetAddressRange(scope,
                                                          range_idx,
                                                          use_inline_block_range,
                                                          gdb_objc_realized_classes_addr_range))
            break;
        
        lldb::addr_t gdb_objc_realized_classes_ptr = gdb_objc_realized_classes_addr_range.GetBaseAddress().GetLoadAddress(&target);
        
        if (gdb_objc_realized_classes_ptr == LLDB_INVALID_ADDRESS)
            break;
    
        // <rdar://problem/10763513>
        
        lldb::addr_t gdb_objc_realized_classes_nxmaptable_ptr;
        
        {
            Error err;
            gdb_objc_realized_classes_nxmaptable_ptr = process_sp->ReadPointerFromMemory(gdb_objc_realized_classes_ptr, err);
            if (!err.Success())
                break;
        }
        
        RemoteNXMapTable gdb_objc_realized_classes(process_sp, gdb_objc_realized_classes_nxmaptable_ptr);
    
        for (RemoteNXMapTable::element elt : gdb_objc_realized_classes)
        {
            if (m_isa_to_descriptor_cache.count(elt.second))
                continue;
            
            ClassDescriptorSP descriptor_sp = ClassDescriptorSP(new ClassDescriptorV2(*this, elt.second));
            
            if (log && log->GetVerbose())
                log->Printf("AppleObjCRuntimeV2 added (ObjCISA)0x%llx (%s) from dynamic table to isa->descriptor cache", elt.second, elt.first.AsCString());
            
            m_isa_to_descriptor_cache[elt.second] = descriptor_sp;
        }
    }
    while(0);
    
    do
    {
        ObjectFile *objc_object = objc_module_sp->GetObjectFile();
        
        if (!objc_object)
            break;
        
        SectionList *section_list = objc_object->GetSectionList();
        
        if (!section_list)
            break;
        
        SectionSP TEXT_section_sp = section_list->FindSectionByName(ConstString("__TEXT"));
        
        if (!TEXT_section_sp)
            break;
        
        SectionList &TEXT_children = TEXT_section_sp->GetChildren();
        
        SectionSP objc_opt_section_sp = TEXT_children.FindSectionByName(ConstString("__objc_opt_ro"));
        
        if (!objc_opt_section_sp)
            break;
        
        lldb::addr_t objc_opt_ptr = objc_opt_section_sp->GetLoadBaseAddress(&target);
        
        if (objc_opt_ptr == LLDB_INVALID_ADDRESS)
            break;
        
        RemoteObjCOpt objc_opt(process_sp, objc_opt_ptr);
        
        for (ObjCLanguageRuntime::ObjCISA objc_isa : objc_opt)
        {
            if (m_isa_to_descriptor_cache.count(objc_isa))
                continue;
            
            ClassDescriptorSP descriptor_sp = ClassDescriptorSP(new ClassDescriptorV2(*this, objc_isa));
            
            if (log && log->GetVerbose())
                log->Printf("AppleObjCRuntimeV2 added (ObjCISA)0x%llx (%s) from static table to isa->descriptor cache", objc_isa, descriptor_sp->GetClassName().AsCString());
            
            m_isa_to_descriptor_cache[objc_isa] = descriptor_sp;
        }
    }
    while (0);
}

// this code relies on the assumption that an Objective-C object always starts
// with an ISA at offset 0. an ISA is effectively a pointer to an instance of
// struct class_t in the ObjCv2 runtime
ObjCLanguageRuntime::ObjCISA
AppleObjCRuntimeV2::GetISA(ValueObject& valobj)
{
//    if (ClangASTType::GetMinimumLanguage(valobj.GetClangAST(),valobj.GetClangType()) != eLanguageTypeObjC)
//        return 0;
//    
    // if we get an invalid VO (which might still happen when playing around
    // with pointers returned by the expression parser, don't consider this
    // a valid ObjC object)
    if (valobj.GetValue().GetContextType() == Value::eContextTypeInvalid)
        return 0;
    
    addr_t isa_pointer = valobj.GetPointerValue();
    
    // tagged pointer
    if (IsTaggedPointer(isa_pointer))
    {
        ClassDescriptorV2Tagged descriptor(valobj);
        
        // probably an invalid tagged pointer - say it's wrong
        if (!descriptor.IsValid())
            return 0;
        
        static const ConstString g_objc_tagged_isa_nsatom_name ("NSAtom");
        static const ConstString g_objc_tagged_isa_nsnumber_name ("NSNumber");
        static const ConstString g_objc_tagged_isa_nsdatets_name ("NSDateTS");
        static const ConstString g_objc_tagged_isa_nsmanagedobject_name ("NSManagedObject");
        static const ConstString g_objc_tagged_isa_nsdate_name ("NSDate");
        
        ConstString class_name_const_string = descriptor.GetClassName();

        if (class_name_const_string == g_objc_tagged_isa_nsatom_name)
            return g_objc_Tagged_ISA_NSAtom;
        if (class_name_const_string == g_objc_tagged_isa_nsnumber_name)
            return g_objc_Tagged_ISA_NSNumber;
        if (class_name_const_string == g_objc_tagged_isa_nsdatets_name)
            return g_objc_Tagged_ISA_NSDateTS;
        if (class_name_const_string == g_objc_tagged_isa_nsmanagedobject_name)
            return g_objc_Tagged_ISA_NSManagedObject;
        if (class_name_const_string == g_objc_tagged_isa_nsdate_name)
            return g_objc_Tagged_ISA_NSDate;
        return g_objc_Tagged_ISA;
    }

    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());

    Process *process = exe_ctx.GetProcessPtr();
    if (process)
    {
        Error error;
        ObjCISA isa = process->ReadPointerFromMemory(isa_pointer, error);
        if (isa != LLDB_INVALID_ADDRESS)
            return isa;
    }
    return 0; // For some reason zero is being used to indicate invalid ISA instead of LLDB_INVALID_ADDRESS
}

// TODO: should we have a transparent_kvo parameter here to say if we
// want to replace the KVO swizzled class with the actual user-level type?
ConstString
AppleObjCRuntimeV2::GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa)
{
    if (!IsValidISA(isa))
        return ConstString();
     
    if (isa == g_objc_Tagged_ISA)
    {
        static const ConstString g_objc_tagged_isa_name ("_lldb_Tagged_ObjC_ISA");
        return g_objc_tagged_isa_name;
    }
    if (isa == g_objc_Tagged_ISA_NSAtom)
    {
        static const ConstString g_objc_tagged_isa_nsatom_name ("NSAtom");
        return g_objc_tagged_isa_nsatom_name;
    }
    if (isa == g_objc_Tagged_ISA_NSNumber)
    {
        static const ConstString g_objc_tagged_isa_nsnumber_name ("NSNumber");
        return g_objc_tagged_isa_nsnumber_name;
    }
    if (isa == g_objc_Tagged_ISA_NSDateTS)
    {
        static const ConstString g_objc_tagged_isa_nsdatets_name ("NSDateTS");
        return g_objc_tagged_isa_nsdatets_name;
    }
    if (isa == g_objc_Tagged_ISA_NSManagedObject)
    {
        static const ConstString g_objc_tagged_isa_nsmanagedobject_name ("NSManagedObject");
        return g_objc_tagged_isa_nsmanagedobject_name;
    }
    if (isa == g_objc_Tagged_ISA_NSDate)
    {
        static const ConstString g_objc_tagged_isa_nsdate_name ("NSDate");
        return g_objc_tagged_isa_nsdate_name;
    }
    
    return ObjCLanguageRuntime::GetActualTypeName(isa);
}

TypeVendor *
AppleObjCRuntimeV2::GetTypeVendor()
{
    if (!m_type_vendor_ap.get())
        m_type_vendor_ap.reset(new AppleObjCTypeVendor(*this));
    
    return m_type_vendor_ap.get();
}
