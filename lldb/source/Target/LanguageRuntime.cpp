//===-- LanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;


class ExceptionSearchFilter : public SearchFilter
{
public:
    ExceptionSearchFilter (const lldb::TargetSP &target_sp,
                           lldb::LanguageType language,
                           bool update_module_list = true) :
        SearchFilter (target_sp),
        m_language (language),
        m_language_runtime (NULL),
        m_filter_sp ()
    {
        if (update_module_list)
            UpdateModuleListIfNeeded ();
    }

    virtual
    ~ExceptionSearchFilter() {};

    bool
    ModulePasses (const lldb::ModuleSP &module_sp) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            return m_filter_sp->ModulePasses (module_sp);
        return false;
    }
    
    bool
    ModulePasses (const FileSpec &spec) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            return m_filter_sp->ModulePasses (spec);
        return false;
        
    }
    
    void
    Search (Searcher &searcher) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            m_filter_sp->Search (searcher);
    }

    void
    GetDescription (Stream *s) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            m_filter_sp->GetDescription (s);
    }
    
protected:
    LanguageType m_language;
    LanguageRuntime *m_language_runtime;
    SearchFilterSP m_filter_sp;

    SearchFilterSP
    DoCopyForBreakpoint(Breakpoint &breakpoint) override
    {
        return SearchFilterSP(new ExceptionSearchFilter(TargetSP(), m_language, false));
    }

    void
    UpdateModuleListIfNeeded ()
    {
        ProcessSP process_sp (m_target_sp->GetProcessSP());
        if (process_sp)
        {
            bool refreash_filter = !m_filter_sp;
            if (m_language_runtime == NULL)
            {
                m_language_runtime = process_sp->GetLanguageRuntime(m_language);
                refreash_filter = true;
            }
            else
            {
                LanguageRuntime *language_runtime = process_sp->GetLanguageRuntime(m_language);
                if (m_language_runtime != language_runtime)
                {
                    m_language_runtime = language_runtime;
                    refreash_filter = true;
                }
            }
            
            if (refreash_filter && m_language_runtime)
            {
                m_filter_sp = m_language_runtime->CreateExceptionSearchFilter ();
            }
        }
        else
        {
            m_filter_sp.reset();
            m_language_runtime = NULL;
        }
    }
};

// The Target is the one that knows how to create breakpoints, so this function
// is meant to be used either by the target or internally in Set/ClearExceptionBreakpoints.
class ExceptionBreakpointResolver : public BreakpointResolver
{
public:
    ExceptionBreakpointResolver (lldb::LanguageType language,
                                 bool catch_bp,
                                 bool throw_bp) :
        BreakpointResolver (NULL, BreakpointResolver::ExceptionResolver),
        m_language (language),
        m_language_runtime (NULL),
        m_catch_bp (catch_bp),
        m_throw_bp (throw_bp)
    {
    }

    virtual
    ~ExceptionBreakpointResolver()
    {
    }
    
    Searcher::CallbackReturn
    SearchCallback (SearchFilter &filter,
                    SymbolContext &context,
                    Address *addr,
                    bool containing) override
    {
        
        if (SetActualResolver())
            return m_actual_resolver_sp->SearchCallback (filter, context, addr, containing);
        else
            return eCallbackReturnStop;
    }
    
    Searcher::Depth
    GetDepth () override
    {
        if (SetActualResolver())
            return m_actual_resolver_sp->GetDepth();
        else
            return eDepthTarget;
    }
    
    void
    GetDescription (Stream *s) override
    {
        s->Printf ("Exception breakpoint (catch: %s throw: %s)",
                   m_catch_bp ? "on" : "off",
                   m_throw_bp ? "on" : "off");
        
        SetActualResolver();
        if (m_actual_resolver_sp)
        {
            s->Printf (" using: ");
            m_actual_resolver_sp->GetDescription (s);
        }
        else
            s->Printf (" the correct runtime exception handler will be determined when you run");
    }

    void
    Dump (Stream *s) const override
    {
    }
    
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const BreakpointResolverName *) { return true; }
    static inline bool classof(const BreakpointResolver *V) {
        return V->getResolverID() == BreakpointResolver::ExceptionResolver;
    }
protected:
    BreakpointResolverSP
    CopyForBreakpoint (Breakpoint &breakpoint) override
    {
        return BreakpointResolverSP(new ExceptionBreakpointResolver(m_language, m_catch_bp, m_throw_bp));
    }

    bool
    SetActualResolver()
    {
        ProcessSP process_sp;
        if (m_breakpoint)
        {
            process_sp = m_breakpoint->GetTarget().GetProcessSP();
            if (process_sp)
            {
                bool refreash_resolver = !m_actual_resolver_sp;
                if (m_language_runtime == NULL)
                {
                    m_language_runtime = process_sp->GetLanguageRuntime(m_language);
                    refreash_resolver = true;
                }
                else
                {
                    LanguageRuntime *language_runtime = process_sp->GetLanguageRuntime(m_language);
                    if (m_language_runtime != language_runtime)
                    {
                        m_language_runtime = language_runtime;
                        refreash_resolver = true;
                    }
                }
                
                if (refreash_resolver && m_language_runtime)
                {
                    m_actual_resolver_sp = m_language_runtime->CreateExceptionResolver (m_breakpoint, m_catch_bp, m_throw_bp);
                }
            }
            else
            {
                m_actual_resolver_sp.reset();
                m_language_runtime = NULL;
            }
        }
        else
        {
            m_actual_resolver_sp.reset();
            m_language_runtime = NULL;
        }
        return (bool)m_actual_resolver_sp;
    }
    lldb::BreakpointResolverSP m_actual_resolver_sp;
    lldb::LanguageType m_language;
    LanguageRuntime *m_language_runtime;
    bool m_catch_bp;
    bool m_throw_bp;
};


LanguageRuntime*
LanguageRuntime::FindPlugin (Process *process, lldb::LanguageType language)
{
    std::unique_ptr<LanguageRuntime> language_runtime_ap;
    LanguageRuntimeCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        language_runtime_ap.reset (create_callback(process, language));

        if (language_runtime_ap.get())
            return language_runtime_ap.release();
    }

    return NULL;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
LanguageRuntime::LanguageRuntime(Process *process) :
    m_process (process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LanguageRuntime::~LanguageRuntime()
{
}

Breakpoint::BreakpointPreconditionSP
LanguageRuntime::CreateExceptionPrecondition (lldb::LanguageType language,
                                              bool catch_bp,
                                              bool throw_bp)
{
    switch (language)
    {
    case eLanguageTypeObjC:
        if (throw_bp)
            return Breakpoint::BreakpointPreconditionSP(new ObjCLanguageRuntime::ObjCExceptionPrecondition ());
        break;
    default:
        break;
    }
    return Breakpoint::BreakpointPreconditionSP();
}

BreakpointSP
LanguageRuntime::CreateExceptionBreakpoint (Target &target,
                                            lldb::LanguageType language,
                                            bool catch_bp,
                                            bool throw_bp,
                                            bool is_internal)
{
    BreakpointResolverSP resolver_sp(new ExceptionBreakpointResolver(language, catch_bp, throw_bp));
    SearchFilterSP filter_sp(new ExceptionSearchFilter(target.shared_from_this(), language));
    bool hardware = false;
    bool resolve_indirect_functions = false;
    BreakpointSP exc_breakpt_sp (target.CreateBreakpoint (filter_sp, resolver_sp, is_internal, hardware, resolve_indirect_functions));
    if (exc_breakpt_sp)
    {
        Breakpoint::BreakpointPreconditionSP precondition_sp = CreateExceptionPrecondition(language, catch_bp, throw_bp);
        if (precondition_sp)
            exc_breakpt_sp->SetPrecondition(precondition_sp);

        if (is_internal)
            exc_breakpt_sp->SetBreakpointKind("exception");
    }
    
    return exc_breakpt_sp;
}

struct language_name_pair {
    const char *name;
    LanguageType type;
};

struct language_name_pair language_names[] =
{
    // To allow GetNameForLanguageType to be a simple array lookup, the first
    // part of this array must follow enum LanguageType exactly.
    {   "unknown",          eLanguageTypeUnknown        },
    {   "c89",              eLanguageTypeC89            },
    {   "c",                eLanguageTypeC              },
    {   "ada83",            eLanguageTypeAda83          },
    {   "c++",              eLanguageTypeC_plus_plus    },
    {   "cobol74",          eLanguageTypeCobol74        },
    {   "cobol85",          eLanguageTypeCobol85        },
    {   "fortran77",        eLanguageTypeFortran77      },
    {   "fortran90",        eLanguageTypeFortran90      },
    {   "pascal83",         eLanguageTypePascal83       },
    {   "modula2",          eLanguageTypeModula2        },
    {   "java",             eLanguageTypeJava           },
    {   "c99",              eLanguageTypeC99            },
    {   "ada95",            eLanguageTypeAda95          },
    {   "fortran95",        eLanguageTypeFortran95      },
    {   "pli",              eLanguageTypePLI            },
    {   "objective-c",      eLanguageTypeObjC           },
    {   "objective-c++",    eLanguageTypeObjC_plus_plus },
    {   "upc",              eLanguageTypeUPC            },
    {   "d",                eLanguageTypeD              },
    {   "python",           eLanguageTypePython         },
    {   "opencl",           eLanguageTypeOpenCL         },
    {   "go",               eLanguageTypeGo             },
    {   "modula3",          eLanguageTypeModula3        },
    {   "haskell",          eLanguageTypeHaskell        },
    {   "c++03",            eLanguageTypeC_plus_plus_03 },
    {   "c++11",            eLanguageTypeC_plus_plus_11 },
    {   "ocaml",            eLanguageTypeOCaml          },
    {   "rust",             eLanguageTypeRust           },
    {   "c11",              eLanguageTypeC11            },
    {   "swift",            eLanguageTypeSwift          },
    {   "julia",            eLanguageTypeJulia          },
    {   "dylan",            eLanguageTypeDylan          },
    {   "c++14",            eLanguageTypeC_plus_plus_14 },
    {   "fortran03",        eLanguageTypeFortran03      },
    {   "fortran08",        eLanguageTypeFortran08      },
    // Vendor Extensions
    {   "mipsassem",        eLanguageTypeMipsAssembler  },
    {   "renderscript",     eLanguageTypeExtRenderScript},
    // Now synonyms, in arbitrary order
    {   "objc",             eLanguageTypeObjC           },
    {   "objc++",           eLanguageTypeObjC_plus_plus }
};

static uint32_t num_languages = sizeof(language_names) / sizeof (struct language_name_pair);

LanguageType
LanguageRuntime::GetLanguageTypeFromString (const char *string)
{
    for (uint32_t i = 0; i < num_languages; i++)
    {
        if (strcasecmp (language_names[i].name, string) == 0)
            return (LanguageType) language_names[i].type;
    }
    return eLanguageTypeUnknown;
}

const char *
LanguageRuntime::GetNameForLanguageType (LanguageType language)
{
    if (language < num_languages)
        return language_names[language].name;
    else
        return language_names[eLanguageTypeUnknown].name;
}

void
LanguageRuntime::PrintAllLanguages (Stream &s, const char *prefix, const char *suffix)
{
    for (uint32_t i = 1; i < num_languages; i++)
    {
        s.Printf("%s%s%s", prefix, language_names[i].name, suffix);
    }
}

bool
LanguageRuntime::LanguageIsCPlusPlus (LanguageType language)
{
    switch (language)
    {
        case eLanguageTypeC_plus_plus:
        case eLanguageTypeC_plus_plus_03:
        case eLanguageTypeC_plus_plus_11:
        case eLanguageTypeC_plus_plus_14:
            return true;
        default:
            return false;
    }
}

void
LanguageRuntime::InitializeCommands (CommandObject* parent)
{
    if (!parent)
        return;

    if (!parent->IsMultiwordObject())
        return;

    LanguageRuntimeCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(idx)) != nullptr;
         ++idx)
    {
        if (LanguageRuntimeGetCommandObject command_callback = 
                PluginManager::GetLanguageRuntimeGetCommandObjectAtIndex(idx))
        {
            CommandObjectSP command = command_callback(parent->GetCommandInterpreter());
            if (command)
            {
                parent->LoadSubCommand(command->GetCommandName(), command);
            }
        }
    }
}

lldb::SearchFilterSP
LanguageRuntime::CreateExceptionSearchFilter ()
{
    return m_process->GetTarget().GetSearchFilterForModule(NULL);
}



