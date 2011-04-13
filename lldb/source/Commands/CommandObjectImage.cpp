//===-- CommandObjectImage.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectImage.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Static Helper functions
//----------------------------------------------------------------------
static void
DumpModuleArchitecture (Stream &strm, Module *module, uint32_t width)
{
    if (module)
    {
        if (width)
            strm.Printf("%-*s", width, module->GetArchitecture().GetArchitectureName());
        else
            strm.PutCString(module->GetArchitecture().GetArchitectureName());
    }
}

static void
DumpModuleUUID (Stream &strm, Module *module)
{
    module->GetUUID().Dump (&strm);
}

static uint32_t
DumpCompileUnitLineTable
(
    CommandInterpreter &interpreter,
    Stream &strm,
    Module *module,
    const FileSpec &file_spec,
    bool load_addresses
)
{
    uint32_t num_matches = 0;
    if (module)
    {
        SymbolContextList sc_list;
        num_matches = module->ResolveSymbolContextsForFileSpec (file_spec,
                                                                0,
                                                                false,
                                                                eSymbolContextCompUnit,
                                                                sc_list);
        
        for (uint32_t i=0; i<num_matches; ++i)
        {
            SymbolContext sc;
            if (sc_list.GetContextAtIndex(i, sc))
            {
                if (i > 0)
                    strm << "\n\n";

                strm << "Line table for " << *static_cast<FileSpec*> (sc.comp_unit) << " in `"
                     << module->GetFileSpec().GetFilename() << "\n";
                LineTable *line_table = sc.comp_unit->GetLineTable();
                if (line_table)
                    line_table->GetDescription (&strm, 
                                                interpreter.GetExecutionContext().target, 
                                                lldb::eDescriptionLevelBrief);
                else
                    strm << "No line table";
            }
        }
    }
    return num_matches;
}

static void
DumpFullpath (Stream &strm, const FileSpec *file_spec_ptr, uint32_t width)
{
    if (file_spec_ptr)
    {
        if (width > 0)
        {
            char fullpath[PATH_MAX];
            if (file_spec_ptr->GetPath(fullpath, sizeof(fullpath)))
            {
                strm.Printf("%-*s", width, fullpath);
                return;
            }
        }
        else
        {
            file_spec_ptr->Dump(&strm);
            return;
        }
    }
    // Keep the width spacing correct if things go wrong...
    if (width > 0)
        strm.Printf("%-*s", width, "");
}

static void
DumpDirectory (Stream &strm, const FileSpec *file_spec_ptr, uint32_t width)
{
    if (file_spec_ptr)
    {
        if (width > 0)
            strm.Printf("%-*s", width, file_spec_ptr->GetDirectory().AsCString(""));
        else
            file_spec_ptr->GetDirectory().Dump(&strm);
        return;
    }
    // Keep the width spacing correct if things go wrong...
    if (width > 0)
        strm.Printf("%-*s", width, "");
}

static void
DumpBasename (Stream &strm, const FileSpec *file_spec_ptr, uint32_t width)
{
    if (file_spec_ptr)
    {
        if (width > 0)
            strm.Printf("%-*s", width, file_spec_ptr->GetFilename().AsCString(""));
        else
            file_spec_ptr->GetFilename().Dump(&strm);
        return;
    }
    // Keep the width spacing correct if things go wrong...
    if (width > 0)
        strm.Printf("%-*s", width, "");
}


static void
DumpModuleSymtab (CommandInterpreter &interpreter, Stream &strm, Module *module, SortOrder sort_order)
{
    if (module)
    {
        ObjectFile *objfile = module->GetObjectFile ();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
                symtab->Dump(&strm, interpreter.GetExecutionContext().target, sort_order);
        }
    }
}

static void
DumpModuleSections (CommandInterpreter &interpreter, Stream &strm, Module *module)
{
    if (module)
    {
        ObjectFile *objfile = module->GetObjectFile ();
        if (objfile)
        {
            SectionList *section_list = objfile->GetSectionList();
            if (section_list)
            {
                strm.PutCString ("Sections for '");
                strm << module->GetFileSpec();
                if (module->GetObjectName())
                    strm << '(' << module->GetObjectName() << ')';
                strm.Printf ("' (%s):\n", module->GetArchitecture().GetArchitectureName());
                strm.IndentMore();
                section_list->Dump(&strm, interpreter.GetExecutionContext().target, true, UINT32_MAX);
                strm.IndentLess();
            }
        }
    }
}

static bool
DumpModuleSymbolVendor (Stream &strm, Module *module)
{
    if (module)
    {
        SymbolVendor *symbol_vendor = module->GetSymbolVendor(true);
        if (symbol_vendor)
        {
            symbol_vendor->Dump(&strm);
            return true;
        }
    }
    return false;
}

static bool
LookupAddressInModule 
(
    CommandInterpreter &interpreter, 
    Stream &strm, 
    Module *module, 
    uint32_t resolve_mask, 
    lldb::addr_t raw_addr, 
    lldb::addr_t offset,
    bool verbose
)
{
    if (module)
    {
        lldb::addr_t addr = raw_addr - offset;
        Address so_addr;
        SymbolContext sc;
        Target *target = interpreter.GetExecutionContext().target;
        if (target && !target->GetSectionLoadList().IsEmpty())
        {
            if (!target->GetSectionLoadList().ResolveLoadAddress (addr, so_addr))
                return false;
            else if (so_addr.GetModule() != module)
                return false;
        }
        else
        {
            if (!module->ResolveFileAddress (addr, so_addr))
                return false;
        }

        // If an offset was given, print out the address we ended up looking up
        if (offset)
            strm.Printf("File Address: 0x%llx\n", addr);

        ExecutionContextScope *exe_scope = interpreter.GetExecutionContext().GetBestExecutionContextScope();
        strm.IndentMore();
        strm.Indent ("    Address: ");
        so_addr.Dump (&strm, exe_scope, Address::DumpStyleSectionNameOffset);
        strm.EOL();
        strm.Indent ("    Summary: ");
        const uint32_t save_indent = strm.GetIndentLevel ();
        strm.SetIndentLevel (save_indent + 11);
        so_addr.Dump (&strm, exe_scope, Address::DumpStyleResolvedDescription);
        strm.SetIndentLevel (save_indent);
        strm.EOL();
        // Print out detailed address information when verbose is enabled
        if (verbose)
        {
            if (so_addr.Dump (&strm, exe_scope, Address::DumpStyleDetailedSymbolContext))
                strm.EOL();
        }
        strm.IndentLess();
        return true;
    }

    return false;
}

static uint32_t
LookupSymbolInModule (CommandInterpreter &interpreter, Stream &strm, Module *module, const char *name, bool name_is_regex)
{
    if (module)
    {
        SymbolContext sc;

        ObjectFile *objfile = module->GetObjectFile ();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
            {
                uint32_t i;
                std::vector<uint32_t> match_indexes;
                ConstString symbol_name (name);
                uint32_t num_matches = 0;
                if (name_is_regex)
                {
                    RegularExpression name_regexp(name);
                    num_matches = symtab->AppendSymbolIndexesMatchingRegExAndType (name_regexp, 
                                                                                   eSymbolTypeAny,
                                                                                   match_indexes);
                }
                else
                {
                    num_matches = symtab->AppendSymbolIndexesWithName (symbol_name, match_indexes);
                }


                if (num_matches > 0)
                {
                    strm.Indent ();
                    strm.Printf("%u symbols match %s'%s' in ", num_matches,
                                name_is_regex ? "the regular expression " : "", name);
                    DumpFullpath (strm, &module->GetFileSpec(), 0);
                    strm.PutCString(":\n");
                    strm.IndentMore ();
                    Symtab::DumpSymbolHeader (&strm);
                    for (i=0; i < num_matches; ++i)
                    {
                        Symbol *symbol = symtab->SymbolAtIndex(match_indexes[i]);
                        strm.Indent ();
                        symbol->Dump (&strm, interpreter.GetExecutionContext().target, i);
                    }
                    strm.IndentLess ();
                    return num_matches;
                }
            }
        }
    }
    return 0;
}


static void
DumpSymbolContextList (CommandInterpreter &interpreter, Stream &strm, SymbolContextList &sc_list, bool prepend_addr)
{
    strm.IndentMore ();
    uint32_t i;
    const uint32_t num_matches = sc_list.GetSize();

    for (i=0; i<num_matches; ++i)
    {
        SymbolContext sc;
        if (sc_list.GetContextAtIndex(i, sc))
        {
            strm.Indent();
            if (prepend_addr)
            {
                if (sc.line_entry.range.GetBaseAddress().IsValid())
                {
                    lldb::addr_t vm_addr = sc.line_entry.range.GetBaseAddress().GetLoadAddress(interpreter.GetExecutionContext().target);
                    int addr_size = sizeof (addr_t);
                    Process *process = interpreter.GetExecutionContext().process;
                    if (process)
                        addr_size = process->GetTarget().GetArchitecture().GetAddressByteSize();
                    if (vm_addr != LLDB_INVALID_ADDRESS)
                        strm.Address (vm_addr, addr_size);
                    else
                        sc.line_entry.range.GetBaseAddress().Dump (&strm, NULL, Address::DumpStyleSectionNameOffset);

                    strm.PutCString(" in ");
                }
            }
            sc.DumpStopContext(&strm, interpreter.GetExecutionContext().process, sc.line_entry.range.GetBaseAddress(), true, true, false);
        }
    }
    strm.IndentLess ();
}

static uint32_t
LookupFunctionInModule (CommandInterpreter &interpreter, Stream &strm, Module *module, const char *name, bool name_is_regex)
{
    if (module && name && name[0])
    {
        SymbolContextList sc_list;
        const bool include_symbols = false;
        const bool append = true;
        uint32_t num_matches = 0;
        if (name_is_regex)
        {
            RegularExpression function_name_regex (name);
            num_matches = module->FindFunctions (function_name_regex, 
                                                 include_symbols,
                                                 append, 
                                                 sc_list);
        }
        else
        {
            ConstString function_name (name);
            num_matches = module->FindFunctions (function_name, 
                                                 eFunctionNameTypeBase | eFunctionNameTypeFull | eFunctionNameTypeMethod | eFunctionNameTypeSelector, 
                                                 include_symbols,
                                                 append, 
                                                 sc_list);
        }

        if (num_matches)
        {
            strm.Indent ();
            strm.Printf("%u match%s found in ", num_matches, num_matches > 1 ? "es" : "");
            DumpFullpath (strm, &module->GetFileSpec(), 0);
            strm.PutCString(":\n");
            DumpSymbolContextList (interpreter, strm, sc_list, true);
        }
        return num_matches;
    }
    return 0;
}

static uint32_t
LookupTypeInModule 
(
    CommandInterpreter &interpreter, 
    Stream &strm, 
    Module *module, 
    const char *name_cstr, 
    bool name_is_regex
)
{
    if (module && name_cstr && name_cstr[0])
    {
        SymbolContextList sc_list;

        SymbolVendor *symbol_vendor = module->GetSymbolVendor();
        if (symbol_vendor)
        {
            TypeList type_list;
            uint32_t num_matches = 0;
            SymbolContext sc;
//            if (name_is_regex)
//            {
//                RegularExpression name_regex (name_cstr);
//                num_matches = symbol_vendor->FindFunctions(sc, name_regex, true, UINT32_MAX, type_list);
//            }
//            else
//            {
                ConstString name(name_cstr);
                num_matches = symbol_vendor->FindTypes(sc, name, true, UINT32_MAX, type_list);
//            }

            if (num_matches)
            {
                strm.Indent ();
                strm.Printf("%u match%s found in ", num_matches, num_matches > 1 ? "es" : "");
                DumpFullpath (strm, &module->GetFileSpec(), 0);
                strm.PutCString(":\n");
                const uint32_t num_types = type_list.GetSize();
                for (uint32_t i=0; i<num_types; ++i)
                {
                    TypeSP type_sp (type_list.GetTypeAtIndex(i));
                    if (type_sp)
                    {
                        // Resolve the clang type so that any forward references
                        // to types that haven't yet been parsed will get parsed.
                        type_sp->GetClangFullType ();
                        type_sp->GetDescription (&strm, eDescriptionLevelFull, true);
                    }
                    strm.EOL();
                }
            }
            return num_matches;
        }
    }
    return 0;
}

static uint32_t
LookupFileAndLineInModule (CommandInterpreter &interpreter, Stream &strm, Module *module, const FileSpec &file_spec, uint32_t line, bool check_inlines)
{
    if (module && file_spec)
    {
        SymbolContextList sc_list;
        const uint32_t num_matches = module->ResolveSymbolContextsForFileSpec(file_spec, line, check_inlines,
                                                                             eSymbolContextEverything, sc_list);
        if (num_matches > 0)
        {
            strm.Indent ();
            strm.Printf("%u match%s found in ", num_matches, num_matches > 1 ? "es" : "");
            strm << file_spec;
            if (line > 0)
                strm.Printf (":%u", line);
            strm << " in ";
            DumpFullpath (strm, &module->GetFileSpec(), 0);
            strm.PutCString(":\n");
            DumpSymbolContextList (interpreter, strm, sc_list, true);
            return num_matches;
        }
    }
    return 0;

}


//----------------------------------------------------------------------
// Image symbol table dumping command
//----------------------------------------------------------------------

class CommandObjectImageDumpModuleList : public CommandObject
{
public:

    CommandObjectImageDumpModuleList (CommandInterpreter &interpreter,
                                      const char *name,
                                      const char *help,
                                      const char *syntax) :
        CommandObject (interpreter, name, help, syntax)
    {
        CommandArgumentEntry arg;
        CommandArgumentData file_arg;

        // Define the first (and only) variant of this arg.
        file_arg.arg_type = eArgTypeFilename;
        file_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectImageDumpModuleList ()
    {
    }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        // Arguments are the standard module completer.
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);

        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eModuleCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }
};

class CommandObjectImageDumpSourceFileList : public CommandObject
{
public:

    CommandObjectImageDumpSourceFileList (CommandInterpreter &interpreter,
                                          const char *name,
                                          const char *help,
                                          const char *syntax) :
        CommandObject (interpreter, name, help, syntax)
    {
        CommandArgumentEntry arg;
        CommandArgumentData source_file_arg;
        
        // Define the first (and only) variant of this arg.
        source_file_arg.arg_type = eArgTypeSourceFile;
        source_file_arg.arg_repetition = eArgRepeatPlus;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (source_file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectImageDumpSourceFileList ()
    {
    }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        // Arguments are the standard source file completer.
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);

        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter, 
                                                             CommandCompletions::eSourceFileCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }
};


class CommandObjectImageDumpSymtab : public CommandObjectImageDumpModuleList
{
public:
    CommandObjectImageDumpSymtab (CommandInterpreter &interpreter) :
        CommandObjectImageDumpModuleList (interpreter,
                                          "image dump symtab",
                                          "Dump the symbol table from one or more executable images.",
                                           NULL),
        m_options (interpreter)
    {
    }

    virtual
    ~CommandObjectImageDumpSymtab ()
    {
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t num_dumped = 0;

            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);

            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    result.GetOutputStream().Printf("Dumping symbol table for %u modules.\n", num_modules);
                    for (uint32_t image_idx = 0;  image_idx<num_modules; ++image_idx)
                    {
                        if (num_dumped > 0)
                        {
                            result.GetOutputStream().EOL();
                            result.GetOutputStream().EOL();
                        }
                        num_dumped++;
                        DumpModuleSymtab (m_interpreter, result.GetOutputStream(), target->GetImages().GetModulePointerAtIndex(image_idx), m_options.m_sort_order);
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);

                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t i=0; i<num_matching_modules; ++i)
                        {
                            Module *image_module = matching_modules.GetModulePointerAtIndex(i);
                            if (image_module)
                            {
                                if (num_dumped > 0)
                                {
                                    result.GetOutputStream().EOL();
                                    result.GetOutputStream().EOL();
                                }
                                num_dumped++;
                                DumpModuleSymtab (m_interpreter, result.GetOutputStream(), image_module, m_options.m_sort_order);
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }

            if (num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no matching executable images found");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter),
            m_sort_order (eSortOrderNone)
        {
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 's':
                {
                    bool found_one = false;
                    m_sort_order = (SortOrder) Args::StringToOptionEnum (option_arg, 
                                                                               g_option_table[option_idx].enum_values, 
                                                                               eSortOrderNone,
                                                                               &found_one);
                    if (!found_one)
                        error.SetErrorStringWithFormat("Invalid enumeration value '%s' for option '%c'.\n", 
                                                       option_arg, 
                                                       short_option);
                }
                break;

            default:
                error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                break;

            }
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_sort_order = eSortOrderNone;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.
        static OptionDefinition g_option_table[];

        SortOrder m_sort_order;
    };

protected:

    CommandOptions m_options;
};

static OptionEnumValueElement
g_sort_option_enumeration[4] =
{
    { eSortOrderNone,       "none",     "No sorting, use the original symbol table order."},
    { eSortOrderByAddress,  "address",  "Sort output by symbol address."},
    { eSortOrderByName,     "name",     "Sort output by symbol name."},
    { 0,                    NULL,       NULL }
};


OptionDefinition
CommandObjectImageDumpSymtab::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "sort", 's', required_argument, g_sort_option_enumeration, 0, eArgTypeSortOrder, "Supply a sort order when dumping the symbol table."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//----------------------------------------------------------------------
// Image section dumping command
//----------------------------------------------------------------------
class CommandObjectImageDumpSections : public CommandObjectImageDumpModuleList
{
public:
    CommandObjectImageDumpSections (CommandInterpreter &interpreter) :
        CommandObjectImageDumpModuleList (interpreter,
                                          "image dump sections",
                                          "Dump the sections from one or more executable images.",
                                          //"image dump sections [<file1> ...]")
                                          NULL)
    {
    }

    virtual
    ~CommandObjectImageDumpSections ()
    {
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t num_dumped = 0;

            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);

            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    result.GetOutputStream().Printf("Dumping sections for %u modules.\n", num_modules);
                    for (uint32_t image_idx = 0;  image_idx<num_modules; ++image_idx)
                    {
                        num_dumped++;
                        DumpModuleSections (m_interpreter, result.GetOutputStream(), target->GetImages().GetModulePointerAtIndex(image_idx));
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);

                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t i=0; i<num_matching_modules; ++i)
                        {
                            Module * image_module = matching_modules.GetModulePointerAtIndex(i);
                            if (image_module)
                            {
                                num_dumped++;
                                DumpModuleSections (m_interpreter, result.GetOutputStream(), image_module);
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }

            if (num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no matching executable images found");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// Image debug symbol dumping command
//----------------------------------------------------------------------
class CommandObjectImageDumpSymfile : public CommandObjectImageDumpModuleList
{
public:
    CommandObjectImageDumpSymfile (CommandInterpreter &interpreter) :
        CommandObjectImageDumpModuleList (interpreter,
                                          "image dump symfile",
                                          "Dump the debug symbol file for one or more executable images.",
                                          //"image dump symfile [<file1> ...]")
                                          NULL)
    {
    }

    virtual
    ~CommandObjectImageDumpSymfile ()
    {
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t num_dumped = 0;

            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);

            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    result.GetOutputStream().Printf("Dumping debug symbols for %u modules.\n", num_modules);
                    for (uint32_t image_idx = 0;  image_idx<num_modules; ++image_idx)
                    {
                        if (DumpModuleSymbolVendor (result.GetOutputStream(), target->GetImages().GetModulePointerAtIndex(image_idx)))
                            num_dumped++;
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);

                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t i=0; i<num_matching_modules; ++i)
                        {
                            Module * image_module = matching_modules.GetModulePointerAtIndex(i);
                            if (image_module)
                            {
                                if (DumpModuleSymbolVendor (result.GetOutputStream(), image_module))
                                    num_dumped++;
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }

            if (num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no matching executable images found");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// Image debug symbol dumping command
//----------------------------------------------------------------------
class CommandObjectImageDumpLineTable : public CommandObjectImageDumpSourceFileList
{
public:
    CommandObjectImageDumpLineTable (CommandInterpreter &interpreter) :
        CommandObjectImageDumpSourceFileList (interpreter,
                                              "image dump line-table",
                                              "Dump the debug symbol file for one or more executable images.",
                                              NULL)
    {
    }

    virtual
    ~CommandObjectImageDumpLineTable ()
    {
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
            uint32_t total_num_dumped = 0;

            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);

            if (command.GetArgumentCount() == 0)
            {
                result.AppendErrorWithFormat ("\nSyntax: %s\n", m_cmd_syntax.c_str());
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (int arg_idx = 0; (arg_cstr = command.GetArgumentAtIndex(arg_idx)) != NULL; ++arg_idx)
                {
                    FileSpec file_spec(arg_cstr, false);
                    const uint32_t num_modules = target->GetImages().GetSize();
                    if (num_modules > 0)
                    {
                        uint32_t num_dumped = 0;
                        for (uint32_t i = 0; i<num_modules; ++i)
                        {
                            if (DumpCompileUnitLineTable (m_interpreter,
                                                          result.GetOutputStream(),
                                                          target->GetImages().GetModulePointerAtIndex(i),
                                                          file_spec,
                                                          exe_ctx.process != NULL && exe_ctx.process->IsAlive()))
                                num_dumped++;
                        }
                        if (num_dumped == 0)
                            result.AppendWarningWithFormat ("No source filenames matched '%s'.\n", arg_cstr);
                        else
                            total_num_dumped += num_dumped;
                    }
                }
            }

            if (total_num_dumped > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
            {
                result.AppendError ("no source filenames matched any command arguments");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// Dump multi-word command
//----------------------------------------------------------------------
class CommandObjectImageDump : public CommandObjectMultiword
{
public:

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectImageDump(CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter, 
                                "image dump",
                                "A set of commands for dumping information about one or more executable images; 'line-table' expects a source file name",
                                "image dump [symtab|sections|symfile|line-table] [<file1> <file2> ...]")
    {
        LoadSubCommand ("symtab",      CommandObjectSP (new CommandObjectImageDumpSymtab (interpreter)));
        LoadSubCommand ("sections",    CommandObjectSP (new CommandObjectImageDumpSections (interpreter)));
        LoadSubCommand ("symfile",     CommandObjectSP (new CommandObjectImageDumpSymfile (interpreter)));
        LoadSubCommand ("line-table",  CommandObjectSP (new CommandObjectImageDumpLineTable (interpreter)));
    }

    virtual
    ~CommandObjectImageDump()
    {
    }
};

//----------------------------------------------------------------------
// List images with associated information
//----------------------------------------------------------------------
class CommandObjectImageList : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter),
            m_format_array()
        {
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            char short_option = (char) m_getopt_table[option_idx].val;
            uint32_t width = 0;
            if (option_arg)
                width = strtoul (option_arg, NULL, 0);
            m_format_array.push_back(std::make_pair(short_option, width));
            Error error;
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_format_array.clear();
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        typedef std::vector< std::pair<char, uint32_t> > FormatWidthCollection;
        FormatWidthCollection m_format_array;
    };

    CommandObjectImageList (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "image list",
                       "List current executable and dependent shared library images.",
                       "image list [<cmd-options>]"),
        m_options (interpreter)
    {
    }

    virtual
    ~CommandObjectImageList ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            // Dump all sections for all modules images
            const uint32_t num_modules = target->GetImages().GetSize();
            if (num_modules > 0)
            {
                Stream &strm = result.GetOutputStream();

                for (uint32_t image_idx = 0; image_idx<num_modules; ++image_idx)
                {
                    Module *module = target->GetImages().GetModulePointerAtIndex(image_idx);
                    strm.Printf("[%3u] ", image_idx);

                    bool dump_object_name = false;
                    if (m_options.m_format_array.empty())
                    {
                        DumpFullpath(strm, &module->GetFileSpec(), 0);
                        dump_object_name = true;
                    }
                    else
                    {
                        const size_t num_entries = m_options.m_format_array.size();
                        for (size_t i=0; i<num_entries; ++i)
                        {
                            if (i > 0)
                                strm.PutChar(' ');
                            char format_char = m_options.m_format_array[i].first;
                            uint32_t width = m_options.m_format_array[i].second;
                            switch (format_char)
                            {
                            case 'a':
                                DumpModuleArchitecture (strm, module, width);
                                break;

                            case 'f':
                                DumpFullpath (strm, &module->GetFileSpec(), width);
                                dump_object_name = true;
                                break;

                            case 'd':
                                DumpDirectory (strm, &module->GetFileSpec(), width);
                                break;

                            case 'b':
                                DumpBasename (strm, &module->GetFileSpec(), width);
                                dump_object_name = true;
                                break;

                            case 's':
                            case 'S':
                                {
                                    SymbolVendor *symbol_vendor = module->GetSymbolVendor();
                                    if (symbol_vendor)
                                    {
                                        SymbolFile *symbol_file = symbol_vendor->GetSymbolFile();
                                        if (symbol_file)
                                        {
                                            if (format_char == 'S')
                                                DumpBasename(strm, &symbol_file->GetObjectFile()->GetFileSpec(), width);
                                            else
                                                DumpFullpath (strm, &symbol_file->GetObjectFile()->GetFileSpec(), width);
                                            dump_object_name = true;
                                            break;
                                        }
                                    }
                                    strm.Printf("%.*s", width, "<NONE>");
                                }
                                break;

                            case 'u':
                                DumpModuleUUID(strm, module);
                                break;

                            default:
                                break;
                            }
                            
                        }
                    }
                    if (dump_object_name)
                    {
                        const char *object_name = module->GetObjectName().GetCString();
                        if (object_name)
                            strm.Printf ("(%s)", object_name);
                    }
                    strm.EOL();
                }
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendError ("the target has no associated executable images");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        return result.Succeeded();
    }
protected:

    CommandOptions m_options;
};

OptionDefinition
CommandObjectImageList::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "arch",       'a', optional_argument, NULL, 0, eArgTypeWidth,   "Display the architecture when listing images."},
{ LLDB_OPT_SET_1, false, "uuid",       'u', no_argument,       NULL, 0, eArgTypeNone,    "Display the UUID when listing images."},
{ LLDB_OPT_SET_1, false, "fullpath",   'f', optional_argument, NULL, 0, eArgTypeWidth,   "Display the fullpath to the image object file."},
{ LLDB_OPT_SET_1, false, "directory",  'd', optional_argument, NULL, 0, eArgTypeWidth,   "Display the directory with optional width for the image object file."},
{ LLDB_OPT_SET_1, false, "basename",   'b', optional_argument, NULL, 0, eArgTypeWidth,   "Display the basename with optional width for the image object file."},
{ LLDB_OPT_SET_1, false, "symfile",    's', optional_argument, NULL, 0, eArgTypeWidth,   "Display the fullpath to the image symbol file with optional width."},
{ LLDB_OPT_SET_1, false, "symfile-basename", 'S', optional_argument, NULL, 0, eArgTypeWidth,   "Display the basename to the image symbol file with optional width."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};



//----------------------------------------------------------------------
// Lookup information in images
//----------------------------------------------------------------------
class CommandObjectImageLookup : public CommandObject
{
public:

    enum
    {
        eLookupTypeInvalid = -1,
        eLookupTypeAddress = 0,
        eLookupTypeSymbol,
        eLookupTypeFileLine,    // Line is optional
        eLookupTypeFunction,
        eLookupTypeType,
        kNumLookupTypes
    };

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            OptionParsingStarting();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;

            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'a':
                m_type = eLookupTypeAddress;
                m_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS);
                if (m_addr == LLDB_INVALID_ADDRESS)
                    error.SetErrorStringWithFormat ("Invalid address string '%s'.\n", option_arg);
                break;

            case 'o':
                m_offset = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS);
                if (m_offset == LLDB_INVALID_ADDRESS)
                    error.SetErrorStringWithFormat ("Invalid offset string '%s'.\n", option_arg);
                break;

            case 's':
                m_str = option_arg;
                m_type = eLookupTypeSymbol;
                break;

            case 'f':
                m_file.SetFile (option_arg, false);
                m_type = eLookupTypeFileLine;
                break;

            case 'i':
                m_check_inlines = false;
                break;

            case 'l':
                m_line_number = Args::StringToUInt32(option_arg, UINT32_MAX);
                if (m_line_number == UINT32_MAX)
                    error.SetErrorStringWithFormat ("Invalid line number string '%s'.\n", option_arg);
                else if (m_line_number == 0)
                    error.SetErrorString ("Zero is an invalid line number.");
                m_type = eLookupTypeFileLine;
                break;

            case 'n':
                m_str = option_arg;
                m_type = eLookupTypeFunction;
                break;

            case 't':
                m_str = option_arg;
                m_type = eLookupTypeType;
                break;

            case 'v':
                m_verbose = 1;
                break;

            case 'r':
                m_use_regex = true;
                break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_type = eLookupTypeInvalid;
            m_str.clear();
            m_file.Clear();
            m_addr = LLDB_INVALID_ADDRESS;
            m_offset = 0;
            m_line_number = 0;
            m_use_regex = false;
            m_check_inlines = true;
            m_verbose = false;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];
        int             m_type;         // Should be a eLookupTypeXXX enum after parsing options
        std::string     m_str;          // Holds name lookup
        FileSpec        m_file;         // Files for file lookups
        lldb::addr_t    m_addr;         // Holds the address to lookup
        lldb::addr_t    m_offset;       // Subtract this offset from m_addr before doing lookups.
        uint32_t        m_line_number;  // Line number for file+line lookups
        bool            m_use_regex;    // Name lookups in m_str are regular expressions.
        bool            m_check_inlines;// Check for inline entries when looking up by file/line.
        bool            m_verbose;      // Enable verbose lookup info

    };

    CommandObjectImageLookup (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "image lookup",
                       "Look up information within executable and dependent shared library images.",
                       NULL),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData file_arg;
        
        // Define the first (and only) variant of this arg.
        file_arg.arg_type = eArgTypeFilename;
        file_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (file_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectImageLookup ()
    {
    }

    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }


    bool
    LookupInModule (CommandInterpreter &interpreter, Module *module, CommandReturnObject &result, bool &syntax_error)
    {
        switch (m_options.m_type)
        {
        case eLookupTypeAddress:
            if (m_options.m_addr != LLDB_INVALID_ADDRESS)
            {
                if (LookupAddressInModule (m_interpreter, 
                                           result.GetOutputStream(), 
                                           module, 
                                           eSymbolContextEverything, 
                                           m_options.m_addr, 
                                           m_options.m_offset,
                                           m_options.m_verbose))
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    return true;
                }
            }
            break;

        case eLookupTypeSymbol:
            if (!m_options.m_str.empty())
            {
                if (LookupSymbolInModule (m_interpreter, result.GetOutputStream(), module, m_options.m_str.c_str(), m_options.m_use_regex))
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    return true;
                }
            }
            break;

        case eLookupTypeFileLine:
            if (m_options.m_file)
            {

                if (LookupFileAndLineInModule (m_interpreter,
                                               result.GetOutputStream(),
                                               module,
                                               m_options.m_file,
                                               m_options.m_line_number,
                                               m_options.m_check_inlines))
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    return true;
                }
            }
            break;

        case eLookupTypeFunction:
            if (!m_options.m_str.empty())
            {
                if (LookupFunctionInModule (m_interpreter,
                                            result.GetOutputStream(),
                                            module,
                                            m_options.m_str.c_str(),
                                            m_options.m_use_regex))
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    return true;
                }
            }
            break;

        case eLookupTypeType:
            if (!m_options.m_str.empty())
            {
                if (LookupTypeInModule (m_interpreter,
                                        result.GetOutputStream(),
                                        module,
                                        m_options.m_str.c_str(),
                                        m_options.m_use_regex))
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    return true;
                }
            }
            break;

        default:
            m_options.GenerateOptionUsage (result.GetErrorStream(), this);
            syntax_error = true;
            break;
        }

        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            bool syntax_error = false;
            uint32_t i;
            uint32_t num_successful_lookups = 0;
            uint32_t addr_byte_size = target->GetArchitecture().GetAddressByteSize();
            result.GetOutputStream().SetAddressByteSize(addr_byte_size);
            result.GetErrorStream().SetAddressByteSize(addr_byte_size);
            // Dump all sections for all modules images

            if (command.GetArgumentCount() == 0)
            {
                // Dump all sections for all modules images
                const uint32_t num_modules = target->GetImages().GetSize();
                if (num_modules > 0)
                {
                    for (i = 0; i<num_modules && syntax_error == false; ++i)
                    {
                        if (LookupInModule (m_interpreter, target->GetImages().GetModulePointerAtIndex(i), result, syntax_error))
                        {
                            result.GetOutputStream().EOL();
                            num_successful_lookups++;
                        }
                    }
                }
                else
                {
                    result.AppendError ("the target has no associated executable images");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // Dump specified images (by basename or fullpath)
                const char *arg_cstr;
                for (i = 0; (arg_cstr = command.GetArgumentAtIndex(i)) != NULL && syntax_error == false; ++i)
                {
                    FileSpec image_file(arg_cstr, false);
                    ModuleList matching_modules;
                    size_t num_matching_modules = target->GetImages().FindModules(&image_file, NULL, NULL, NULL, matching_modules);

                    // Not found in our module list for our target, check the main
                    // shared module list in case it is a extra file used somewhere
                    // else
                    if (num_matching_modules == 0)
                        num_matching_modules = ModuleList::FindSharedModules (image_file, 
                                                                              target->GetArchitecture(), 
                                                                              NULL, 
                                                                              NULL, 
                                                                              matching_modules);
                    
                    if (num_matching_modules > 0)
                    {
                        for (size_t j=0; j<num_matching_modules; ++j)
                        {
                            Module * image_module = matching_modules.GetModulePointerAtIndex(j);
                            if (image_module)
                            {
                                if (LookupInModule (m_interpreter, image_module, result, syntax_error))
                                {
                                    result.GetOutputStream().EOL();
                                    num_successful_lookups++;
                                }
                            }
                        }
                    }
                    else
                        result.AppendWarningWithFormat("Unable to find an image that matches '%s'.\n", arg_cstr);
                }
            }

            if (num_successful_lookups > 0)
                result.SetStatus (eReturnStatusSuccessFinishResult);
            else
                result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
protected:

    CommandOptions m_options;
};

OptionDefinition
CommandObjectImageLookup::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1,   true,  "address",    'a', required_argument, NULL, 0, eArgTypeAddress,      "Lookup an address in one or more executable images."},
{ LLDB_OPT_SET_1,   false, "offset",     'o', required_argument, NULL, 0, eArgTypeOffset,       "When looking up an address subtract <offset> from any addresses before doing the lookup."},
{ LLDB_OPT_SET_2,   true,  "symbol",     's', required_argument, NULL, 0, eArgTypeSymbol,       "Lookup a symbol by name in the symbol tables in one or more executable images."},
{ LLDB_OPT_SET_2,   false, "regex",      'r', no_argument,       NULL, 0, eArgTypeNone,         "The <name> argument for name lookups are regular expressions."},
{ LLDB_OPT_SET_3,   true,  "file",       'f', required_argument, NULL, 0, eArgTypeFilename,     "Lookup a file by fullpath or basename in one or more executable images."},
{ LLDB_OPT_SET_3,   false, "line",       'l', required_argument, NULL, 0, eArgTypeLineNum,      "Lookup a line number in a file (must be used in conjunction with --file)."},
{ LLDB_OPT_SET_3,   false, "no-inlines", 'i', no_argument,       NULL, 0, eArgTypeNone,         "Check inline line entries (must be used in conjunction with --file)."},
{ LLDB_OPT_SET_4,   true,  "function",   'n', required_argument, NULL, 0, eArgTypeFunctionName, "Lookup a function by name in the debug symbols in one or more executable images."},
{ LLDB_OPT_SET_5,   true,  "type",       't', required_argument, NULL, 0, eArgTypeName,         "Lookup a type by name in the debug symbols in one or more executable images."},
{ LLDB_OPT_SET_ALL, false, "verbose",    'v', no_argument,       NULL, 0, eArgTypeNone,         "Enable verbose lookup information."},
{ 0, false, NULL,           0, 0,                 NULL, 0, eArgTypeNone, NULL }
};





//----------------------------------------------------------------------
// CommandObjectImage constructor
//----------------------------------------------------------------------
CommandObjectImage::CommandObjectImage(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "image",
                            "A set of commands for accessing information for one or more executable images.",
                            "image <sub-command> ...")
{
    LoadSubCommand ("dump",    CommandObjectSP (new CommandObjectImageDump (interpreter)));
    LoadSubCommand ("list",    CommandObjectSP (new CommandObjectImageList (interpreter)));
    LoadSubCommand ("lookup",  CommandObjectSP (new CommandObjectImageLookup (interpreter)));
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommandObjectImage::~CommandObjectImage()
{
}

