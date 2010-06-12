//===-- CommandCompletions.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/CommandCompletions.h"


using namespace lldb_private;

CommandCompletions::CommonCompletionElement
CommandCompletions::g_common_completions[] =
{
    {eCustomCompletion,     NULL},
    {eSourceFileCompletion, CommandCompletions::SourceFiles},
    {eDiskFileCompletion,   NULL},
    {eSymbolCompletion,     CommandCompletions::Symbols},
    {eModuleCompletion,     CommandCompletions::Modules},
    {eNoCompletion,         NULL}      // This one has to be last in the list.
};

bool
CommandCompletions::InvokeCommonCompletionCallbacks (uint32_t completion_mask,
                                                const char *completion_str,
                                                int match_start_point,
                                                int max_return_elements,
                                                lldb_private::CommandInterpreter *interpreter,
                                                SearchFilter *searcher,
                                                lldb_private::StringList &matches)
{
    bool handled = false;

    if (completion_mask & eCustomCompletion)
        return false;

    for (int i = 0; ; i++)
    {
        if (g_common_completions[i].type == eNoCompletion)
            break;
         else if ((g_common_completions[i].type & completion_mask) == g_common_completions[i].type
                   && g_common_completions[i].callback != NULL)
         {
            handled = true;
            g_common_completions[i].callback (completion_str,
                                              match_start_point,
                                              max_return_elements,
                                              interpreter,
                                              searcher,
                                              matches);
        }
    }
    return handled;
}

int
CommandCompletions::SourceFiles (const char *partial_file_name,
                    int match_start_point,
                    int max_return_elements,
                    lldb_private::CommandInterpreter *interpreter,
                    SearchFilter *searcher,
                    lldb_private::StringList &matches)
{
    // Find some way to switch "include support files..."
    SourceFileCompleter completer (false, partial_file_name, match_start_point, max_return_elements, interpreter,
                                   matches);

    if (searcher == NULL)
    {
        lldb::TargetSP target_sp = interpreter->Context()->GetTarget()->GetSP();
        SearchFilter null_searcher (target_sp);
        completer.DoCompletion (&null_searcher);
    }
    else
    {
        completer.DoCompletion (searcher);
    }
    return matches.GetSize();
}

int
CommandCompletions::Modules (const char *partial_file_name,
                    int match_start_point,
                    int max_return_elements,
                    lldb_private::CommandInterpreter *interpreter,
                    SearchFilter *searcher,
                    lldb_private::StringList &matches)
{
    ModuleCompleter completer(partial_file_name, match_start_point, max_return_elements, interpreter, matches);

    if (searcher == NULL)
    {
        lldb::TargetSP target_sp = interpreter->Context()->GetTarget()->GetSP();
        SearchFilter null_searcher (target_sp);
        completer.DoCompletion (&null_searcher);
    }
    else
    {
        completer.DoCompletion (searcher);
    }
    return matches.GetSize();
}

int
CommandCompletions::Symbols (const char *partial_file_name,
                    int match_start_point,
                    int max_return_elements,
                    lldb_private::CommandInterpreter *interpreter,
                    SearchFilter *searcher,
                    lldb_private::StringList &matches)
{
    SymbolCompleter completer(partial_file_name, match_start_point, max_return_elements, interpreter, matches);

    if (searcher == NULL)
    {
        lldb::TargetSP target_sp = interpreter->Context()->GetTarget()->GetSP();
        SearchFilter null_searcher (target_sp);
        completer.DoCompletion (&null_searcher);
    }
    else
    {
        completer.DoCompletion (searcher);
    }
    return matches.GetSize();
}

CommandCompletions::Completer::Completer (
    const char *completion_str,
    int match_start_point,
    int max_return_elements,
    CommandInterpreter *interpreter,
    StringList &matches
) :
    m_completion_str (completion_str),
    m_match_start_point (match_start_point),
    m_max_return_elements (max_return_elements),
    m_interpreter (interpreter),
    m_matches (matches)
{
}

CommandCompletions::Completer::~Completer ()
{

}

//----------------------------------------------------------------------
// SourceFileCompleter
//----------------------------------------------------------------------

CommandCompletions::SourceFileCompleter::SourceFileCompleter (
    bool include_support_files,
    const char *completion_str,
    int match_start_point,
    int max_return_elements,
    CommandInterpreter *interpreter,
    StringList &matches
) :
    CommandCompletions::Completer (completion_str, match_start_point, max_return_elements, interpreter, matches),
    m_include_support_files (include_support_files),
    m_matching_files()
{
    FileSpec partial_spec (m_completion_str.c_str());
    m_file_name = partial_spec.GetFilename().GetCString();
    m_dir_name = partial_spec.GetDirectory().GetCString();
}

Searcher::Depth
CommandCompletions::SourceFileCompleter::GetDepth()
{
    return eDepthCompUnit;
}

Searcher::CallbackReturn
CommandCompletions::SourceFileCompleter::SearchCallback (
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool complete
)
{
    if (context.comp_unit != NULL)
    {
        if (m_include_support_files)
        {
            FileSpecList supporting_files = context.comp_unit->GetSupportFiles();
            for (size_t sfiles = 0; sfiles < supporting_files.GetSize(); sfiles++)
            {
                const FileSpec &sfile_spec = supporting_files.GetFileSpecAtIndex(sfiles);
                const char *sfile_file_name = sfile_spec.GetFilename().GetCString();
                const char *sfile_dir_name = sfile_spec.GetFilename().GetCString();
                bool match = false;
                if (m_file_name && sfile_file_name
                    && strstr (sfile_file_name, m_file_name) == sfile_file_name)
                    match = true;
                if (match && m_dir_name && sfile_dir_name
                    && strstr (sfile_dir_name, m_dir_name) != sfile_dir_name)
                    match = false;

                if (match)
                {
                    m_matching_files.AppendIfUnique(sfile_spec);
                }
            }

        }
        else
        {
            const char *cur_file_name = context.comp_unit->GetFilename().GetCString();
            const char *cur_dir_name = context.comp_unit->GetDirectory().GetCString();

            bool match = false;
            if (m_file_name && cur_file_name
                && strstr (cur_file_name, m_file_name) == cur_file_name)
                match = true;

            if (match && m_dir_name && cur_dir_name
                && strstr (cur_dir_name, m_dir_name) != cur_dir_name)
                match = false;

            if (match)
            {
                m_matching_files.AppendIfUnique(context.comp_unit);
            }
        }
    }
    return Searcher::eCallbackReturnContinue;
}

size_t
CommandCompletions::SourceFileCompleter::DoCompletion (SearchFilter *filter)
{
    filter->Search (*this);
    // Now convert the filelist to completions:
    for (size_t i = 0; i < m_matching_files.GetSize(); i++)
    {
        m_matches.AppendString (m_matching_files.GetFileSpecAtIndex(i).GetFilename().GetCString());
    }
    return m_matches.GetSize();

}

//----------------------------------------------------------------------
// SymbolCompleter
//----------------------------------------------------------------------

static bool
regex_chars (const char comp)
{
    if (comp == '[' || comp == ']' || comp == '(' || comp == ')')
        return true;
    else
        return false;
}
CommandCompletions::SymbolCompleter::SymbolCompleter (
    const char *completion_str,
    int match_start_point,
    int max_return_elements,
    CommandInterpreter *interpreter,
    StringList &matches
) :
    CommandCompletions::Completer (completion_str, match_start_point, max_return_elements, interpreter, matches)
{
    std::string regex_str ("^");
    regex_str.append(completion_str);
    regex_str.append(".*");
    std::string::iterator pos;

    pos = find_if(regex_str.begin(), regex_str.end(), regex_chars);
    while (pos < regex_str.end()) {
        pos = regex_str.insert(pos, '\\');
        pos += 2;
        pos = find_if(pos, regex_str.end(), regex_chars);
    }
    m_regex.Compile(regex_str.c_str());
}

Searcher::Depth
CommandCompletions::SymbolCompleter::GetDepth()
{
    return eDepthModule;
}

Searcher::CallbackReturn
CommandCompletions::SymbolCompleter::SearchCallback (
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool complete
)
{
    SymbolContextList func_list;
    SymbolContextList sym_list;

    if (context.module_sp != NULL)
    {
        if (context.module_sp)
        {
            context.module_sp->FindSymbolsMatchingRegExAndType (m_regex, lldb::eSymbolTypeCode, sym_list);
            context.module_sp->FindFunctions (m_regex, true, func_list);
        }

        SymbolContext sc;
        // Now add the functions & symbols to the list - only add if unique:
        for (int i = 0; i < func_list.GetSize(); i++)
        {
            if (func_list.GetContextAtIndex(i, sc))
            {
                if (sc.function)
                {
                    m_match_set.insert (sc.function->GetMangled().GetDemangledName());
                }
            }
        }

        for (int i = 0; i < sym_list.GetSize(); i++)
        {
            if (sym_list.GetContextAtIndex(i, sc))
            {
                if (sc.symbol && sc.symbol->GetAddressRangePtr())
                {
                    m_match_set.insert (sc.symbol->GetMangled().GetDemangledName());
                }
            }
        }
    }
    return Searcher::eCallbackReturnContinue;
}

size_t
CommandCompletions::SymbolCompleter::DoCompletion (SearchFilter *filter)
{
    filter->Search (*this);
    collection::iterator pos = m_match_set.begin(), end = m_match_set.end();
    for (pos = m_match_set.begin(); pos != end; pos++)
        m_matches.AppendString((*pos).GetCString());

    return m_matches.GetSize();
}

//----------------------------------------------------------------------
// ModuleCompleter
//----------------------------------------------------------------------
CommandCompletions::ModuleCompleter::ModuleCompleter (
    const char *completion_str,
    int match_start_point,
    int max_return_elements,
    CommandInterpreter *interpreter,
    StringList &matches
) :
    CommandCompletions::Completer (completion_str, match_start_point, max_return_elements, interpreter, matches)
{
    FileSpec partial_spec (m_completion_str.c_str());
    m_file_name = partial_spec.GetFilename().GetCString();
    m_dir_name = partial_spec.GetDirectory().GetCString();
}

Searcher::Depth
CommandCompletions::ModuleCompleter::GetDepth()
{
    return eDepthModule;
}

Searcher::CallbackReturn
CommandCompletions::ModuleCompleter::SearchCallback (
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool complete
)
{
    if (context.module_sp != NULL)
    {
        const char *cur_file_name = context.module_sp->GetFileSpec().GetFilename().GetCString();
        const char *cur_dir_name = context.module_sp->GetFileSpec().GetDirectory().GetCString();

        bool match = false;
        if (m_file_name && cur_file_name
            && strstr (cur_file_name, m_file_name) == cur_file_name)
            match = true;

        if (match && m_dir_name && cur_dir_name
            && strstr (cur_dir_name, m_dir_name) != cur_dir_name)
            match = false;

        if (match)
        {
            m_matches.AppendString (cur_file_name);
        }
    }
    return Searcher::eCallbackReturnContinue;
}

size_t
CommandCompletions::ModuleCompleter::DoCompletion (SearchFilter *filter)
{
    filter->Search (*this);
    return m_matches.GetSize();
}



