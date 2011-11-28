//===-- SymbolFile.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolFile.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/ObjectFile.h"

using namespace lldb_private;

SymbolFile*
SymbolFile::FindPlugin (ObjectFile* obj_file)
{
    std::auto_ptr<SymbolFile> best_symfile_ap;
    if (obj_file != NULL)
    {
        // TODO: Load any plug-ins in the appropriate plug-in search paths and
        // iterate over all of them to find the best one for the job.

        //----------------------------------------------------------------------
        // We currently only have one debug symbol parser...
        //----------------------------------------------------------------------
        uint32_t best_symfile_abilities = 0;

        SymbolFileCreateInstance create_callback;
        for (uint32_t idx = 0; (create_callback = PluginManager::GetSymbolFileCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            std::auto_ptr<SymbolFile> curr_symfile_ap(create_callback(obj_file));

            if (curr_symfile_ap.get())
            {
                uint32_t sym_file_abilities = curr_symfile_ap->GetAbilities();
                if (sym_file_abilities > best_symfile_abilities)
                {
                    best_symfile_abilities = sym_file_abilities;
                    best_symfile_ap = curr_symfile_ap;
                }
            }
        }
        if (best_symfile_ap.get())
        {
            // Let the winning symbol file parser initialize itself more 
            // completely now that it has been chosen
            best_symfile_ap->InitializeObject();
        }
    }
    return best_symfile_ap.release();
}

TypeList *
SymbolFile::GetTypeList ()
{
    if (m_obj_file)
        return m_obj_file->GetModule()->GetTypeList();
    return NULL;
}

lldb_private::ClangASTContext &       
SymbolFile::GetClangASTContext ()
{
    return m_obj_file->GetModule()->GetClangASTContext();
}


void
SymbolFile::ReportError (const char *format, ...)
{
    StreamString module_description;
    m_obj_file->GetModule()->GetDescription (&module_description, lldb::eDescriptionLevelBrief);
    ::fprintf (stderr, "error: %s ", module_description.GetString().c_str());
    
    va_list args;
    va_start (args, format);
    vfprintf (stderr, format, args);
    va_end (args);
}

void
SymbolFile::ReportWarning (const char *format, ...)
{
    StreamString module_description;
    m_obj_file->GetModule()->GetDescription (&module_description, lldb::eDescriptionLevelBrief);
    ::fprintf (stderr, "warning: %s ", module_description.GetString().c_str());
    
    va_list args;
    va_start (args, format);
    vfprintf (stderr, format, args);
    va_end (args);
}

void
SymbolFile::LogMessage (Log *log, const char *format, ...)
{
    if (log)
    {
        StreamString log_message;
        m_obj_file->GetModule()->GetDescription (&log_message, lldb::eDescriptionLevelBrief);
        log_message.PutChar(' ');
        va_list args;
        va_start (args, format);
        log_message.PrintfVarArg (format, args);
        va_end (args);
        log->PutCString (log_message.GetString().c_str());
    }
}
