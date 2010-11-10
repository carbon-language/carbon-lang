//===-- SymbolFile.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/ObjectFile.h"

using namespace lldb_private;

SymbolFile*
SymbolFile::FindPlugin (ObjectFile* obj_file)
{
    std::auto_ptr<SymbolFile> best_sym_file_ap;
    if (obj_file != NULL)
    {
        // TODO: Load any plug-ins in the appropriate plug-in search paths and
        // iterate over all of them to find the best one for the job.

        //----------------------------------------------------------------------
        // We currently only have one debug symbol parser...
        //----------------------------------------------------------------------
        std::auto_ptr<SymbolFile> best_symfile_ap;
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
                    best_sym_file_ap = curr_symfile_ap;
                }
            }
        }
    }
    return best_sym_file_ap.release();
}

TypeList *
SymbolFile::GetTypeList ()
{
    return m_obj_file->GetModule()->GetTypeList();
}
