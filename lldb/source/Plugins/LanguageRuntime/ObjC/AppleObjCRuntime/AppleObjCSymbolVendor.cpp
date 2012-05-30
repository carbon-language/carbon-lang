//===-- AppleObjCSymbolVendor.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCSymbolVendor.h"

#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "clang/AST/ASTContext.h"

using namespace lldb_private;

AppleObjCSymbolVendor::AppleObjCSymbolVendor(Process *process) :
    SymbolVendor(lldb::ModuleSP()),
    m_process(process->shared_from_this()),
    m_ast_ctx(process->GetTarget().GetArchitecture().GetTriple().getTriple().c_str())
{
}

uint32_t
AppleObjCSymbolVendor::FindTypes (const SymbolContext& sc, 
                                  const ConstString &name,
                                  const ClangNamespaceDecl *namespace_decl, 
                                  bool append, 
                                  uint32_t max_matches, 
                                  TypeList& types)
{
    lldb::LogSP log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_SYMBOLS));  // FIXME - a more appropriate log channel?
        
    if (log)
        log->Printf("ObjC SymbolVendor asked for '%s'", 
                    name.AsCString());
    
    if (!append)
        types.Clear();
    
    uint32_t ret = 0;
    
    ModuleList &target_modules = m_process->GetTarget().GetImages();
    Mutex::Locker modules_locker(target_modules.GetMutex());
    
    for (size_t image_index = 0, end_index = target_modules.GetSize();
         image_index < end_index;
         ++image_index)
    {
        Module *image = target_modules.GetModulePointerAtIndexUnlocked(image_index);
        
        if (!image)
            continue;
        
        SymbolVendor *symbol_vendor = image->GetSymbolVendor();
        
        if (!symbol_vendor)
            continue;
        
        SymbolFile *symbol_file = image->GetSymbolVendor()->GetSymbolFile();
        
        // Don't use a symbol file if it actually has types. We are specifically
        // looking for something in runtime information, not from debug information,
        // as the data in debug information will get parsed by the debug info
        // symbol files. So we veto any symbol file that has actual variable
        // type parsing abilities.
        if (symbol_file == NULL || (symbol_file->GetAbilities() & SymbolFile::VariableTypes))
            continue;
        
        const bool inferior_append = true;
        
        ret += symbol_file->FindTypes (sc, name, namespace_decl, inferior_append, max_matches - ret, types);
        
        if (ret >= max_matches)
            break;
    }
    
    return ret;
}
