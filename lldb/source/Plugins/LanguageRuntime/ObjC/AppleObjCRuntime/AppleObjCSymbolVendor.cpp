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
    SymbolVendor(NULL),
    m_process(process->GetSP()),
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
    
    ModuleList &images = m_process->GetTarget().GetImages();
    
    for (size_t image_index = 0, end_index = images.GetSize();
         image_index < end_index;
         ++image_index)
    {
        Module *image = images.GetModulePointerAtIndex(image_index);
        
        if (!image)
            continue;
        
        SymbolVendor *symbol_vendor = image->GetSymbolVendor();
        
        if (!symbol_vendor)
            continue;
        
        SymbolFile *symbol_file = image->GetSymbolVendor()->GetSymbolFile();
        
        if (!symbol_file || !(symbol_file->GetAbilities() & SymbolFile::RuntimeTypes))
            continue;
        
        const bool inferior_append = true;
        
        ret += symbol_file->FindTypes (sc, name, namespace_decl, inferior_append, max_matches - ret, types);
        
        if (ret >= max_matches)
            break;
    }
    
    return ret;
}
