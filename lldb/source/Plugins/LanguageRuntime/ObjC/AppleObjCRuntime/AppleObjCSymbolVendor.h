//===-- AppleObjCSymbolVendor.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCSymbolVendor_h_
#define liblldb_AppleObjCSymbolVendor_h_

// C Includes
// C++ Includes

#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/SymbolFile.h"

namespace lldb_private {
    
class AppleObjCSymbolVendor : public SymbolVendor
{
public:
    AppleObjCSymbolVendor(Process* process);
    
    virtual uint32_t
    FindTypes (const SymbolContext& sc, 
               const ConstString &name,
               const ClangNamespaceDecl *namespace_decl, 
               bool append, 
               uint32_t max_matches, 
               TypeList& types);
    
private:
    lldb::ProcessSP                     m_process;
    ClangASTContext                     m_ast_ctx;
};

} // namespace lldb_private

#endif  // liblldb_AppleObjCSymbolVendor_h_
