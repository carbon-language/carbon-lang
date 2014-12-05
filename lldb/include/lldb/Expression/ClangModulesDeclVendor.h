//===-- ClangModulesDeclVendor.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _liblldb_ClangModulesDeclVendor_
#define _liblldb_ClangModulesDeclVendor_

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/DeclVendor.h"
#include "lldb/Target/Platform.h"

#include <vector>

namespace lldb_private
{
    
class ClangModulesDeclVendor : public DeclVendor
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ClangModulesDeclVendor();
    
    virtual
    ~ClangModulesDeclVendor();
    
    static ClangModulesDeclVendor *
    Create(Target &target);
    
    //------------------------------------------------------------------
    /// Add a module to the list of modules to search.
    ///
    /// @param[in] path
    ///     The path to the exact module to be loaded.  E.g., if the desired
    ///     module is std.io, then this should be { "std", "io" }.
    ///
    /// @param[in] error_stream
    ///     A stream to populate with the output of the Clang parser when
    ///     it tries to load the module.
    ///
    /// @return
    ///     True if the module could be loaded; false if not.  If the
    ///     compiler encountered a fatal error during a previous module
    ///     load, then this will always return false for this ModuleImporter.
    //------------------------------------------------------------------
    virtual bool
    AddModule(std::vector<llvm::StringRef> &path, Stream &error_stream) = 0;
};
    
}
#endif /* defined(_lldb_ClangModulesDeclVendor_) */
