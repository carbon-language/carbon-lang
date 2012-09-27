//===-- Symbols.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Symbols_h_
#define liblldb_Symbols_h_

// C Includes
#include <stdint.h>
#include <sys/time.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Host/FileSpec.h"

namespace lldb_private {

class Symbols
{
public:
    //----------------------------------------------------------------------
    // Locate the executable file given a module specification.
    //
    // Locating the file should happen only on the local computer or using
    // the current computers global settings.
    //----------------------------------------------------------------------
    static FileSpec
    LocateExecutableObjectFile (const ModuleSpec &module_spec);

    //----------------------------------------------------------------------
    // Locate the symbol file given a module specification.
    //
    // Locating the file should happen only on the local computer or using
    // the current computers global settings.
    //----------------------------------------------------------------------
    static FileSpec
    LocateExecutableSymbolFile (const ModuleSpec &module_spec);
    
    static FileSpec
    FindSymbolFileInBundle (const FileSpec& dsym_bundle_fspec,
                            const lldb_private::UUID *uuid,
                            const ArchSpec *arch);
    
    //----------------------------------------------------------------------
    // Locate the object and symbol file given a module specification.
    //
    // Locating the file can try to download the file from a corporate build
    // respository, or using any other meeans necessary to locate both the
    // unstripped object file and the debug symbols.
    //----------------------------------------------------------------------
    static bool
    DownloadObjectAndSymbolFile (ModuleSpec &module_spec);
                                 
};

} // namespace lldb_private


#endif  // liblldb_Symbols_h_
