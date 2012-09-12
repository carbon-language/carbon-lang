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
    static FileSpec
    LocateExecutableObjectFile (const ModuleSpec &module_spec);

    static FileSpec
    LocateExecutableSymbolFile (const ModuleSpec &module_spec);
    
    static FileSpec
    FindSymbolFileInBundle (const FileSpec& dsym_bundle_fspec,
                            const lldb_private::UUID *uuid,
                            const ArchSpec *arch);
};

} // namespace lldb_private


#endif  // liblldb_Symbols_h_
