//===-- SymbolVendorELF.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolVendorELF_h_
#define liblldb_SymbolVendorELF_h_

#include "lldb/lldb-private.h"
#include "lldb/Symbol/SymbolVendor.h"

class SymbolVendorELF : public lldb_private::SymbolVendor
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SymbolVendor*
    CreateInstance (const lldb::ModuleSP &module_sp, lldb_private::Stream *feedback_strm);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SymbolVendorELF (const lldb::ModuleSP &module_sp);

    virtual
    ~SymbolVendorELF();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

private:
    DISALLOW_COPY_AND_ASSIGN (SymbolVendorELF);
};

#endif  // liblldb_SymbolVendorELF_h_
