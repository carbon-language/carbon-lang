//===-- ABI.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABI_h_
#define liblldb_ABI_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ABI :
    public PluginInterface
{
public:
    virtual
    ~ABI();

    virtual size_t
    GetRedZoneSize () const = 0;

    virtual bool
    PrepareTrivialCall (Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t functionAddress,
                        lldb::addr_t returnAddress, 
                        lldb::addr_t arg,
                        lldb::addr_t *this_arg) const = 0;
    
    virtual bool
    GetArgumentValues (Thread &thread,
                       ValueList &values) const = 0;
    
    virtual bool
    GetReturnValue (Thread &thread,
                    Value &value) const = 0;
    
    static ABI* 
    FindPlugin (const ConstString &triple);
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ABI can see and modify these
    //------------------------------------------------------------------
    ABI();
private:
    DISALLOW_COPY_AND_ASSIGN (ABI);
};

} // namespace lldb_private

#endif  // liblldb_ABI_h_
