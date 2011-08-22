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
                        lldb::addr_t *arg1_ptr = NULL,
                        lldb::addr_t *arg2_ptr = NULL,
                        lldb::addr_t *arg3_ptr = NULL,
                        lldb::addr_t *arg4_ptr = NULL,
                        lldb::addr_t *arg5_ptr = NULL,
                        lldb::addr_t *arg6_ptr = NULL) const = 0;

    virtual bool
    GetArgumentValues (Thread &thread,
                       ValueList &values) const = 0;
    
    virtual bool
    GetReturnValue (Thread &thread,
                    Value &value) const = 0;

    virtual bool
    CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan) = 0;

    virtual bool
    CreateDefaultUnwindPlan (UnwindPlan &unwind_plan) = 0;

    virtual bool
    RegisterIsVolatile (const RegisterInfo *reg_info) = 0;

    virtual bool
    StackUsesFrames () = 0;

    virtual bool
    CallFrameAddressIsValid (lldb::addr_t cfa) = 0;

    virtual bool
    CodeAddressIsValid (lldb::addr_t pc) = 0;    

    virtual const RegisterInfo *
    GetRegisterInfoArray (uint32_t &count) = 0;

    
    
    bool
    GetRegisterInfoByName (const ConstString &name, RegisterInfo &info);

    bool
    GetRegisterInfoByKind (lldb::RegisterKind reg_kind, 
                           uint32_t reg_num, 
                           RegisterInfo &info);

    static lldb::ABISP
    FindPlugin (const ArchSpec &arch);
    
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
