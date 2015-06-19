//===-- ABISysV_mips64.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABISysV_mips64_h_
#define liblldb_ABISysV_mips64_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"

class ABISysV_mips64 :
    public lldb_private::ABI
{
public:

    ~ABISysV_mips64()
    {
    }

    virtual size_t
    GetRedZoneSize () const;

    virtual bool
    PrepareTrivialCall (lldb_private::Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t functionAddress,
                        lldb::addr_t returnAddress, 
                        llvm::ArrayRef<lldb::addr_t> args) const;

    virtual bool
    GetArgumentValues (lldb_private::Thread &thread,
                       lldb_private::ValueList &values) const;

    virtual lldb_private::Error
    SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value);

protected:
    lldb::ValueObjectSP
    GetReturnValueObjectSimple (lldb_private::Thread &thread,
                                lldb_private::ClangASTType &ast_type) const;

public:    
    virtual lldb::ValueObjectSP
    GetReturnValueObjectImpl (lldb_private::Thread &thread,
                          lldb_private::ClangASTType &type) const;

    virtual bool
    CreateFunctionEntryUnwindPlan (lldb_private::UnwindPlan &unwind_plan);
    
    virtual bool
    CreateDefaultUnwindPlan (lldb_private::UnwindPlan &unwind_plan);

    virtual bool
    RegisterIsVolatile (const lldb_private::RegisterInfo *reg_info);

    // The SysV mips ABI requires that stack frames be 16 byte aligned.
    // When there is a trap handler on the stack, e.g. _sigtramp in userland
    // code, we've seen that the stack pointer is often not aligned properly
    // before the handler is invoked.  This means that lldb will stop the unwind
    // early -- before the function which caused the trap.
    //
    // To work around this, we relax that alignment to be just word-size (8-bytes).
    // Whitelisting the trap handlers for user space would be easy (_sigtramp) but
    // in other environments there can be a large number of different functions
    // involved in async traps.
    virtual bool
    CallFrameAddressIsValid (lldb::addr_t cfa)
    {
        // Make sure the stack call frame addresses are 8 byte aligned
        if (cfa & (8ull - 1ull))
            return false;   // Not 8 byte aligned
        if (cfa == 0)
            return false;   // Zero is not a valid stack address
        return true;
    }

    virtual bool
    CodeAddressIsValid (lldb::addr_t pc)
    {
       if (pc & (4ull - 1ull))
           return false;   // Not 4 byte aligned
        
        // Anything else if fair game..
        return true;
    }

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoArray (uint32_t &count);
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb::ABISP
    CreateInstance (const lldb_private::ArchSpec &arch);

    static lldb_private::ConstString
    GetPluginNameStatic();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

protected:
    void
    CreateRegisterMapIfNeeded ();

    bool
    RegisterIsCalleeSaved (const lldb_private::RegisterInfo *reg_info);

private:
    ABISysV_mips64() : lldb_private::ABI() { } // Call CreateInstance instead.
};

#endif  // liblldb_ABI_h_
