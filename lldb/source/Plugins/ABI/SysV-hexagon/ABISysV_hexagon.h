//===-- ABISysV_hexagon.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABISysV_hexagon_h_
#define liblldb_ABISysV_hexagon_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"

class ABISysV_hexagon :
    public lldb_private::ABI
{
public:

    ~ABISysV_hexagon( void )
    {
    }

    virtual size_t
    GetRedZoneSize ( void ) const;

    virtual bool
    PrepareTrivialCall ( lldb_private::Thread &thread, 
                         lldb::addr_t sp,
                         lldb::addr_t functionAddress,
                         lldb::addr_t returnAddress, 
                         llvm::ArrayRef<lldb::addr_t> args ) const;
    
    // special thread plan for GDB style non-jit function calls
    virtual bool
    PrepareTrivialCall ( lldb_private::Thread &thread, 
                         lldb::addr_t sp,
                         lldb::addr_t functionAddress,
                         lldb::addr_t returnAddress,
                         llvm::Type &prototype,
                         llvm::ArrayRef<ABI::CallArgument> args ) const;

    virtual bool
    GetArgumentValues ( lldb_private::Thread &thread,
                        lldb_private::ValueList &values ) const;
    
    virtual lldb_private::Error
    SetReturnValueObject ( lldb::StackFrameSP &frame_sp,
                           lldb::ValueObjectSP &new_value );

protected:
    lldb::ValueObjectSP
    GetReturnValueObjectSimple ( lldb_private::Thread &thread,
                                 lldb_private::ClangASTType &ast_type ) const;
    
public:    
    virtual lldb::ValueObjectSP
    GetReturnValueObjectImpl ( lldb_private::Thread &thread,
                               lldb_private::ClangASTType &type ) const;
        
    // specialized to work with llvm IR types
    virtual lldb::ValueObjectSP
    GetReturnValueObjectImpl ( lldb_private::Thread &thread, llvm::Type &type ) const;

    virtual bool
    CreateFunctionEntryUnwindPlan ( lldb_private::UnwindPlan &unwind_plan );
    
    virtual bool
    CreateDefaultUnwindPlan ( lldb_private::UnwindPlan &unwind_plan );
        
    virtual bool
    RegisterIsVolatile ( const lldb_private::RegisterInfo *reg_info );

    virtual bool
    CallFrameAddressIsValid ( lldb::addr_t cfa )
    {
        // Make sure the stack call frame addresses are 8 byte aligned
        if (cfa & 0x07)
            return false;   // Not 8 byte aligned
        if (cfa == 0)
            return false;   // Zero is not a valid stack address
        return true;
    }
    
    virtual bool
    CodeAddressIsValid ( lldb::addr_t pc )
    {
        // We have a 64 bit address space, so anything is valid as opcodes
        // aren't fixed width...
        return true;
    }

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoArray ( uint32_t &count );

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize ( void );

    static void
    Terminate ( void );

    static lldb::ABISP
    CreateInstance ( const lldb_private::ArchSpec &arch );

    static lldb_private::ConstString
    GetPluginNameStatic ( void );
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName ( void );

    virtual uint32_t
    GetPluginVersion ( void );

protected:
    void
    CreateRegisterMapIfNeeded ( void );

    bool
    RegisterIsCalleeSaved (const lldb_private::RegisterInfo *reg_info);

private:
    ABISysV_hexagon ( void ) : lldb_private::ABI() { } // Call CreateInstance instead.
};

#endif  // liblldb_ABISysV_hexagon_h_
