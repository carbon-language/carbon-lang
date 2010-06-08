//===-- LibUnwindRegisterContext.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_LibUnwindRegisterContext_h_
#define lldb_LibUnwindRegisterContext_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/RegisterContext.h"

#include "libunwind.h"

class LibUnwindRegisterContext : public lldb_private::RegisterContext
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    LibUnwindRegisterContext (lldb_private::Thread &thread,
                              lldb_private::StackFrame *frame,
                              const lldb_private::unw_cursor_t &unwind_cursor);

    virtual
    ~LibUnwindRegisterContext ();

    //------------------------------------------------------------------
    // Subclasses must override these functions
    //------------------------------------------------------------------
    virtual void
    Invalidate ();

    virtual size_t
    GetRegisterCount ();

    virtual const lldb::RegisterInfo *
    GetRegisterInfoAtIndex (uint32_t reg);

    virtual size_t
    GetRegisterSetCount ();

    virtual const lldb::RegisterSet *
    GetRegisterSet (uint32_t reg_set);

    virtual bool
    ReadRegisterValue (uint32_t reg, lldb_private::Scalar &value);

    virtual bool
    ReadRegisterBytes (uint32_t reg, lldb_private::DataExtractor &data);

    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);

    virtual bool
    WriteRegisterValue (uint32_t reg, const lldb_private::Scalar &value);

    virtual bool
    WriteRegisterBytes (uint32_t reg, lldb_private::DataExtractor &data, uint32_t data_offset);

    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);
    
private:
    lldb_private::unw_cursor_t m_unwind_cursor;
    bool m_unwind_cursor_is_valid;
    //------------------------------------------------------------------
    // For LibUnwindRegisterContext only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (LibUnwindRegisterContext);
};

#endif  // lldb_LibUnwindRegisterContext_h_
