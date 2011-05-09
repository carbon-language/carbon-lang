//===-- RegisterContextLLDB.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterContextLLDB_h_
#define lldb_RegisterContextLLDB_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Symbol/SymbolContext.h"

class RegisterContextLLDB : public lldb_private::RegisterContext
{
public:
    typedef lldb::SharedPtr<RegisterContextLLDB>::Type SharedPtr;

    RegisterContextLLDB (lldb_private::Thread &thread,
                         const SharedPtr& next_frame,
                         lldb_private::SymbolContext& sym_ctx,
                         uint32_t frame_number);

    ///
    // pure virtual functions from the base class that we must implement
    ///

    virtual
    ~RegisterContextLLDB () { }

    virtual void
    InvalidateAllRegisters ();

    virtual size_t
    GetRegisterCount ();

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex (uint32_t reg);

    virtual size_t
    GetRegisterSetCount ();

    virtual const lldb_private::RegisterSet *
    GetRegisterSet (uint32_t reg_set);

    virtual bool
    ReadRegister (const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister (const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);

    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);

    bool
    IsValid () const;

    bool
    GetCFA (lldb::addr_t& cfa);

    bool
    GetStartPC (lldb::addr_t& start_pc);

    bool
    ReadPC (lldb::addr_t& start_pc);

private:

    enum FrameType
    {
        eNormalFrame,
        eSigtrampFrame,
        eDebuggerFrame,  // a debugger inferior function call frame; we get caller's registers from debugger
        eNotAValidFrame  // this frame is invalid for some reason - most likely it is past the top (end) of the stack
    };

    enum RegisterLocationTypes
    {
        eRegisterNotSaved = 0,          // register was not preserved by callee.  If volatile reg, is unavailable
        eRegisterSavedAtMemoryLocation, // register is saved at a specific word of target mem (target_memory_location)
        eRegisterInRegister,            // register is available in a (possible other) register (register_number)
        eRegisterSavedAtHostMemoryLocation, // register is saved at a word in lldb's address space
        eRegisterValueInferred          // register val was computed (and is in inferred_value)
    };

    struct RegisterLocation
    {
        int type;
        union
        {
            lldb::addr_t target_memory_location;
            uint32_t     register_number;       // in eRegisterKindLLDB register numbering system
            void*        host_memory_location;
            uint64_t     inferred_value;        // eRegisterValueInferred - e.g. stack pointer == cfa + offset
        } location;
    };


    // Indicates whether this frame is frame zero -- the currently
    // executing frame -- or not.  If it is not frame zero, m_next_frame's
    // shared pointer holds a pointer to the RegisterContextLLDB
    // object "below" this frame, i.e. this frame called m_next_frame's
    // function.
    bool
    IsFrameZero () const;

    void 
    InitializeZerothFrame ();

    void 
    InitializeNonZerothFrame();

    // Provide a location for where THIS function saved the CALLER's register value
    // Or a frame "below" this one saved it, i.e. a function called by this one, preserved a register that this
    // function didn't modify/use.
    //
    // The RegisterLocation type may be set to eRegisterNotAvailable -- this will happen for a volatile register 
    // being queried mid-stack.  Instead of floating frame 0's contents of that register up the stack (which may
    // or may not be the value of that reg when the function was executing), we won't return any value.
    //
    // If a non-volatile register (a "preserved" register) is requested mid-stack and no frames "below" the requested
    // stack have saved the register anywhere, it is safe to assume that frame 0's register values are still the same
    // as the requesting frame's.
    //
    bool
    SavedLocationForRegister (uint32_t lldb_regnum, RegisterLocation &regloc);

    bool
    ReadRegisterValueFromRegisterLocation (RegisterLocation regloc, 
                                           const lldb_private::RegisterInfo *reg_info,
                                           lldb_private::RegisterValue &value);

    bool
    WriteRegisterValueToRegisterLocation (RegisterLocation regloc, 
                                          const lldb_private::RegisterInfo *reg_info,
                                          const lldb_private::RegisterValue &value);

    // Get the contents of a general purpose (address-size) register for this frame 
    // (usually retrieved from the m_next_frame)
    // m_base_reg_ectx and m_next_frame should both be initialized appropriately before calling.
    bool
    ReadGPRValue (int register_kind, uint32_t regnum, lldb::addr_t &value);

    lldb::UnwindPlanSP
    GetFastUnwindPlanForFrame ();

    lldb::UnwindPlanSP
    GetFullUnwindPlanForFrame ();

    lldb_private::Thread& m_thread;
    
    SharedPtr m_next_frame;

    ///
    // The following tell us how to retrieve the CALLER's register values (ie the "previous" frame, aka the frame above)
    // i.e. where THIS frame saved them
    ///

    lldb::UnwindPlanSP m_fast_unwind_plan_sp;  // may be NULL
    lldb::UnwindPlanSP m_full_unwind_plan_sp;
    bool m_all_registers_available;               // Can we retrieve all regs or just nonvolatile regs?
    int m_frame_type;                             // enum FrameType

    lldb::addr_t m_cfa;
    lldb_private::Address m_start_pc;
    lldb_private::Address m_current_pc;

    int m_current_offset;                         // how far into the function we've executed; -1 if unknown
                                                  // 0 if no instructions have been executed yet.

    int m_current_offset_backed_up_one;           // how far into the function we've executed; -1 if unknown
                                                  // 0 if no instructions have been executed yet.
                                                  // On architectures where the return address on the stack points
                                                  // to the instruction after the CALL, this value will have 1 
                                                  // subtracted from it.  Else a function that ends in a CALL will
                                                  // have an offset pointing into the next function's address range.
                                                  // m_current_pc has the actual address of the "current" pc.

    lldb_private::SymbolContext& m_sym_ctx;
    bool m_sym_ctx_valid;                         // if ResolveSymbolContextForAddress fails, don't try to use m_sym_ctx

    uint32_t m_frame_number;                      // What stack frame level this frame is - used for debug logging

    std::map<uint32_t, RegisterLocation> m_registers; // where to find reg values for this frame

    //------------------------------------------------------------------
    // For RegisterContextLLDB only
    //------------------------------------------------------------------

    DISALLOW_COPY_AND_ASSIGN (RegisterContextLLDB);
};

#endif  // lldb_RegisterContextLLDB_h_
