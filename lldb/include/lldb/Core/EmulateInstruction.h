//===-- EmulateInstruction.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_EmulateInstruction_h_
#define lldb_EmulateInstruction_h_

#include <string>

#include "lldb/lldb-public.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Interpreter/NamedOptionValue.h"

//----------------------------------------------------------------------
/// @class EmulateInstruction EmulateInstruction.h "lldb/Core/EmulateInstruction.h"
/// @brief A class that allows emulation of CPU opcodes.
///
/// This class is a plug-in interface that is accessed through the 
/// standard static FindPlugin function call in the EmulateInstruction
/// class. The FindPlugin takes a target triple and returns a new object
/// if there is a plug-in that supports the architecture and OS. Four
/// callbacks and a baton are provided. The four callbacks are read 
/// register, write register, read memory and write memory.
///
/// This class is currently designed for these main use cases:
/// - Auto generation of Call Frame Information (CFI) from assembly code
/// - Predicting single step breakpoint locations
/// - Emulating instructions for breakpoint traps
///
/// Objects can be asked to read an instruction which will cause a call
/// to the read register callback to get the PC, followed by a read 
/// memory call to read the opcode. If ReadInstruction () returns true, 
/// then a call to EmulateInstruction::EvaluateInstruction () can be 
/// made. At this point the EmulateInstruction subclass will use all of
/// the callbacks to emulate an instruction.
///
/// Clients that provide the callbacks can either do the read/write 
/// registers/memory to actually emulate the instruction on a real or
/// virtual CPU, or watch for the EmulateInstruction::Context which
/// is context for the read/write register/memory which explains why
/// the callback is being called. Examples of a context are:
/// "pushing register 3 onto the stack at offset -12", or "adjusting
/// stack pointer by -16". This extra context allows the generation of
/// CFI information from assembly code without having to actually do
/// the read/write register/memory.
///
/// Clients must be prepared that not all instructions for an 
/// Instruction Set Architecture (ISA) will be emulated. 
///
/// Subclasses at the very least should implement the instructions that
/// save and restore regiters onto the stack and adjustment to the stack
/// pointer. By just implementing a few instructions for an ISA that are
/// the typical prologue opcodes, you can then generate CFI using a 
/// class that will soon be available.
/// 
/// Implmenting all of the instructions that affect the PC can then
/// allow single step prediction support.
///
/// Implmenting all of the instructions allows for emulation of opcodes
/// for breakpoint traps and will pave the way for "thread centric"
/// debugging. The current debugging model is "process centric" where
/// all threads must be stopped when any thread is stopped since when
/// hitting software breakpoints once must disable the breakpoint by
/// restoring the original breakpoint opcde, single stepping and 
/// restoring the breakpoint trap. If all threads were allowed to run
/// then other threads could miss the breakpoint. 
///
/// This class centralizes the code that usually is done in separate 
/// code paths in a debugger (single step prediction, finding save
/// restore locations of registers for unwinding stack frame variables,
/// and emulating the intruction is just a bonus.
//----------------------------------------------------------------------

namespace lldb_private {

class EmulateInstruction :
    public PluginInterface
{
public:

    static EmulateInstruction*
    FindPlugin (const ArchSpec &arch, const char *plugin_name);

    enum ContextType
    {
        eContextInvalid = 0,
        // Read an instruciton opcode from memory
        eContextReadOpcode,
        
        // Usually used for writing a register value whose source value in an 
        // immediate
        eContextImmediate,

        // Exclusively used when saving a register to the stack as part of the 
        // prologue
        eContextPushRegisterOnStack,

        // Exclusively used when restoring a register off the stack as part of 
        // the epilogue
        eContextPopRegisterOffStack,

        // Add or subtract a value from the stack
        eContextAdjustStackPointer,
        
        // Add or subtract a value from a base address register (other than SP)
        eContextAdjustBaseRegister,

        // Add or subtract a value from the PC or store a value to the PC.
        eContextAdjustPC,
        
        // Used in WriteRegister callbacks to indicate where the 
        eContextRegisterPlusOffset,

        // Used in WriteMemory callback to indicate where the data came from
        eContextRegisterStore,
        
        eContextRegisterLoad,
        
        // Used when performing a PC-relative branch where the
        eContextRelativeBranchImmediate,

        // Used when performing an absolute branch where the
        eContextAbsoluteBranchRegister,

        // Used when performing a supervisor call to an operating system to
        // provide a service:
        eContextSupervisorCall,

        // Used when performing a MemU operation to read the PC-relative offset
        // from an address.
        eContextTableBranchReadMemory,
        
        // Used when random bits are written into a register
        eContextWriteRegisterRandomBits,
        
        // Used when random bits are written to memory
        eContextWriteMemoryRandomBits,
        
        eContextMultiplication,

        eContextAddition,
        
        eContextSubtraction,
        
        eContextAdvancePC,

        eContextReturnFromException
    };
    
    enum InfoType {
        eInfoTypeRegisterPlusOffset,
        eInfoTypeRegisterPlusIndirectOffset,
        eInfoTypeRegisterToRegisterPlusOffset,
        eInfoTypeRegisterToRegisterPlusIndirectOffset,
        eInfoTypeRegisterRegisterOperands,
        eInfoTypeOffset,
        eInfoTypeRegister,
        eInfoTypeImmediate,
        eInfoTypeImmediateSigned,
        eInfoTypeAddress,
        eInfoTypeModeAndImmediate,
        eInfoTypeModeAndImmediateSigned,
        eInfoTypeMode,
        eInfoTypeNoArgs
    } InfoType;
    
    struct Register
    {
        uint32_t kind;
        uint32_t num;
        

        void
        SetRegister (uint32_t reg_kind, uint32_t reg_num)
        {
            kind = reg_kind;
            num = reg_num;
        }
    };
    
    struct Context
    {
        ContextType type;
        enum InfoType info_type;
        union
        {
            struct RegisterPlusOffset 
            {
                Register reg;          // base register
                int64_t signed_offset; // signed offset added to base register
            } RegisterPlusOffset;
            
            struct RegisterPlusIndirectOffset 
            {
                Register base_reg;    // base register number
                Register offset_reg;  // offset register kind
            } RegisterPlusIndirectOffset;
            
            struct RegisterToRegisterPlusOffset 
            {
                Register data_reg;  // source/target register for data
                Register base_reg;  // base register for address calculation
                int64_t offset;     // offset for address calculation
            } RegisterToRegisterPlusOffset;
            
            struct RegisterToRegisterPlusIndirectOffset
            {
                Register base_reg;   // base register for address calculation
                Register offset_reg; // offset register for address calculation
                Register data_reg;   // source/target register for data
            } RegisterToRegisterPlusIndirectOffset;
            
            struct RegisterRegisterOperands
            {
                Register operand1;  // register containing first operand for binary op
                Register operand2;  // register containing second operand for binary op
            } RegisterRegisterOperands;
            
            int64_t signed_offset; // signed offset by which to adjust self (for registers only)
            
            Register reg;          // plain register
            
            uint64_t immediate;    // immediate value
            
            int64_t signed_immediate; // signed immediate value
            
            lldb::addr_t address;        // direct address
            
            struct ModeAndImmediate 
            {
                uint32_t mode;        // eModeARM or eModeThumb
                uint32_t data_value;  // immdiate data
            } ModeAndImmediate;
            
            struct ModeAndImmediateSigned 
            {
                uint32_t mode;             // eModeARM or eModeThumb
                int32_t signed_data_value; // signed immdiate data
            } ModeAndImmediateSigned;
            
            uint32_t mode;         // eModeARM or eModeThumb
                        
        } info;
        
        void 
        SetRegisterPlusOffset (Register base_reg,
                               int64_t signed_offset)
        {
            info_type = eInfoTypeRegisterPlusOffset;
            info.RegisterPlusOffset.reg = base_reg;
            info.RegisterPlusOffset.signed_offset = signed_offset;
        }

        void
        SetRegisterPlusIndirectOffset (Register base_reg,
                                       Register offset_reg)
        {
            info_type = eInfoTypeRegisterPlusIndirectOffset;
            info.RegisterPlusIndirectOffset.base_reg   = base_reg;
            info.RegisterPlusIndirectOffset.offset_reg = offset_reg;
        }
        
        void
        SetRegisterToRegisterPlusOffset (Register data_reg,
                                         Register base_reg,
                                         int64_t offset)
        {
            info_type = eInfoTypeRegisterToRegisterPlusOffset;
            info.RegisterToRegisterPlusOffset.data_reg = data_reg;
            info.RegisterToRegisterPlusOffset.base_reg = base_reg;
            info.RegisterToRegisterPlusOffset.offset   = offset;
        }
        
        void
        SetRegisterToRegisterPlusIndirectOffset (Register base_reg,
                                                 Register offset_reg,
                                                 Register data_reg)
        {
            info_type = eInfoTypeRegisterToRegisterPlusIndirectOffset;
            info.RegisterToRegisterPlusIndirectOffset.base_reg   = base_reg;
            info.RegisterToRegisterPlusIndirectOffset.offset_reg = offset_reg;
            info.RegisterToRegisterPlusIndirectOffset.data_reg   = data_reg;
        }
        
        void
        SetRegisterRegisterOperands (Register op1_reg,
                                     Register op2_reg)
        {
            info_type = eInfoTypeRegisterRegisterOperands;
            info.RegisterRegisterOperands.operand1 = op1_reg;
            info.RegisterRegisterOperands.operand2 = op2_reg;
        }
        
        void
        SetOffset (int64_t signed_offset)
        {
            info_type = eInfoTypeOffset;
            info.signed_offset = signed_offset;
        }
        
        void
        SetRegister (Register reg)
        {
            info_type = eInfoTypeRegister;
            info.reg = reg;
        }
        
        void
        SetImmediate (uint64_t immediate)
        {
            info_type = eInfoTypeImmediate;
            info.immediate = immediate;
        }
        
        void
        SetImmediateSigned (int64_t signed_immediate)
        {
            info_type = eInfoTypeImmediateSigned;
            info.signed_immediate = signed_immediate;
        }
        
        void
        SetAddress (lldb::addr_t address)
        {
            info_type = eInfoTypeAddress;
            info.address = address;
        }
        void
        SetModeAndImmediate (uint32_t mode, uint32_t data_value)
        {
            info_type = eInfoTypeModeAndImmediate;
            info.ModeAndImmediate.mode = mode;
            info.ModeAndImmediate.data_value = data_value;
        }
        
        void
        SetModeAndImmediateSigned (uint32_t mode, int32_t signed_data_value)
        {
            info_type = eInfoTypeModeAndImmediateSigned;
            info.ModeAndImmediateSigned.mode = mode;
            info.ModeAndImmediateSigned.signed_data_value = signed_data_value;
        }
        
        void
        SetMode (uint32_t mode)
        {
            info_type = eInfoTypeMode;
            info.mode = mode;
        }
        
        void
        SetNoArgs ()
        {
            info_type = eInfoTypeNoArgs;
        }
        
    };

    static void
    PrintContext (const char *context_type, const Context &context);
    
    typedef size_t (*ReadMemory) (void *baton,
                                  const Context &context, 
                                  lldb::addr_t addr, 
                                  void *dst,
                                  size_t length);
    
    typedef size_t (*WriteMemory) (void *baton,
                                   const Context &context, 
                                   lldb::addr_t addr, 
                                   const void *dst,
                                   size_t length);
    
    typedef bool   (*ReadRegister)  (void *baton,
                                     uint32_t reg_kind, 
                                     uint32_t reg_num,
                                     uint64_t &reg_value);

    typedef bool   (*WriteRegister) (void *baton,
                                     const Context &context, 
                                     uint32_t reg_kind, 
                                     uint32_t reg_num,
                                     uint64_t reg_value);

    EmulateInstruction (lldb::ByteOrder byte_order,
                        uint32_t addr_byte_size,
                        const ArchSpec &arch,
                        void *baton,
                        ReadMemory read_mem_callback,
                        WriteMemory write_mem_callback,
                        ReadRegister read_reg_callback,
                        WriteRegister write_reg_callback);
                        
    EmulateInstruction (lldb::ByteOrder byte_order,
                        uint32_t addr_byte_size,
                        const ArchSpec &arch);

    virtual ~EmulateInstruction()
    {
    }
    
    virtual bool
    SetTargetTriple (const ArchSpec &arch) = 0;
    
    virtual bool 
    ReadInstruction () = 0;

    virtual bool
    SetInstruction (const Opcode &insn_opcode, const Address &inst_addr) = 0;

    virtual bool
    EvaluateInstruction () = 0;
    
    virtual bool
    TestEmulation (Stream *out_stream, ArchSpec &arch, OptionValueDictionary *test_data) = 0;
    
    bool
    GetAdvancePC () { return m_advance_pc; }
    
    void
    SetAdvancePC (bool value) { m_advance_pc = value; }
    
    static void
    TranslateRegister (uint32_t reg_kind, uint32_t reg_num, std::string &reg_name);
    
    uint64_t
    ReadRegisterUnsigned (uint32_t reg_kind, 
                          uint32_t reg_num, 
                          uint64_t fail_value, 
                          bool *success_ptr);

    bool
    WriteRegisterUnsigned (const Context &context, 
                           uint32_t reg_kind, 
                           uint32_t reg_num, 
                           uint64_t reg_value);

    uint64_t
    ReadMemoryUnsigned (const Context &context, 
                        lldb::addr_t addr, 
                        size_t byte_size, 
                        uint64_t fail_value, 
                        bool *success_ptr);

    bool
    WriteMemoryUnsigned (const Context &context, 
                         lldb::addr_t addr, 
                         uint64_t uval,
                         size_t uval_byte_size);

    uint32_t
    GetAddressByteSize () const
    {
        return m_addr_byte_size;
    }

    lldb::ByteOrder
    GetByteOrder () const
    {
        return m_byte_order;
    }

    const Opcode &
    GetOpcode () const
    {
        return m_opcode;
    }


    static size_t 
    ReadMemoryFrame (void *baton,
                     const Context &context, 
                     lldb::addr_t addr, 
                     void *dst,
                     size_t length);
    
    static size_t 
    WriteMemoryFrame (void *baton,
                      const Context &context, 
                      lldb::addr_t addr, 
                      const void *dst,
                      size_t length);
    
    static bool   
    ReadRegisterFrame  (void *baton,
                        uint32_t reg_kind, 
                        uint32_t reg_num,
                        uint64_t &reg_value);
    
    
    static bool   
    WriteRegisterFrame (void *baton,
                        const Context &context, 
                        uint32_t reg_kind, 
                        uint32_t reg_num,
                        uint64_t reg_value);
                          
    static size_t 
    ReadMemoryDefault (void *baton,
                       const Context &context, 
                       lldb::addr_t addr, 
                       void *dst,
                       size_t length);
    
    static size_t 
    WriteMemoryDefault (void *baton,
                        const Context &context, 
                        lldb::addr_t addr, 
                        const void *dst,
                        size_t length);
    
    static bool   
    ReadRegisterDefault  (void *baton,
                          uint32_t reg_kind, 
                          uint32_t reg_num,
                          uint64_t &reg_value);
    
    
    static bool   
    WriteRegisterDefault (void *baton,
                          const Context &context, 
                          uint32_t reg_kind, 
                          uint32_t reg_num,
                          uint64_t reg_value);
   
    void
    SetBaton (void *baton);
    
    void
    SetCallbacks (ReadMemory read_mem_callback,
                  WriteMemory write_mem_callback,
                  ReadRegister read_reg_callback,
                  WriteRegister write_reg_callback);
                  
    void
    SetReadMemCallback (ReadMemory read_mem_callback);
    
    void
    SetWriteMemCallback (WriteMemory write_mem_callback);
    
    void
    SetReadRegCallback (ReadRegister read_reg_callback);
    
    void
    SetWriteRegCallback (WriteRegister write_reg_callback);
    

protected:
    lldb::ByteOrder     m_byte_order;
    uint32_t            m_addr_byte_size;
    ArchSpec            m_arch;
    void *              m_baton;
    ReadMemory          m_read_mem_callback;
    WriteMemory         m_write_mem_callback;
    ReadRegister        m_read_reg_callback;
    WriteRegister       m_write_reg_callback;
    lldb::addr_t        m_opcode_pc;
    Opcode              m_opcode;
    bool                m_advance_pc;
    //------------------------------------------------------------------
    // For EmulateInstruction only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (EmulateInstruction);
};

}   // namespace lldb_private

#endif  // lldb_EmulateInstruction_h_
