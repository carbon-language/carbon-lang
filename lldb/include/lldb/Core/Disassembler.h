//===-- Disassembler.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Disassembler_h_
#define liblldb_Disassembler_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Core/PluginInterface.h"

namespace lldb_private {

class Instruction
{
public:
    Instruction (const Address &address);

    virtual
   ~Instruction();

    const Address &
    GetAddress () const
    {
        return m_address;
    }

    void
    SetAddress (const Address &addr)
    {
        m_address = addr;
    }

    virtual void
    Dump (Stream *s,
          bool show_address,
          bool show_bytes,
          const ExecutionContext *exe_ctx, 
          bool raw) = 0;
    
    virtual bool
    DoesBranch () const = 0;

    virtual size_t
    Extract (const Disassembler &disassembler, 
             const DataExtractor& data, 
             uint32_t data_offset) = 0;

    const Opcode &
    GetOpcode () const
    {
        return m_opcode;
    }

protected:
    Address m_address; // The section offset address of this instruction
    Opcode m_opcode; // The opcode for this instruction
};


class InstructionList
{
public:
    InstructionList();
    ~InstructionList();

    size_t
    GetSize() const;

    lldb::InstructionSP
    GetInstructionAtIndex (uint32_t idx) const;

    void
    Clear();

    void
    Append (lldb::InstructionSP &inst_sp);

private:
    typedef std::vector<lldb::InstructionSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    collection m_instructions;
};

class Disassembler :
    public PluginInterface
{
public:


    static Disassembler*
    FindPlugin (const ArchSpec &arch, const char *plugin_name);

    static lldb::DisassemblerSP
    DisassembleRange (const ArchSpec &arch,
                      const char *plugin_name,
                      const ExecutionContext &exe_ctx,
                      const AddressRange &disasm_range);

    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const ExecutionContext &exe_ctx,
                 const AddressRange &range,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 bool show_bytes,
                 bool raw,
                 Stream &strm);

    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const ExecutionContext &exe_ctx,
                 const Address &start,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 bool show_bytes,
                 bool raw,
                 Stream &strm);

    static size_t
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const ExecutionContext &exe_ctx,
                 SymbolContextList &sc_list,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 bool show_bytes,
                 bool raw,
                 Stream &strm);
    
    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const ExecutionContext &exe_ctx,
                 const ConstString &name,
                 Module *module,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 bool show_bytes,
                 bool raw,
                 Stream &strm);

    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const ExecutionContext &exe_ctx,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 bool show_bytes,
                 bool raw,
                 Stream &strm);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    Disassembler(const ArchSpec &arch);
    virtual ~Disassembler();

    typedef const char * (*SummaryCallback)(const Instruction& inst, ExecutionContext *exe_context, void *user_data);

    static bool 
    PrintInstructions (Disassembler *disasm_ptr,
                       Debugger &debugger,
                       const ArchSpec &arch,
                       const ExecutionContext &exe_ctx,
                       const  Address &start_addr,
                       uint32_t num_instructions,
                       uint32_t num_mixed_context_lines,
                       bool show_bytes,
                       bool raw,
                       Stream &strm);
    
    size_t
    ParseInstructions (const ExecutionContext *exe_ctx,
                       const AddressRange &range,
                       DataExtractor& data);

    size_t
    ParseInstructions (const ExecutionContext *exe_ctx,
                       const Address &range,
                       uint32_t num_instructions,
                       DataExtractor& data);

    virtual size_t
    DecodeInstructions (const Address &base_addr,
                        const DataExtractor& data,
                        uint32_t data_offset,
                        uint32_t num_instructions,
                        bool append) = 0;
    
    InstructionList &
    GetInstructionList ();

    const InstructionList &
    GetInstructionList () const;

    const ArchSpec &
    GetArchitecture () const
    {
        return m_arch;
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from Disassembler can see and modify these
    //------------------------------------------------------------------
    const ArchSpec m_arch;
    InstructionList m_instruction_list;
    lldb::addr_t m_base_addr;

private:
    //------------------------------------------------------------------
    // For Disassembler only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (Disassembler);
};

} // namespace lldb_private

#endif  // liblldb_Disassembler_h_
