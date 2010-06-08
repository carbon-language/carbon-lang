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
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginInterface.h"

namespace lldb_private {

class ExecutionContext;

class Disassembler :
    public PluginInterface
{
public:
    class Instruction
    {
    public:
        typedef lldb::SharedPtr<Instruction>::Type shared_ptr;

        Instruction();

        virtual
       ~Instruction();

        virtual size_t
        GetByteSize() const = 0;

        virtual void
        Dump (Stream *s, lldb::addr_t base_address, DataExtractor *bytes, uint32_t bytes_offset, const lldb_private::ExecutionContext exe_ctx, bool raw) = 0;

        virtual bool
        DoesBranch () const = 0;

        virtual size_t
        Extract (const DataExtractor& data, uint32_t data_offset) = 0;
    };


    class InstructionList
    {
    public:
        InstructionList();
        ~InstructionList();

        size_t
        GetSize() const;

        Instruction *
        GetInstructionAtIndex (uint32_t idx);

        const Instruction *
        GetInstructionAtIndex (uint32_t idx) const;

    void
    Clear();

    void
    AppendInstruction (Instruction::shared_ptr &inst_sp);

    private:
        typedef std::vector<Instruction::shared_ptr> collection;
        typedef collection::iterator iterator;
        typedef collection::const_iterator const_iterator;

        collection m_instructions;
    };


    static Disassembler*
    FindPlugin (const ArchSpec &arch);

    static bool
    Disassemble (const ArchSpec &arch,
                 const ExecutionContext &exe_ctx,
                 uint32_t mixed_context_lines,
                 Stream &strm);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    Disassembler(const ArchSpec &arch);
    virtual ~Disassembler();

    typedef const char * (*SummaryCallback)(const Instruction& inst, ExecutionContext *exe_context, void *user_data);

    size_t
    ParseInstructions (const ExecutionContext *exe_ctx,
                       lldb::AddressType addr_type,
                       lldb::addr_t addr,
                       size_t byte_size,
                       DataExtractor& data);

    virtual size_t
    ParseInstructions (const DataExtractor& data,
                       uint32_t data_offset,
                       uint32_t num_instructions,
                       lldb::addr_t base_addr) = 0;

    InstructionList &
    GetInstructionList ();

    const InstructionList &
    GetInstructionList () const;

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
