//===-- lldb_EmulationStateARM.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_EmulationStateARM_h_
#define lldb_EmulationStateARM_h_

#include <map>

#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Opcode.h"

namespace lldb_private {

class EmulationStateARM {
public: 
    
    EmulationStateARM ();
    
    virtual
    ~EmulationStateARM ();
    
    bool
    StorePseudoRegisterValue (uint32_t reg_num, uint64_t value);
    
    uint64_t
    ReadPseudoRegisterValue (uint32_t reg_num, bool &success);
    
    bool
    StoreToPseudoAddress (lldb::addr_t p_address, uint64_t value, uint32_t size);
    
    uint32_t
    ReadFromPseudoAddress (lldb::addr_t p_address, uint32_t size, bool &success);
    
    void
    ClearPseudoRegisters ();
    
    void
    ClearPseudoMemory ();
    
    bool
    LoadPseudoRegistersFromFrame (StackFrame &frame);
    
    bool
    LoadState (FILE *test_file);
    
    static bool
    LoadRegisterStatesFromTestFile (FILE *test_file, EmulationStateARM &before_state, EmulationStateARM &after_state);
    
    bool
    CompareState (EmulationStateARM &other_state);

    static size_t
    ReadPseudoMemory (void *baton,
                      const EmulateInstruction::Context &context,
                      lldb::addr_t addr,
                      void *dst,
                      size_t length);
    
    static size_t
    WritePseudoMemory (void *baton,
                       const EmulateInstruction::Context &context,
                       lldb::addr_t addr,
                       const void *dst,
                       size_t length);
    
    static bool
    ReadPseudoRegister (void *baton,
                        uint32_t reg_kind,
                        uint32_t reg_num,
                        uint64_t &reg_value);
    
    static bool
    WritePseudoRegister (void *baton,
                         const EmulateInstruction::Context &context,
                         uint32_t reg_kind,
                         uint32_t reg_num,
                         uint64_t reg_value);
private:
    uint32_t m_gpr[17];
    struct sd_regs
    {
        union 
        {
            uint32_t s_reg[2];
            uint64_t d_reg;
        } sd_regs[16];  // sregs 0 - 31 & dregs 0 - 15
        
        uint64_t d_regs[16]; // dregs 16-31
 
    } m_vfp_regs;
    
    std::map<lldb::addr_t, uint32_t> m_memory; // Eventually will want to change uint32_t to a data buffer heap type.
    
    DISALLOW_COPY_AND_ASSIGN (EmulationStateARM);
};
 
}   // namespace lldb_private

#endif  // lldb_EmulationStateARM_h_
