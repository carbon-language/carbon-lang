//===-- RecordingMemoryManager.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RecordingMemoryManager_h_
#define lldb_RecordingMemoryManager_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Log.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionParser.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class RecordingMemoryManager RecordingMemoryManager.h "lldb/Expression/RecordingMemoryManager.h"
/// @brief Passthrough memory manager for the JIT that records what was allocated where
///
/// The LLVM JIT is built to compile code for execution in the current
/// process, so it needs to be able to allocate memory.  Because different
/// clients have different requirements for the locations of JIT compiled
/// code, the interface for allocating memory has been abstracted out and
/// can be implemented by any client.
///
/// LLDB, however, needs to move JIT-compiled code into the target process.
/// Because writing individual bytes of code hasn't been abstracted out of
/// the JIT, LLDB instead implements a custom memory allocator that records 
/// what regions have been allocated for code.  When JIT compilation is
/// complete, these regions are then copied as necessary into the target
/// process.
///
/// Ideally the memory manager would handle this copying, but this class has 
/// to be built without RTTI, which means it cannot include Process.h.  As a
/// result, ClangExpression::WriteJITCode() accesses the stored mappings 
/// directly.
//----------------------------------------------------------------------
class RecordingMemoryManager : public llvm::JITMemoryManager
{
friend Error ClangExpressionParser::MakeJIT (uint64_t &, 
                                             uint64_t&, 
                                             ExecutionContext &,
                                             lldb::ClangExpressionVariableSP *);

public:
    //------------------------------------------------------------------
    /// Constructor
    //------------------------------------------------------------------
    RecordingMemoryManager ();
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual ~RecordingMemoryManager();

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void setMemoryWritable ();

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void setMemoryExecutable ();

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void setPoisonMemory (bool poison)
    {
        m_default_mm_ap->setPoisonMemory (poison);
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void AllocateGOT()
    {
        m_default_mm_ap->AllocateGOT();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual uint8_t *getGOTBase() const
    {
        return m_default_mm_ap->getGOTBase();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual uint8_t *startFunctionBody(const llvm::Function *F,
                                       uintptr_t &ActualSize);

    //------------------------------------------------------------------
    /// Allocate room for a dyld stub for a lazy-referenced function,
    /// and add it to the m_stubs map
    ///
    /// @param[in] F
    ///     The function being referenced.
    ///
    /// @param[in] StubSize
    ///     The size of the stub.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the stub.
    ///
    /// @return
    ///     Allocated space for the stub.
    //------------------------------------------------------------------
    virtual uint8_t *allocateStub(const llvm::GlobalValue* F, 
                                  unsigned StubSize,
                                  unsigned Alignment);

    //------------------------------------------------------------------
    /// Complete the body of a function, and add it to the m_functions map
    ///
    /// @param[in] F
    ///     The function being completed.
    ///
    /// @param[in] FunctionStart
    ///     The first instruction of the function.
    ///
    /// @param[in] FunctionEnd
    ///     The last byte of the last instruction of the function.
    //------------------------------------------------------------------
    virtual void endFunctionBody(const llvm::Function *F, 
                                 uint8_t *FunctionStart,
                                 uint8_t *FunctionEnd);
    //------------------------------------------------------------------
    /// Allocate space for an unspecified purpose, and add it to the
    /// m_spaceBlocks map
    ///
    /// @param[in] Size
    ///     The size of the area.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the area.
    ///
    /// @return
    ///     Allocated space.
    //------------------------------------------------------------------
    virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment);

    //------------------------------------------------------------------
    /// Allocate space for a global variable, and add it to the
    /// m_spaceBlocks map
    ///
    /// @param[in] Size
    ///     The size of the variable.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the variable.
    ///
    /// @return
    ///     Allocated space for the global.
    //------------------------------------------------------------------
    virtual uint8_t *allocateGlobal(uintptr_t Size, 
                                    unsigned Alignment);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void deallocateFunctionBody(void *Body);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual uint8_t* startExceptionTable(const llvm::Function* F,
                                         uintptr_t &ActualSize);

    //------------------------------------------------------------------
    /// Complete the exception table for a function, and add it to the
    /// m_exception_tables map
    ///
    /// @param[in] F
    ///     The function whose exception table is being written.
    ///
    /// @param[in] TableStart
    ///     The first byte of the exception table.
    ///
    /// @param[in] TableEnd
    ///     The last byte of the exception table.
    ///
    /// @param[in] FrameRegister
    ///     I don't know what this does, but it's passed through.
    //------------------------------------------------------------------
    virtual void endExceptionTable(const llvm::Function *F, 
                                   uint8_t *TableStart,
                                   uint8_t *TableEnd, 
                                   uint8_t* FrameRegister);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void deallocateExceptionTable(void *ET);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual size_t GetDefaultCodeSlabSize() {
        return m_default_mm_ap->GetDefaultCodeSlabSize();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual size_t GetDefaultDataSlabSize() {
        return m_default_mm_ap->GetDefaultDataSlabSize();
    }

    virtual size_t GetDefaultStubSlabSize() {
        return m_default_mm_ap->GetDefaultStubSlabSize();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual unsigned GetNumCodeSlabs() {
        return m_default_mm_ap->GetNumCodeSlabs();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual unsigned GetNumDataSlabs() {
        return m_default_mm_ap->GetNumDataSlabs();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual unsigned GetNumStubSlabs() {
        return m_default_mm_ap->GetNumStubSlabs();
    }

    //------------------------------------------------------------------
    /// [Convenience method for ClangExpression] Look up the object in
    /// m_address_map that contains a given address, find where it was
    /// copied to, and return the remote address at the same offset into
    /// the copied entity
    ///
    /// @param[in] local_address
    ///     The address in the debugger.
    ///
    /// @return
    ///     The address in the target process.
    //------------------------------------------------------------------
    lldb::addr_t
    GetRemoteAddressForLocal (lldb::addr_t local_address);
    
    //------------------------------------------------------------------
    /// [Convenience method for ClangExpression] Look up the object in
    /// m_address_map that contains a given address, find where it was
    /// copied to, and return its address range in the target process
    ///
    /// @param[in] local_address
    ///     The address in the debugger.
    ///
    /// @return
    ///     The range of the containing object in the target process.
    //------------------------------------------------------------------
    std::pair <lldb::addr_t, lldb::addr_t>
    GetRemoteRangeForLocal (lldb::addr_t local_address);
private:
    std::auto_ptr<JITMemoryManager> m_default_mm_ap;    ///< The memory allocator to use in actually creating space.  All calls are passed through to it.
    std::map<uint8_t *, uint8_t *> m_functions;         ///< A map from function base addresses to their end addresses.
    std::map<uint8_t *, intptr_t> m_spaceBlocks;        ///< A map from the base addresses of generic allocations to their sizes.
    std::map<uint8_t *, unsigned> m_stubs;              ///< A map from the base addresses of stubs to their sizes.
    std::map<uint8_t *, uintptr_t> m_globals;           ///< A map from the base addresses of globals to their sizes.
    std::map<uint8_t *, uint8_t *> m_exception_tables;  ///< A map from the base addresses of exception tables to their end addresses.
    
    lldb::LogSP m_log; ///< The log to use when printing log messages.  May be NULL.

    //----------------------------------------------------------------------
    /// @class LocalToRemoteAddressRange RecordingMemoryManager.h "lldb/Expression/RecordingMemoryManager.h"
    /// @brief A record of an allocated region that has been copied into the target
    ///
    /// The RecordingMemoryManager makes records of all regions that need copying;
    /// then, ClangExpression copies these regions into the target.  It records
    /// what was copied where in records of type LocalToRemoteAddressRange.
    //----------------------------------------------------------------------
    struct LocalToRemoteAddressRange
    {
        lldb::addr_t m_local_start;     ///< The base address of the local allocation
        size_t       m_size;            ///< The size of the allocation
        lldb::addr_t m_remote_start;    ///< The base address of the remote allocation

        //------------------------------------------------------------------
        /// Constructor
        //------------------------------------------------------------------
        LocalToRemoteAddressRange (lldb::addr_t lstart, size_t size, lldb::addr_t rstart) :
            m_local_start (lstart),
            m_size (size),
            m_remote_start (rstart)
        {}

    };

    //------------------------------------------------------------------
    /// Add a range to the list of copied ranges.
    ///
    /// @param[in] lstart
    ///     The base address of the local allocation.
    /// 
    /// @param[in] size
    ///     The size of the allocation.
    ///
    /// @param[in] rstart
    ///     The base address of the remote allocation.
    //------------------------------------------------------------------
    void
    AddToLocalToRemoteMap (lldb::addr_t lstart, size_t size, lldb::addr_t rstart);

    std::vector<LocalToRemoteAddressRange> m_address_map;   ///< The base address of the remote allocation
};

} // namespace lldb_private

#endif  // lldb_RecordingMemoryManager_h_
