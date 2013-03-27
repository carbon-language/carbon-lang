//===-- IRExecutionUnit.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_IRExecutionUnit_h_
#define lldb_IRExecutionUnit_h_

// C Includes
// C++ Includes
#include <atomic>
#include <string>
#include <vector>
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Host/Mutex.h"

namespace llvm {
    
class Module;
class ExecutionEngine;
    
}

namespace lldb_private {

class Error;
    
//----------------------------------------------------------------------
/// @class IRExecutionUnit IRExecutionUnit.h "lldb/Expression/IRExecutionUnit.h"
/// @brief Contains the IR and, optionally, JIT-compiled code for a module.
///
/// This class encapsulates the compiled version of an expression, in IR
/// form (for interpretation purposes) and in raw machine code form (for
/// execution in the target).
///
/// This object wraps an IR module that comes from the expression parser,
/// and knows how to use the JIT to make it into executable code.  It can
/// then be used as input to the IR interpreter, or the address of the
/// executable code can be passed to a thread plan to run in the target.
/// 
/// This class creates a subclass of LLVM's JITMemoryManager, because that is
/// how the JIT emits code.  Because LLDB needs to move JIT-compiled code
/// into the target process, the IRExecutionUnit knows how to copy the
/// emitted code into the target process.
//----------------------------------------------------------------------
class IRExecutionUnit 
{
public:
    //------------------------------------------------------------------
    /// Constructor
    //------------------------------------------------------------------
    IRExecutionUnit (std::auto_ptr<llvm::Module> &module_ap,
                     ConstString &name,
                     lldb::ProcessSP process_sp,
                     std::vector<std::string> &cpu_features);
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~IRExecutionUnit();
        
    llvm::Module *GetModule()
    {
        return m_module;
    }
    
    void GetRunnableInfo(Error &error,
                         lldb::addr_t &func_addr,
                         lldb::addr_t &func_end);
    
    //------------------------------------------------------------------
    /// Accessors for IRForTarget and other clients that may want binary
    /// data placed on their behalf.  The binary data is owned by the
    /// IRExecutionUnit unless the client explicitly chooses to free it.
    //------------------------------------------------------------------
    
    lldb::addr_t WriteNow(const uint8_t *bytes,
                          size_t size,
                          Error &error);
    
    void FreeNow(lldb::addr_t allocation);
    
private:
    //------------------------------------------------------------------
    /// Look up the object in m_address_map that contains a given address,
    /// find where it was copied to, and return the remote address at the
    /// same offset into the copied entity
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
    /// Look up the object in m_address_map that contains a given address,
    /// find where it was copied to, and return its address range in the
    /// target process
    ///
    /// @param[in] local_address
    ///     The address in the debugger.
    ///
    /// @return
    ///     The range of the containing object in the target process.
    //------------------------------------------------------------------
    typedef std::pair <lldb::addr_t, uintptr_t> AddrRange;
    AddrRange
    GetRemoteRangeForLocal (lldb::addr_t local_address);
    
    //------------------------------------------------------------------
    /// Commit all allocations to the process and record where they were stored.
    ///
    /// @param[in] process
    ///     The process to allocate memory in.
    ///
    /// @return
    ///     True <=> all allocations were performed successfully.
    ///     This method will attempt to free allocated memory if the
    ///     operation fails.
    //------------------------------------------------------------------
    bool
    CommitAllocations (lldb::ProcessSP &process_sp);
    
    //------------------------------------------------------------------
    /// Report all committed allocations to the execution engine.
    ///
    /// @param[in] engine
    ///     The execution engine to notify.
    //------------------------------------------------------------------
    void
    ReportAllocations (llvm::ExecutionEngine &engine);
    
    //------------------------------------------------------------------
    /// Write the contents of all allocations to the process. 
    ///
    /// @param[in] local_address
    ///     The process containing the allocations.
    ///
    /// @return
    ///     True <=> all allocations were performed successfully.
    //------------------------------------------------------------------
    bool
    WriteData (lldb::ProcessSP &process_sp);
    
    Error
    DisassembleFunction (Stream &stream,
                         lldb::ProcessSP &process_sp);

    class MemoryManager : public llvm::JITMemoryManager
    {
    public:
        MemoryManager (IRExecutionUnit &parent);
        
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
        /// Allocate space for executable code, and add it to the
        /// m_spaceBlocks map
        ///
        /// @param[in] Size
        ///     The size of the area.
        ///
        /// @param[in] Alignment
        ///     The required alignment of the area.
        ///
        /// @param[in] SectionID
        ///     A unique identifier for the section.
        ///
        /// @return
        ///     Allocated space.
        //------------------------------------------------------------------
        virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                             unsigned SectionID);
        
        //------------------------------------------------------------------
        /// Allocate space for data, and add it to the m_spaceBlocks map
        ///
        /// @param[in] Size
        ///     The size of the area.
        ///
        /// @param[in] Alignment
        ///     The required alignment of the area.
        ///
        /// @param[in] SectionID
        ///     A unique identifier for the section.
        ///
        /// @param[in] IsReadOnly
        ///     Flag indicating the section is read-only.
        ///
        /// @return
        ///     Allocated space.
        //------------------------------------------------------------------
        virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                             unsigned SectionID, bool IsReadOnly);
        
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
        /// Called when object loading is complete and section page
        /// permissions can be applied. Currently unimplemented for LLDB.
        ///
        /// @param[out] ErrMsg
        ///     The error that prevented the page protection from succeeding.
        ///
        /// @return
        ///     True in case of failure, false in case of success.
        //------------------------------------------------------------------
        bool applyPermissions(std::string *ErrMsg) { return false; }
        
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
        /// Passthrough interface stub
        //------------------------------------------------------------------
        virtual void *getPointerToNamedFunction(const std::string &Name,
                                                bool AbortOnFailure = true) {
            return m_default_mm_ap->getPointerToNamedFunction(Name, AbortOnFailure);
        }
    private:
        std::auto_ptr<JITMemoryManager>     m_default_mm_ap;    ///< The memory allocator to use in actually creating space.  All calls are passed through to it.
        IRExecutionUnit                    &m_parent;           ///< The execution unit this is a proxy for.
    };
    
    //----------------------------------------------------------------------
    /// @class Allocation IRExecutionUnit.h "lldb/Expression/IRExecutionUnit.h"
    /// @brief A record of a region that has been allocated by the JIT.
    ///
    /// The IRExecutionUnit makes records of all regions that need copying;
    /// upon requests, it allocates and 
    //----------------------------------------------------------------------
    struct Allocation
    {
        lldb::addr_t    m_remote_allocation;///< The (unaligned) base for the remote allocation
        lldb::addr_t    m_remote_start;     ///< The base address of the remote allocation
        uintptr_t       m_local_start;      ///< The base address of the local allocation
        uintptr_t       m_size;             ///< The size of the allocation
        unsigned        m_section_id;       ///< The ID of the section
        unsigned        m_alignment;        ///< The required alignment for the allocation
        bool            m_executable;       ///< True <=> the allocation must be executable in the target
        bool            m_allocated;        ///< True <=> the allocation has been propagated to the target

        std::unique_ptr<DataBufferHeap> m_data;   ///< If non-NULL, a local data buffer containing the written bytes.  Only populated by WriteNow.
        
        static const unsigned eSectionIDNone = (unsigned)-1;
        
        //------------------------------------------------------------------
        /// Constructor
        //------------------------------------------------------------------
        Allocation () :
            m_remote_allocation(0),
            m_remote_start(0),
            m_local_start(0),
            m_size(0),
            m_section_id(eSectionIDNone),
            m_alignment(0),
            m_executable(false),
            m_allocated(false)
        {
        }
        
        void dump (Log *log);
    };
    
    //----------------------------------------------------------------------
    /// @class JittedFunction ClangExpressionParser.h "lldb/Expression/ClangExpressionParser.h"
    /// @brief Encapsulates a single function that has been generated by the JIT.
    ///
    /// Functions that have been generated by the JIT are first resident in the
    /// local process, and then placed in the target process.  JittedFunction
    /// represents a function possibly resident in both.
    //----------------------------------------------------------------------
    struct JittedFunction {
        std::string m_name;             ///< The function's name
        lldb::addr_t m_local_addr;      ///< The address of the function in LLDB's memory
        lldb::addr_t m_remote_addr;     ///< The address of the function in the target's memory
        
        //------------------------------------------------------------------
        /// Constructor
        ///
        /// Initializes class variabes.
        ///
        /// @param[in] name
        ///     The name of the function.
        ///
        /// @param[in] local_addr
        ///     The address of the function in LLDB, or LLDB_INVALID_ADDRESS if
        ///     it is not present in LLDB's memory.
        ///
        /// @param[in] remote_addr
        ///     The address of the function in the target, or LLDB_INVALID_ADDRESS
        ///     if it is not present in the target's memory.
        //------------------------------------------------------------------
        JittedFunction (const char *name,
                        lldb::addr_t local_addr = LLDB_INVALID_ADDRESS,
                        lldb::addr_t remote_addr = LLDB_INVALID_ADDRESS) :
            m_name (name),
            m_local_addr (local_addr),
            m_remote_addr (remote_addr)
        {
        }
    };

    lldb::ProcessWP                         m_process_wp;
    typedef std::vector<Allocation>         AllocationList;
    AllocationList                          m_allocations;          ///< The base address of the remote allocation
    std::auto_ptr<llvm::ExecutionEngine>    m_execution_engine_ap;
    std::auto_ptr<llvm::Module>             m_module_ap;            ///< Holder for the module until it's been handed off
    llvm::Module                           *m_module;               ///< Owned by the execution engine
    std::vector<std::string>                m_cpu_features;
    llvm::SmallVector<JittedFunction, 1>    m_jitted_functions;     ///< A vector of all functions that have been JITted into machine code
    const ConstString                       m_name;
    
    std::atomic<bool>                       m_did_jit;
    Mutex                                   m_jit_mutex;

    lldb::addr_t                            m_function_load_addr;
    lldb::addr_t                            m_function_end_load_addr;
};

} // namespace lldb_private

#endif  // lldb_IRExecutionUnit_h_
