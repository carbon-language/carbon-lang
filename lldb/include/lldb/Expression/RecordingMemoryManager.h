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
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "lldb/Expression/ClangExpression.h"

namespace lldb_private {

class RecordingMemoryManager : public llvm::JITMemoryManager
{

// I can't write the JIT code in this class because this class has to be
// built without RTTI, which means I can't include Process.h.  But I don't
// want to write iterators or "do_over_regions" functions right now, so I'm
// just going to let the ClangExpression handle it using our data members directly.

friend bool ClangExpression::WriteJITCode (const ExecutionContext &exc_context);

public:
    RecordingMemoryManager ();
    virtual ~RecordingMemoryManager();

    virtual void setMemoryWritable ();

    virtual void setMemoryExecutable ();

    virtual void setPoisonMemory (bool poison)
    {
        m_default_mm_ap->setPoisonMemory (poison);
    }

    virtual void AllocateGOT()
    {
        m_default_mm_ap->AllocateGOT();
    }


    virtual uint8_t *getGOTBase() const
    {
        return m_default_mm_ap->getGOTBase();
    }

    virtual uint8_t *startFunctionBody(const llvm::Function *F,
                                 uintptr_t &ActualSize);

    virtual uint8_t *allocateStub(const llvm::GlobalValue* F, unsigned StubSize,
                            unsigned Alignment);

    virtual void endFunctionBody(const llvm::Function *F, uint8_t *FunctionStart,
                           uint8_t *FunctionEnd);

    virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment);

    virtual uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment);

    virtual void deallocateFunctionBody(void *Body);

    virtual uint8_t* startExceptionTable(const llvm::Function* F,
                                   uintptr_t &ActualSize);

    virtual void endExceptionTable(const llvm::Function *F, uint8_t *TableStart,
                             uint8_t *TableEnd, uint8_t* FrameRegister);

    virtual void deallocateExceptionTable(void *ET);

    virtual size_t GetDefaultCodeSlabSize() {
        return m_default_mm_ap->GetDefaultCodeSlabSize();
    }

    virtual size_t GetDefaultDataSlabSize() {
        return m_default_mm_ap->GetDefaultDataSlabSize();
    }

    virtual size_t GetDefaultStubSlabSize() {
        return m_default_mm_ap->GetDefaultStubSlabSize();
    }

    virtual unsigned GetNumCodeSlabs() {
        return m_default_mm_ap->GetNumCodeSlabs();
    }

    virtual unsigned GetNumDataSlabs() {
        return m_default_mm_ap->GetNumDataSlabs();
    }

    virtual unsigned GetNumStubSlabs() {
        return m_default_mm_ap->GetNumStubSlabs();
    }

    // These are methods I've added so we can transfer the memory we've remembered down
    // to the target program.  For now I'm assuming all this code is PIC without fixups,
    // so I'll just copy it blind, but if we need to we can do fixups later.

    lldb::addr_t
    GetRemoteAddressForLocal (lldb::addr_t local_address);

    bool
    WriteJITRegions (const ExecutionContext &exc_context);


private:
    std::auto_ptr<JITMemoryManager> m_default_mm_ap;
    std::map<uint8_t *, uint8_t *> m_functions;
    std::map<uint8_t *, intptr_t> m_spaceBlocks;
    std::map<uint8_t *, unsigned> m_stubs;
    std::map<uint8_t *, uintptr_t> m_globals;
    std::map<uint8_t *, uint8_t *> m_exception_tables;

    struct LocalToRemoteAddressRange
    {
        lldb::addr_t m_local_start;
        size_t       m_size;
        lldb::addr_t m_remote_start;

        LocalToRemoteAddressRange (lldb::addr_t lstart, size_t size, lldb::addr_t rstart) :
            m_local_start (lstart),
            m_size (size),
            m_remote_start (rstart)
        {}

    };

    void
    AddToLocalToRemoteMap (lldb::addr_t lstart, size_t size, lldb::addr_t rstart);

    // We should probably store this by address so we can efficiently
    // search it but there really won't be many elements in this array
    // at present.  So we can put that off for now.
    std::vector<LocalToRemoteAddressRange> m_address_map;

};

} // namespace lldb_private
#endif  // lldb_RecordingMemoryManager_h_
