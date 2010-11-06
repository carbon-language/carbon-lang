//===-- RecordingMemoryManager.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Expression/RecordingMemoryManager.h"

using namespace lldb_private;

RecordingMemoryManager::RecordingMemoryManager () :
    llvm::JITMemoryManager(),
    m_default_mm_ap (llvm::JITMemoryManager::CreateDefaultMemManager()),
    m_log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS))
{
}

RecordingMemoryManager::~RecordingMemoryManager ()
{
}

void
RecordingMemoryManager::setMemoryWritable ()
{
    m_default_mm_ap->setMemoryWritable();
}

void
RecordingMemoryManager::setMemoryExecutable ()
{
    m_default_mm_ap->setMemoryExecutable();
}


uint8_t *
RecordingMemoryManager::startFunctionBody(const llvm::Function *F,
                     uintptr_t &ActualSize)
{
    uint8_t *return_value = m_default_mm_ap->startFunctionBody(F, ActualSize);
    return return_value;
}

uint8_t *
RecordingMemoryManager::allocateStub(const llvm::GlobalValue* F, unsigned StubSize,
                unsigned Alignment)
{
    uint8_t *return_value = m_default_mm_ap->allocateStub(F, StubSize, Alignment);
    m_stubs.insert (std::pair<uint8_t *,unsigned>(return_value, StubSize));
    return return_value;
}

void
RecordingMemoryManager::endFunctionBody(const llvm::Function *F, uint8_t *FunctionStart,
               uint8_t *FunctionEnd)
{
    m_default_mm_ap->endFunctionBody(F, FunctionStart, FunctionEnd);
    if (m_log)
        m_log->Printf("Adding [%p-%p] to m_functions",
                      FunctionStart, FunctionEnd);
    m_functions.insert(std::pair<uint8_t *, uint8_t *>(FunctionStart, FunctionEnd));
}

uint8_t *
RecordingMemoryManager::allocateSpace(intptr_t Size, unsigned Alignment)
{
    uint8_t *return_value = m_default_mm_ap->allocateSpace(Size, Alignment);
    if (m_log)
        m_log->Printf("RecordingMemoryManager::allocateSpace(Size=0x%llx, Alignment=%u) = %p",
                      (uint64_t)Size, Alignment, return_value);
    m_spaceBlocks.insert (std::pair<uint8_t *, intptr_t>(return_value, Size));
    return return_value;
}

uint8_t *
RecordingMemoryManager::allocateGlobal(uintptr_t Size, unsigned Alignment)
{
    uint8_t *return_value = m_default_mm_ap->allocateGlobal(Size, Alignment);
    if (m_log)
        m_log->Printf("RecordingMemoryManager::allocateGlobal(Size=0x%llx, Alignment=%u) = %p",
                      (uint64_t)Size, Alignment, return_value);
    m_globals.insert (std::pair<uint8_t *, uintptr_t>(return_value, Size));
    return return_value;
}

void
RecordingMemoryManager::deallocateFunctionBody(void *Body)
{
    m_default_mm_ap->deallocateFunctionBody(Body);
}

uint8_t*
RecordingMemoryManager::startExceptionTable(const llvm::Function* F,
                       uintptr_t &ActualSize)
{
    uint8_t *return_value = m_default_mm_ap->startExceptionTable(F, ActualSize);
    return return_value;
}

void
RecordingMemoryManager::endExceptionTable(const llvm::Function *F, uint8_t *TableStart,
                 uint8_t *TableEnd, uint8_t* FrameRegister)
{
    m_default_mm_ap->endExceptionTable(F, TableStart, TableEnd, FrameRegister);
    m_exception_tables.insert (std::pair<uint8_t *, uint8_t *>(TableStart, TableEnd));
}

void
RecordingMemoryManager::deallocateExceptionTable(void *ET)
{
    m_default_mm_ap->deallocateExceptionTable (ET);
}

lldb::addr_t
RecordingMemoryManager::GetRemoteAddressForLocal (lldb::addr_t local_address)
{
    std::vector<LocalToRemoteAddressRange>::iterator pos, end = m_address_map.end();
    for (pos = m_address_map.begin(); pos < end; pos++)
    {
        lldb::addr_t lstart = (*pos).m_local_start;
        if (local_address >= lstart && local_address < lstart + (*pos).m_size)
        {
            return (*pos).m_remote_start + (local_address - lstart);
        }
    }
    return LLDB_INVALID_ADDRESS;
}

std::pair <lldb::addr_t, lldb::addr_t>
RecordingMemoryManager::GetRemoteRangeForLocal (lldb::addr_t local_address)
{
    std::vector<LocalToRemoteAddressRange>::iterator pos, end = m_address_map.end();
    
    for (pos = m_address_map.begin(); pos < end; ++pos)
    {
        lldb::addr_t lstart = pos->m_local_start;
        lldb::addr_t lend = lstart + pos->m_size;
        
        if (local_address >= lstart && local_address < lend)
        {
            return std::pair <lldb::addr_t, lldb::addr_t> (pos->m_remote_start, pos->m_remote_start + pos->m_size);
        }
    }
    
    return std::pair <lldb::addr_t, lldb::addr_t> (0, 0);
}

void
RecordingMemoryManager::AddToLocalToRemoteMap (lldb::addr_t lstart, size_t size, lldb::addr_t rstart)
{
    if (m_log)
        m_log->Printf("Adding local [0x%llx-0x%llx], remote [0x%llx-0x%llx] to local->remote map", lstart, lstart + size, rstart, rstart + size);
    
    m_address_map.push_back (LocalToRemoteAddressRange(lstart, size, rstart));
}

