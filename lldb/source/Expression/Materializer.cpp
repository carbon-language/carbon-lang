//===-- Materializer.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Log.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb_private;

uint32_t
Materializer::AddStructMember (Entity &entity)
{
    uint32_t size = entity.GetSize();
    uint32_t alignment = entity.GetAlignment();
    
    uint32_t ret;
    
    if (m_current_offset == 0)
        m_struct_alignment = alignment;
    
    if (m_current_offset % alignment)
        m_current_offset += (alignment - (m_current_offset % alignment));
    
    ret = m_current_offset;
    
    m_current_offset += size;
    
    return ret;
}

void
Materializer::Entity::SetSizeAndAlignmentFromType (ClangASTType &type)
{
    m_size = type.GetTypeByteSize();
    
    uint32_t bit_alignment = type.GetTypeBitAlign();
    
    if (bit_alignment % 8)
    {
        bit_alignment += 8;
        bit_alignment &= ~((uint32_t)0x111u);
    }
    
    m_alignment = bit_alignment / 8;
}

class EntityPersistentVariable : public Materializer::Entity
{
public:
    EntityPersistentVariable (lldb::ClangExpressionVariableSP &persistent_variable_sp) :
        Entity(),
        m_persistent_variable_sp(persistent_variable_sp)
    {
        ClangASTType type(m_persistent_variable_sp->GetClangAST(),
                          m_persistent_variable_sp->GetClangType());

        SetSizeAndAlignmentFromType(type);
    }
    
    void MakeAllocation (IRMemoryMap &map, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        // Allocate a spare memory area to store the persistent variable's contents.
        
        Error allocate_error;
        
        lldb::addr_t mem = map.Malloc(m_persistent_variable_sp->GetByteSize(),
                                      8,
                                      lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                      IRMemoryMap::eAllocationPolicyMirror,
                                      allocate_error);
        
        if (!allocate_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't allocate a memory area to store %s: %s", m_persistent_variable_sp->GetName().GetCString(), allocate_error.AsCString());
            return;
        }
        
        if (log)
            log->Printf("Allocated %s (0x%" PRIx64 ") sucessfully", m_persistent_variable_sp->GetName().GetCString(), mem);
        
        // Put the location of the spare memory into the live data of the ValueObject.
                
        m_persistent_variable_sp->m_live_sp = ValueObjectConstResult::Create (map.GetBestExecutionContextScope(),
                                                                              m_persistent_variable_sp->GetTypeFromUser().GetASTContext(),
                                                                              m_persistent_variable_sp->GetTypeFromUser().GetOpaqueQualType(),
                                                                              m_persistent_variable_sp->GetName(),
                                                                              mem,
                                                                              eAddressTypeLoad,
                                                                              m_persistent_variable_sp->GetByteSize());
        
        // Clear the flag if the variable will never be deallocated.
        
        if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVKeepInTarget)
            m_persistent_variable_sp->m_flags &= ~ClangExpressionVariable::EVNeedsAllocation;
        
        // Write the contents of the variable to the area.
        
        Error write_error;
        
        map.WriteMemory (mem,
                         m_persistent_variable_sp->GetValueBytes(),
                         m_persistent_variable_sp->GetByteSize(),
                         write_error);
        
        if (!write_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat ("Couldn't write %s to the target: %s", m_persistent_variable_sp->GetName().AsCString(),
                                          write_error.AsCString());
            return;
        }
    }
    
    void DestroyAllocation (IRMemoryMap &map, Error &err)
    {
        Error deallocate_error;
        
        map.Free((lldb::addr_t)m_persistent_variable_sp->m_live_sp->GetValue().GetScalar().ULongLong(), deallocate_error);
            
        if (!deallocate_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat ("Couldn't deallocate memory for %s: %s", m_persistent_variable_sp->GetName().GetCString(), deallocate_error.AsCString());
        }
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("EntityPersistentVariable::Materialize [process_address = 0x%llx, m_name = %s, m_flags = 0x%hx]",
                        (uint64_t)process_address,
                        m_persistent_variable_sp->GetName().AsCString(),
                        m_persistent_variable_sp->m_flags);
        }
        
        if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVNeedsAllocation)
        {
            MakeAllocation(map, err);
            if (!err.Success())
                return;
        }
        
        if ((m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsProgramReference && m_persistent_variable_sp->m_live_sp) ||
            m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsLLDBAllocated)
        {
            Error write_error;
            
            map.WriteScalarToMemory(process_address + m_offset,
                                    m_persistent_variable_sp->m_live_sp->GetValue().GetScalar(),
                                    m_persistent_variable_sp->m_live_sp->GetProcessSP()->GetAddressByteSize(),
                                    write_error);
            
            if (!write_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the location of %s to memory: %s", m_persistent_variable_sp->GetName().AsCString(), write_error.AsCString());
            }
        }
        else
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("No materialization happened for persistent variable %s", m_persistent_variable_sp->GetName().AsCString());
            return;
        }
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
        {
            log->Printf("EntityPersistentVariable::Dematerialize [process_address = 0x%llx, m_name = %s, m_flags = 0x%hx]",
                        (uint64_t)process_address,
                        m_persistent_variable_sp->GetName().AsCString(),
                        m_persistent_variable_sp->m_flags);
        }
        
        if ((m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsLLDBAllocated) ||
            (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsProgramReference))
        {
            if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsProgramReference &&
                !m_persistent_variable_sp->m_live_sp)
            {
                // If the reference comes from the program, then the ClangExpressionVariable's
                // live variable data hasn't been set up yet.  Do this now.
                
                Scalar location_scalar;
                Error read_error;
                
                map.ReadScalarFromMemory(location_scalar, process_address + m_offset, map.GetAddressByteSize(), read_error);
                                
                if (!read_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't read the address of program-allocated variable %s: %s", m_persistent_variable_sp->GetName().GetCString(), read_error.AsCString());
                    return;
                }
                
                lldb::addr_t location = location_scalar.ULongLong();
                
                m_persistent_variable_sp->m_live_sp = ValueObjectConstResult::Create (map.GetBestExecutionContextScope (),
                                                                                      m_persistent_variable_sp->GetTypeFromUser().GetASTContext(),
                                                                                      m_persistent_variable_sp->GetTypeFromUser().GetOpaqueQualType(),
                                                                                      m_persistent_variable_sp->GetName(),
                                                                                      location,
                                                                                      eAddressTypeLoad,
                                                                                      m_persistent_variable_sp->GetByteSize());
                
                if (frame_top != LLDB_INVALID_ADDRESS &&
                    frame_bottom != LLDB_INVALID_ADDRESS &&
                    location >= frame_bottom &&
                    location <= frame_top)
                {
                    // If the variable is resident in the stack frame created by the expression,
                    // then it cannot be relied upon to stay around.  We treat it as needing
                    // reallocation.
                    m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
                    m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
                    m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsFreezeDry;
                    m_persistent_variable_sp->m_flags &= ~ClangExpressionVariable::EVIsProgramReference;
                }
            }
            
            lldb::addr_t mem = m_persistent_variable_sp->m_live_sp->GetValue().GetScalar().ULongLong();
            
            if (!m_persistent_variable_sp->m_live_sp)
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't find the memory area used to store %s", m_persistent_variable_sp->GetName().GetCString());
                return;
            }
            
            if (m_persistent_variable_sp->m_live_sp->GetValue().GetValueAddressType() != eAddressTypeLoad)
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("The address of the memory area for %s is in an incorrect format", m_persistent_variable_sp->GetName().GetCString());
                return;
            }
            
            if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVNeedsFreezeDry ||
                m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVKeepInTarget)
            {                
                if (log)
                    log->Printf("Dematerializing %s from 0x%" PRIx64 " (size = %llu)", m_persistent_variable_sp->GetName().GetCString(), (uint64_t)mem, (unsigned long long)m_persistent_variable_sp->GetByteSize());
                
                // Read the contents of the spare memory area
                
                m_persistent_variable_sp->ValueUpdated ();
                
                Error read_error;
                
                map.ReadMemory(m_persistent_variable_sp->GetValueBytes(),
                               mem,
                               m_persistent_variable_sp->GetByteSize(),
                               read_error);
                
                if (!read_error.Success())
                {
                    err.SetErrorStringWithFormat ("Couldn't read the contents of %s from memory: %s", m_persistent_variable_sp->GetName().GetCString(), read_error.AsCString());
                    return;
                }
                    
                m_persistent_variable_sp->m_flags &= ~ClangExpressionVariable::EVNeedsFreezeDry;
            }
        }
        else
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("No dematerialization happened for persistent variable %s", m_persistent_variable_sp->GetName().AsCString());
            return;
        }
        
        if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVNeedsAllocation &&
            !(m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVKeepInTarget))
        {
            DestroyAllocation(map, err);
            if (!err.Success())
                return;
        }
    }
private:
    lldb::ClangExpressionVariableSP m_persistent_variable_sp;
};

uint32_t
Materializer::AddPersistentVariable (lldb::ClangExpressionVariableSP &persistent_variable_sp, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityPersistentVariable (persistent_variable_sp));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntityVariable : public Materializer::Entity
{
public:
    EntityVariable (lldb::VariableSP &variable_sp) :
        Entity(),
        m_variable_sp(variable_sp),
        m_is_reference(false),
        m_temporary_allocation(LLDB_INVALID_ADDRESS)
    {
        // Hard-coding to maximum size of a pointer since all variables are materialized by reference
        m_size = 8;
        m_alignment = 8;
        m_is_reference = ClangASTContext::IsReferenceType(m_variable_sp->GetType()->GetClangForwardType());
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
        {
            log->Printf("EntityVariable::Materialize [process_address = 0x%llx, m_variable_sp = %s]",
                        (uint64_t)process_address,
                        m_variable_sp->GetName().AsCString());
        }
        
        lldb::ValueObjectSP valobj_sp = ValueObjectVariable::Create(frame_sp.get(), m_variable_sp);
        
        if (!valobj_sp)
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't get a value object for variable %s", m_variable_sp->GetName().AsCString());
            return;
        }
        
        if (m_is_reference)
        {
            DataExtractor valobj_extractor;
            valobj_sp->GetData(valobj_extractor);
            lldb::offset_t offset = 0;
            lldb::addr_t reference_addr = valobj_extractor.GetAddress(&offset);
            
            Error write_error;
            map.WritePointerToMemory(process_address + m_offset, reference_addr, write_error);
            
            if (!write_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the contents of reference variable %s to memory: %s", m_variable_sp->GetName().AsCString(), write_error.AsCString());
                return;
            }
        }
        else
        {
            Error get_address_error;
            lldb::ValueObjectSP addr_of_valobj_sp = valobj_sp->AddressOf(get_address_error);
            if (get_address_error.Success())
            {
                DataExtractor valobj_extractor;
                addr_of_valobj_sp->GetData(valobj_extractor);
                lldb::offset_t offset = 0;
                lldb::addr_t addr_of_valobj_addr = valobj_extractor.GetAddress(&offset);
                
                Error write_error;
                map.WritePointerToMemory(process_address + m_offset, addr_of_valobj_addr, write_error);
                
                if (!write_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't write the address of variable %s to memory: %s", m_variable_sp->GetName().AsCString(), write_error.AsCString());
                    return;
                }
            }
            else
            {
                DataExtractor data;
                valobj_sp->GetData(data);
                
                if (m_temporary_allocation == LLDB_INVALID_ADDRESS)
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Trying to create a temporary region for %s but one exists", m_variable_sp->GetName().AsCString());
                    return;
                }
                
                if (data.GetByteSize() != m_variable_sp->GetType()->GetByteSize())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Size of variable %s disagrees with the ValueObject's size", m_variable_sp->GetName().AsCString());
                    return;
                }
                
                size_t bit_align = ClangASTType::GetTypeBitAlign(m_variable_sp->GetType()->GetClangAST(), m_variable_sp->GetType()->GetClangLayoutType());
                size_t byte_align = (bit_align + 7) / 8;
                
                Error alloc_error;
                
                m_temporary_allocation = map.Malloc(data.GetByteSize(), byte_align, lldb::ePermissionsReadable | lldb::ePermissionsWritable, IRMemoryMap::eAllocationPolicyMirror, alloc_error);
                
                if (!alloc_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't allocate a temporary region for %s: %s", m_variable_sp->GetName().AsCString(), alloc_error.AsCString());
                    return;
                }
                
                Error write_error;
                
                map.WriteMemory(m_temporary_allocation, data.GetDataStart(), data.GetByteSize(), write_error);
                
                if (!write_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't write to the temporary region for %s: %s", m_variable_sp->GetName().AsCString(), write_error.AsCString());
                    return;
                }
                
                Error pointer_write_error;
                
                map.WritePointerToMemory(process_address + m_offset, m_temporary_allocation, pointer_write_error);
                
                if (!pointer_write_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't write the address of the temporary region for %s: %s", m_variable_sp->GetName().AsCString(), pointer_write_error.AsCString());
                }
            }
        }
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("EntityVariable::Dematerialize [process_address = 0x%llx, m_variable_sp = %s]",
                        (uint64_t)process_address,
                        m_variable_sp->GetName().AsCString());
        }
        
        if (m_temporary_allocation != LLDB_INVALID_ADDRESS)
        {
            lldb::ValueObjectSP valobj_sp = ValueObjectVariable::Create(frame_sp.get(), m_variable_sp);
            
            if (!valobj_sp)
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't get a value object for variable %s", m_variable_sp->GetName().AsCString());
                return;
            }
            
            lldb_private::DataExtractor data;
            
            Error extract_error;
            
            map.GetMemoryData(data, m_temporary_allocation, valobj_sp->GetByteSize(), extract_error);
            
            if (!extract_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't get the data for variable %s", m_variable_sp->GetName().AsCString());
                return;
            }
            
            Error set_error;
            
            valobj_sp->SetData(data, set_error);
            
            if (!set_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the new contents of %s back into the variable", m_variable_sp->GetName().AsCString());
                return;
            }
            
            Error free_error;
            
            map.Free(m_temporary_allocation, free_error);
            
            if (!free_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn'tfree the temporary region for %s: %s", m_variable_sp->GetName().AsCString(), free_error.AsCString());
                return;
            }
            
            m_temporary_allocation = LLDB_INVALID_ADDRESS;
        }
    }
private:
    lldb::VariableSP    m_variable_sp;
    bool                m_is_reference;
    lldb::addr_t        m_temporary_allocation;
};

uint32_t
Materializer::AddVariable (lldb::VariableSP &variable_sp, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityVariable (variable_sp));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntityResultVariable : public Materializer::Entity
{
public:
    EntityResultVariable (const ClangASTType &type, bool keep_in_memory) :
        Entity(),
        m_type(type),
        m_keep_in_memory(keep_in_memory)
    {
        SetSizeAndAlignmentFromType(m_type);
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
    }
private:
    ClangASTType    m_type;
    bool            m_keep_in_memory;
};

uint32_t
Materializer::AddResultVariable (const ClangASTType &type, bool keep_in_memory, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityResultVariable (type, keep_in_memory));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntitySymbol : public Materializer::Entity
{
public:
    EntitySymbol (const Symbol &symbol) :
        Entity(),
        m_symbol(symbol)
    {
        // Hard-coding to maximum size of a symbol
        m_size = 8;
        m_alignment = 8;
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
    }
private:
    Symbol m_symbol;
};

uint32_t
Materializer::AddSymbol (const Symbol &symbol_sp, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntitySymbol (symbol_sp));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntityRegister : public Materializer::Entity
{
public:
    EntityRegister (const RegisterInfo &register_info) :
        Entity(),
        m_register_info(register_info)
    {
        // Hard-coding alignment conservatively
        m_size = m_register_info.byte_size;
        m_alignment = m_register_info.byte_size;
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
    }
private:
    RegisterInfo m_register_info;
};

uint32_t
Materializer::AddRegister (const RegisterInfo &register_info, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityRegister (register_info));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

Materializer::Materializer () :
    m_needs_dematerialize(Mutex::eMutexTypeNormal),
    m_current_offset(0),
    m_struct_alignment(8)
{
}


Materializer::Dematerializer
Materializer::Materialize (lldb::StackFrameSP &frame_sp, lldb::ClangExpressionVariableSP &result_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &error)
{
    for (EntityUP &entity_up : m_entities)
    {
        entity_up->Materialize(frame_sp, map, process_address, error);
        
        if (!error.Success())
            return Dematerializer (*this, frame_sp, map, LLDB_INVALID_ADDRESS);
    }
    
    m_needs_dematerialize.Lock();
    
    return Dematerializer (*this, frame_sp, map, process_address);
}

void
Materializer::Dematerializer::Dematerialize (Error &error, lldb::addr_t frame_top, lldb::addr_t frame_bottom)
{
    lldb::StackFrameSP frame_sp = m_frame_wp.lock();
    
    if (!frame_sp)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Couldn't dematerialize: frame is gone");
    }
    else
    {
        for (EntityUP &entity_up : m_materializer.m_entities)
        {
            entity_up->Dematerialize (frame_sp, m_map, m_process_address, frame_top, frame_bottom, error);
            
            if (!error.Success())
                break;
        }
    }
    
    m_materializer.m_needs_dematerialize.Unlock();
}
