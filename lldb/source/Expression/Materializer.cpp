//===-- Materializer.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/ExecutionContext.h"

using namespace lldb_private;

uint32_t
Materializer::AddStructMember (Entity &entity)
{
    uint32_t size = entity.GetSize();
    uint32_t alignment = entity.GetAlignment();
    
    uint32_t ret;
    
    if (!m_current_offset)
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
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
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
        m_variable_sp(variable_sp)
    {
        Type *type = variable_sp->GetType();
        
        assert(type);
        
        if (type)
        {
            ClangASTType clang_type(type->GetClangAST(),
                                    type->GetClangLayoutType());
            
            SetSizeAndAlignmentFromType(clang_type);
        }
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
private:
    lldb::VariableSP m_variable_sp;
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
    EntityResultVariable (const ClangASTType &type) :
        Entity(),
        m_type(type)
    {
        SetSizeAndAlignmentFromType(m_type);
    }
    
    virtual void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
    }
private:
    ClangASTType m_type;
};

uint32_t
Materializer::AddResultVariable (const ClangASTType &type, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityResultVariable (type));
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
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
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
    
    virtual void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
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
Materializer::Dematerializer::Dematerialize (Error &error)
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
            entity_up->Dematerialize (frame_sp, m_map, m_process_address, error);
            
            if (!error.Success())
                break;
        }
    }
    
    m_materializer.m_needs_dematerialize.Unlock();
}
