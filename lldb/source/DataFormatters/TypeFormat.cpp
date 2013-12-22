//===-- TypeFormat.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

// C Includes

// C++ Includes

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

TypeFormatImpl::TypeFormatImpl (lldb::Format f,
                                const Flags& flags) :
m_flags(flags),
m_format (f)
{
}

bool
TypeFormatImpl::FormatObject (ValueObject *valobj,
                              std::string& dest) const
{
    if (!valobj)
        return false;
    if (valobj->GetClangType().IsAggregateType () == false)
    {
        const Value& value(valobj->GetValue());
        const Value::ContextType context_type = value.GetContextType();
        ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
        DataExtractor data;
        
        if (context_type == Value::eContextTypeRegisterInfo)
        {
            const RegisterInfo *reg_info = value.GetRegisterInfo();
            if (reg_info)
            {
                valobj->GetData(data);
                
                StreamString reg_sstr;
                data.Dump (&reg_sstr,
                           0,
                           GetFormat(),
                           reg_info->byte_size,
                           1,
                           UINT32_MAX,
                           LLDB_INVALID_ADDRESS,
                           0,
                           0,
                           exe_ctx.GetBestExecutionContextScope());
                dest.swap(reg_sstr.GetString());
            }
        }
        else
        {
            ClangASTType clang_type = valobj->GetClangType ();
            if (clang_type)
            {
                // put custom bytes to display in the DataExtractor to override the default value logic
                if (GetFormat() == eFormatCString)
                {
                    lldb_private::Flags type_flags(clang_type.GetTypeInfo(NULL)); // disambiguate w.r.t. TypeFormatImpl::Flags
                    if (type_flags.Test(ClangASTType::eTypeIsPointer) && !type_flags.Test(ClangASTType::eTypeIsObjC))
                    {
                        // if we are dumping a pointer as a c-string, get the pointee data as a string
                        TargetSP target_sp(valobj->GetTargetSP());
                        if (target_sp)
                        {
                            size_t max_len = target_sp->GetMaximumSizeOfStringSummary();
                            Error error;
                            DataBufferSP buffer_sp(new DataBufferHeap(max_len+1,0));
                            Address address(valobj->GetPointerValue());
                            if (target_sp->ReadCStringFromMemory(address, (char*)buffer_sp->GetBytes(), max_len, error) && error.Success())
                                data.SetData(buffer_sp);
                        }
                    }
                }
                else
                    valobj->GetData(data);
                
                StreamString sstr;
                clang_type.DumpTypeValue (&sstr,                         // The stream to use for display
                                          GetFormat(),                  // Format to display this type with
                                          data,                         // Data to extract from
                                          0,                             // Byte offset into "m_data"
                                          valobj->GetByteSize(),                 // Byte size of item in "m_data"
                                          valobj->GetBitfieldBitSize(),          // Bitfield bit size
                                          valobj->GetBitfieldBitOffset(),        // Bitfield bit offset
                                          exe_ctx.GetBestExecutionContextScope());
                // Given that we do not want to set the ValueObject's m_error
                // for a formatting error (or else we wouldn't be able to reformat
                // until a next update), an empty string is treated as a "false"
                // return from here, but that's about as severe as we get
                // ClangASTType::DumpTypeValue() should always return
                // something, even if that something is an error message
                if (sstr.GetString().empty())
                    dest.clear();
                else
                    dest.swap(sstr.GetString());
            }
        }
        return !dest.empty();
    }
    else
        return false;
}

std::string
TypeFormatImpl::GetDescription()
{
    StreamString sstr;
    sstr.Printf ("%s%s%s%s",
                 FormatManager::GetFormatAsCString (GetFormat()),
                 Cascades() ? "" : " (not cascading)",
                 SkipsPointers() ? " (skip pointers)" : "",
                 SkipsReferences() ? " (skip references)" : "");
    return sstr.GetString();
}

