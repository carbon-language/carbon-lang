//===-- NSSet.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "llvm/Support/ConvertUTF.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

template<bool cf_style>
bool
lldb_private::formatters::NSSetSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    bool is_64bit = (ptr_size == 8);
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"__NSSetI"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSSetM"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSCFSet"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + (is_64bit ? 20 : 12), ptr_size, 0, error);
        if (error.Fail())
            return false;
        if (is_64bit)
            value &= ~0x1fff000000000000UL;
    }
    else if (!strcmp(class_name,"NSCountedSet"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value = process_sp->ReadUnsignedIntegerFromMemory(value + (is_64bit ? 20 : 12), ptr_size, 0, error);
        if (error.Fail())
            return false;
        if (is_64bit)
            value &= ~0x1fff000000000000UL;
    }
    else
    {
        if (!ExtractValueFromObjCExpression(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("%s%" PRIu64 " %s%s",
                  (cf_style ? "@\"" : ""),
                  value,
                  (cf_style ? (value == 1 ? "value" : "values") : (value == 1 ? "object" : "objects")),
                  (cf_style ? "\"" : ""));
    return true;
}

template bool
lldb_private::formatters::NSSetSummaryProvider<true> (ValueObject& valobj, Stream& stream);

template bool
lldb_private::formatters::NSSetSummaryProvider<false> (ValueObject& valobj, Stream& stream);
