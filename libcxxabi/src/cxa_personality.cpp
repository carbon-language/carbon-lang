//===------------------------- cxa_exception.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//  
//  This file implements the "Exception Handling APIs"
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html
//  http://www.intel.com/design/itanium/downloads/245358.htm
//  
//===----------------------------------------------------------------------===//

#include "unwind.h"
#include "cxa_exception.hpp"
#include <typeinfo>
#include <stdlib.h>
#include <assert.h>

namespace __cxxabiv1
{

extern "C"
{

// private API

// Heavily borrowed from llvm/examples/ExceptionDemo/ExceptionDemo.cpp

// DWARF Constants
enum
{
    DW_EH_PE_absptr   = 0x00,
    DW_EH_PE_uleb128  = 0x01,
    DW_EH_PE_udata2   = 0x02,
    DW_EH_PE_udata4   = 0x03,
    DW_EH_PE_udata8   = 0x04,
    DW_EH_PE_sleb128  = 0x09,
    DW_EH_PE_sdata2   = 0x0A,
    DW_EH_PE_sdata4   = 0x0B,
    DW_EH_PE_sdata8   = 0x0C,
    DW_EH_PE_pcrel    = 0x10,
    DW_EH_PE_textrel  = 0x20,
    DW_EH_PE_datarel  = 0x30,
    DW_EH_PE_funcrel  = 0x40,
    DW_EH_PE_aligned  = 0x50,
    DW_EH_PE_indirect = 0x80,
    DW_EH_PE_omit     = 0xFF
};

/// Read a uleb128 encoded value and advance pointer 
/// See Variable Length Data Appendix C in: 
/// @link http://dwarfstd.org/Dwarf4.pdf @unlink
/// @param data reference variable holding memory pointer to decode from
/// @returns decoded value
static
uintptr_t
readULEB128(const uint8_t** data)
{
    uintptr_t result = 0;
    uintptr_t shift = 0;
    unsigned char byte;
    const uint8_t *p = *data;
    do
    {
        byte = *p++;
        result |= static_cast<uintptr_t>(byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    *data = p;
    return result;
}

/// Read a sleb128 encoded value and advance pointer 
/// See Variable Length Data Applendix C in: 
/// @link http://dwarfstd.org/Dwarf4.pdf @unlink
/// @param data reference variable holding memory pointer to decode from
/// @returns decoded value
static
uintptr_t
readSLEB128(const uint8_t** data)
{
    uintptr_t result = 0;
    uintptr_t shift = 0;
    unsigned char byte;
    const uint8_t *p = *data;
    do
    {
        byte = *p++;
        result |= static_cast<uintptr_t>(byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    *data = p;
    if ((byte & 0x40) && (shift < (sizeof(result) << 3)))
        result |= static_cast<uintptr_t>(~0) << shift;
    return result;
}

/// Read a pointer encoded value and advance pointer 
/// See Variable Length Data in: 
/// @link http://dwarfstd.org/Dwarf3.pdf @unlink
/// @param data reference variable holding memory pointer to decode from
/// @param encoding dwarf encoding type
/// @returns decoded value
static
uintptr_t
readEncodedPointer(const uint8_t** data, uint8_t encoding)
{
    uintptr_t result = 0;
    const uint8_t* p = *data;
    if (encoding == DW_EH_PE_omit) 
        return result;
    // first get value 
    switch (encoding & 0x0F)
    {
    case DW_EH_PE_absptr:
        result = *((uintptr_t*)p);
        p += sizeof(uintptr_t);
        break;
    case DW_EH_PE_uleb128:
        result = readULEB128(&p);
        break;
    case DW_EH_PE_sleb128:
        result = readSLEB128(&p);
        break;
    case DW_EH_PE_udata2:
        result = *((uint16_t*)p);
        p += sizeof(uint16_t);
        break;
    case DW_EH_PE_udata4:
        result = *((uint32_t*)p);
        p += sizeof(uint32_t);
        break;
    case DW_EH_PE_udata8:
        result = *((uint64_t*)p);
        p += sizeof(uint64_t);
        break;
    case DW_EH_PE_sdata2:
        result = *((int16_t*)p);
        p += sizeof(int16_t);
        break;
    case DW_EH_PE_sdata4:
        result = *((int32_t*)p);
        p += sizeof(int32_t);
        break;
    case DW_EH_PE_sdata8:
        result = *((int64_t*)p);
        p += sizeof(int64_t);
        break;
    default:
        // not supported 
        abort();
        break;
    }
    // then add relative offset 
    switch (encoding & 0x70)
    {
    case DW_EH_PE_absptr:
        // do nothing 
        break;
    case DW_EH_PE_pcrel:
        result += (uintptr_t)(*data);
        break;
    case DW_EH_PE_textrel:
    case DW_EH_PE_datarel:
    case DW_EH_PE_funcrel:
    case DW_EH_PE_aligned:
    default:
        // not supported 
        abort();
        break;
    }
    // then apply indirection 
    if (encoding & DW_EH_PE_indirect)
        result = *((uintptr_t*)result);
    *data = p;
    return result;
}

/// Deals with Dwarf actions matching our type infos 
/// (OurExceptionType_t instances). Returns whether or not a dwarf emitted 
/// action matches the supplied exception type. If such a match succeeds, 
/// the handlerSwitchValue will be set with > 0 index value. Only 
/// corresponding llvm.eh.selector type info arguments, cleanup arguments 
/// are supported. Filters are not supported.
/// See Variable Length Data in: 
/// @link http://dwarfstd.org/Dwarf3.pdf @unlink
/// Also see @link http://refspecs.freestandards.org/abi-eh-1.21.html @unlink
/// @param classInfo our array of type info pointers (to globals)
/// @param actionEntry index into above type info array or 0 (clean up). 
///        We do not support filters.
/// @param exceptionObject thrown _Unwind_Exception instance.
/// @returns whether or not a type info was found. False is returned if only
///          a cleanup was found
static
bool
handleActionValue(const std::type_info** classInfo, uintptr_t actionEntry,
                  _Unwind_Exception* exceptionObject)
{
    __cxa_exception* excp = (__cxa_exception*)(exceptionObject+1) - 1;
    const std::type_info* excpType = excp->exceptionType;
    const uint8_t* actionPos = (uint8_t*)actionEntry;
    for (int i = 0; true; ++i)
    {
        // Each emitted dwarf action corresponds to a 2 tuple of
        // type info address offset, and action offset to the next
        // emitted action.
        int64_t typeOffset = readSLEB128(&actionPos);
        const uint8_t* tempActionPos = actionPos;
        int64_t actionOffset = readSLEB128(&tempActionPos);
        assert((typeOffset >= 0) && "handleActionValue(...):filters are not supported.");
        // Note: A typeOffset == 0 implies that a cleanup llvm.eh.selector
        //       argument has been matched.
        if ((typeOffset > 0) && (excpType == classInfo[-typeOffset]))
        {
            excp->handlerSwitchValue = i + 1;
            return true;
        }
        if (!actionOffset)
            break;
        actionPos += actionOffset;
    }
    return false;
}

// Return true if there is a handler and false otherwise
// cache handlerSwitchValue, actionRecord, languageSpecificData,
//    catchTemp and adjustedPtr here.
static
bool
contains_handler(_Unwind_Exception* exceptionObject, _Unwind_Context* context)
{
    const uint8_t* lsda = (const uint8_t*)_Unwind_GetLanguageSpecificData(context);
    if (lsda)
    {
        // Get the current instruction pointer and offset it before next
        // instruction in the current frame which threw the exception.
        uintptr_t pc = _Unwind_GetIP(context) - 1;
        // Get beginning current frame's code (as defined by the 
        // emitted dwarf code)
        uintptr_t funcStart = _Unwind_GetRegionStart(context);
        uintptr_t pcOffset = pc - funcStart;
        const std::type_info** classInfo = NULL;
        // Note: See JITDwarfEmitter::EmitExceptionTable(...) for corresponding
        //       dwarf emission
        // Parse LSDA header.
        uint8_t lpStartEncoding = *lsda++;
        if (lpStartEncoding != DW_EH_PE_omit)
            (void)readEncodedPointer(&lsda, lpStartEncoding); 
        uint8_t ttypeEncoding = *lsda++;
        if (ttypeEncoding != DW_EH_PE_omit)
        {
            // Calculate type info locations in emitted dwarf code which
            // were flagged by type info arguments to llvm.eh.selector
            // intrinsic
            uintptr_t classInfoOffset = readULEB128(&lsda);
            classInfo = (const std::type_info**)(lsda + classInfoOffset);
        }
        // Walk call-site table looking for range that 
        // includes current PC. 
        uint8_t callSiteEncoding = *lsda++;
        uint32_t callSiteTableLength = readULEB128(&lsda);
        const uint8_t* callSiteTableStart = lsda;
        const uint8_t* callSiteTableEnd = callSiteTableStart + callSiteTableLength;
        const uint8_t* actionTableStart = callSiteTableEnd;
        const uint8_t* callSitePtr = callSiteTableStart;
        __cxa_exception* excp = (__cxa_exception*)(exceptionObject+1) - 1;
        while (callSitePtr < callSiteTableEnd)
        {
            uintptr_t start = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t length = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t landingPad = readEncodedPointer(&callSitePtr, callSiteEncoding);
            // Note: Action value
            uintptr_t actionEntry = readULEB128(&callSitePtr);
            if (landingPad == 0)
                continue; // no landing pad for this entry
            if (actionEntry)
                actionEntry += ((uintptr_t)actionTableStart) - 1;
            bool exceptionMatched = false;
            if ((start <= pcOffset) && (pcOffset < (start + length)))
            {
                excp->actionRecord = (const unsigned char*)(funcStart + landingPad);
                if (actionEntry)
                    return handleActionValue(classInfo, 
                                             actionEntry, 
                                             exceptionObject);
                // Note: Only non-clean up handlers are marked as
                //       found. Otherwise the clean up handlers will be 
                //       re-found and executed during the clean up 
                //       phase.
                break;
            }
        }
    }
    return false;
}

static
_Unwind_Reason_Code
transfer_control_to_landing_pad(_Unwind_Exception* exceptionObject,
                                _Unwind_Context* context)
{
    __cxa_exception* excp = (__cxa_exception*)(exceptionObject+1) - 1;
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)exceptionObject);
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), excp->handlerSwitchValue);
    _Unwind_SetIP(context, (uintptr_t)excp->actionRecord);
    return _URC_INSTALL_CONTEXT;
}

static
_Unwind_Reason_Code
perform_cleanup(_Unwind_Exception* exceptionObject, _Unwind_Context* context)
{
    __cxa_exception* excp = (__cxa_exception*)(exceptionObject+1) - 1;
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)exceptionObject);
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), 0);
    _Unwind_SetIP(context, (uintptr_t)excp->actionRecord);
    return _URC_INSTALL_CONTEXT;
}

// public API

// Requires:  version == 1
//            actions == _UA_SEARCH_PHASE, or
//                    == _UA_CLEANUP_PHASE, or
//                    == _UA_CLEANUP_PHASE | _UA_HANDLER_FRAME, or
//                    == _UA_CLEANUP_PHASE | _UA_FORCE_UNWIND
//            exceptionObject != nullptr
//            context != nullptr
_Unwind_Reason_Code
__gxx_personality_v0(int version, _Unwind_Action actions, uint64_t exceptionClass,
                     _Unwind_Exception* exceptionObject, _Unwind_Context* context)
{
    if (version == 1 && exceptionObject != 0 && context != 0)
    {
        bool native_exception = (exceptionClass & 0xFFF0) == 0x432B2B00;
        bool force_unwind = actions & _UA_FORCE_UNWIND;
        if (native_exception && !force_unwind)
        {
            if (actions & _UA_SEARCH_PHASE)
            {
                if (actions & _UA_CLEANUP_PHASE)
                    return _URC_FATAL_PHASE1_ERROR;
                if (contains_handler(exceptionObject, context))
                    return _URC_HANDLER_FOUND;
                return _URC_CONTINUE_UNWIND;
            }
            if (actions & _UA_CLEANUP_PHASE)
            {
                if (actions & _UA_HANDLER_FRAME)
                {
                    // return _URC_INSTALL_CONTEXT or _URC_FATAL_PHASE2_ERROR
                    return transfer_control_to_landing_pad(exceptionObject, context);
                }
                // return _URC_CONTINUE_UNWIND or _URC_FATAL_PHASE2_ERROR
                return perform_cleanup(exceptionObject, context);
            }
        }
        else // foreign exception or force_unwind
        {
            if (actions & _UA_SEARCH_PHASE)
            {
                if (actions & _UA_CLEANUP_PHASE)
                    return _URC_FATAL_PHASE1_ERROR;
                return _URC_CONTINUE_UNWIND;
            }
            if (actions & _UA_CLEANUP_PHASE)
            {
                if (actions & _UA_HANDLER_FRAME)
                    return _URC_FATAL_PHASE2_ERROR;
                // return _URC_CONTINUE_UNWIND or _URC_FATAL_PHASE2_ERROR
                return perform_cleanup(exceptionObject, context);
            }
        }
    }
    return _URC_FATAL_PHASE1_ERROR;
}

}  // extern "C"

}  // __cxxabiv1
