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
#include "cxa_handlers.hpp"
#include "private_typeinfo.h"
#include <typeinfo>
#include <stdlib.h>
#include <assert.h>

#include <stdio.h>

/*
    Exception Header Layout:

+---------------------------+-----------------------------+---------------+
| __cxa_exception           | _Unwind_Exception CLNGC++\0 | thrown object |
+---------------------------+-----------------------------+---------------+
                                                          ^
                                                          |
  +-------------------------------------------------------+
  |
+---------------------------+-----------------------------+
| __cxa_dependent_exception | _Unwind_Exception CLNGC++\1 |
+---------------------------+-----------------------------+

    Exception Handling Table Layout:

+-----------------+--------+
| lpStartEncoding | (char) |
+---------+-------+--------+---------------+-----------------------+
| lpStart | (encoded wtih lpStartEncoding) | defaults to funcStart |
+---------+-----+--------+-----------------+---------------+-------+
| ttypeEncoding | (char) | Encoding of the type_info table |
+---------------+-+------+----+----------------------------+----------------+
| classInfoOffset | (ULEB128) | Offset to type_info table, defaults to null |
+-----------------++--------+-+----------------------------+----------------+
| callSiteEncoding | (char) | Encoding for Call Site Table |
+------------------+--+-----+-----+------------------------+--------------------------+
| callSiteTableLength | (ULEB128) | Call Site Table length, used to find Action table |
+---------------------+-----------+------------------------------------------------+--+
| Beginning of Call Site Table            If the current ip lies within the        |
| ...                                     (start, length) range of one of these    |
|                                         call sites, there may be action needed.  |
| +-------------+---------------------------------+------------------------------+ |
| | start       | (encoded with callSiteEncoding) | offset relative to funcStart | |
| | length      | (encoded with callSiteEncoding) | lenght of code fragment      | |
| | landingPad  | (encoded with callSiteEncoding) | offset relative to lpStart   | |
| | actionEntry | (ULEB128)                       | Action Table Index 1-based   | |
| |             |                                 | actionEntry == 0 -> cleanup  | |
| +-------------+---------------------------------+------------------------------+ |
| ...                                                                              |
+---------------------------------------------------------------------+------------+
| Beginning of Action Table       ttypeIndex == 0 : cleanup           |
| ...                             ttypeIndex  > 0 : catch             |
|                                 ttypeIndex  < 0 : exception spec    |
| +--------------+-----------+--------------------------------------+ |
| | ttypeIndex   | (SLEB128) | Index into type_info Table (1-based) | |
| | actionOffset | (SLEB128) | Offset into next Action Table entry  | |
| +--------------+-----------+--------------------------------------+ |
| ...                                                                 |
+---------------------------------------------------------------------+-----------------+
| type_info Table, but classInfoOffset does *not* point here!                           |
| +----------------+------------------------------------------------+-----------------+ |
| | Nth type_info* | Encoded with ttypeEncoding, 0 means catch(...) | ttypeIndex == N | |
| +----------------+------------------------------------------------+-----------------+ |
| ...                                                                                   |
| +----------------+------------------------------------------------+-----------------+ |
| | 1st type_info* | Encoded with ttypeEncoding, 0 means catch(...) | ttypeIndex == 1 | |
| +----------------+------------------------------------------------+-----------------+ |
| +---------------------------------------+-----------+------------------------------+  |
| | 1st ttypeIndex for 1st exception spec | (ULEB128) | classInfoOffset points here! |  |
| | ...                                   | (ULEB128) |                              |  |
| | Mth ttypeIndex for 1st exception spec | (ULEB128) |                              |  |
| | 0                                     | (ULEB128) |                              |  |
| +---------------------------------------+------------------------------------------+  |
| ...                                                                                   |
| +---------------------------------------+------------------------------------------+  |
| | 0                                     | (ULEB128) | throw()                      |  |
| +---------------------------------------+------------------------------------------+  |
| ...                                                                                   |
| +---------------------------------------+------------------------------------------+  |
| | 1st ttypeIndex for Nth exception spec | (ULEB128) |                              |  |
| | ...                                   | (ULEB128) |                              |  |
| | Mth ttypeIndex for Nth exception spec | (ULEB128) |                              |  |
| | 0                                     | (ULEB128) |                              |  |
| +---------------------------------------+------------------------------------------+  |
+---------------------------------------------------------------------------------------+

Notes:

*  ttypeIndex in the Action Table, and in the exception spec table, is an index,
     not a byte count, if positive.  It is a negative index offset of
     classInfoOffset and the sizeof entry depends on ttypeEncoding.
   But if ttypeIndex is negative, it is a positive 1-based byte offset into the
     type_info Table.
   And if ttypeIndex is zero, it refers to a catch (...).

*  landingPad can be 0, this implies there is nothing to be done.

*  landingPad != 0 and actionEntry == 0 implies a cleanup needs to be done
     @landingPad.

*  A cleanup can also be found under landingPad != 0 and actionEntry != 0 in
     the Action Table with ttypeIndex == 0.
*/

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
// TODO:  Not quite rgiht.  This should be able to read a 0 from the TType table
//                          and not dereference it.  Pasted in temporayr workaround
// TODO:  Sometimes this is clearly not always reading an encoded pointer, for
//        example a length in the call site table.  Needs new name?
    uintptr_t result = 0;
    if (encoding == DW_EH_PE_omit) 
        return result;
    const uint8_t* p = *data;
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
        if (result)
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
    if (result && (encoding & DW_EH_PE_indirect))
        result = *((uintptr_t*)result);
    *data = p;
    return result;
}

static
void
call_terminate(bool native_exception, _Unwind_Exception* unwind_exception)
{
    __cxa_begin_catch(unwind_exception);
    if (native_exception)
    {
        // Use the stored terminate_handler if possible
        __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
        std::__terminate(exception_header->terminateHandler);
    }
    std::terminate();
}

static
const __shim_type_info*
get_shim_type_info(int64_t ttypeIndex, const uint8_t* classInfo,
                   uint8_t ttypeEncoding, bool native_exception,
                   _Unwind_Exception* unwind_exception)
{
    // TODO:  Move this check sooner
    if (classInfo == 0)
    {
        // this should not happen
        call_terminate(native_exception, unwind_exception);
    }
    switch (ttypeEncoding & 0x0F)
    {
    case DW_EH_PE_absptr:
        ttypeIndex *= sizeof(void*);
        break;
    case DW_EH_PE_udata2:
    case DW_EH_PE_sdata2:
        ttypeIndex *= 2;
        break;
    case DW_EH_PE_udata4:
    case DW_EH_PE_sdata4:
        ttypeIndex *= 4;
        break;
    case DW_EH_PE_udata8:
    case DW_EH_PE_sdata8:
        ttypeIndex *= 8;
        break;
    default:
        // TODO:  Move this check sooner
        // this should not happen
        call_terminate(native_exception, unwind_exception);
    }
    classInfo -= ttypeIndex;
    return (const __shim_type_info*)readEncodedPointer(&classInfo, ttypeEncoding);
}

static
bool
exception_spec_can_catch(int64_t specIndex, const uint8_t* classInfo,
                         uint8_t ttypeEncoding, const __shim_type_info* excpType,
                         void* adjustedPtr, _Unwind_Exception* unwind_exception)
{
    // TODO:  Move this check sooner
    if (classInfo == 0)
    {
        // this should not happen
        call_terminate(false, unwind_exception);
    }
    // specIndex is 1-based byte offset into classInfo;
    --specIndex;
    const uint8_t* temp = classInfo + specIndex;
    // If any type in the spec list can catch excpType, return false, else return true
    //    adjustments to adjustedPtr are ignored.
    while (true)
    {
        uint64_t ttypeIndex = readULEB128(&temp);
        if (ttypeIndex == 0)
            break;
        const __shim_type_info* catchType = get_shim_type_info(ttypeIndex,
                                                               classInfo,
                                                               ttypeEncoding,
                                                               true,
                                                               unwind_exception);
        void* tempPtr = adjustedPtr;
        if (catchType->can_catch(excpType, tempPtr))
            return false;
    }
    return true;
}

static
const uint8_t*
getTTypeEntry(int64_t ttypeIndex, const uint8_t* classInfo, uint8_t ttypeEncoding)
{
    switch (ttypeEncoding & 0x0F)
    {
    case DW_EH_PE_absptr:
        ttypeIndex *= sizeof(void*);
        break;
    case DW_EH_PE_udata2:
    case DW_EH_PE_sdata2:
        ttypeIndex *= 2;
        break;
    case DW_EH_PE_udata4:
    case DW_EH_PE_sdata4:
        ttypeIndex *= 4;
        break;
    case DW_EH_PE_udata8:
    case DW_EH_PE_sdata8:
        ttypeIndex *= 8;
        break;
    }
    return classInfo - ttypeIndex;
}

static
void
save_state(__cxa_exception* exception_header, int handlerSwitchValue,
           const uint8_t* actionRecord, const uint8_t* languageSpecificData,
           void* landingPad, void* adjustedPtr)
{
    exception_header->handlerSwitchValue = handlerSwitchValue;
    exception_header->actionRecord = actionRecord;
    exception_header->languageSpecificData = languageSpecificData;
    exception_header->catchTemp = landingPad;
    exception_header->adjustedPtr = adjustedPtr;
}

static
void
save_state_for_exception_spec(__cxa_exception* exception_header,
                              int handlerSwitchValue,
                              const void* classInfo,
                              uint8_t ttypeEncoding,
                              void* adjustedPtr)
{
    exception_header->handlerSwitchValue = handlerSwitchValue;
    exception_header->languageSpecificData = static_cast<const uint8_t*>(classInfo);
    exception_header->catchTemp = (void*)(uintptr_t)ttypeEncoding;
    exception_header->adjustedPtr = adjustedPtr;
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
/// @param actionEntry index into above type info array or 0 (cleanup). 
///        We do not support filters.
/// @param unwind_exception thrown _Unwind_Exception instance.
/// @returns whether or not a type info was found. False is returned if only
///          a cleanup was found
static
bool
handleActionValue(const uint8_t* classInfo, uintptr_t actionEntry,
                  _Unwind_Exception* unwind_exception, uint8_t ttypeEncoding)
{
    __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
    void* thrown_object =
        unwind_exception->exception_class == kOurDependentExceptionClass ?
            ((__cxa_dependent_exception*)exception_header)->primaryException :
            exception_header + 1;
    const __shim_type_info* excpType =
        static_cast<const __shim_type_info*>(exception_header->exceptionType);
    const uint8_t* actionPos = (uint8_t*)actionEntry;
    while (true)
    {
        // Each emitted dwarf action corresponds to a 2 tuple of
        // type info address offset, and action offset to the next
        // emitted action.
        const uint8_t* SactionPos = actionPos;
        int64_t ttypeIndex = readSLEB128(&actionPos);
        const uint8_t* tempActionPos = actionPos;
        int64_t actionOffset = readSLEB128(&tempActionPos);
        if (ttypeIndex > 0)  // a catch handler
        {
            const uint8_t* TTypeEntry = getTTypeEntry(ttypeIndex, classInfo,
                                                      ttypeEncoding);
            const __shim_type_info* catchType =
                       (const __shim_type_info*)readEncodedPointer(&TTypeEntry,
                                                                 ttypeEncoding);
            void* adjustedPtr = thrown_object;
            // catchType == 0 -> catch (...)
            if (catchType == 0 || catchType->can_catch(excpType, adjustedPtr))
            {
                exception_header->handlerSwitchValue = ttypeIndex;
                exception_header->actionRecord = SactionPos;  // unnecessary?
                // used by __cxa_get_exception_ptr and __cxa_begin_catch
                exception_header->adjustedPtr = adjustedPtr;
                return true;
            }
        }
        else if (ttypeIndex < 0)  // an exception spec
        {
        }
        else  // ttypeIndex == 0  // a cleanup
        {
        }
        if (actionOffset == 0)
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
contains_handler(_Unwind_Exception* unwind_exception, _Unwind_Context* context)
{
    __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
    const uint8_t* lsda = (const uint8_t*)_Unwind_GetLanguageSpecificData(context);
    exception_header->languageSpecificData = lsda;
    if (lsda)
    {
        // Get the current instruction pointer and offset it before next
        // instruction in the current frame which threw the exception.
        uintptr_t ip = _Unwind_GetIP(context) - 1;
        // Get beginning current frame's code (as defined by the 
        // emitted dwarf code)
        uintptr_t funcStart = _Unwind_GetRegionStart(context);
        uintptr_t ipOffset = ip - funcStart;
        const uint8_t* classInfo = NULL;
        // Note: See JITDwarfEmitter::EmitExceptionTable(...) for corresponding
        //       dwarf emission
        // Parse LSDA header.
        uint8_t lpStartEncoding = *lsda++;
        const uint8_t* lpStart = (const uint8_t*)readEncodedPointer(&lsda, lpStartEncoding);
        if (lpStart == 0)
            lpStart = (const uint8_t*)funcStart;
        uint8_t ttypeEncoding = *lsda++;
        // TODO:  preflight ttypeEncoding here and return error if there's a problem
        if (ttypeEncoding != DW_EH_PE_omit)
        {
            // Calculate type info locations in emitted dwarf code which
            // were flagged by type info arguments to llvm.eh.selector
            // intrinsic
            uintptr_t classInfoOffset = readULEB128(&lsda);
            classInfo = lsda + classInfoOffset;
        }
        // Walk call-site table looking for range that 
        // includes current PC. 
        uint8_t callSiteEncoding = *lsda++;
        uint32_t callSiteTableLength = readULEB128(&lsda);
        const uint8_t* callSiteTableStart = lsda;
        const uint8_t* callSiteTableEnd = callSiteTableStart + callSiteTableLength;
        const uint8_t* actionTableStart = callSiteTableEnd;
        const uint8_t* callSitePtr = callSiteTableStart;
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
            if ((start <= ipOffset) && (ipOffset < (start + length)))
            {
                exception_header->catchTemp = (void*)(lpStart + landingPad);
                if (actionEntry)
                    return handleActionValue(classInfo, 
                                             actionEntry, 
                                             unwind_exception,
                                             ttypeEncoding);
                // Note: Only non-cleanup handlers are marked as
                //       found. Otherwise the cleanup handlers will be 
                //       re-found and executed during the cleanup 
                //       phase.
                return false;  // Won't find another call site in range of ipOffset
            }
        }
        // Not found, need to properly terminate
    }
    return false;
}

static
_Unwind_Reason_Code
transfer_control_to_landing_pad(_Unwind_Exception* unwind_exception,
                                _Unwind_Context* context)
{
    __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)unwind_exception);
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), exception_header->handlerSwitchValue);
    _Unwind_SetIP(context, (uintptr_t)exception_header->catchTemp);
    return _URC_INSTALL_CONTEXT;
}

static
_Unwind_Reason_Code
perform_cleanup(_Unwind_Exception* unwind_exception, _Unwind_Context* context)
{
    __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)unwind_exception);
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), 0);
    _Unwind_SetIP(context, (uintptr_t)exception_header->catchTemp);
    return _URC_INSTALL_CONTEXT;
}

// public API

/*
A foreign exception is defined by by one with an exceptionClass that doesn't
have 'C++' in the 3 low order bytes 3 - 1:

  big end  |   | ... | C | + | + |1/0|  little end

The lowest order byte may be 0 or 1.

The personality function branches on actions like so:

_UA_SEARCH_PHASE

    If _UA_CLEANUP_PHASE or _UA_HANDLER_FRAME or _UA_FORCE_UNWIND there's
      an error from above, return _URC_FATAL_PHASE1_ERROR.

    Scan for anything that could stop unwinding:

       1.  A catch clause that will catch this exception
           (will never catch foreign).
       2.  A catch (...) (will always catch foreign).
       3.  An exception spec that will catch this exception
           (will always catch foreign).
    If a handler is found
        If not foreign
            Save state in header
        return _URC_HANDLER_FOUND
    Else a handler not found
        return _URC_CONTINUE_UNWIND

_UA_CLEANUP_PHASE

    If _UA_HANDLER_FRAME
        If _UA_FORCE_UNWIND
            How did this happen?  return _URC_FATAL_PHASE2_ERROR
        If foreign
            Do _UA_SEARCH_PHASE to recover state
        else
            Recover state from header
        Transfer control to landing pad.  return _URC_INSTALL_CONTEXT
    
    Else
    
        Scan for anything that can not stop unwinding:
    
            1.  A cleanup.
        
        If a cleanup is found
            transfer control to it. return _URC_INSTALL_CONTEXT
        Else a cleanup is not found: return _URC_CONTINUE_UNWIND
*/

_Unwind_Reason_Code
__gxx_personality_v0(int version, _Unwind_Action actions, uint64_t exceptionClass,
                     _Unwind_Exception* unwind_exception, _Unwind_Context* context)
{
printf("__gxx_personality_v0 start with unwind_exception = %p\n", unwind_exception);
    if (version != 1 || unwind_exception == 0 || context == 0)
        return _URC_FATAL_PHASE1_ERROR;
    bool native_exception = (exceptionClass     & get_language) ==
                            (kOurExceptionClass & get_language);
    if (actions & _UA_SEARCH_PHASE)
    {
printf("__gxx_personality_v0 start phase 1\n");
        // Do Phase 1
        if (actions & (_UA_CLEANUP_PHASE | _UA_HANDLER_FRAME | _UA_FORCE_UNWIND))
        {
            // None of these flags should be set during Phase 1
            return _URC_FATAL_PHASE1_ERROR;
        }
        // Scan for handlers
        //   If a handler is found return _URC_HANDLER_FOUND else return _URC_CONTINUE_UNWIND
        const uint8_t* lsda = (const uint8_t*)_Unwind_GetLanguageSpecificData(context);
        if (lsda == 0)
        {
            // No LanguageSpecificData means no handlers
printf("__gxx_personality_v0 phase 1 no lsda, returning _URC_CONTINUE_UNWIND\n");
            return _URC_CONTINUE_UNWIND;
        }
        const uint8_t* languageSpecificData = lsda;
        // Get the current instruction pointer and offset it before next
        // instruction in the current frame which threw the exception.
        uintptr_t ip = _Unwind_GetIP(context) - 1;
        // Get beginning current frame's code (as defined by the 
        // emitted dwarf code)
        uintptr_t funcStart = _Unwind_GetRegionStart(context);
        uintptr_t ipOffset = ip - funcStart;
        const uint8_t* classInfo = NULL;
        // Note: See JITDwarfEmitter::EmitExceptionTable(...) for corresponding
        //       dwarf emission
        // Parse LSDA header.
        uint8_t lpStartEncoding = *lsda++;
        const uint8_t* lpStart = (const uint8_t*)readEncodedPointer(&lsda, lpStartEncoding);
        if (lpStart == 0)
            lpStart = (const uint8_t*)funcStart;
        uint8_t ttypeEncoding = *lsda++;
        if (ttypeEncoding != DW_EH_PE_omit)
        {
            // Calculate type info locations in emitted dwarf code which
            // were flagged by type info arguments to llvm.eh.selector
            // intrinsic
            uintptr_t classInfoOffset = readULEB128(&lsda);
            classInfo = lsda + classInfoOffset;
        }
        // Walk call-site table looking for range that 
        // includes current PC. 
        uint8_t callSiteEncoding = *lsda++;
        uint32_t callSiteTableLength = readULEB128(&lsda);
        const uint8_t* callSiteTableStart = lsda;
        const uint8_t* callSiteTableEnd = callSiteTableStart + callSiteTableLength;
        const uint8_t* actionTableStart = callSiteTableEnd;
        const uint8_t* callSitePtr = callSiteTableStart;
        while (true)
        {
            // There is one entry per call site.
            // The call sites are non-overlapping in [start, start+length)
            // The call sites are ordered in increasing value of start
            uintptr_t start = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t length = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t landingPad = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t actionEntry = readULEB128(&callSitePtr);
            if ((start <= ipOffset) && (ipOffset < (start + length)))
            {
                // Found the call site containing ip.
                if (landingPad == 0 || actionEntry == 0)
                {
                    // No handler here
 printf("__gxx_personality_v0 phase 1 no landingPad or no actionEntry, returning _URC_CONTINUE_UNWIND\n");
                   return _URC_CONTINUE_UNWIND;
                }
                // Convert 1-based byte offset into
                const uint8_t* action = actionTableStart + (actionEntry - 1);
                // Scan action entries until you find a matching handler, or they end
                while (true)
                {
                    const uint8_t* actionRecord = action;
                    int64_t ttypeIndex = readSLEB128(&action);
                    const uint8_t* temp = action;
                    int64_t actionOffset = readSLEB128(&temp);
                    if (ttypeIndex > 0)
                    {
                        // Does this handler match?
                        // First check for catch (...)
                        const __shim_type_info* catchType =
                            get_shim_type_info(ttypeIndex, classInfo,
                                               ttypeEncoding, native_exception,
                                               unwind_exception);
                        if (catchType == 0)
                        {
                            // catch (...) catches everything, including foreign exceptions
                            // If not foreign, safe state before returning
                            if (native_exception)
                            {
                                __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                                void* adjustedPtr =
                                    unwind_exception->exception_class == kOurDependentExceptionClass ?
                                        ((__cxa_dependent_exception*)exception_header)->primaryException :
                                        exception_header + 1;
                                save_state(exception_header, static_cast<int>(ttypeIndex),
                                           actionRecord, languageSpecificData,
                                           const_cast<uint8_t*>(lpStart + landingPad),
                                           adjustedPtr);
                            }
 printf("__gxx_personality_v0 phase 1 catch (...), returning _URC_HANDLER_FOUND\n");
                            return _URC_HANDLER_FOUND;
                        }
                        // Else this is a catch (T) clause and will never
                        //    catch a foreign exception
                        if (native_exception)
                        {
                            __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                            void* adjustedPtr =
                                unwind_exception->exception_class == kOurDependentExceptionClass ?
                                    ((__cxa_dependent_exception*)exception_header)->primaryException :
                                    exception_header + 1;
                            const __shim_type_info* excpType =
                                static_cast<const __shim_type_info*>(exception_header->exceptionType);
                            if (adjustedPtr == 0 || excpType == 0)
                            {
                                // Something very bad happened
                                call_terminate(native_exception, unwind_exception);
                            }
                            if (catchType->can_catch(excpType, adjustedPtr))
                            {
                                // Found a matching handler
                                save_state(exception_header, static_cast<int>(ttypeIndex),
                                           actionRecord, languageSpecificData,
                                           const_cast<uint8_t*>(lpStart + landingPad),
                                           adjustedPtr);
 printf("__gxx_personality_v0 phase 1 catch (T), returning _URC_HANDLER_FOUND\n");
                                return _URC_HANDLER_FOUND;
                            }
                        }
                        // Scan next action ...
                    }
                    else if (ttypeIndex < 0)
                    {
                        // Found an exception spec.  If this is a foreign exception,
                        //   it is always caught.
                        if (!native_exception)
                        {
 printf("__gxx_personality_v0 phase 1 exception spec for foreign, returning _URC_HANDLER_FOUND\n");
                            return _URC_HANDLER_FOUND;
                        }
                        __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                        void* adjustedPtr =
                            unwind_exception->exception_class == kOurDependentExceptionClass ?
                                ((__cxa_dependent_exception*)exception_header)->primaryException :
                                exception_header + 1;
                        const __shim_type_info* excpType =
                            static_cast<const __shim_type_info*>(exception_header->exceptionType);
                        if (adjustedPtr == 0 || excpType == 0)
                        {
                            // Something very bad happened
                            call_terminate(native_exception, unwind_exception);
                        }
                        if (exception_spec_can_catch(ttypeIndex, classInfo,
                                                     ttypeEncoding, excpType,
                                                     adjustedPtr, unwind_exception))
                        {
                            // The state saved is a little different for exception specs
                            save_state(exception_header,
                                                          ttypeIndex,
                                                          actionRecord,
                                                          languageSpecificData,
                                                          const_cast<uint8_t*>(lpStart + landingPad),
                                                          adjustedPtr);
 printf("__gxx_personality_v0 phase 1 exception spec for native, returning _URC_HANDLER_FOUND\n");
                                return _URC_HANDLER_FOUND;
                        }
                        // Scan next action ...
                    }
                    if (actionOffset == 0)
                    {
                        // End of action list, no matching handler found
 printf("__gxx_personality_v0 phase 1 no handler found, returning _URC_CONTINUE_UNWIND\n");
                        return _URC_CONTINUE_UNWIND;
                    }
                    // Go to next action
                    action += actionOffset;
                }
            }
            else if (ipOffset < start)
            {
                // There is no call site for this ip
                // Something bad has happened.  We should never get here.
                // Possible stack corruption.
                call_terminate(native_exception, unwind_exception);
            }
        }
    }
    if (actions & _UA_CLEANUP_PHASE)
    {
        if (actions & _UA_HANDLER_FRAME)
        {
printf("__gxx_personality_v0 start phase 2 handling\n");
            // Search phase found a handler, now install it
            if (actions & _UA_FORCE_UNWIND)
            {
                // This should never happen.  The search phase isn't executed
                //   for forced unwinding, so no handler could have been found.
                call_terminate(native_exception, unwind_exception);
            }
            // This is the state we need:
            uintptr_t handlerSwitchValue;
            uintptr_t landingPad;
            if (native_exception)
            {
                // Just retrieve it from the exception_header
                __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                handlerSwitchValue = static_cast<uintptr_t>(static_cast<intptr_t>(exception_header->handlerSwitchValue));
                landingPad = reinterpret_cast<uintptr_t>(exception_header->catchTemp);
            }
            else
            {
                // Else a foreign exception, we need to find the handler that caught it
                const uint8_t* lsda = (const uint8_t*)_Unwind_GetLanguageSpecificData(context);
                if (lsda == 0)
                {
                    // If we don't find a handler, something bad happened
                    call_terminate(native_exception, unwind_exception);
                }
                // Get the current instruction pointer and offset it before next
                // instruction in the current frame which threw the exception.
                uintptr_t ip = _Unwind_GetIP(context) - 1;
                // Get beginning current frame's code (as defined by the 
                // emitted dwarf code)
                uintptr_t funcStart = _Unwind_GetRegionStart(context);
                uintptr_t ipOffset = ip - funcStart;
                const uint8_t* classInfo = NULL;
                // Note: See JITDwarfEmitter::EmitExceptionTable(...) for corresponding
                //       dwarf emission
                // Parse LSDA header.
                uint8_t lpStartEncoding = *lsda++;
                const uint8_t* lpStart = (const uint8_t*)readEncodedPointer(&lsda, lpStartEncoding);
                if (lpStart == 0)
                    lpStart = (const uint8_t*)funcStart;
                uint8_t ttypeEncoding = *lsda++;
                if (ttypeEncoding != DW_EH_PE_omit)
                {
                    // Calculate type info locations in emitted dwarf code which
                    // were flagged by type info arguments to llvm.eh.selector
                    // intrinsic
                    uintptr_t classInfoOffset = readULEB128(&lsda);
                    classInfo = lsda + classInfoOffset;
                }
                // Walk call-site table looking for range that 
                // includes current PC. 
                uint8_t callSiteEncoding = *lsda++;
                uint32_t callSiteTableLength = readULEB128(&lsda);
                const uint8_t* callSiteTableStart = lsda;
                const uint8_t* callSiteTableEnd = callSiteTableStart + callSiteTableLength;
                const uint8_t* actionTableStart = callSiteTableEnd;
                const uint8_t* callSitePtr = callSiteTableStart;
                while (true)
                {
                    // There is one entry per call site.
                    // The call sites are non-overlapping in [start, start+length)
                    // The call sites are ordered in increasing value of start
                    uintptr_t start = readEncodedPointer(&callSitePtr, callSiteEncoding);
                    uintptr_t length = readEncodedPointer(&callSitePtr, callSiteEncoding);
                    landingPad = readEncodedPointer(&callSitePtr, callSiteEncoding);
                    uintptr_t actionEntry = readULEB128(&callSitePtr);
                    if ((start <= ipOffset) && (ipOffset < (start + length)))
                    {
                        // Found the call site containing ip.
                        if (landingPad == 0 || actionEntry == 0)
                        {
                            // No handler here
                            // If we don't find a handler, something bad happened
                            call_terminate(native_exception, unwind_exception);
                        }
                        landingPad = (uintptr_t)lpStart + landingPad;
                        // Convert 1-based byte offset into
                        const uint8_t* action = actionTableStart + (actionEntry - 1);
                        // Scan action entries until you find a matching handler, or they end
                        while (true)
                        {
                            int64_t ttypeIndex = readSLEB128(&action);
                            const uint8_t* temp = action;
                            int64_t actionOffset = readSLEB128(&temp);
                            if (ttypeIndex > 0)
                            {
                                // Does this handler match?
                                // First check for catch (...)
                                const __shim_type_info* catchType =
                                    get_shim_type_info(ttypeIndex, classInfo,
                                                       ttypeEncoding, native_exception,
                                                       unwind_exception);
                                if (catchType == 0)
                                {
                                    // catch (...) catches everything, including foreign exceptions
                                    handlerSwitchValue = ttypeIndex;
                                    goto install_handler;
                                }
                                // Else this is a catch (T) clause and will never
                                //    catch a foreign exception
                                // Scan next action ...
                            }
                            else if (ttypeIndex < 0)
                            {
                                // Found an exception spec.  This is a foreign exception,
                                //   and thus is always caught.
                                //   However the landing pad is going to call either
                                //   __cxa_call_unexpected (for a throw spec) or
                                //   std::terminate (for noexcept).  We
                                //   don't know which.  And __cxa_call_unexpected
                                //   lacks the API to recover ttypeIndex.  However
                                //   if we were to call a variant of __cxa_call_unexpected
                                //   from here, and if it throws an exception,
                                //   that won't work either.  We can't propagate
                                //   an exception out of here.  So just call
                                //   the landing pad and let __cxa_call_unexpected
                                //   force terminate.  There's nothing else we
                                //   can do.
                                handlerSwitchValue = ttypeIndex;
                                goto install_handler;
                            }
                            if (actionOffset == 0)
                            {
                                // End of action list, no matching handler found
                                // If we don't find a handler, something bad happened
                                call_terminate(native_exception, unwind_exception);
                            }
                            // Go to next action
                            action += actionOffset;
                        }
                    }
                    else if (ipOffset < start)
                    {
                        // There is no call site for this ip
                        // Something bad has happened.  We should never get here.
                        // Possible stack corruption.
                        call_terminate(native_exception, unwind_exception);
                    }
                }
            }
        install_handler:
            _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)unwind_exception);
            _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), handlerSwitchValue);
            _Unwind_SetIP(context, landingPad);
 printf("__gxx_personality_v0 phase 2 handler found, returning _URC_INSTALL_CONTEXT\n");
            return _URC_INSTALL_CONTEXT;
        }
printf("__gxx_personality_v0 start phase 2 cleanup\n");
        // Else scan for a cleanup.
        //  If handler found and !_UA_FORCE_UNWIND, terminate.
        //  If cleanup found, install it.
        //  If nothing found return _URC_CONTINUE_UNWIND
        const uint8_t* lsda = (const uint8_t*)_Unwind_GetLanguageSpecificData(context);
        if (lsda == 0)
        {
            // No LanguageSpecificData means no handlers
 printf("__gxx_personality_v0 phase 2 no lsda, returning _URC_CONTINUE_UNWIND\n");
            return _URC_CONTINUE_UNWIND;
        }
        // Get the current instruction pointer and offset it before next
        // instruction in the current frame which threw the exception.
        uintptr_t ip = _Unwind_GetIP(context) - 1;
        // Get beginning current frame's code (as defined by the 
        // emitted dwarf code)
        uintptr_t funcStart = _Unwind_GetRegionStart(context);
        uintptr_t ipOffset = ip - funcStart;
        const uint8_t* classInfo = NULL;
        // Note: See JITDwarfEmitter::EmitExceptionTable(...) for corresponding
        //       dwarf emission
        // Parse LSDA header.
        uint8_t lpStartEncoding = *lsda++;
        const uint8_t* lpStart = (const uint8_t*)readEncodedPointer(&lsda, lpStartEncoding);
        if (lpStart == 0)
            lpStart = (const uint8_t*)funcStart;
        uint8_t ttypeEncoding = *lsda++;
        if (ttypeEncoding != DW_EH_PE_omit)
        {
            // Calculate type info locations in emitted dwarf code which
            // were flagged by type info arguments to llvm.eh.selector
            // intrinsic
            uintptr_t classInfoOffset = readULEB128(&lsda);
            classInfo = lsda + classInfoOffset;
        }
        // Walk call-site table looking for range that 
        // includes current PC. 
        uint8_t callSiteEncoding = *lsda++;
        uint32_t callSiteTableLength = readULEB128(&lsda);
        const uint8_t* callSiteTableStart = lsda;
        const uint8_t* callSiteTableEnd = callSiteTableStart + callSiteTableLength;
        const uint8_t* actionTableStart = callSiteTableEnd;
        const uint8_t* callSitePtr = callSiteTableStart;
        while (true)
        {
            // There is one entry per call site.
            // The call sites are non-overlapping in [start, start+length)
            // The call sites are ordered in increasing value of start
            uintptr_t start = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t length = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t landingPad = readEncodedPointer(&callSitePtr, callSiteEncoding);
            uintptr_t actionEntry = readULEB128(&callSitePtr);
            if ((start <= ipOffset) && (ipOffset < (start + length)))
            {
                // Found the call site containing ip.
                if (landingPad == 0)
                {
                    // No handler here
 printf("__gxx_personality_v0 phase 2 no landingPad, returning _URC_CONTINUE_UNWIND\n");
                    return _URC_CONTINUE_UNWIND;
                }
                landingPad = (uintptr_t)lpStart + landingPad;
                if (actionEntry == 0)
                {
                    // Found a cleanup, install it:
                    _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)unwind_exception);
                    _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), 0);
                    _Unwind_SetIP(context, landingPad);
 printf("__gxx_personality_v0 phase 2 found cleanup 1, returning _URC_INSTALL_CONTEXT\n");
                    return _URC_INSTALL_CONTEXT;
                }
                // Convert 1-based byte offset into
                const uint8_t* action = actionTableStart + (actionEntry - 1);
                // Scan action entries until you find a matching handler, or they end
                while (true)
                {
                    int64_t ttypeIndex = readSLEB128(&action);
                    const uint8_t* temp = action;
                    int64_t actionOffset = readSLEB128(&temp);
                    if (ttypeIndex > 0)
                    {
                        // Does this handler match?
                        // First check for catch (...)
                        const __shim_type_info* catchType =
                            get_shim_type_info(ttypeIndex, classInfo,
                                               ttypeEncoding, native_exception,
                                               unwind_exception);
                        if (catchType == 0)
                        {
                            // catch (...) catches everything, including foreign exceptions
                            if (!(actions & _UA_FORCE_UNWIND))
                            {
                                // Something bad has happened.  We should never get here.
                                // We should have found this handler in phase 1
                                // Possible stack corruption.
                                call_terminate(native_exception, unwind_exception);
                            }
                            // Ignoring this handler because we are forced
                        }
                        else
                        {
                            // Else this is a catch (T) clause and will never
                            //    catch a foreign exception
                            if (native_exception)
                            {
                                __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                                void* adjustedPtr =
                                    unwind_exception->exception_class == kOurDependentExceptionClass ?
                                        ((__cxa_dependent_exception*)exception_header)->primaryException :
                                        exception_header + 1;
                                const __shim_type_info* excpType =
                                    static_cast<const __shim_type_info*>(exception_header->exceptionType);
                                if (adjustedPtr == 0 || excpType == 0)
                                {
                                    // Something very bad happened
                                    call_terminate(native_exception, unwind_exception);
                                }
                                if (catchType->can_catch(excpType, adjustedPtr))
                                {
                                    // Found a matching handler
                                    if (!(actions & _UA_FORCE_UNWIND))
                                    {
                                        // Something bad has happened.  We should never get here.
                                        // We should have found this handler in phase 1
                                        // Possible stack corruption.
                                        call_terminate(native_exception, unwind_exception);
                                    }
                                    // Ignoring this handler because we are forced
                                }
                            }
                        }
                        // Scan next action ...
                    }
                    else if (ttypeIndex < 0)
                    {
                        // Found an exception spec.  If this is a foreign exception,
                        //   it is always caught.
                        if (!native_exception)
                        {
                            if (!(actions & _UA_FORCE_UNWIND))
                            {
                                // Something bad has happened.  We should never get here.
                                // We should have found this handler in phase 1
                                // Possible stack corruption.
                                call_terminate(native_exception, unwind_exception);
                            }
                        }
                        else
                        {
                            __cxa_exception* exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
                            void* adjustedPtr =
                                unwind_exception->exception_class == kOurDependentExceptionClass ?
                                    ((__cxa_dependent_exception*)exception_header)->primaryException :
                                    exception_header + 1;
                            const __shim_type_info* excpType =
                                static_cast<const __shim_type_info*>(exception_header->exceptionType);
                            if (adjustedPtr == 0 || excpType == 0)
                            {
                                // Something very bad happened
                                call_terminate(native_exception, unwind_exception);
                            }
                            if (exception_spec_can_catch(ttypeIndex, classInfo,
                                                         ttypeEncoding, excpType,
                                                         adjustedPtr, unwind_exception))
                            {
                                if (!(actions & _UA_FORCE_UNWIND))
                                {
                                    // Something bad has happened.  We should never get here.
                                    // We should have found this handler in phase 1
                                    // Possible stack corruption.
                                    call_terminate(native_exception, unwind_exception);
                                }
                            }
                        }
                        // Scan next action ...
                    }
                    else  // ttypeIndex == 0
                    {
                        // Found a cleanup, install it:
                        _Unwind_SetGR(context, __builtin_eh_return_data_regno(0), (uintptr_t)unwind_exception);
                        _Unwind_SetGR(context, __builtin_eh_return_data_regno(1), 0);
                        _Unwind_SetIP(context, landingPad);
 printf("__gxx_personality_v0 phase 2 found cleanup 2, returning _URC_INSTALL_CONTEXT\n");
                        return _URC_INSTALL_CONTEXT;
                    }
                    if (actionOffset == 0)
                    {
                        // End of action list, no matching handler or cleanup found
 printf("__gxx_personality_v0 phase 2 found no cleanups, returning _URC_CONTINUE_UNWIND\n");
                        return _URC_CONTINUE_UNWIND;
                    }
                    // Go to next action
                    action += actionOffset;
                }
            }
            else if (ipOffset < start)
            {
                // There is no call site for this ip
                // Something bad has happened.  We should never get here.
                // Possible stack corruption.
                call_terminate(native_exception, unwind_exception);
            }
        }
    }
    // Neither _UA_SEARCH_PHASE nor _UA_CLEANUP_PHASE
    return _URC_FATAL_PHASE1_ERROR;
}

__attribute__((noreturn))
void
__cxa_call_unexpected(void* arg)
{
printf("__cxa_call_unexpected A\n");
    _Unwind_Exception* unwind_exception = static_cast<_Unwind_Exception*>(arg);
    if (unwind_exception == 0)
        call_terminate(true, unwind_exception);
    __cxa_begin_catch(unwind_exception);
    bool native_old_exception = (unwind_exception->exception_class & get_language) ==
                                (kOurExceptionClass                & get_language);
    std::unexpected_handler u_handler;
    std::terminate_handler t_handler;
    __cxa_exception* old_exception_header = 0;
    if (native_old_exception)
    {
        old_exception_header = (__cxa_exception*)(unwind_exception+1) - 1;
        t_handler = old_exception_header->terminateHandler;
        u_handler = old_exception_header->unexpectedHandler;
    }
    else
    {
        t_handler = std::get_terminate();
        u_handler = std::get_unexpected();
    }
    try
    {
        std::__unexpected(u_handler);
    }
    catch (...)
    {
        // If the old exception is foreign, then all we can do is terminate.
        //   We have no way to recover the needed old exception spec.  There's
        //   no way to pass that information here.  And the personality routine
        //   can't call us directly and do anything but terminate() if we throw
        //   from here.
        if (native_old_exception)
        {
            int64_t ttypeIndex = old_exception_header->handlerSwitchValue;
            // Have:
            //   old_exception_header->languageSpecificData
            //   old_exception_header->actionRecord
            // Need
            //   const uint8_t* classInfo
            //   uint8_t ttypeEncoding
            const uint8_t* lsda = old_exception_header->languageSpecificData;
            uint8_t lpStartEncoding = *lsda++;
            const uint8_t* lpStart = (const uint8_t*)readEncodedPointer(&lsda, lpStartEncoding);
            uint8_t ttypeEncoding = *lsda++;
            if (ttypeEncoding == DW_EH_PE_omit)
                std::__terminate(t_handler);
            uintptr_t classInfoOffset = readULEB128(&lsda);
            const uint8_t* classInfo = lsda + classInfoOffset;
            // Is this new exception catchable by the exception spec at ttypeIndex?
            // The answer is obviously yes if the new and old exceptions are the same exception
            // If no
            //    throw;
            __cxa_eh_globals* globals = __cxa_get_globals_fast();
            __cxa_exception* new_exception_header = globals->caughtExceptions;
            if (new_exception_header == 0)
                // This shouldn't be able to happen!
                std::__terminate(t_handler);
            bool native_new_exception =
                (new_exception_header->unwindHeader.exception_class & get_language) ==
                                                (kOurExceptionClass & get_language);
            void* adjustedPtr;
            if (native_new_exception && new_exception_header != old_exception_header)
            {
                const __shim_type_info* excpType =
                    static_cast<const __shim_type_info*>(new_exception_header->exceptionType);
                adjustedPtr =
                    new_exception_header->unwindHeader.exception_class == kOurDependentExceptionClass ?
                        ((__cxa_dependent_exception*)new_exception_header)->primaryException :
                        new_exception_header + 1;
                if (!exception_spec_can_catch(ttypeIndex, classInfo, ttypeEncoding,
                                              excpType, adjustedPtr, unwind_exception))
                {
                    // We need to __cxa_end_catch, but for the old exception,
                    //   not the new one.  This is a little tricky ...
                    // Disguise new_exception_header as a rethrown exception, but
                    //   don't actually rethrow it.  This means you can temporarily
                    //   end the catch clause enclosing new_exception_header without
                    //   __cxa_end_catch destroying new_exception_header.
                    new_exception_header->handlerCount = -new_exception_header->handlerCount;
                    globals->uncaughtExceptions += 1;
                    // Call __cxa_end_catch for new_exception_header
                    __cxa_end_catch();
                    // Call __cxa_end_catch for old_exception_header
                    __cxa_end_catch();
                    // Renter this catch clause with new_exception_header
                    __cxa_begin_catch(&new_exception_header->unwindHeader);
                    // Rethrow new_exception_header
                    throw;
                }
            }
            // Will a std::bad_exception be catchable by the exception spec at
            //   ttypeIndex?
            // If no
            //    throw std::bad_exception();
            const __shim_type_info* excpType =
                static_cast<const __shim_type_info*>(&typeid(std::bad_exception));
            std::bad_exception be;
            adjustedPtr = &be;
            if (!exception_spec_can_catch(ttypeIndex, classInfo, ttypeEncoding,
                                          excpType, adjustedPtr, unwind_exception))
            {
                // We need to __cxa_end_catch for both the old exception and the
                //   new exception.  Technically we should do it in that order.
                //   But it is expedent to do it in the opposite order:
                // Call __cxa_end_catch for new_exception_header
                __cxa_end_catch();
                // Throw std::bad_exception will __cxa_end_catch for
                //   old_exception_header
                throw be;
            }
        }
    }
    std::__terminate(t_handler);
}

/*
_Unwind_Reason_Code
__gxx_personality_v0(int version, _Unwind_Action actions, uint64_t exceptionClass,
                     _Unwind_Exception* unwind_exception, _Unwind_Context* context)
{
    if (version == 1 && unwind_exception != 0 && context != 0)
    {
        bool native_exception = (exceptionClass & 0xFFFFFF00) == 0x432B2B00;
        bool force_unwind = actions & _UA_FORCE_UNWIND;
        if (native_exception && !force_unwind)
        {
            if (actions & _UA_SEARCH_PHASE)
            {
                if (actions & _UA_CLEANUP_PHASE)
                    return _URC_FATAL_PHASE1_ERROR;
                if (contains_handler(unwind_exception, context))
                    return _URC_HANDLER_FOUND;
                return _URC_CONTINUE_UNWIND;
            }
            if (actions & _UA_CLEANUP_PHASE)
            {
                if (actions & _UA_HANDLER_FRAME)
                {
                    // return _URC_INSTALL_CONTEXT or _URC_FATAL_PHASE2_ERROR
                    return transfer_control_to_landing_pad(unwind_exception, context);
                }
                // return _URC_CONTINUE_UNWIND or _URC_FATAL_PHASE2_ERROR
                return perform_cleanup(unwind_exception, context);
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
                return perform_cleanup(unwind_exception, context);
            }
        }
    }
    return _URC_FATAL_PHASE1_ERROR;
}
*/
}  // extern "C"

}  // __cxxabiv1
