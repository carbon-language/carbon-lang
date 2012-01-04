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
//  
//===----------------------------------------------------------------------===//

#include "unwind.h"

namespace __cxxabiv1
{

extern "C"
{

// private API

// Return true if there is a handler and false otherwise
// cache handlerSwitchValue, actionRecord, languageSpecificData,
//    catchTemp and adjustedPtr here.
static
bool
contains_handler(_Unwind_Exception* exceptionObject, _Unwind_Context* context)
{
}

// return _URC_INSTALL_CONTEXT or _URC_FATAL_PHASE2_ERROR
static
_Unwind_Reason_Code
transfer_control_to_landing_pad(_Unwind_Context* context)
{
}

// return _URC_CONTINUE_UNWIND or _URC_FATAL_PHASE2_ERROR
static
_Unwind_Reason_Code
perform_cleanup(_Unwind_Context* context)
{
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
                    return _URC_HANDLER_FOUND
                return _URC_CONTINUE_UNWIND;
            }
            if (actions & _UA_CLEANUP_PHASE)
            {
                if (actions & _UA_HANDLER_FRAME)
                {
                    // return _URC_INSTALL_CONTEXT or _URC_FATAL_PHASE2_ERROR
                    return transfer_control_to_landing_pad(context);
                }
                // return _URC_CONTINUE_UNWIND or _URC_FATAL_PHASE2_ERROR
                return perform_cleanup();
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
                return perform_cleanup();
            }
        }
    }
    return _URC_FATAL_PHASE1_ERROR;
}

}  // extern "C"

}  // __cxxabiv1
