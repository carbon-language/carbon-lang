//===-- State.cpp -----------------------------------------------*- C++ -*-===//
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
#include "lldb/Core/State.h"
#include <stdio.h>

using namespace lldb;
using namespace lldb_private;

const char *
lldb_private::StateAsCString (StateType state)
{
    switch (state)
    {
    case eStateInvalid:     return "invalid";
    case eStateUnloaded:    return "unloaded";
    case eStateConnected:   return "connected";
    case eStateAttaching:   return "attaching";
    case eStateLaunching:   return "launching";
    case eStateStopped:     return "stopped";
    case eStateRunning:     return "running";
    case eStateStepping:    return "stepping";
    case eStateCrashed:     return "crashed";
    case eStateDetached:    return "detached";
    case eStateExited:      return "exited";
    case eStateSuspended:   return "suspended";
    }
    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "StateType = %i", state);
    return unknown_state_string;
}

const char *
lldb_private::GetFormatAsCString (lldb::Format format)
{
    switch (format)
    {
        case eFormatDefault:        return "default";
        case eFormatBoolean:        return "boolean";
        case eFormatBinary:         return "binary";
        case eFormatBytes:          return "bytes";
        case eFormatBytesWithASCII: return "bytes with ASCII";
        case eFormatChar:           return "character";
        case eFormatCharArray:      return "character array";
        case eFormatCharPrintable:  return "printable character";
        case eFormatComplexFloat:   return "complet float";
        case eFormatCString:        return "c-string";
        case eFormatDecimal:        return "signed decimal";
        case eFormatEnum:           return "enumeration";
        case eFormatHex:            return "hex";
        case eFormatFloat:          return "float";
        case eFormatOctal:          return "octal";
        case eFormatOSType:         return "OSType";
        case eFormatUnicode16:      return "Unicode16";
        case eFormatUnicode32:      return "Unicode32";
        case eFormatUnsigned:       return "unsigned decimal";
        case eFormatPointer:        return "pointer";
        case eFormatVectorOfChar:   return "vector of characters";
        case eFormatVectorOfSInt8:  return "vector of int8_t";
        case eFormatVectorOfUInt8:  return "vector of uint8_t";
        case eFormatVectorOfSInt16: return "vector of int16_t";
        case eFormatVectorOfUInt16: return "vector of uint16_t";
        case eFormatVectorOfSInt32: return "vector of int32_t";
        case eFormatVectorOfUInt32: return "vector of uint32_t";
        case eFormatVectorOfSInt64: return "vector of int64_t";
        case eFormatVectorOfUInt64: return "vector of uint64_t";
        case eFormatVectorOfFloat32:return "vector of float32";
        case eFormatVectorOfFloat64:return "vector of float64";
        case eFormatVectorOfUInt128:return "vector of uint128_t";
        case eFormatComplexInteger: return "complex integer";
        default: break;
    }
    static char unknown_format_string[64];
    snprintf(unknown_format_string, sizeof (unknown_format_string), "Format = %u", format);
    return unknown_format_string;
}


const char *
lldb_private::GetPermissionsAsCString (uint32_t permissions)
{
    switch (permissions)
    {
        case 0:                      return "---";
        case ePermissionsWritable:   return "-w-";
        case ePermissionsReadable:   return "r--";
        case ePermissionsExecutable: return "--x";
        case ePermissionsReadable | 
             ePermissionsWritable:   return "rw-";
        case ePermissionsReadable | 
             ePermissionsExecutable: return "r-x";
        case ePermissionsWritable | 
             ePermissionsExecutable: return "-wx";        
        case ePermissionsReadable | 
             ePermissionsWritable | 
             ePermissionsExecutable: return "rwx";
        default: 
            break;
    }
    return "???";
}

bool
lldb_private::StateIsRunningState (StateType state)
{
    switch (state)
    {
    case eStateAttaching:
    case eStateLaunching:
    case eStateRunning:
    case eStateStepping:
        return true;

    case eStateConnected:
    case eStateDetached:
    case eStateInvalid:
    case eStateUnloaded:
    case eStateStopped:
    case eStateCrashed:
    case eStateExited:
    case eStateSuspended:
    default:
        break;
    }
    return false;
}

bool
lldb_private::StateIsStoppedState (StateType state)
{
    switch (state)
    {
    case eStateInvalid:
    case eStateConnected:
    case eStateAttaching:
    case eStateLaunching:
    case eStateRunning:
    case eStateStepping:
    case eStateDetached:
    default:
        break;

    case eStateUnloaded:
    case eStateStopped:
    case eStateCrashed:
    case eStateExited:
    case eStateSuspended:
        return true;
    }
    return false;
}
