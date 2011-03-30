//===-- ProcessMessage.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessMessage.h"

using namespace lldb_private;

const char *
ProcessMessage::GetCrashReasonString(CrashReason reason)
{
    const char *str = NULL;

    switch (reason)
    {
    default:
        assert(false && "invalid CrashReason");
        break;

    case eInvalidAddress:
        str = "invalid address";
        break;
    case ePrivilegedAddress:
        str = "address access protected";
        break;
    case eIllegalOpcode:
        str = "illegal instruction";
        break;
    case eIllegalOperand:
        str = "illegal instruction operand";
        break;
    case eIllegalAddressingMode:
        str = "illegal addressing mode";
        break;
    case eIllegalTrap:
        str = "illegal trap";
        break;
    case ePrivilegedOpcode:
        str = "privileged instruction";
        break;
    case ePrivilegedRegister:
        str = "privileged register";
        break;
    case eCoprocessorError:
        str = "coprocessor error";
        break;
    case eInternalStackError:
        str = "internal stack error";
        break;
    case eIllegalAlignment:
        str = "illegal alignment";
        break;
    case eIllegalAddress:
        str = "illegal address";
        break;
    case eHardwareError:
        str = "hardware error";
        break;
    case eIntegerDivideByZero:
        str = "integer divide by zero";
        break;
    case eIntegerOverflow:
        str = "integer overflow";
        break;
    case eFloatDivideByZero:
        str = "floating point divide by zero";
        break;
    case eFloatOverflow:
        str = "floating point overflow";
        break;
    case eFloatUnderflow:
        str = "floating point underflow";
        break;
    case eFloatInexactResult:
        str = "inexact floating point result";
        break;
    case eFloatInvalidOperation:
        str = "invalid floating point operation";
        break;
    case eFloatSubscriptRange:
        str = "invalid floating point subscript range";
        break;
    }

    return str;
}
