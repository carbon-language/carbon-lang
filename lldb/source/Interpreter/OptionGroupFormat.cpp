//===-- OptionGroupFormat.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupFormat.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Utility/Utils.h"

using namespace lldb;
using namespace lldb_private;

OptionGroupFormat::OptionGroupFormat (lldb::Format default_format,
                                      uint64_t default_byte_size,
                                      uint64_t default_count) :
    m_format (default_format, default_format),
    m_byte_size (default_byte_size, default_byte_size),
    m_count (default_count, default_count),
    m_prev_gdb_format('x'),
    m_prev_gdb_size('w')
{
}

OptionGroupFormat::~OptionGroupFormat ()
{
}

static OptionDefinition 
g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "format"    ,'f', required_argument, NULL, 0, eArgTypeFormat   , "Specify a format to be used for display."},
{ LLDB_OPT_SET_1|
  LLDB_OPT_SET_2|
  LLDB_OPT_SET_3, false, "gdb-format",'G', required_argument, NULL, 0, eArgTypeGDBFormat, "Specify a format using a GDB format specifier string."},
{ LLDB_OPT_SET_2, false, "size"      ,'s', required_argument, NULL, 0, eArgTypeByteSize , "The size in bytes to use when displaying with the selected format."},
{ LLDB_OPT_SET_3, false, "count"     ,'c', required_argument, NULL, 0, eArgTypeCount    , "The number of total items to display."},
};

uint32_t
OptionGroupFormat::GetNumDefinitions ()
{
    if (m_byte_size.GetDefaultValue() < UINT64_MAX)
    {
        if (m_count.GetDefaultValue() < UINT64_MAX)
            return 4;
        else
            return 3;
    }
    return 2;
}

const OptionDefinition *
OptionGroupFormat::GetDefinitions ()
{
    return g_option_table;
}

Error
OptionGroupFormat::SetOptionValue (CommandInterpreter &interpreter,
                                   uint32_t option_idx,
                                   const char *option_arg)
{
    Error error;
    char short_option = (char) g_option_table[option_idx].short_option;

    switch (short_option)
    {
        case 'f':
            error = m_format.SetValueFromCString (option_arg);
            break;

        case 'c':
            if (m_count.GetDefaultValue() == 0)
            {
                error.SetErrorString ("--count option is disabled");
            }
            else
            {
                error = m_count.SetValueFromCString (option_arg);
                if (m_count.GetCurrentValue() == 0)
                    error.SetErrorStringWithFormat("invalid --count option value '%s'", option_arg);
            }
            break;
            
        case 's':
            if (m_byte_size.GetDefaultValue() == 0)
            {
                error.SetErrorString ("--size option is disabled");
            }
            else
            {
                error = m_byte_size.SetValueFromCString (option_arg);
                if (m_byte_size.GetCurrentValue() == 0)
                    error.SetErrorStringWithFormat("invalid --size option value '%s'", option_arg);
            }
            break;

        case 'G':
            {
                char *end = NULL;
                const char *gdb_format_cstr = option_arg; 
                uint64_t count = 0;
                if (::isdigit (gdb_format_cstr[0]))
                {
                    count = strtoull (gdb_format_cstr, &end, 0);

                    if (option_arg != end)
                        gdb_format_cstr = end;  // We have a valid count, advance the string position
                    else
                        count = 0;
                }

                Format format = SetFormatUsingGDBFormatLetter (gdb_format_cstr[0]);
                if (format != eFormatInvalid)
                    ++gdb_format_cstr;

                uint32_t byte_size = SetByteSizeUsingGDBSizeLetter (gdb_format_cstr[0]);
                if (byte_size > 0)
                    ++gdb_format_cstr;
                
                // We the first character of the "gdb_format_cstr" is not the 
                // NULL terminator, we didn't consume the entire string and 
                // something is wrong. Also, if none of the format, size or count
                // was specified correctly, then abort.
                if (gdb_format_cstr[0] || (format == eFormatInvalid && byte_size == 0 && count == 0))
                {
                    // Nothing got set correctly
                    error.SetErrorStringWithFormat ("invalid gdb format string '%s'", option_arg);
                    return error;
                }

                // At least one of the format, size or count was set correctly.
                // Anything that wasn't set correctly should be set to the
                // previous default
                if (format == eFormatInvalid)
                    format = SetFormatUsingGDBFormatLetter (m_prev_gdb_format);
                
                const bool byte_size_enabled = m_byte_size.GetDefaultValue() < UINT64_MAX;
                const bool count_enabled = m_count.GetDefaultValue() < UINT64_MAX;
                if (byte_size_enabled)
                {
                    // Byte size is enabled
                    if (byte_size == 0)
                        byte_size = SetByteSizeUsingGDBSizeLetter (m_prev_gdb_size);
                }
                else
                {
                    // Byte size is disabled, make sure it wasn't specified
                    if (byte_size > 0)
                    {
                        error.SetErrorString ("this command doesn't support specifying a byte size");
                        return error;
                    }
                }

                if (count_enabled)
                {
                    // Count is enabled and was not set, set it to the default
                    if (count == 0)
                        count = m_count.GetDefaultValue();
                }
                else
                {
                    // Count is disabled, make sure it wasn't specified
                    if (count > 0)
                    {
                        error.SetErrorString ("this command doesn't support specifying a count");
                        return error;
                    }
                }

                m_format.SetCurrentValue (format);
                m_format.SetOptionWasSet ();
                if (byte_size_enabled)
                {
                    m_byte_size.SetCurrentValue (byte_size);
                    m_byte_size.SetOptionWasSet ();
                }
                if (count_enabled)
                {
                    m_count.SetCurrentValue(count);
                    m_count.SetOptionWasSet ();
                }
            }
            break;

        default:
            error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
            break;
    }

    return error;
}

Format
OptionGroupFormat::SetFormatUsingGDBFormatLetter (char format_letter)
{
    Format format = eFormatInvalid;
    switch (format_letter)
    {
        case 'o': format = eFormatOctal;        break; 
        case 'x': format = eFormatHex;          break;
        case 'd': format = eFormatDecimal;      break;
        case 'u': format = eFormatUnsigned;     break;
        case 't': format = eFormatBinary;       break;
        case 'f': format = eFormatFloat;        break;
        case 'a': format = eFormatHex;          break; // TODO: add a new format: eFormatAddress
        case 'i': format = eFormatHex;          break; // TODO: add a new format: eFormatInstruction
        case 'c': format = eFormatChar;         break;
        case 's': format = eFormatCString;      break;
        case 'T': format = eFormatOSType;       break;
        case 'A': format = eFormatHex;          break; // TODO: add a new format: eFormatHexFloat
        default:  break;
    }
    if (format != eFormatInvalid)
        m_prev_gdb_format = format_letter;
    return format;
}

uint32_t
OptionGroupFormat::SetByteSizeUsingGDBSizeLetter (char size_letter)
{
    uint32_t byte_size = 0;
    switch (size_letter)
    {
        case 'b': // byte
            byte_size = 1; 
            break;

        case 'h': // halfword
            byte_size = 2;
            break;

        case 'w': // word
            byte_size = 4;
            break;

        case 'g': // giant
            byte_size = 8;
            break;

        default:
            break;
    }
    if (byte_size)
        m_prev_gdb_size = size_letter;
    return byte_size;
}


void
OptionGroupFormat::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_format.Clear();
    m_byte_size.Clear();
    m_count.Clear();
}
