//===-- CommandObjectMemory.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectMemory.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Read memory from the inferior process
//----------------------------------------------------------------------
class CommandObjectMemoryRead : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            ResetOptionValues();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'f':
                error = Args::StringToFormat (option_arg, m_format);

                switch (m_format)
                {
                default:
                    break;

                case eFormatBoolean:
                    if (m_byte_size == 0)
                        m_byte_size = 1;
                    if (m_num_per_line == 0)
                        m_num_per_line = 1;
                    break;

                case eFormatCString:
                    if (m_num_per_line == 0)
                        m_num_per_line = 1;
                    break;

                case eFormatPointer:
                    break;

                case eFormatBinary:
                case eFormatFloat:
                case eFormatOctal:
                case eFormatDecimal:
                case eFormatEnum:
                case eFormatUnicode16:
                case eFormatUnicode32:
                case eFormatUnsigned:
                    if (m_byte_size == 0)
                        m_byte_size = 4;
                    if (m_num_per_line == 0)
                        m_num_per_line = 1;
                    break;

                case eFormatBytes:
                case eFormatBytesWithASCII:
                case eFormatChar:
                case eFormatCharPrintable:
                    if (m_byte_size == 0)
                        m_byte_size = 1;
                    break;
                case eFormatComplex:
                    if (m_byte_size == 0)
                        m_byte_size = 8;
                    break;
                case eFormatHex:
                    if (m_byte_size == 0)
                        m_byte_size = 4;
                    break;

                case eFormatVectorOfChar:
                case eFormatVectorOfSInt8:
                case eFormatVectorOfUInt8:
                case eFormatVectorOfSInt16:
                case eFormatVectorOfUInt16:
                case eFormatVectorOfSInt32:
                case eFormatVectorOfUInt32:
                case eFormatVectorOfSInt64:
                case eFormatVectorOfUInt64:
                case eFormatVectorOfFloat32:
                case eFormatVectorOfFloat64:
                case eFormatVectorOfUInt128:
                    break;
                }
                break;

            case 'l':
                m_num_per_line = Args::StringToUInt32 (option_arg, 0);
                if (m_num_per_line == 0)
                    error.SetErrorStringWithFormat("Invalid value for --num-per-line option '%s'. Must be positive integer value.\n", option_arg);
                break;

            case 'c':
                m_count = Args::StringToUInt32 (option_arg, 0);
                if (m_count == 0)
                    error.SetErrorStringWithFormat("Invalid value for --count option '%s'. Must be positive integer value.\n", option_arg);
                break;

            case 's':
                m_byte_size = Args::StringToUInt32 (option_arg, 0);
                if (m_byte_size == 0)
                    error.SetErrorStringWithFormat("Invalid value for --size option '%s'. Must be positive integer value.\n", option_arg);
                break;

            case 'o':
                m_outfile_filespec.SetFile (option_arg, true);
                break;

            case 'b':
                m_output_as_binary = true;
                break;

            case 'a':
                m_append_to_outfile = true;
                break;
            
            default:
                error.SetErrorStringWithFormat("Unrecognized short option '%c'.\n", short_option);
                break;
            }
            return error;
        }

        void
        ResetOptionValues ()
        {
            m_format = eFormatBytesWithASCII;
            m_byte_size = 0;
            m_count = 0;
            m_num_per_line = 0;
            m_outfile_filespec.Clear();
            m_append_to_outfile = false;
            m_output_as_binary = false;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        lldb::Format m_format;
        uint32_t m_byte_size;
        uint32_t m_count;
        uint32_t m_num_per_line;
        FileSpec m_outfile_filespec;
        bool m_append_to_outfile;
        bool m_output_as_binary;
    };

    CommandObjectMemoryRead (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "memory read",
                       "Read from the memory of the process being debugged.",
                       NULL,
                       eFlagProcessMustBeLaunched),
        m_options (interpreter)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData start_addr_arg;
        CommandArgumentData end_addr_arg;
        
        // Define the first (and only) variant of this arg.
        start_addr_arg.arg_type = eArgTypeStartAddress;
        start_addr_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (start_addr_arg);
        
        // Define the first (and only) variant of this arg.
        end_addr_arg.arg_type = eArgTypeEndAddress;
        end_addr_arg.arg_repetition = eArgRepeatOptional;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (end_addr_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectMemoryRead ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError("need a process to read memory");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        const size_t argc = command.GetArgumentCount();

        if (argc == 0 || argc > 2)
        {
            result.AppendErrorWithFormat ("%s takes 1 or two args.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        size_t item_byte_size = m_options.m_byte_size;
        if (item_byte_size == 0)
        {
            if (m_options.m_format == eFormatPointer)
                item_byte_size = process->GetTarget().GetArchitecture().GetAddressByteSize();
            else
                item_byte_size = 1;
        }

        size_t item_count = m_options.m_count;

        size_t num_per_line = m_options.m_num_per_line;
        if (num_per_line == 0)
        {
            num_per_line = (16/item_byte_size);
            if (num_per_line == 0)
                num_per_line = 1;
        }

        size_t total_byte_size = m_options.m_count * item_byte_size;
        if (total_byte_size == 0)
            total_byte_size = 32;

        lldb::addr_t addr = Args::StringToUInt64(command.GetArgumentAtIndex(0), LLDB_INVALID_ADDRESS, 0);

        if (addr == LLDB_INVALID_ADDRESS)
        {
            result.AppendErrorWithFormat("invalid start address string '%s'.\n", command.GetArgumentAtIndex(0));
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        if (argc == 2)
        {
            lldb::addr_t end_addr = Args::StringToUInt64(command.GetArgumentAtIndex(1), LLDB_INVALID_ADDRESS, 0);
            if (end_addr == LLDB_INVALID_ADDRESS)
            {
                result.AppendErrorWithFormat("Invalid end address string '%s'.\n", command.GetArgumentAtIndex(1));
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            else if (end_addr <= addr)
            {
                result.AppendErrorWithFormat("End address (0x%llx) must be greater that the start address (0x%llx).\n", end_addr, addr);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            else if (item_count != 0)
            {
                result.AppendErrorWithFormat("Specify either the end address (0x%llx) or the count (--count %u), not both.\n", end_addr, item_count);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }

            total_byte_size = end_addr - addr;
            item_count = total_byte_size / item_byte_size;
        }
        else
        {
            if (item_count == 0)
                item_count = 32;
        }

        DataBufferSP data_sp(new DataBufferHeap (total_byte_size, '\0'));
        Error error;
        size_t bytes_read = process->ReadMemory(addr, data_sp->GetBytes (), data_sp->GetByteSize(), error);
        if (bytes_read == 0)
        {
            result.AppendWarningWithFormat("Read from 0x%llx failed.\n", addr);
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        if (bytes_read < total_byte_size)
            result.AppendWarningWithFormat("Not all bytes (%u/%u) were able to be read from 0x%llx.\n", bytes_read, total_byte_size, addr);

        result.SetStatus(eReturnStatusSuccessFinishResult);
        DataExtractor data (data_sp, 
                            process->GetTarget().GetArchitecture().GetByteOrder(), 
                            process->GetTarget().GetArchitecture().GetAddressByteSize());

        StreamFile outfile_stream;
        Stream *output_stream = NULL;

        if (m_options.m_outfile_filespec)
        {
            char path[PATH_MAX];
            m_options.m_outfile_filespec.GetPath (path, sizeof(path));
            char mode[16] = { 'w', '\0' };
            if (m_options.m_append_to_outfile)
                mode[0] = 'a';
                
            if (outfile_stream.GetFile ().Open (path, File::eOpenOptionWrite | File::eOpenOptionCanCreate).Success())
            {
                if (m_options.m_output_as_binary)
                {
                    int bytes_written = outfile_stream.Write (data_sp->GetBytes(), bytes_read);
                    if (bytes_written > 0)
                    {
                        result.GetOutputStream().Printf ("%i bytes %s to '%s'\n", 
                                                         bytes_written, 
                                                         m_options.m_append_to_outfile ? "appended" : "written", 
                                                         path);
                        return true;
                    }
                    else 
                    {
                        result.AppendErrorWithFormat("Failed to write %zu bytes to '%s'.\n", bytes_read, path);
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
                else
                {
                    // We are going to write ASCII to the file just point the
                    // output_stream to our outfile_stream...
                    output_stream = &outfile_stream;
                }
            }
            else 
            {
                result.AppendErrorWithFormat("Failed to open file '%s' with a mode of '%s'.\n", path, mode);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        else 
        {
            output_stream = &result.GetOutputStream();
        }

        assert (output_stream);
        data.Dump (output_stream,
                   0,
                   m_options.m_format,
                   item_byte_size,
                   item_count,
                   num_per_line,
                   addr,
                   0,
                   0);
        output_stream->EOL();
        return true;
    }

protected:
    CommandOptions m_options;
};

#define SET1 LLDB_OPT_SET_1
#define SET2 LLDB_OPT_SET_2

OptionDefinition
CommandObjectMemoryRead::CommandOptions::g_option_table[] =
{
{ SET1       , false, "format",       'f', required_argument, NULL, 0, eArgTypeFormat,       "The format that will be used to display the memory. Defaults to bytes with ASCII (--format=Y)."},
{ SET1       , false, "size",         's', required_argument, NULL, 0, eArgTypeByteSize,     "The size in bytes to use when displaying with the selected format."},
{ SET1       , false, "num-per-line", 'l', required_argument, NULL, 0, eArgTypeNumberPerLine,"The number of items per line to display."},
{ SET1       , false, "count",        'c', required_argument, NULL, 0, eArgTypeCount,        "The number of total items to display."},
{ SET1 | SET2, false, "outfile",      'o', required_argument, NULL, 0, eArgTypeFilename,     "Dump memory read results into a file."},
{ SET1 | SET2, false, "append",       'a', no_argument,       NULL, 0, eArgTypeNone,         "Append memory read results to 'outfile'."},
{        SET2, false, "binary",       'b', no_argument,       NULL, 0, eArgTypeNone,         "If true, memory will be saved as binary. If false, the memory is saved save as an ASCII dump that uses the format, size, count and number per line settings."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#undef SET1
#undef SET2

//----------------------------------------------------------------------
// Write memory to the inferior process
//----------------------------------------------------------------------
class CommandObjectMemoryWrite : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            ResetOptionValues();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            switch (short_option)
            {
            case 'f':
                error = Args::StringToFormat (option_arg, m_format);
                break;

            case 's':
                m_byte_size = Args::StringToUInt32 (option_arg, 0);
                if (m_byte_size == 0)
                    error.SetErrorStringWithFormat("Invalid value for --size option '%s'.  Must be positive integer value.\n", option_arg);
                break;

            case 'i':
                m_infile.SetFile (option_arg, true);
                if (!m_infile.Exists())
                {
                    m_infile.Clear();
                    error.SetErrorStringWithFormat("Input file does not exist: '%s'\n", option_arg);
                }
                break;
            
            case 'o':
                {
                    bool success;
                    m_infile_offset = Args::StringToUInt64(option_arg, 0, 0, &success);
                    if (!success)
                    {
                        error.SetErrorStringWithFormat("Invalid offset string '%s'\n", option_arg);
                    }
                }
                break;

            default:
                error.SetErrorStringWithFormat("Unrecognized short option '%c'\n", short_option);
                break;
            }
            return error;
        }

        void
        ResetOptionValues ()
        {
            m_format = eFormatBytes;
            m_byte_size = 1;
            m_infile.Clear();
            m_infile_offset = 0;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        lldb::Format m_format;
        uint32_t m_byte_size;
        FileSpec m_infile;
        off_t m_infile_offset;
    };

    CommandObjectMemoryWrite (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "memory write",
                       "Write to the memory of the process being debugged.",
                       //"memory write [<cmd-options>] <addr> [value1 value2 ...]",
                       NULL,
                       eFlagProcessMustBeLaunched),
        m_options (interpreter)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData addr_arg;
        CommandArgumentData value_arg;
        
        // Define the first (and only) variant of this arg.
        addr_arg.arg_type = eArgTypeAddress;
        addr_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (addr_arg);
        
        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlus;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectMemoryWrite ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    bool
    UIntValueIsValidForSize (uint64_t uval64, size_t total_byte_size)
    {
        if (total_byte_size > 8)
            return false;

        if (total_byte_size == 8)
            return true;

        const uint64_t max = ((uint64_t)1 << (uint64_t)(total_byte_size * 8)) - 1;
        return uval64 <= max;
    }

    bool
    SIntValueIsValidForSize (int64_t sval64, size_t total_byte_size)
    {
        if (total_byte_size > 8)
            return false;

        if (total_byte_size == 8)
            return true;

        const int64_t max = ((int64_t)1 << (uint64_t)(total_byte_size * 8 - 1)) - 1;
        const int64_t min = ~(max);
        return min <= sval64 && sval64 <= max;
    }

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError("need a process to read memory");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        const size_t argc = command.GetArgumentCount();

        if (m_options.m_infile)
        {
            if (argc < 1)
            {
                result.AppendErrorWithFormat ("%s takes a destination address when writing file contents.\n", m_cmd_name.c_str());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }       
        }
        else if (argc < 2)
        {
            result.AppendErrorWithFormat ("%s takes a destination address and at least one value.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        StreamString buffer (Stream::eBinary,
                             process->GetTarget().GetArchitecture().GetAddressByteSize(),
                             process->GetTarget().GetArchitecture().GetByteOrder());

        size_t item_byte_size = m_options.m_byte_size;

        lldb::addr_t addr = Args::StringToUInt64(command.GetArgumentAtIndex(0), LLDB_INVALID_ADDRESS, 0);

        if (addr == LLDB_INVALID_ADDRESS)
        {
            result.AppendErrorWithFormat("Invalid address string '%s'.\n", command.GetArgumentAtIndex(0));
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        if (m_options.m_infile)
        {
            size_t length = SIZE_MAX;
            if (m_options.m_byte_size > 0)
                length = m_options.m_byte_size;
            lldb::DataBufferSP data_sp (m_options.m_infile.ReadFileContents (m_options.m_infile_offset, length));
            if (data_sp)
            {
                length = data_sp->GetByteSize();
                if (length > 0)
                {
                    Error error;
                    size_t bytes_written = process->WriteMemory (addr, data_sp->GetBytes(), length, error);
                    
                    if (bytes_written == length)
                    {
                        // All bytes written
                        result.GetOutputStream().Printf("%zu bytes were written to 0x%llx\n", bytes_written, addr);
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                    }
                    else if (bytes_written > 0)
                    {
                        // Some byte written
                        result.GetOutputStream().Printf("%zu bytes of %zu requested were written to 0x%llx\n", bytes_written, length, addr);
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                    }
                    else 
                    {
                        result.AppendErrorWithFormat ("Memory write to 0x%llx failed: %s.\n", addr, error.AsCString());
                        result.SetStatus(eReturnStatusFailed);
                    }
                }
            }
            else
            {
                result.AppendErrorWithFormat ("Unable to read contents of file.\n");
                result.SetStatus(eReturnStatusFailed);
            }
            return result.Succeeded();
        }
        else if (m_options.m_byte_size == 0)
        {
            if (m_options.m_format == eFormatPointer)
                item_byte_size = buffer.GetAddressByteSize();
            else
                item_byte_size = 1;
        }

        command.Shift(); // shift off the address argument
        uint64_t uval64;
        int64_t sval64;
        bool success = false;
        const uint32_t num_value_args = command.GetArgumentCount();
        uint32_t i;
        for (i=0; i<num_value_args; ++i)
        {
            const char *value_str = command.GetArgumentAtIndex(i);

            switch (m_options.m_format)
            {
            case eFormatFloat:  // TODO: add support for floats soon
            case eFormatCharPrintable:
            case eFormatBytesWithASCII:
            case eFormatComplex:
            case eFormatEnum:
            case eFormatUnicode16:
            case eFormatUnicode32:
            case eFormatVectorOfChar:
            case eFormatVectorOfSInt8:
            case eFormatVectorOfUInt8:
            case eFormatVectorOfSInt16:
            case eFormatVectorOfUInt16:
            case eFormatVectorOfSInt32:
            case eFormatVectorOfUInt32:
            case eFormatVectorOfSInt64:
            case eFormatVectorOfUInt64:
            case eFormatVectorOfFloat32:
            case eFormatVectorOfFloat64:
            case eFormatVectorOfUInt128:
            case eFormatOSType:
            case eFormatComplexInteger:
                result.AppendError("unsupported format for writing memory");
                result.SetStatus(eReturnStatusFailed);
                return false;

            case eFormatDefault:
            case eFormatBytes:
            case eFormatHex:
            case eFormatPointer:
                
                // Decode hex bytes
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 16, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid hex string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value 0x%llx is too large to fit in a %u byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatBoolean:
                uval64 = Args::StringToBoolean(value_str, false, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid boolean string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatBinary:
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 2, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid binary string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value 0x%llx is too large to fit in a %u byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatChar:
            case eFormatCString:
                if (value_str[0])
                {
                    size_t len = strlen (value_str);
                    // Include the NULL for C strings...
                    if (m_options.m_format == eFormatCString)
                        ++len;
                    Error error;
                    if (process->WriteMemory (addr, value_str, len, error) == len)
                    {
                        addr += len;
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("Memory write to 0x%llx failed: %s.\n", addr, error.AsCString());
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
                break;

            case eFormatDecimal:
                sval64 = Args::StringToSInt64(value_str, INT64_MAX, 0, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid signed decimal value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!SIntValueIsValidForSize (sval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value %lli is too large or small to fit in a %u byte signed integer value.\n", sval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (sval64, item_byte_size);
                break;

            case eFormatUnsigned:
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 0, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid unsigned decimal string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value %llu is too large to fit in a %u byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatOctal:
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 8, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid octal string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value %llo is too large to fit in a %u byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;
            }
        }

        if (!buffer.GetString().empty())
        {
            Error error;
            if (process->WriteMemory (addr, buffer.GetString().c_str(), buffer.GetString().size(), error) == buffer.GetString().size())
                return true;
            else
            {
                result.AppendErrorWithFormat ("Memory write to 0x%llx failed: %s.\n", addr, error.AsCString());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        return true;
    }

protected:
    CommandOptions m_options;
};

#define SET1 LLDB_OPT_SET_1
#define SET2 LLDB_OPT_SET_2

OptionDefinition
CommandObjectMemoryWrite::CommandOptions::g_option_table[] =
{
{ SET1       , false, "format", 'f', required_argument, NULL, 0, eArgTypeFormat,   "The format value types that will be decoded and written to memory."},
{ SET1 | SET2, false, "size",   's', required_argument, NULL, 0, eArgTypeByteSize, "The size in bytes of the values to write to memory."},
{        SET2, true,  "infile", 'i', required_argument, NULL, 0, eArgTypeFilename, "Write memory using the contents of a file."},
{        SET2, false, "offset", 'o', required_argument, NULL, 0, eArgTypeOffset,   "Start writng bytes from an offset within the input file."},
{ 0          , false, NULL    ,  0 , 0                , NULL, 0, eArgTypeNone,     NULL }
};

#undef SET1
#undef SET2

//-------------------------------------------------------------------------
// CommandObjectMemory
//-------------------------------------------------------------------------

CommandObjectMemory::CommandObjectMemory (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "memory",
                            "A set of commands for operating on memory.",
                            "memory <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("read",  CommandObjectSP (new CommandObjectMemoryRead (interpreter)));
    LoadSubCommand ("write", CommandObjectSP (new CommandObjectMemoryWrite (interpreter)));
}

CommandObjectMemory::~CommandObjectMemory ()
{
}
