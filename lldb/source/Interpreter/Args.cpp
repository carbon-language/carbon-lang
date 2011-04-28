//===-- Args.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <getopt.h>
#include <cstdlib>
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Args constructor
//----------------------------------------------------------------------
Args::Args (const char *command) :
    m_args(),
    m_argv(),
    m_args_quote_char()
{
    if (command)
        SetCommandString (command);
}


Args::Args (const char *command, size_t len) :
    m_args(),
    m_argv(),
    m_args_quote_char()
{
    if (command && len)
        SetCommandString (command, len);
}

//----------------------------------------------------------------------
// We have to be very careful on the copy constructor of this class
// to make sure we copy all of the string values, but we can't copy the
// rhs.m_argv into m_argv since it will point to the "const char *" c 
// strings in rhs.m_args. We need to copy the string list and update our
// own m_argv appropriately. 
//----------------------------------------------------------------------
Args::Args (const Args &rhs) :
    m_args (rhs.m_args),
    m_argv (),
    m_args_quote_char(rhs.m_args_quote_char)
{
    UpdateArgvFromArgs();
}

//----------------------------------------------------------------------
// We have to be very careful on the copy constructor of this class
// to make sure we copy all of the string values, but we can't copy the
// rhs.m_argv into m_argv since it will point to the "const char *" c 
// strings in rhs.m_args. We need to copy the string list and update our
// own m_argv appropriately. 
//----------------------------------------------------------------------
const Args &
Args::operator= (const Args &rhs)
{
    // Make sure we aren't assigning to self
    if (this != &rhs)
    {
        m_args = rhs.m_args;
        m_args_quote_char = rhs.m_args_quote_char;
        UpdateArgvFromArgs();
    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Args::~Args ()
{
}

void
Args::Dump (Stream *s)
{
//    int argc = GetArgumentCount();
//
//    arg_sstr_collection::const_iterator pos, begin = m_args.begin(), end = m_args.end();
//    for (pos = m_args.begin(); pos != end; ++pos)
//    {
//        s->Indent();
//        s->Printf("args[%zu]=%s\n", std::distance(begin, pos), pos->c_str());
//    }
//    s->EOL();
    const int argc = m_argv.size();
    for (int i=0; i<argc; ++i)
    {
        s->Indent();
        const char *arg_cstr = m_argv[i];
        if (arg_cstr)
            s->Printf("argv[%i]=\"%s\"\n", i, arg_cstr);
        else
            s->Printf("argv[%i]=NULL\n", i);
    }
    s->EOL();
}

bool
Args::GetCommandString (std::string &command)
{
    command.clear();
    int argc = GetArgumentCount();
    for (int i=0; i<argc; ++i)
    {
        if (i > 0)
            command += ' ';
        command += m_argv[i];
    }
    return argc > 0;
}

bool
Args::GetQuotedCommandString (std::string &command)
{
    command.clear ();
    size_t argc = GetArgumentCount ();
    for (size_t i = 0; i < argc; ++i)
    {
        if (i > 0)
            command.append (1, ' ');
        char quote_char = GetArgumentQuoteCharAtIndex(i);
        if (quote_char)
        {
            command.append (1, quote_char);
            command.append (m_argv[i]);
            command.append (1, quote_char);
        }
        else
            command.append (m_argv[i]);
    }
    return argc > 0;
}

void
Args::SetCommandString (const char *command, size_t len)
{
    // Use std::string to make sure we get a NULL terminated string we can use
    // as "command" could point to a string within a large string....
    std::string null_terminated_command(command, len);
    SetCommandString(null_terminated_command.c_str());
}

void
Args::SetCommandString (const char *command)
{
    m_args.clear();
    m_argv.clear();
    m_args_quote_char.clear();

    if (command && command[0])
    {
        static const char *k_space_separators = " \t";
        static const char *k_space_separators_with_slash_and_quotes = " \t \\'\"`";
        const char *arg_end = NULL;
        const char *arg_pos;
        for (arg_pos = command;
             arg_pos && arg_pos[0];
             arg_pos = arg_end)
        {
            // Skip any leading space separators
            const char *arg_start = ::strspn (arg_pos, k_space_separators) + arg_pos;
            
            // If there were only space separators to the end of the line, then
            // we're done.
            if (*arg_start == '\0')
                break;

            // Arguments can be split into multiple discontiguous pieces,
            // for example:
            //  "Hello ""World"
            // this would result in a single argument "Hello World" (without/
            // the quotes) since the quotes would be removed and there is 
            // not space between the strings. So we need to keep track of the
            // current start of each argument piece in "arg_piece_start"
            const char *arg_piece_start = arg_start;
            arg_pos = arg_piece_start;

            std::string arg;
            // Since we can have multiple quotes that form a single command
            // in a command like: "Hello "world'!' (which will make a single
            // argument "Hello world!") we remember the first quote character
            // we encounter and use that for the quote character.
            char first_quote_char = '\0';
            char quote_char = '\0';
            bool arg_complete = false;

            do
            {
                arg_end = ::strcspn (arg_pos, k_space_separators_with_slash_and_quotes) + arg_pos;

                switch (arg_end[0])
                {
                default:
                    assert (!"Unhandled case statement, we must handle this...");
                    break;

                case '\0':
                    // End of C string
                    if (arg_piece_start && arg_piece_start[0])
                        arg.append (arg_piece_start);
                    arg_complete = true;
                    break;
                    
                case '\\':
                    // Backslash character
                    switch (arg_end[1])
                    {
                        case '\0':
                            arg.append (arg_piece_start);
                            arg_complete = true;
                            break;

                        default:
                            arg_pos = arg_end + 2;
                            break;
                    }
                    break;
                
                case '"':
                case '\'':
                case '`':
                    // Quote characters 
                    if (quote_char)
                    {
                        // We found a quote character while inside a quoted
                        // character argument. If it matches our current quote
                        // character, this ends the effect of the quotes. If it
                        // doesn't we ignore it.
                        if (quote_char == arg_end[0])
                        {
                            arg.append (arg_piece_start, arg_end - arg_piece_start);
                            // Clear the quote character and let parsing
                            // continue (we need to watch for things like:
                            // "Hello ""World"
                            // "Hello "World
                            // "Hello "'World'
                            // All of which will result in a single argument "Hello World"
                            quote_char = '\0'; // Note that we are no longer inside quotes
                            arg_pos = arg_end + 1; // Skip the quote character
                            arg_piece_start = arg_pos; // Note we are starting from later in the string
                        }
                        else
                        {
                            // different quote, skip it and keep going
                            arg_pos = arg_end + 1;
                        }
                    }
                    else
                    {
                        // We found the start of a quote scope.
                        // Make sure there isn't a string that precedes
                        // the start of a quote scope like:
                        // Hello" World"
                        // If so, then add the "Hello" to the arg
                        if (arg_end > arg_piece_start)
                            arg.append (arg_piece_start, arg_end - arg_piece_start);
                            
                        // Enter into a quote scope
                        quote_char = arg_end[0];
                        
                        if (first_quote_char == '\0')
                            first_quote_char = quote_char;

                        arg_pos = arg_end;
                        
                        if (quote_char != '`')
                            ++arg_pos; // Skip the quote character if it is not a backtick

                        arg_piece_start = arg_pos; // Note we are starting from later in the string
                        
                        // Skip till the next quote character
                        const char *end_quote = ::strchr (arg_piece_start, quote_char);
                        while (end_quote && end_quote[-1] == '\\')
                        {
                            // Don't skip the quote character if it is 
                            // preceded by a '\' character
                            end_quote = ::strchr (end_quote + 1, quote_char);
                        }
                        
                        if (end_quote)
                        {
                            if (end_quote > arg_piece_start)
                            {
                                // Keep the backtick quote on commands
                                if (quote_char == '`')
                                    arg.append (arg_piece_start, end_quote + 1 - arg_piece_start);
                                else
                                    arg.append (arg_piece_start, end_quote - arg_piece_start);
                            }

                            // If the next character is a space or the end of 
                            // string, this argument is complete...
                            if (end_quote[1] == ' ' || end_quote[1] == '\t' || end_quote[1] == '\0')
                            {
                                arg_complete = true;
                                arg_end = end_quote + 1;
                            }
                            else
                            {
                                arg_pos = end_quote + 1;
                                arg_piece_start = arg_pos;
                            }
                            quote_char = '\0';
                        }
                    }
                    break;

                case ' ':
                case '\t':
                    if (quote_char)
                    {
                        // We are currently processing a quoted character and found
                        // a space character, skip any spaces and keep trying to find
                        // the end of the argument. 
                        arg_pos = ::strspn (arg_end, k_space_separators) + arg_end;
                    }
                    else
                    {
                        // We are not inside any quotes, we just found a space after an
                        // argument
                        if (arg_end > arg_piece_start)
                            arg.append (arg_piece_start, arg_end - arg_piece_start);
                        arg_complete = true;
                    }
                    break;
                }
            } while (!arg_complete);

            m_args.push_back(arg);
            m_args_quote_char.push_back (first_quote_char);
        }
        UpdateArgvFromArgs();
    }
}

void
Args::UpdateArgsAfterOptionParsing()
{
    // Now m_argv might be out of date with m_args, so we need to fix that
    arg_cstr_collection::const_iterator argv_pos, argv_end = m_argv.end();
    arg_sstr_collection::iterator args_pos;
    arg_quote_char_collection::iterator quotes_pos;

    for (argv_pos = m_argv.begin(), args_pos = m_args.begin(), quotes_pos = m_args_quote_char.begin();
         argv_pos != argv_end && args_pos != m_args.end();
         ++argv_pos)
    {
        const char *argv_cstr = *argv_pos;
        if (argv_cstr == NULL)
            break;

        while (args_pos != m_args.end())
        {
            const char *args_cstr = args_pos->c_str();
            if (args_cstr == argv_cstr)
            {
                // We found the argument that matches the C string in the
                // vector, so we can now look for the next one
                ++args_pos;
                ++quotes_pos;
                break;
            }
            else
            {
                quotes_pos = m_args_quote_char.erase (quotes_pos);
                args_pos = m_args.erase (args_pos);
            }
        }
    }

    if (args_pos != m_args.end())
        m_args.erase (args_pos, m_args.end());

    if (quotes_pos != m_args_quote_char.end())
        m_args_quote_char.erase (quotes_pos, m_args_quote_char.end());
}

void
Args::UpdateArgvFromArgs()
{
    m_argv.clear();
    arg_sstr_collection::const_iterator pos, end = m_args.end();
    for (pos = m_args.begin(); pos != end; ++pos)
        m_argv.push_back(pos->c_str());
    m_argv.push_back(NULL);
    // Make sure we have enough arg quote chars in the array
    if (m_args_quote_char.size() < m_args.size())
        m_args_quote_char.resize (m_argv.size());
}

size_t
Args::GetArgumentCount() const
{
    if (m_argv.empty())
        return 0;
    return m_argv.size() - 1;
}

const char *
Args::GetArgumentAtIndex (size_t idx) const
{
    if (idx < m_argv.size())
        return m_argv[idx];
    return NULL;
}

char
Args::GetArgumentQuoteCharAtIndex (size_t idx) const
{
    if (idx < m_args_quote_char.size())
        return m_args_quote_char[idx];
    return '\0';
}

char **
Args::GetArgumentVector()
{
    if (!m_argv.empty())
        return (char **)&m_argv[0];
    return NULL;
}

const char **
Args::GetConstArgumentVector() const
{
    if (!m_argv.empty())
        return (const char **)&m_argv[0];
    return NULL;
}

void
Args::Shift ()
{
    // Don't pop the last NULL terminator from the argv array
    if (m_argv.size() > 1)
    {
        m_argv.erase(m_argv.begin());
        m_args.pop_front();
        if (!m_args_quote_char.empty())
            m_args_quote_char.erase(m_args_quote_char.begin());
    }
}

const char *
Args::Unshift (const char *arg_cstr, char quote_char)
{
    m_args.push_front(arg_cstr);
    m_argv.insert(m_argv.begin(), m_args.front().c_str());
    m_args_quote_char.insert(m_args_quote_char.begin(), quote_char);
    return GetArgumentAtIndex (0);
}

void
Args::AppendArguments (const Args &rhs)
{
    const size_t rhs_argc = rhs.GetArgumentCount();
    for (size_t i=0; i<rhs_argc; ++i)
        AppendArgument(rhs.GetArgumentAtIndex(i));
}

const char *
Args::AppendArgument (const char *arg_cstr, char quote_char)
{
    return InsertArgumentAtIndex (GetArgumentCount(), arg_cstr, quote_char);
}

const char *
Args::InsertArgumentAtIndex (size_t idx, const char *arg_cstr, char quote_char)
{
    // Since we are using a std::list to hold onto the copied C string and
    // we don't have direct access to the elements, we have to iterate to
    // find the value.
    arg_sstr_collection::iterator pos, end = m_args.end();
    size_t i = idx;
    for (pos = m_args.begin(); i > 0 && pos != end; ++pos)
        --i;

    pos = m_args.insert(pos, arg_cstr);
    
    if (idx >= m_args_quote_char.size())
    {
        m_args_quote_char.resize(idx + 1);
        m_args_quote_char[idx] = quote_char;
    }
    else
        m_args_quote_char.insert(m_args_quote_char.begin() + idx, quote_char);
    
    UpdateArgvFromArgs();
    return GetArgumentAtIndex(idx);
}

const char *
Args::ReplaceArgumentAtIndex (size_t idx, const char *arg_cstr, char quote_char)
{
    // Since we are using a std::list to hold onto the copied C string and
    // we don't have direct access to the elements, we have to iterate to
    // find the value.
    arg_sstr_collection::iterator pos, end = m_args.end();
    size_t i = idx;
    for (pos = m_args.begin(); i > 0 && pos != end; ++pos)
        --i;

    if (pos != end)
    {
        pos->assign(arg_cstr);
        assert(idx < m_argv.size() - 1);
        m_argv[idx] = pos->c_str();
        if (idx >= m_args_quote_char.size())
            m_args_quote_char.resize(idx + 1);
        m_args_quote_char[idx] = quote_char;
        return GetArgumentAtIndex(idx);
    }
    return NULL;
}

void
Args::DeleteArgumentAtIndex (size_t idx)
{
    // Since we are using a std::list to hold onto the copied C string and
    // we don't have direct access to the elements, we have to iterate to
    // find the value.
    arg_sstr_collection::iterator pos, end = m_args.end();
    size_t i = idx;
    for (pos = m_args.begin(); i > 0 && pos != end; ++pos)
        --i;

    if (pos != end)
    {
        m_args.erase (pos);
        assert(idx < m_argv.size() - 1);
        m_argv.erase(m_argv.begin() + idx);
        if (idx < m_args_quote_char.size())
            m_args_quote_char.erase(m_args_quote_char.begin() + idx);
    }
}

void
Args::SetArguments (int argc, const char **argv)
{
    // m_argv will be rebuilt in UpdateArgvFromArgs() below, so there is
    // no need to clear it here.
    m_args.clear();
    m_args_quote_char.clear();

    // Make a copy of the arguments in our internal buffer
    size_t i;
    // First copy each string
    for (i=0; i<argc; ++i)
    {
        m_args.push_back (argv[i]);
        if ((argv[i][0] == '\'') || (argv[i][0] == '"') || (argv[i][0] == '`'))
            m_args_quote_char.push_back (argv[i][0]);
        else
            m_args_quote_char.push_back ('\0');
    }

    UpdateArgvFromArgs();
}


Error
Args::ParseOptions (Options &options)
{
    StreamString sstr;
    Error error;
    struct option *long_options = options.GetLongOptions();
    if (long_options == NULL)
    {
        error.SetErrorStringWithFormat("Invalid long options.\n");
        return error;
    }

    for (int i=0; long_options[i].name != NULL; ++i)
    {
        if (long_options[i].flag == NULL)
        {
            sstr << (char)long_options[i].val;
            switch (long_options[i].has_arg)
            {
            default:
            case no_argument:                       break;
            case required_argument: sstr << ':';    break;
            case optional_argument: sstr << "::";   break;
            }
        }
    }
#ifdef __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    int val;
    while (1)
    {
        int long_options_index = -1;
        val = ::getopt_long(GetArgumentCount(), GetArgumentVector(), sstr.GetData(), long_options,
                            &long_options_index);
        if (val == -1)
            break;

        // Did we get an error?
        if (val == '?')
        {
            error.SetErrorStringWithFormat("Unknown or ambiguous option.\n");
            break;
        }
        // The option auto-set itself
        if (val == 0)
            continue;

        ((Options *) &options)->OptionSeen (val);

        // Lookup the long option index
        if (long_options_index == -1)
        {
            for (int i=0;
                 long_options[i].name || long_options[i].has_arg || long_options[i].flag || long_options[i].val;
                 ++i)
            {
                if (long_options[i].val == val)
                {
                    long_options_index = i;
                    break;
                }
            }
        }
        // Call the callback with the option
        if (long_options_index >= 0)
        {
            error = options.SetOptionValue(long_options_index,
                                           long_options[long_options_index].has_arg == no_argument ? NULL : optarg);
        }
        else
        {
            error.SetErrorStringWithFormat("Invalid option with value '%i'.\n", val);
        }
        if (error.Fail())
            break;
    }

    // Update our ARGV now that get options has consumed all the options
    m_argv.erase(m_argv.begin(), m_argv.begin() + optind);
    UpdateArgsAfterOptionParsing ();
    return error;
}

void
Args::Clear ()
{
    m_args.clear ();
    m_argv.clear ();
    m_args_quote_char.clear();
}

int32_t
Args::StringToSInt32 (const char *s, int32_t fail_value, int base, bool *success_ptr)
{
    if (s && s[0])
    {
        char *end = NULL;
        int32_t uval = ::strtol (s, &end, base);
        if (*end == '\0')
        {
            if (success_ptr) *success_ptr = true;
            return uval; // All characters were used, return the result
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}

uint32_t
Args::StringToUInt32 (const char *s, uint32_t fail_value, int base, bool *success_ptr)
{
    if (s && s[0])
    {
        char *end = NULL;
        uint32_t uval = ::strtoul (s, &end, base);
        if (*end == '\0')
        {
            if (success_ptr) *success_ptr = true;
            return uval; // All characters were used, return the result
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}


int64_t
Args::StringToSInt64 (const char *s, int64_t fail_value, int base, bool *success_ptr)
{
    if (s && s[0])
    {
        char *end = NULL;
        int64_t uval = ::strtoll (s, &end, base);
        if (*end == '\0')
        {
            if (success_ptr) *success_ptr = true;
            return uval; // All characters were used, return the result
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}

uint64_t
Args::StringToUInt64 (const char *s, uint64_t fail_value, int base, bool *success_ptr)
{
    if (s && s[0])
    {
        char *end = NULL;
        uint64_t uval = ::strtoull (s, &end, base);
        if (*end == '\0')
        {
            if (success_ptr) *success_ptr = true;
            return uval; // All characters were used, return the result
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}

lldb::addr_t
Args::StringToAddress (const char *s, lldb::addr_t fail_value, bool *success_ptr)
{
    if (s && s[0])
    {
        char *end = NULL;
        lldb::addr_t addr = ::strtoull (s, &end, 0);
        if (*end == '\0')
        {
            if (success_ptr) *success_ptr = true;
            return addr; // All characters were used, return the result
        }
        // Try base 16 with no prefix...
        addr = ::strtoull (s, &end, 16);
        if (*end == '\0')
        {
            if (success_ptr) *success_ptr = true;
            return addr; // All characters were used, return the result
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}

bool
Args::StringToBoolean (const char *s, bool fail_value, bool *success_ptr)
{
    if (s && s[0])
    {
        if (::strcasecmp (s, "false") == 0 ||
            ::strcasecmp (s, "off") == 0 ||
            ::strcasecmp (s, "no") == 0 ||
                ::strcmp (s, "0") == 0)
        {
            if (success_ptr)
                *success_ptr = true;
            return false;
        }
        else
        if (::strcasecmp (s, "true") == 0 ||
            ::strcasecmp (s, "on") == 0 ||
            ::strcasecmp (s, "yes") == 0 ||
                ::strcmp (s, "1") == 0)
        {
            if (success_ptr) *success_ptr = true;
            return true;
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}

const char *
Args::StringToVersion (const char *s, uint32_t &major, uint32_t &minor, uint32_t &update)
{
    major = UINT32_MAX;
    minor = UINT32_MAX;
    update = UINT32_MAX;

    if (s && s[0])
    {
        char *pos = NULL;
        uint32_t uval32;
        uval32 = ::strtoul (s, &pos, 0);
        if (pos == s)
            return s;
        major = uval32;
        if (*pos == '\0')
        {
            return pos;   // Decoded major and got end of string
        }
        else if (*pos == '.')
        {
            const char *minor_cstr = pos + 1;
            uval32 = ::strtoul (minor_cstr, &pos, 0);
            if (pos == minor_cstr)
                return pos; // Didn't get any digits for the minor version...
            minor = uval32;
            if (*pos == '.')
            {
                const char *update_cstr = pos + 1;
                uval32 = ::strtoul (update_cstr, &pos, 0);
                if (pos == update_cstr)
                    return pos;
                update = uval32;
            }
            return pos;
        }
    }
    return 0;
}


int32_t
Args::StringToOptionEnum (const char *s, OptionEnumValueElement *enum_values, int32_t fail_value, bool *success_ptr)
{    
    if (enum_values && s && s[0])
    {
        for (int i = 0; enum_values[i].string_value != NULL ; i++) 
        {
            if (strstr(enum_values[i].string_value, s) == enum_values[i].string_value)
            {
                if (success_ptr) *success_ptr = true;
                return enum_values[i].value;
            }
        }
    }
    if (success_ptr) *success_ptr = false;
    
    return fail_value;
}

ScriptLanguage
Args::StringToScriptLanguage (const char *s, ScriptLanguage fail_value, bool *success_ptr)
{
    if (s && s[0])
    {
        if ((::strcasecmp (s, "python") == 0) ||
            (::strcasecmp (s, "default") == 0 && eScriptLanguagePython == eScriptLanguageDefault))
        {
            if (success_ptr) *success_ptr = true;
            return eScriptLanguagePython;
        }
        if (::strcasecmp (s, "none"))
        {
            if (success_ptr) *success_ptr = true;
            return eScriptLanguageNone;
        }
    }
    if (success_ptr) *success_ptr = false;
    return fail_value;
}

Error
Args::StringToFormat
(
    const char *s,
    lldb::Format &format,
    uint32_t *byte_size_ptr
)
{
    format = eFormatInvalid;
    Error error;

    if (s && s[0])
    {
        if (byte_size_ptr)
        {
            if (isdigit (s[0]))
            {
                char *format_char = NULL;
                unsigned long byte_size = ::strtoul (s, &format_char, 0);
                if (byte_size != ULONG_MAX)
                    *byte_size_ptr = byte_size;
                s = format_char;
            }
            else
                *byte_size_ptr = 0;
        }

        bool success = s[1] == '\0';
        if (success)
        {
            switch (s[0])
            {
            case 'y': format = eFormatBytes;            break;
            case 'Y': format = eFormatBytesWithASCII;   break;
            case 'b': format = eFormatBinary;           break;
            case 'B': format = eFormatBoolean;          break;
            case 'c': format = eFormatChar;             break;
            case 'C': format = eFormatCharPrintable;    break;
            case 'o': format = eFormatOctal;            break;
            case 'O': format = eFormatOSType;           break;
            case 'i':
            case 'd': format = eFormatDecimal;          break;
            case 'I': format = eFormatComplexInteger;   break;
            case 'u': format = eFormatUnsigned;         break;
            case 'x': format = eFormatHex;              break;
            case 'X': format = eFormatComplex;          break;
            case 'f':
            case 'e':
            case 'g': format = eFormatFloat;            break;
            case 'p': format = eFormatPointer;          break;
            case 's': format = eFormatCString;          break;
            default:
                success = false;
                break;
            }
        }
        if (!success)
            error.SetErrorStringWithFormat ("Invalid format specification '%s'. Valid values are:\n"
                                            "  b - binary\n"
                                            "  B - boolean\n"
                                            "  c - char\n"
                                            "  C - printable char\n"
                                            "  d - signed decimal\n"
                                            "  e - float\n"
                                            "  f - float\n"
                                            "  g - float\n"
                                            "  i - signed decimal\n"
                                            "  i - complex integer\n"
                                            "  o - octal\n"
                                            "  O - OSType\n"
                                            "  p - pointer\n"
                                            "  s - c-string\n"
                                            "  u - unsigned decimal\n"
                                            "  x - hex\n"
                                            "  X - complex float\n"
                                            "  y - bytes\n"
                                            "  Y - bytes with ASCII\n%s",
                                            s, 
                                            byte_size_ptr ? "An optional byte size can precede the format character.\n" : "");


        if (error.Fail())
            return error;
    }
    else
    {
        error.SetErrorStringWithFormat("%s option string.\n", s ? "empty" : "invalid");
    }
    return error;
}

void
Args::LongestCommonPrefix (std::string &common_prefix)
{
    arg_sstr_collection::iterator pos, end = m_args.end();
    pos = m_args.begin();
    if (pos == end)
        common_prefix.clear();
    else
        common_prefix = (*pos);

    for (++pos; pos != end; ++pos)
    {
        size_t new_size = (*pos).size();

        // First trim common_prefix if it is longer than the current element:
        if (common_prefix.size() > new_size)
            common_prefix.erase (new_size);

        // Then trim it at the first disparity:

        for (size_t i = 0; i < common_prefix.size(); i++)
        {
            if ((*pos)[i]  != common_prefix[i])
            {
                common_prefix.erase(i);
                break;
            }
        }

        // If we've emptied the common prefix, we're done.
        if (common_prefix.empty())
            break;
    }
}

size_t
Args::FindArgumentIndexForOption (struct option *long_options, int long_options_index)
{
    char short_buffer[3];
    char long_buffer[255];
    ::snprintf (short_buffer, sizeof (short_buffer), "-%c", (char) long_options[long_options_index].val);
    ::snprintf (long_buffer, sizeof (long_buffer),  "--%s", long_options[long_options_index].name);
    size_t end = GetArgumentCount ();
    size_t idx = 0;
    while (idx < end)
    {   
        if ((::strncmp (GetArgumentAtIndex (idx), short_buffer, strlen (short_buffer)) == 0)
            || (::strncmp (GetArgumentAtIndex (idx), long_buffer, strlen (long_buffer)) == 0))
            {
                return idx;
            }
        ++idx;
    }

    return end;
}

bool
Args::IsPositionalArgument (const char *arg)
{
    if (arg == NULL)
        return false;
        
    bool is_positional = true;
    char *cptr = (char *) arg;
    
    if (cptr[0] == '%')
    {
        ++cptr;
        while (isdigit (cptr[0]))
            ++cptr;
        if (cptr[0] != '\0')
            is_positional = false;
    }
    else
        is_positional = false;

    return is_positional;
}

void
Args::ParseAliasOptions (Options &options,
                         CommandReturnObject &result,
                         OptionArgVector *option_arg_vector,
                         std::string &raw_input_string)
{
    StreamString sstr;
    int i;
    struct option *long_options = options.GetLongOptions();

    if (long_options == NULL)
    {
        result.AppendError ("invalid long options");
        result.SetStatus (eReturnStatusFailed);
        return;
    }

    for (i = 0; long_options[i].name != NULL; ++i)
    {
        if (long_options[i].flag == NULL)
        {
            sstr << (char) long_options[i].val;
            switch (long_options[i].has_arg)
            {
                default:
                case no_argument:
                    break;
                case required_argument:
                    sstr << ":";
                    break;
                case optional_argument:
                    sstr << "::";
                    break;
            }
        }
    }

#ifdef __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    int val;
    while (1)
    {
        int long_options_index = -1;
        val = ::getopt_long (GetArgumentCount(), GetArgumentVector(), sstr.GetData(), long_options,
                             &long_options_index);

        if (val == -1)
            break;

        if (val == '?')
        {
            result.AppendError ("unknown or ambiguous option");
            result.SetStatus (eReturnStatusFailed);
            break;
        }

        if (val == 0)
            continue;

        ((Options *) &options)->OptionSeen (val);

        // Look up the long option index
        if (long_options_index == -1)
        {
            for (int j = 0;
                 long_options[j].name || long_options[j].has_arg || long_options[j].flag || long_options[j].val;
                 ++j)
            {
                if (long_options[j].val == val)
                {
                    long_options_index = j;
                    break;
                }
            }
        }

        // See if the option takes an argument, and see if one was supplied.
        if (long_options_index >= 0)
        {
            StreamString option_str;
            option_str.Printf ("-%c", (char) val);

            switch (long_options[long_options_index].has_arg)
            {
            case no_argument:
                option_arg_vector->push_back (OptionArgPair (std::string (option_str.GetData()), 
                                                             OptionArgValue (no_argument, "<no-argument>")));
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
                break;
            case required_argument:
                if (optarg != NULL)
                {
                    option_arg_vector->push_back (OptionArgPair (std::string (option_str.GetData()),
                                                                 OptionArgValue (required_argument, 
                                                                                 std::string (optarg))));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    result.AppendErrorWithFormat ("Option '%s' is missing argument specifier.\n",
                                                 option_str.GetData());
                    result.SetStatus (eReturnStatusFailed);
                }
                break;
            case optional_argument:
                if (optarg != NULL)
                {
                    option_arg_vector->push_back (OptionArgPair (std::string (option_str.GetData()),
                                                                 OptionArgValue (optional_argument, 
                                                                                 std::string (optarg))));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    option_arg_vector->push_back (OptionArgPair (std::string (option_str.GetData()),
                                                                 OptionArgValue (optional_argument, "<no-argument>")));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                break;
            default:
                result.AppendErrorWithFormat
                ("error with options table; invalid value in has_arg field for option '%c'.\n",
                 (char) val);
                result.SetStatus (eReturnStatusFailed);
                break;
            }
        }
        else
        {
            result.AppendErrorWithFormat ("Invalid option with value '%c'.\n", (char) val);
            result.SetStatus (eReturnStatusFailed);
        }

        if (long_options_index >= 0)
        {
            // Find option in the argument list; also see if it was supposed to take an argument and if one was
            // supplied.  Remove option (and argument, if given) from the argument list.  Also remove them from
            // the raw_input_string, if one was passed in.
            size_t idx = FindArgumentIndexForOption (long_options, long_options_index);
            if (idx < GetArgumentCount())
            {
                if (raw_input_string.size() > 0)
                {
                    const char *tmp_arg = GetArgumentAtIndex (idx);
                    size_t pos = raw_input_string.find (tmp_arg);
                    if (pos != std::string::npos)
                        raw_input_string.erase (pos, strlen (tmp_arg));
                }
                ReplaceArgumentAtIndex (idx, "");
                if ((long_options[long_options_index].has_arg != no_argument)
                    && (optarg != NULL)
                    && (idx+1 < GetArgumentCount())
                    && (strcmp (optarg, GetArgumentAtIndex(idx+1)) == 0))
                {
                    if (raw_input_string.size() > 0)
                    {
                        const char *tmp_arg = GetArgumentAtIndex (idx+1);
                        size_t pos = raw_input_string.find (tmp_arg);
                        if (pos != std::string::npos)
                            raw_input_string.erase (pos, strlen (tmp_arg));
                    }
                    ReplaceArgumentAtIndex (idx+1, "");
                }
            }
        }

        if (!result.Succeeded())
            break;
    }
}

void
Args::ParseArgsForCompletion
(
    Options &options,
    OptionElementVector &option_element_vector,
    uint32_t cursor_index
)
{
    StreamString sstr;
    struct option *long_options = options.GetLongOptions();
    option_element_vector.clear();

    if (long_options == NULL)
    {
        return;
    }

    // Leading : tells getopt to return a : for a missing option argument AND
    // to suppress error messages.

    sstr << ":";
    for (int i = 0; long_options[i].name != NULL; ++i)
    {
        if (long_options[i].flag == NULL)
        {
            sstr << (char) long_options[i].val;
            switch (long_options[i].has_arg)
            {
                default:
                case no_argument:
                    break;
                case required_argument:
                    sstr << ":";
                    break;
                case optional_argument:
                    sstr << "::";
                    break;
            }
        }
    }

#ifdef __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    opterr = 0;

    int val;
    const OptionDefinition *opt_defs = options.GetDefinitions();

    // Fooey... getopt_long permutes the GetArgumentVector to move the options to the front.
    // So we have to build another Arg and pass that to getopt_long so it doesn't
    // change the one we have.

    std::vector<const char *> dummy_vec (GetArgumentVector(), GetArgumentVector() + GetArgumentCount() + 1);

    bool failed_once = false;
    uint32_t dash_dash_pos = -1;
        
    while (1)
    {
        bool missing_argument = false;
        int parse_start = optind;
        int long_options_index = -1;
        
        val = ::getopt_long (dummy_vec.size() - 1,
                             (char *const *) &dummy_vec.front(), 
                             sstr.GetData(), 
                             long_options,
                             &long_options_index);

        if (val == -1)
        {
            // When we're completing a "--" which is the last option on line, 
            if (failed_once)
                break;
                
            failed_once = true;
            
            // If this is a bare  "--" we mark it as such so we can complete it successfully later.
            // Handling the "--" is a little tricky, since that may mean end of options or arguments, or the
            // user might want to complete options by long name.  I make this work by checking whether the
            // cursor is in the "--" argument, and if so I assume we're completing the long option, otherwise
            // I let it pass to getopt_long which will terminate the option parsing.
            // Note, in either case we continue parsing the line so we can figure out what other options
            // were passed.  This will be useful when we come to restricting completions based on what other
            // options we've seen on the line.

            if (optind < dummy_vec.size() - 1 
                && (strcmp (dummy_vec[optind-1], "--") == 0))
            {
                dash_dash_pos = optind - 1;
                if (optind - 1 == cursor_index)
                {
                    option_element_vector.push_back (OptionArgElement (OptionArgElement::eBareDoubleDash, optind - 1, 
                                                                   OptionArgElement::eBareDoubleDash));
                    continue;
                }
                else
                    break;
            }
            else
                break;
        }
        else if (val == '?')
        {
            option_element_vector.push_back (OptionArgElement (OptionArgElement::eUnrecognizedArg, optind - 1, 
                                                               OptionArgElement::eUnrecognizedArg));
            continue;
        }
        else if (val == 0)
        {
            continue;
        }
        else if (val == ':')
        {
            // This is a missing argument.
            val = optopt;
            missing_argument = true;
        }

        ((Options *) &options)->OptionSeen (val);

        // Look up the long option index
        if (long_options_index == -1)
        {
            for (int j = 0;
                 long_options[j].name || long_options[j].has_arg || long_options[j].flag || long_options[j].val;
                 ++j)
            {
                if (long_options[j].val == val)
                {
                    long_options_index = j;
                    break;
                }
            }
        }

        // See if the option takes an argument, and see if one was supplied.
        if (long_options_index >= 0)
        {
            int opt_defs_index = -1;
            for (int i = 0; ; i++)
            {
                if (opt_defs[i].short_option == 0)
                    break;
                else if (opt_defs[i].short_option == val)
                {
                    opt_defs_index = i;
                    break;
                }
            }

            switch (long_options[long_options_index].has_arg)
            {
            case no_argument:
                option_element_vector.push_back (OptionArgElement (opt_defs_index, parse_start, 0));
                break;
            case required_argument:
                if (optarg != NULL)
                {
                    int arg_index;
                    if (missing_argument)
                        arg_index = -1;
                    else
                        arg_index = optind - 1;

                    option_element_vector.push_back (OptionArgElement (opt_defs_index, optind - 2, arg_index));
                }
                else
                {
                    option_element_vector.push_back (OptionArgElement (opt_defs_index, optind - 1, -1));
                }
                break;
            case optional_argument:
                if (optarg != NULL)
                {
                    option_element_vector.push_back (OptionArgElement (opt_defs_index, optind - 2, optind - 1));
                }
                else
                {
                    option_element_vector.push_back (OptionArgElement (opt_defs_index, optind - 2, optind - 1));
                }
                break;
            default:
                // The options table is messed up.  Here we'll just continue
                option_element_vector.push_back (OptionArgElement (OptionArgElement::eUnrecognizedArg, optind - 1, 
                                                                   OptionArgElement::eUnrecognizedArg));
                break;
            }
        }
        else
        {
            option_element_vector.push_back (OptionArgElement (OptionArgElement::eUnrecognizedArg, optind - 1, 
                                                               OptionArgElement::eUnrecognizedArg));
        }
    }
    
    // Finally we have to handle the case where the cursor index points at a single "-".  We want to mark that in
    // the option_element_vector, but only if it is not after the "--".  But it turns out that getopt_long just ignores
    // an isolated "-".  So we have to look it up by hand here.  We only care if it is AT the cursor position.
    
    if ((dash_dash_pos == -1 || cursor_index < dash_dash_pos)
         && strcmp (GetArgumentAtIndex(cursor_index), "-") == 0)
    {
        option_element_vector.push_back (OptionArgElement (OptionArgElement::eBareDash, cursor_index, 
                                                           OptionArgElement::eBareDash));
        
    }
}
