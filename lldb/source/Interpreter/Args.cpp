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

static const char *k_space_characters = "\t\n\v\f\r ";
static const char *k_space_characters_with_slash = "\t\n\v\f\r \\";


//----------------------------------------------------------------------
// Args constructor
//----------------------------------------------------------------------
Args::Args (const char *command) :
    m_args(),
    m_argv()
{
    SetCommandString (command);
}


Args::Args (const char *command, size_t len) :
    m_args(),
    m_argv()
{
    SetCommandString (command, len);
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
    if (command && command[0])
    {
        const char *arg_start;
        const char *next_arg_start;
        for (arg_start = command, next_arg_start = NULL;
             arg_start && arg_start[0];
             arg_start = next_arg_start, next_arg_start = NULL)
        {
            // Skip any leading space characters
            arg_start = ::strspn (arg_start, k_space_characters) + arg_start;

            // If there were only space characters to the end of the line, then
            // we're done.
            if (*arg_start == '\0')
                break;

            std::string arg;
            const char *arg_end = NULL;

            switch (*arg_start)
            {
            case '\'':
            case '"':
            case '`':
                {
                    // Look for either a quote character, or the backslash
                    // character
                    const char quote_char = *arg_start;
                    char find_chars[3] = { quote_char, '\\' , '\0'};
                    bool is_backtick = (quote_char == '`');
                    if (quote_char == '"' || quote_char == '`')
                        m_args_quote_char.push_back(quote_char);
                    else
                        m_args_quote_char.push_back('\0');

                    while (*arg_start != '\0')
                    {
                        arg_end = ::strcspn (arg_start + 1, find_chars) + arg_start + 1;

                        if (*arg_end == '\0')
                        {
                            arg.append (arg_start);
                            break;
                        }

                        // Watch out for quote characters prefixed with '\'
                        if (*arg_end == '\\')
                        {
                            if (arg_end[1] == quote_char)
                            {
                                // The character following the '\' is our quote
                                // character so strip the backslash character
                                arg.append (arg_start, arg_end);
                            }
                            else
                            {
                                // The character following the '\' is NOT our
                                // quote character, so include the backslash
                                // and continue
                                arg.append (arg_start, arg_end + 1);
                            }
                            arg_start = arg_end + 1;
                            continue;
                        }
                        else
                        {
                            arg.append (arg_start, arg_end + 1);
                            next_arg_start = arg_end + 1;
                            break;
                        }
                    }

                    // Skip single and double quotes, but leave backtick quotes
                    if (!is_backtick)
                    {
                        char first_c = arg[0];
                        arg.erase(0,1);
                        // Only erase the last character if it is the same as the first.
                        // Otherwise, we're parsing an incomplete command line, and we
                        // would be stripping off the last character of that string.
                        if (arg[arg.size() - 1] == first_c)
                            arg.erase(arg.size() - 1, 1);
                    }
                }
                break;
            default:
                {
                    m_args_quote_char.push_back('\0');
                    // Look for the next non-escaped space character
                    while (*arg_start != '\0')
                    {
                        arg_end = ::strcspn (arg_start, k_space_characters_with_slash) + arg_start;

                        if (arg_end == NULL)
                        {
                            arg.append(arg_start);
                            break;
                        }

                        if (*arg_end == '\\')
                        {
                            // Append up to the '\' char
                            arg.append (arg_start, arg_end);

                            if (arg_end[1] == '\0')
                                break;

                            // Append the character following the '\' if it isn't
                            // the end of the string
                            arg.append (1, arg_end[1]);
                            arg_start = arg_end + 2;
                            continue;
                        }
                        else
                        {
                            arg.append (arg_start, arg_end);
                            next_arg_start = arg_end;
                            break;
                        }
                    }
                }
                break;
            }

            m_args.push_back(arg);
        }
    }
    UpdateArgvFromArgs();
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
        if ((argv[i][0] == '"') || (argv[i][0] == '`'))
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

int32_t
Args::StringToOptionEnum (const char *s, lldb::OptionEnumValueElement *enum_values, int32_t fail_value, bool *success_ptr)
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
    lldb::Format &format
)
{
    format = eFormatInvalid;
    Error error;

    if (s && s[0])
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
        case 'i':
        case 'd': format = eFormatDecimal;          break;
        case 'u': format = eFormatUnsigned;         break;
        case 'x': format = eFormatHex;              break;
        case 'f':
        case 'e':
        case 'g': format = eFormatFloat;            break;
        case 'p': format = eFormatPointer;          break;
        case 's': format = eFormatCString;          break;
        default:
            error.SetErrorStringWithFormat("Invalid format character '%c'. Valid values are:\n"
                                            "  b - binary\n"
                                            "  B - boolean\n"
                                            "  c - char\n"
                                            "  C - printable char\n"
                                            "  d - signed decimal\n"
                                            "  e - float\n"
                                            "  f - float\n"
                                            "  g - float\n"
                                            "  i - signed decimal\n"
                                            "  o - octal\n"
                                            "  s - c-string\n"
                                            "  u - unsigned decimal\n"
                                            "  x - hex\n"
                                            "  y - bytes\n"
                                            "  Y - bytes with ASCII\n", s[0]);
            break;
        }

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
                         OptionArgVector *option_arg_vector)
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
            // supplied.  Remove option (and argument, if given) from the argument list.
            size_t idx = FindArgumentIndexForOption (long_options, long_options_index);
            if (idx < GetArgumentCount())
            {
                ReplaceArgumentAtIndex (idx, "");
                if ((long_options[long_options_index].has_arg != no_argument)
                    && (optarg != NULL)
                    && (idx+1 < GetArgumentCount())
                    && (strcmp (optarg, GetArgumentAtIndex(idx+1)) == 0))
                    ReplaceArgumentAtIndex (idx+1, "");
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
