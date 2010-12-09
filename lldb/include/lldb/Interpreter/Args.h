//===-- Args.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Command_h_
#define liblldb_Command_h_

// C Includes
#include <getopt.h>

// C++ Includes
#include <list>
#include <string>
#include <vector>
#include <utility>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Error.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

typedef std::pair<int, std::string> OptionArgValue;
typedef std::pair<std::string, OptionArgValue> OptionArgPair;
typedef std::vector<OptionArgPair> OptionArgVector;
typedef lldb::SharedPtr<OptionArgVector>::Type OptionArgVectorSP;

struct OptionArgElement
{
    enum {
        eUnrecognizedArg = -1,
        eBareDash = -2,
        eBareDoubleDash = -3
    };
    
    OptionArgElement (int defs_index, int pos, int arg_pos) :
        opt_defs_index(defs_index),
        opt_pos (pos),
        opt_arg_pos (arg_pos)
    {
    }

    int opt_defs_index;
    int opt_pos;
    int opt_arg_pos;
};

typedef std::vector<OptionArgElement> OptionElementVector;

//----------------------------------------------------------------------
/// @class Args Args.h "lldb/Interpreter/Args.h"
/// @brief A command line argument class.
///
/// The Args class is designed to be fed a command line. The
/// command line is copied into an internal buffer and then split up
/// into arguments. Arguments are space delimited if there are no quotes
/// (single, double, or backtick quotes) surrounding the argument. Spaces
/// can be escaped using a \ character to avoid having to surround an
/// argument that contains a space with quotes.
//----------------------------------------------------------------------
class Args
{
public:

    //------------------------------------------------------------------
    /// Construct with an option command string.
    ///
    /// @param[in] command
    ///     A NULL terminated command that will be copied and split up
    ///     into arguments.
    ///
    /// @see Args::SetCommandString(const char *)
    //------------------------------------------------------------------
    Args (const char *command = NULL);

    Args (const char *command, size_t len);

    //------------------------------------------------------------------
    /// Destructor.
    //------------------------------------------------------------------
    ~Args();

    //------------------------------------------------------------------
    /// Dump all arguments to the stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to dump all arguments in the argument
    ///     vector.
    //------------------------------------------------------------------
    void
    Dump (Stream *s);

    //------------------------------------------------------------------
    /// Sets the command string contained by this object.
    ///
    /// The command string will be copied and split up into arguments
    /// that can be accessed via the accessor functions.
    ///
    /// @param[in] command
    ///     A NULL terminated command that will be copied and split up
    ///     into arguments.
    ///
    /// @see Args::GetArgumentCount() const
    /// @see Args::GetArgumentAtIndex (size_t) const
    /// @see Args::GetArgumentVector ()
    /// @see Args::Shift ()
    /// @see Args::Unshift (const char *)
    //------------------------------------------------------------------
    void
    SetCommandString (const char *command);

    void
    SetCommandString (const char *command, size_t len);

    bool
    GetCommandString (std::string &command);

    //------------------------------------------------------------------
    /// Gets the number of arguments left in this command object.
    ///
    /// @return
    ///     The number or arguments in this object.
    //------------------------------------------------------------------
    size_t
    GetArgumentCount () const;

    //------------------------------------------------------------------
    /// Gets the NULL terminated C string argument pointer for the
    /// argument at index \a idx.
    ///
    /// @return
    ///     The NULL terminated C string argument pointer if \a idx is a
    ///     valid argument index, NULL otherwise.
    //------------------------------------------------------------------
    const char *
    GetArgumentAtIndex (size_t idx) const;

    char
    GetArgumentQuoteCharAtIndex (size_t idx) const;

    //------------------------------------------------------------------
    /// Gets the argument vector.
    ///
    /// The value returned by this function can be used by any function
    /// that takes and vector. The return value is just like \a argv
    /// in the standard C entry point function:
    ///     \code
    ///         int main (int argc, const char **argv);
    ///     \endcode
    ///
    /// @return
    ///     An array of NULL terminated C string argument pointers that
    ///     also has a terminating NULL C string pointer
    //------------------------------------------------------------------
    char **
    GetArgumentVector ();

    //------------------------------------------------------------------
    /// Gets the argument vector.
    ///
    /// The value returned by this function can be used by any function
    /// that takes and vector. The return value is just like \a argv
    /// in the standard C entry point function:
    ///     \code
    ///         int main (int argc, const char **argv);
    ///     \endcode
    ///
    /// @return
    ///     An array of NULL terminate C string argument pointers that
    ///     also has a terminating NULL C string pointer
    //------------------------------------------------------------------
    const char **
    GetConstArgumentVector () const;


    //------------------------------------------------------------------
    /// Appends a new argument to the end of the list argument list.
    ///
    /// @param[in] arg_cstr
    ///     The new argument as a NULL terminated C string.
    ///
    /// @param[in] quote_char
    ///     If the argument was originally quoted, put in the quote char here.
    ///
    /// @return
    ///     The NULL terminated C string of the copy of \a arg_cstr.
    //------------------------------------------------------------------
    const char *
    AppendArgument (const char *arg_cstr, char quote_char = '\0');

    void
    AppendArguments (const Args &rhs);
    //------------------------------------------------------------------
    /// Insert the argument value at index \a idx to \a arg_cstr.
    ///
    /// @param[in] idx
    ///     The index of where to insert the argument.
    ///
    /// @param[in] arg_cstr
    ///     The new argument as a NULL terminated C string.
    ///
    /// @param[in] quote_char
    ///     If the argument was originally quoted, put in the quote char here.
    ///
    /// @return
    ///     The NULL terminated C string of the copy of \a arg_cstr.
    //------------------------------------------------------------------
    const char *
    InsertArgumentAtIndex (size_t idx, const char *arg_cstr, char quote_char = '\0');

    //------------------------------------------------------------------
    /// Replaces the argument value at index \a idx to \a arg_cstr
    /// if \a idx is a valid argument index.
    ///
    /// @param[in] idx
    ///     The index of the argument that will have its value replaced.
    ///
    /// @param[in] arg_cstr
    ///     The new argument as a NULL terminated C string.
    ///
    /// @param[in] quote_char
    ///     If the argument was originally quoted, put in the quote char here.
    ///
    /// @return
    ///     The NULL terminated C string of the copy of \a arg_cstr if
    ///     \a idx was a valid index, NULL otherwise.
    //------------------------------------------------------------------
    const char *
    ReplaceArgumentAtIndex (size_t idx, const char *arg_cstr, char quote_char = '\0');

    //------------------------------------------------------------------
    /// Deletes the argument value at index
    /// if \a idx is a valid argument index.
    ///
    /// @param[in] idx
    ///     The index of the argument that will have its value replaced.
    ///
    //------------------------------------------------------------------
    void
    DeleteArgumentAtIndex (size_t idx);

    //------------------------------------------------------------------
    /// Sets the argument vector value, optionally copying all
    /// arguments into an internal buffer.
    ///
    /// Sets the arguments to match those found in \a argv. All argument
    /// strings will be copied into an internal buffers.
    //
    //  FIXME: Handle the quote character somehow.
    //------------------------------------------------------------------
    void
    SetArguments (int argc, const char **argv);

    //------------------------------------------------------------------
    /// Shifts the first argument C string value of the array off the
    /// argument array.
    ///
    /// The string value will be freed, so a copy of the string should
    /// be made by calling Args::GetArgumentAtIndex (size_t) const
    /// first and copying the returned value before calling
    /// Args::Shift().
    ///
    /// @see Args::GetArgumentAtIndex (size_t) const
    //------------------------------------------------------------------
    void
    Shift ();

    //------------------------------------------------------------------
    /// Inserts a class owned copy of \a arg_cstr at the beginning of
    /// the argument vector.
    ///
    /// A copy \a arg_cstr will be made.
    ///
    /// @param[in] arg_cstr
    ///     The argument to push on the front the the argument stack.
    ///
    /// @param[in] quote_char
    ///     If the argument was originally quoted, put in the quote char here.
    ///
    /// @return
    ///     A pointer to the copy of \a arg_cstr that was made.
    //------------------------------------------------------------------
    const char *
    Unshift (const char *arg_cstr, char quote_char = '\0');

    //------------------------------------------------------------------
    /// Parse the arguments in the contained arguments.
    ///
    /// The arguments that are consumed by the argument parsing process
    /// will be removed from the argument vector. The arguements that
    /// get processed start at the second argument. The first argument
    /// is assumed to be the command and will not be touched.
    ///
    /// @see class Options
    //------------------------------------------------------------------
    Error
    ParseOptions (Options &options);
    
    size_t
    FindArgumentIndexForOption (struct option *long_options, int long_options_index);
    
    bool
    IsPositionalArgument (const char *arg);

    // The following works almost identically to ParseOptions, except that no option is required to have arguments,
    // and it builds up the option_arg_vector as it parses the options.

    void
    ParseAliasOptions (Options &options, CommandReturnObject &result, OptionArgVector *option_arg_vector, 
                       std::string &raw_input_line);

    void
    ParseArgsForCompletion (Options &options, OptionElementVector &option_element_vector, uint32_t cursor_index);

    //------------------------------------------------------------------
    // Clear the arguments.
    //
    // For re-setting or blanking out the list of arguments.
    //------------------------------------------------------------------
    void
    Clear ();

    static int32_t
    StringToSInt32 (const char *s, int32_t fail_value = 0, int base = 0, bool *success_ptr = NULL);

    static uint32_t
    StringToUInt32 (const char *s, uint32_t fail_value = 0, int base = 0, bool *success_ptr = NULL);

    static int64_t
    StringToSInt64 (const char *s, int64_t fail_value = 0, int base = 0, bool *success_ptr = NULL);

    static uint64_t
    StringToUInt64 (const char *s, uint64_t fail_value = 0, int base = 0, bool *success_ptr = NULL);

    static lldb::addr_t
    StringToAddress (const char *s, lldb::addr_t fail_value = LLDB_INVALID_ADDRESS, bool *success_ptr = NULL);

    static bool
    StringToBoolean (const char *s, bool fail_value, bool *success_ptr);
    
    static int32_t
    StringToOptionEnum (const char *s, lldb::OptionEnumValueElement *enum_values, int32_t fail_value, bool *success_ptr);

    static lldb::ScriptLanguage
    StringToScriptLanguage (const char *s, lldb::ScriptLanguage fail_value, bool *success_ptr);

    static Error
    StringToFormat (const char *s, lldb::Format &format);

    // This one isn't really relevant to Arguments per se, but we're using the Args as a
    // general strings container, so...
    void
    LongestCommonPrefix (std::string &common_prefix);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from Args can see and modify these
    //------------------------------------------------------------------
    typedef std::list<std::string> arg_sstr_collection;
    typedef std::vector<const char *> arg_cstr_collection;
    typedef std::vector<char> arg_quote_char_collection;
    arg_sstr_collection m_args;
    arg_cstr_collection m_argv; ///< The current argument vector.
    arg_quote_char_collection m_args_quote_char;

    void
    UpdateArgsAfterOptionParsing ();

    void
    UpdateArgvFromArgs ();
};

} // namespace lldb_private

#endif  // liblldb_Command_h_
