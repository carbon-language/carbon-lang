//===-- Options.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/Options.h"

// C Includes
// C++ Includes
#include <bitset>

// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// Options
//-------------------------------------------------------------------------
Options::Options () :
    m_getopt_table ()
{
    BuildValidOptionSets();
}

Options::~Options ()
{
}


void
Options::ResetOptionValues ()
{
    m_seen_options.clear();
}

void
Options::OptionSeen (int option_idx)
{
    m_seen_options.insert ((char) option_idx);
}

// Returns true is set_a is a subset of set_b;  Otherwise returns false.

bool
Options::IsASubset (const OptionSet& set_a, const OptionSet& set_b)
{
    bool is_a_subset = true;
    OptionSet::const_iterator pos_a;
    OptionSet::const_iterator pos_b;

    // set_a is a subset of set_b if every member of set_a is also a member of set_b

    for (pos_a = set_a.begin(); pos_a != set_a.end() && is_a_subset; ++pos_a)
    {
        pos_b = set_b.find(*pos_a);
        if (pos_b == set_b.end())
            is_a_subset = false;
    }

    return is_a_subset;
}

// Returns the set difference set_a - set_b, i.e. { x | ElementOf (x, set_a) && !ElementOf (x, set_b) }

size_t
Options::OptionsSetDiff (const OptionSet& set_a, const OptionSet& set_b, OptionSet& diffs)
{
    size_t num_diffs = 0;
    OptionSet::const_iterator pos_a;
    OptionSet::const_iterator pos_b;

    for (pos_a = set_a.begin(); pos_a != set_a.end(); ++pos_a)
    {
        pos_b = set_b.find(*pos_a);
        if (pos_b == set_b.end())
        {
            ++num_diffs;
            diffs.insert(*pos_a);
        }
    }

    return num_diffs;
}

// Returns the union of set_a and set_b.  Does not put duplicate members into the union.

void
Options::OptionsSetUnion (const OptionSet &set_a, const OptionSet &set_b, OptionSet &union_set)
{
    OptionSet::const_iterator pos;
    OptionSet::iterator pos_union;

    // Put all the elements of set_a into the union.

    for (pos = set_a.begin(); pos != set_a.end(); ++pos)
        union_set.insert(*pos);

    // Put all the elements of set_b that are not already there into the union.
    for (pos = set_b.begin(); pos != set_b.end(); ++pos)
    {
        pos_union = union_set.find(*pos);
        if (pos_union == union_set.end())
            union_set.insert(*pos);
    }
}

bool
Options::VerifyOptions (CommandReturnObject &result)
{
    bool options_are_valid = false;

    int num_levels = GetRequiredOptions().size();
    if (num_levels)
    {
        for (int i = 0; i < num_levels && !options_are_valid; ++i)
        {
            // This is the correct set of options if:  1). m_seen_options contains all of m_required_options[i]
            // (i.e. all the required options at this level are a subset of m_seen_options); AND
            // 2). { m_seen_options - m_required_options[i] is a subset of m_options_options[i] (i.e. all the rest of
            // m_seen_options are in the set of optional options at this level.

            // Check to see if all of m_required_options[i] are a subset of m_seen_options
            if (IsASubset (GetRequiredOptions()[i], m_seen_options))
            {
                // Construct the set difference: remaining_options = {m_seen_options} - {m_required_options[i]}
                OptionSet remaining_options;
                OptionsSetDiff (m_seen_options, GetRequiredOptions()[i], remaining_options);
                // Check to see if remaining_options is a subset of m_optional_options[i]
                if (IsASubset (remaining_options, GetOptionalOptions()[i]))
                    options_are_valid = true;
            }
        }
    }
    else
    {
        options_are_valid = true;
    }

    if (options_are_valid)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        result.AppendError ("invalid combination of options for the given command");
        result.SetStatus (eReturnStatusFailed);
    }

    return options_are_valid;
}

// This is called in the Options constructor, though we could call it lazily if that ends up being
// a performance problem.

void
Options::BuildValidOptionSets ()
{
    // Check to see if we already did this.
    if (m_required_options.size() != 0)
        return;

    // Check to see if there are any options.
    int num_options = NumCommandOptions ();
    if (num_options == 0)
        return;

    const lldb::OptionDefinition *full_options_table = GetDefinitions();
    m_required_options.resize(1);
    m_optional_options.resize(1);
    
    // First count the number of option sets we've got.  Ignore LLDB_ALL_OPTION_SETS...
    
    uint32_t num_option_sets = 0;
    
    for (int i = 0; i < num_options; i++)
    {
        uint32_t this_usage_mask = full_options_table[i].usage_mask;
        if (this_usage_mask == LLDB_OPT_SET_ALL)
        {
            if (num_option_sets == 0)
                num_option_sets = 1;
        }
        else
        {
            for (int j = 0; j < LLDB_MAX_NUM_OPTION_SETS; j++)
            {
                if (this_usage_mask & (1 << j))
                {
                    if (num_option_sets <= j)
                        num_option_sets = j + 1;
                }
            }
        }
    }

    if (num_option_sets > 0)
    {
        m_required_options.resize(num_option_sets);
        m_optional_options.resize(num_option_sets);
        
        for (int i = 0; i < num_options; ++i)
        {
            for (int j = 0; j < num_option_sets; j++)
            {
                if (full_options_table[i].usage_mask & 1 << j)
                {
                    if (full_options_table[i].required)
                        m_required_options[j].insert(full_options_table[i].short_option);
                    else
                        m_optional_options[j].insert(full_options_table[i].short_option);
                }
            }
        }
    }
}

uint32_t
Options::NumCommandOptions ()
{
    const lldb::OptionDefinition *full_options_table = GetDefinitions ();
    if (full_options_table == NULL) 
        return 0;
        
    int i = 0;

    if (full_options_table != NULL)
    {
        while (full_options_table[i].long_option != NULL)
            ++i;
    }

    return i;
}

struct option *
Options::GetLongOptions ()
{
    // Check to see if this has already been done.
    if (m_getopt_table.empty())
    {
        // Check to see if there are any options.
        const uint32_t num_options = NumCommandOptions();
        if (num_options == 0)
            return NULL;

        uint32_t i;
        uint32_t j;
        const lldb::OptionDefinition *full_options_table = GetDefinitions();

        std::bitset<256> option_seen;

        m_getopt_table.resize(num_options + 1);
        for (i = 0, j = 0; i < num_options; ++i)
        {
            char short_opt = full_options_table[i].short_option;

            if (option_seen.test(short_opt) == false)
            {
                m_getopt_table[j].name    = full_options_table[i].long_option;
                m_getopt_table[j].has_arg = full_options_table[i].option_has_arg;
                m_getopt_table[j].flag    = NULL;
                m_getopt_table[j].val     = full_options_table[i].short_option;
                option_seen.set(short_opt);
                ++j;
            }
        }

        //getopt_long requires a NULL final entry in the table:

        m_getopt_table[j].name    = NULL;
        m_getopt_table[j].has_arg = 0;
        m_getopt_table[j].flag    = NULL;
        m_getopt_table[j].val     = 0;
    }

    if (m_getopt_table.empty())
        return NULL;

    return &m_getopt_table.front();
}


// This function takes INDENT, which tells how many spaces to output at the front of each line; SPACES, which is
// a string containing 80 spaces; and TEXT, which is the text that is to be output.   It outputs the text, on
// multiple lines if necessary, to RESULT, with INDENT spaces at the front of each line.  It breaks lines on spaces,
// tabs or newlines, shortening the line if necessary to not break in the middle of a word.  It assumes that each
// output line should contain a maximum of OUTPUT_MAX_COLUMNS characters.


void
Options::OutputFormattedUsageText
(
    Stream &strm,
    const char *text,
    uint32_t output_max_columns
)
{
    int len = strlen (text);

    // Will it all fit on one line?

    if ((len + strm.GetIndentLevel()) < output_max_columns)
    {
        // Output it as a single line.
        strm.Indent (text);
        strm.EOL();
    }
    else
    {
        // We need to break it up into multiple lines.

        int text_width = output_max_columns - strm.GetIndentLevel() - 1;
        int start = 0;
        int end = start;
        int final_end = strlen (text);
        int sub_len;

        while (end < final_end)
        {
            // Don't start the 'text' on a space, since we're already outputting the indentation.
            while ((start < final_end) && (text[start] == ' '))
                start++;

            end = start + text_width;
            if (end > final_end)
                end = final_end;
            else
            {
                // If we're not at the end of the text, make sure we break the line on white space.
                while (end > start
                       && text[end] != ' ' && text[end] != '\t' && text[end] != '\n')
                    end--;
            }

            sub_len = end - start;
            if (start != 0)
                strm.EOL();
            strm.Indent();
            assert (start < final_end);
            assert (start + sub_len <= final_end);
            strm.Write(text + start, sub_len);
            start = end + 1;
        }
        strm.EOL();
    }
}

void
Options::GenerateOptionUsage
(
    Stream &strm,
    CommandObject *cmd,
    const char *program_name)
{
    uint32_t screen_width = 80;
    const lldb::OptionDefinition *full_options_table = GetDefinitions();
    const uint32_t save_indent_level = strm.GetIndentLevel();
    const char *name;

    if (cmd)
      name = cmd->GetCommandName();
    else
      name = program_name;

    strm.PutCString ("\nCommand Options Usage:\n");

    strm.IndentMore(2);

    // First, show each usage level set of options, e.g. <cmd> [options-for-level-0]
    //                                                   <cmd> [options-for-level-1]
    //                                                   etc.

    const uint32_t num_options = NumCommandOptions();
    if (num_options == 0)
        return;
        
    int num_option_sets = GetRequiredOptions().size();
    
    uint32_t i;
    
    for (uint32_t opt_set = 0; opt_set < num_option_sets; ++opt_set)
    {
        uint32_t opt_set_mask;
        
        opt_set_mask = 1 << opt_set;
        if (opt_set > 0)
            strm.Printf ("\n");
        strm.Indent (name);
        
        for (i = 0; i < num_options; ++i)
        {
            if (full_options_table[i].usage_mask & opt_set_mask)
            {
                // Add current option to the end of out_stream.

                if (full_options_table[i].required)
                {
                    if (full_options_table[i].option_has_arg == required_argument)
                    {
                        strm.Printf (" -%c %s",
                                    full_options_table[i].short_option,
                                    full_options_table[i].argument_name);
                    }
                    else if (full_options_table[i].option_has_arg == optional_argument)
                    {
                        strm.Printf (" -%c [%s]",
                                     full_options_table[i].short_option,
                                     full_options_table[i].argument_name);
                    }
                    else
                        strm.Printf (" -%c", full_options_table[i].short_option);
                }
                else
                {
                    if (full_options_table[i].option_has_arg == required_argument)
                        strm.Printf (" [-%c %s]", full_options_table[i].short_option,
                                           full_options_table[i].argument_name);
                    else if (full_options_table[i].option_has_arg == optional_argument)
                        strm.Printf (" [-%c [%s]]", full_options_table[i].short_option,
                                           full_options_table[i].argument_name);
                    else
                        strm.Printf (" [-%c]", full_options_table[i].short_option);
                }
            }
        }
    }
    strm.Printf ("\n\n");

    // Now print out all the detailed information about the various options:  long form, short form and help text:
    //   -- long_name <argument>
    //   - short <argument>
    //   help text

    // This variable is used to keep track of which options' info we've printed out, because some options can be in
    // more than one usage level, but we only want to print the long form of its information once.

    OptionSet options_seen;
    OptionSet::iterator pos;
    strm.IndentMore (5);

    int first_option_printed = 1;
    for (i = 0; i < num_options; ++i)
    {
        // Only print out this option if we haven't already seen it.
        pos = options_seen.find (full_options_table[i].short_option);
        if (pos == options_seen.end())
        {
            // Put a newline separation between arguments
            if (first_option_printed)
                first_option_printed = 0;
            else
                strm.EOL();
      
            options_seen.insert (full_options_table[i].short_option);
            strm.Indent ();
            strm.Printf ("-%c ", full_options_table[i].short_option);
            if (full_options_table[i].argument_name != NULL)
                strm.PutCString(full_options_table[i].argument_name);
            strm.EOL();
            strm.Indent ();
            strm.Printf ("--%s ", full_options_table[i].long_option);
            if (full_options_table[i].argument_name != NULL)
                strm.PutCString(full_options_table[i].argument_name);
            strm.EOL();

            strm.IndentMore (5);
            
            if (full_options_table[i].usage_text)
                    OutputFormattedUsageText (strm,
                                              full_options_table[i].usage_text,
                                              screen_width);
            if (full_options_table[i].enum_values != NULL)
            {
                strm.Indent ();
                strm.Printf("Values: ");
                for (int j = 0; full_options_table[i].enum_values[j].string_value != NULL; j++) 
                {
                    if (j == 0)
                        strm.Printf("%s", full_options_table[i].enum_values[j].string_value);
                    else
                        strm.Printf(" | %s", full_options_table[i].enum_values[j].string_value);
                }
                strm.EOL();
            }
            strm.IndentLess (5);
        }
    }

    // Restore the indent level
    strm.SetIndentLevel (save_indent_level);
}

// This function is called when we have been given a potentially incomplete set of
// options, such as when an alias has been defined (more options might be added at
// at the time the alias is invoked).  We need to verify that the options in the set
// m_seen_options are all part of a set that may be used together, but m_seen_options
// may be missing some of the "required" options.

bool
Options::VerifyPartialOptions (CommandReturnObject &result)
{
    bool options_are_valid = false;

    int num_levels = GetRequiredOptions().size();
    if (num_levels)
      {
        for (int i = 0; i < num_levels && !options_are_valid; ++i)
          {
            // In this case we are treating all options as optional rather than required.
            // Therefore a set of options is correct if m_seen_options is a subset of the
            // union of m_required_options and m_optional_options.
            OptionSet union_set;
            OptionsSetUnion (GetRequiredOptions()[i], GetOptionalOptions()[i], union_set);
            if (IsASubset (m_seen_options, union_set))
                options_are_valid = true;
          }
      }

    return options_are_valid;
}

bool
Options::HandleOptionCompletion
(
    CommandInterpreter &interpreter,
    Args &input,
    OptionElementVector &opt_element_vector,
    int cursor_index,
    int char_pos,
    int match_start_point,
    int max_return_elements,
    bool &word_complete,
    lldb_private::StringList &matches
)
{
    word_complete = true;
    
    // For now we just scan the completions to see if the cursor position is in
    // an option or its argument.  Otherwise we'll call HandleArgumentCompletion.
    // In the future we can use completion to validate options as well if we want.

    const OptionDefinition *opt_defs = GetDefinitions();

    std::string cur_opt_std_str (input.GetArgumentAtIndex(cursor_index));
    cur_opt_std_str.erase(char_pos);
    const char *cur_opt_str = cur_opt_std_str.c_str();

    for (int i = 0; i < opt_element_vector.size(); i++)
    {
        int opt_pos = opt_element_vector[i].opt_pos;
        int opt_arg_pos = opt_element_vector[i].opt_arg_pos;
        int opt_defs_index = opt_element_vector[i].opt_defs_index;
        if (opt_pos == cursor_index)
        {
            // We're completing the option itself.

            if (opt_defs_index == OptionArgElement::eBareDash)
            {
                // We're completing a bare dash.  That means all options are open.
                // FIXME: We should scan the other options provided and only complete options
                // within the option group they belong to.
                char opt_str[3] = {'-', 'a', '\0'};
                
                for (int j = 0 ; opt_defs[j].short_option != 0 ; j++)
                {   
                    opt_str[1] = opt_defs[j].short_option;
                    matches.AppendString (opt_str);
                }
                return true;
            }
            else if (opt_defs_index == OptionArgElement::eBareDoubleDash)
            {
                std::string full_name ("--");
                for (int j = 0 ; opt_defs[j].short_option != 0 ; j++)
                {   
                    full_name.erase(full_name.begin() + 2, full_name.end());
                    full_name.append (opt_defs[j].long_option);
                    matches.AppendString (full_name.c_str());
                }
                return true;
            }
            else if (opt_defs_index != OptionArgElement::eUnrecognizedArg)
            {
                // We recognized it, if it an incomplete long option, complete it anyway (getopt_long is
                // happy with shortest unique string, but it's still a nice thing to do.)  Otherwise return
                // The string so the upper level code will know this is a full match and add the " ".
                if (cur_opt_str && strlen (cur_opt_str) > 2
                    && cur_opt_str[0] == '-' && cur_opt_str[1] == '-'
                    && strcmp (opt_defs[opt_defs_index].long_option, cur_opt_str) != 0)
                {
                        std::string full_name ("--");
                        full_name.append (opt_defs[opt_defs_index].long_option);
                        matches.AppendString(full_name.c_str());
                        return true;
                }
                else
                {
                    matches.AppendString(input.GetArgumentAtIndex(cursor_index));
                    return true;
                }
            }
            else
            {
                // FIXME - not handling wrong options yet:
                // Check to see if they are writing a long option & complete it.
                // I think we will only get in here if the long option table has two elements
                // that are not unique up to this point.  getopt_long does shortest unique match
                // for long options already.

                if (cur_opt_str && strlen (cur_opt_str) > 2
                    && cur_opt_str[0] == '-' && cur_opt_str[1] == '-')
                {
                    for (int j = 0 ; opt_defs[j].short_option != 0 ; j++)
                    {
                        if (strstr(opt_defs[j].long_option, cur_opt_str + 2) == opt_defs[j].long_option)
                        {
                            std::string full_name ("--");
                            full_name.append (opt_defs[j].long_option);
                            // The options definitions table has duplicates because of the
                            // way the grouping information is stored, so only add once.
                            bool duplicate = false;
                            for (int k = 0; k < matches.GetSize(); k++)
                            {
                                if (matches.GetStringAtIndex(k) == full_name)
                                {
                                    duplicate = true;
                                    break;
                                }
                            }
                            if (!duplicate)
                                matches.AppendString(full_name.c_str());
                        }
                    }
                }
                return true;
            }


        }
        else if (opt_arg_pos == cursor_index)
        {
            // Okay the cursor is on the completion of an argument.
            // See if it has a completion, otherwise return no matches.

            if (opt_defs_index != -1)
            {
                HandleOptionArgumentCompletion (interpreter, 
                                                input,
                                                cursor_index,
                                                strlen (input.GetArgumentAtIndex(cursor_index)),
                                                opt_element_vector,
                                                i,
                                                match_start_point,
                                                max_return_elements,
                                                word_complete,
                                                matches);
                return true;
            }
            else
            {
                // No completion callback means no completions...
                return true;
            }

        }
        else
        {
            // Not the last element, keep going.
            continue;
        }
    }
    return false;
}

bool
Options::HandleOptionArgumentCompletion
(
    CommandInterpreter &interpreter,
    Args &input,
    int cursor_index,
    int char_pos,
    OptionElementVector &opt_element_vector,
    int opt_element_index,
    int match_start_point,
    int max_return_elements,
    bool &word_complete,
    lldb_private::StringList &matches
)
{
    const OptionDefinition *opt_defs = GetDefinitions();
    std::auto_ptr<SearchFilter> filter_ap;

    int opt_arg_pos = opt_element_vector[opt_element_index].opt_arg_pos;
    int opt_defs_index = opt_element_vector[opt_element_index].opt_defs_index;
    
    // See if this is an enumeration type option, and if so complete it here:
    
    OptionEnumValueElement *enum_values = opt_defs[opt_defs_index].enum_values;
    if (enum_values != NULL)
    {
        bool return_value = false;
        std::string match_string(input.GetArgumentAtIndex (opt_arg_pos), input.GetArgumentAtIndex (opt_arg_pos) + char_pos);
        for (int i = 0; enum_values[i].string_value != NULL; i++)
        {
            if (strstr(enum_values[i].string_value, match_string.c_str()) == enum_values[i].string_value)
            {
                matches.AppendString (enum_values[i].string_value);
                return_value = true;
            }
        }
        return return_value;
    }

    // If this is a source file or symbol type completion, and  there is a
    // -shlib option somewhere in the supplied arguments, then make a search filter
    // for that shared library.
    // FIXME: Do we want to also have an "OptionType" so we don't have to match string names?

    uint32_t completion_mask = opt_defs[opt_defs_index].completionType;
    if (completion_mask & CommandCompletions::eSourceFileCompletion
        || completion_mask & CommandCompletions::eSymbolCompletion)
    {
        for (int i = 0; i < opt_element_vector.size(); i++)
        {
            int cur_defs_index = opt_element_vector[i].opt_defs_index;
            int cur_arg_pos    = opt_element_vector[i].opt_arg_pos;
            const char *cur_opt_name = opt_defs[cur_defs_index].long_option;

            // If this is the "shlib" option and there was an argument provided,
            // restrict it to that shared library.
            if (strcmp(cur_opt_name, "shlib") == 0 && cur_arg_pos != -1)
            {
                const char *module_name = input.GetArgumentAtIndex(cur_arg_pos);
                if (module_name)
                {
                    FileSpec module_spec(module_name);
                    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
                    // Search filters require a target...
                    if (target_sp != NULL)
                        filter_ap.reset (new SearchFilterByModule (target_sp, module_spec));
                }
                break;
            }
        }
    }

    return CommandCompletions::InvokeCommonCompletionCallbacks (interpreter,
                                                                completion_mask,
                                                                input.GetArgumentAtIndex (opt_arg_pos),
                                                                match_start_point,
                                                                max_return_elements,
                                                                filter_ap.get(),
                                                                word_complete,
                                                                matches);
    
}
