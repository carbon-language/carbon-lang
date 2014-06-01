//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_env.h"
#include <string.h>
#include <ctype.h>
#include "offload_util.h"
#include "liboffload_error_codes.h"

// for environment variables valid on all cards
const int MicEnvVar::any_card = -1;

MicEnvVar::~MicEnvVar()
{
    for (std::list<MicEnvVar::CardEnvVars*>::const_iterator
         it = card_spec_list.begin();
         it != card_spec_list.end(); it++) {
        CardEnvVars *card_data = *it;
        delete card_data;
    }
}

MicEnvVar::VarValue::~VarValue()
{
    free(env_var_value);
}

MicEnvVar::CardEnvVars::~CardEnvVars()
{
    for (std::list<MicEnvVar::VarValue*>::const_iterator it = env_vars.begin();
        it != env_vars.end(); it++) {
            VarValue *var_value = *it;
            delete var_value;
    }
}

// Searching for card in "card_spec_list" list with the same "number"

MicEnvVar::CardEnvVars* MicEnvVar::get_card(int number)
{
    if (number == any_card) {
        return &common_vars;
    }
    for (std::list<MicEnvVar::CardEnvVars*>::const_iterator
         it = card_spec_list.begin();
         it != card_spec_list.end(); it++) {
        CardEnvVars *card_data = *it;
        if (card_data->card_number == number) {
            return card_data;
        }
    }
    return NULL;
}

// Searching for environment variable in "env_var" list with the same name

MicEnvVar::VarValue* MicEnvVar::CardEnvVars::find_var(
    char* env_var_name,
    int env_var_name_length
)
{
    for (std::list<MicEnvVar::VarValue*>::const_iterator it = env_vars.begin();
        it != env_vars.end(); it++) {
            VarValue *var_value = *it;
            if (var_value->length == env_var_name_length &&
                !strncmp(var_value->env_var, env_var_name,
                         env_var_name_length)) {
                return var_value;
            }
    }
    return NULL;
}

void MicEnvVar::analyze_env_var(char *env_var_string)
{
    char          *env_var_name;
    char          *env_var_def;
    int           card_number;
    int           env_var_name_length;
    MicEnvVarKind env_var_kind;

    env_var_kind = get_env_var_kind(env_var_string,
                                    &card_number,
                                    &env_var_name,
                                    &env_var_name_length,
                                    &env_var_def);
    switch (env_var_kind) {
        case c_mic_var:
        case c_mic_card_var:
            add_env_var(card_number,
                        env_var_name,
                        env_var_name_length,
                        env_var_def);
            break;
        case c_mic_card_env:
            mic_parse_env_var_list(card_number, env_var_def);
            break;
        case c_no_mic:
        default:
            break;
    }
}

void MicEnvVar::add_env_var(
    int card_number,
    char *env_var_name,
    int env_var_name_length,
    char *env_var_def
)
{
    VarValue *var;
    CardEnvVars *card;

    // The case corresponds to common env var definition of kind
    // <mic-prefix>_<var>
    if (card_number == any_card) {
        card = &common_vars;
    }
    else {
        card = get_card(card_number);
        if (!card) {
            // definition for new card occurred
            card = new CardEnvVars(card_number);
            card_spec_list.push_back(card);
        }

    }
    var = card->find_var(env_var_name, env_var_name_length);
    if (!var) {
        // put new env var definition in "env_var" list
        var = new VarValue(env_var_name, env_var_name_length, env_var_def);
        card->env_vars.push_back(var);
    }
}

// The routine analyses string pointed by "env_var_string" argument
// according to the following syntax:
//
// Specification of prefix for MIC environment variables
// MIC_ENV_PREFIX=<mic-prefix>
//
// Setting single MIC environment variable
// <mic-prefix>_<var>=<value>
// <mic-prefix>_<card-number>_<var>=<value>

// Setting multiple MIC environment variables
// <mic-prefix>_<card-number>_ENV=<env-vars>

MicEnvVarKind MicEnvVar::get_env_var_kind(
    char *env_var_string,
    int *card_number,
    char **env_var_name,
    int *env_var_name_length,
    char **env_var_def
)
{
    int len = strlen(prefix);
    char *c = env_var_string;
    int num = 0;
    bool card_is_set = false;

    if (strncmp(c, prefix, len) != 0 || c[len] != '_') {
            return c_no_mic;
    }
    c += len + 1;

    *card_number = any_card;
    if (isdigit(*c)) {
        while (isdigit (*c)) {
            num = (*c++ - '0') + (num * 10);
        }
    if (*c != '_') {
        return c_no_mic;
    }
    c++;
        *card_number = num;
        card_is_set = true;
    }
    if (!isalpha(*c)) {
        return c_no_mic;
    }
    *env_var_name = *env_var_def = c;
    if (strncmp(c, "ENV=", 4) == 0) {
        if (!card_is_set) {
            *env_var_name_length = 3;
            *env_var_name = *env_var_def = c;
            *env_var_def = strdup(*env_var_def);
            return  c_mic_var;
        }
        *env_var_def = c + strlen("ENV=");
        *env_var_def = strdup(*env_var_def);
        return c_mic_card_env;
    }
    if (isalpha(*c)) {
        *env_var_name_length = 0;
        while (isalnum(*c) || *c == '_') {
            c++;
            (*env_var_name_length)++;
        }
    }
    if (*c != '=') {
        return c_no_mic;
    }
    *env_var_def = strdup(*env_var_def);
    return card_is_set? c_mic_card_var : c_mic_var;
}

// analysing <env-vars> in form:
// <mic-prefix>_<card-number>_ENV=<env-vars>
// where:
//
// <env-vars>:
//                <env-var>
//                <env-vars> | <env-var>
//
// <env-var>:
//                variable=value
//                variable="value"
//                variable=

void MicEnvVar::mic_parse_env_var_list(
    int card_number, char *env_vars_def_list)
{
    char *c = env_vars_def_list;
    char *env_var_name;
    int  env_var_name_length;
    char *env_var_def;
    bool var_is_quoted;

    if (*c == '"') {
        c++;
    }
    while (*c != 0) {
        var_is_quoted = false;
        env_var_name = c;
        env_var_name_length = 0;
        if (isalpha(*c)) {
            while (isalnum(*c) || *c == '_') {
                c++;
                env_var_name_length++;
            }
        }
        else {
            LIBOFFLOAD_ERROR(c_mic_parse_env_var_list1);
            return;
        }
        if (*c != '=') {
            LIBOFFLOAD_ERROR(c_mic_parse_env_var_list2);
            return;
        }
        c++;

        if (*c == '"') {
            var_is_quoted = true;
            c++;
        }
        // Environment variable values that contain | will need to be escaped.
        while (*c != 0 && *c != '|' &&
               (!var_is_quoted || *c != '"'))
        {
            // skip escaped symbol
            if (*c == '\\') {
                c++;
            }
            c++;
        }
        if (var_is_quoted) {
            c++; // for "
            while (*c != 0 && *c != '|') {
                c++;
            }
        }

        int sz = c - env_var_name;
        env_var_def = (char*)malloc(sz);
        memcpy(env_var_def, env_var_name, sz);
        env_var_def[sz] = 0;

        if (*c == '|') {
            c++;
            while (*c != 0 && *c == ' ') {
                c++;
            }
        }
        add_env_var(card_number,
                    env_var_name,
                    env_var_name_length,
                    env_var_def);
    }
}

// Collect all definitions for the card with number "card_num".
// The returned result is vector of string pointers defining one
// environment variable. The vector is terminated by NULL pointer.
// In the beginning of the vector there are env vars defined as
// <mic-prefix>_<card-number>_<var>=<value>
// or
// <mic-prefix>_<card-number>_ENV=<env-vars>
// where <card-number> is equal to "card_num"
// They are followed by definitions valid for any card
// and absent in previous definitions.

char** MicEnvVar::create_environ_for_card(int card_num)
{
    VarValue *var_value;
    VarValue *var_value_find;
    CardEnvVars *card_data = get_card(card_num);
    CardEnvVars *card_data_common;
    std::list<char*> new_env;
    char **rez;

    if (!prefix) {
        return NULL;
    }
    // There is no personel env var definitions for the card with
    // number "card_num"
    if (!card_data) {
        return create_environ_for_card(any_card);
    }

    for (std::list<MicEnvVar::VarValue*>::const_iterator
         it = card_data->env_vars.begin();
         it != card_data->env_vars.end(); it++) {
        var_value = *it;
        new_env.push_back(var_value->env_var_value);
    }

    if (card_num != any_card) {
        card_data_common = get_card(any_card);
        for (std::list<MicEnvVar::VarValue*>::const_iterator
             it = card_data_common->env_vars.begin();
             it != card_data_common->env_vars.end(); it++) {
            var_value = *it;
            var_value_find = card_data->find_var(var_value->env_var,
                                                 var_value->length);
            if (!var_value_find) {
                new_env.push_back(var_value->env_var_value);
            }
        }
    }

    int new_env_size = new_env.size();
    rez = (char**) malloc((new_env_size + 1) * sizeof(char*));
    std::copy(new_env.begin(), new_env.end(), rez);
    rez[new_env_size] = 0;
    return rez;
}
