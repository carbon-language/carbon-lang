//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef OFFLOAD_ENV_H_INCLUDED
#define OFFLOAD_ENV_H_INCLUDED

#include <list>

// data structure and routines to parse MIC user environment and pass to MIC

enum MicEnvVarKind
{
    c_no_mic,         // not MIC env var
    c_mic_var,        // for <mic-prefix>_<var>
    c_mic_card_var,   // for <mic-prefix>_<card-number>_<var>
    c_mic_card_env    // for <mic-prefix>_<card-number>_ENV
};

struct MicEnvVar {
public:
    MicEnvVar() : prefix(0) {}
    ~MicEnvVar();

    void analyze_env_var(char *env_var_string);
    char** create_environ_for_card(int card_num);
    MicEnvVarKind get_env_var_kind(
        char *env_var_string,
        int *card_number,
        char **env_var_name,
        int *env_var_name_length,
        char **env_var_def
    );
    void add_env_var(
        int card_number,
        char *env_var_name,
        int env_var_name_length,
        char *env_var_def
    );

    void set_prefix(const char *pref) {
        prefix = (pref && *pref != '\0') ? pref : 0;
    }

    struct VarValue {
    public:
        char* env_var;
        int   length;
        char* env_var_value;

        VarValue(char* var, int ln, char* value)
        {
            env_var = var;
            length = ln;
            env_var_value = value;
        }
        ~VarValue();
    };

    struct CardEnvVars {
    public:

        int card_number;
        std::list<struct VarValue*> env_vars;

        CardEnvVars() { card_number = any_card; }
        CardEnvVars(int num) { card_number = num; }
        ~CardEnvVars();

        void add_new_env_var(int number, char *env_var, int length,
                             char *env_var_value);
        VarValue* find_var(char* env_var_name, int env_var_name_length);
    };
    static const int any_card;

private:
    void         mic_parse_env_var_list(int card_number, char *env_var_def);
    CardEnvVars* get_card(int number);

    const char *prefix;
    std::list<struct CardEnvVars *> card_spec_list;
    CardEnvVars common_vars;
};

#endif // OFFLOAD_ENV_H_INCLUDED
