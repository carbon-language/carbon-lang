/*
 * kmp_settings.h -- Initialize environment variables
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_SETTINGS_H
#define KMP_SETTINGS_H

void __kmp_reset_global_vars(void);
void __kmp_env_initialize(char const *);
void __kmp_env_print();
void __kmp_env_print_2();
void __kmp_display_env_impl(int display_env, int display_env_verbose);
#if OMPD_SUPPORT
void __kmp_env_dump();
#endif

int __kmp_initial_threads_capacity(int req_nproc);
void __kmp_init_dflt_team_nth();
int __kmp_convert_to_milliseconds(char const *);
int __kmp_default_tp_capacity(int, int, int);

#if KMP_MIC
#define KMP_STR_BUF_PRINT_NAME                                                 \
  __kmp_str_buf_print(buffer, "  %s %s", KMP_I18N_STR(Device), name)
#define KMP_STR_BUF_PRINT_NAME_EX(x)                                           \
  __kmp_str_buf_print(buffer, "  %s %s='", KMP_I18N_STR(Device), x)
#define KMP_STR_BUF_PRINT_BOOL_EX(n, v, t, f)                                  \
  __kmp_str_buf_print(buffer, "  %s %s='%s'\n", KMP_I18N_STR(Device), n,       \
                      (v) ? t : f)
#define KMP_STR_BUF_PRINT_BOOL                                                 \
  KMP_STR_BUF_PRINT_BOOL_EX(name, value, "TRUE", "FALSE")
#define KMP_STR_BUF_PRINT_INT                                                  \
  __kmp_str_buf_print(buffer, "  %s %s='%d'\n", KMP_I18N_STR(Device), name,    \
                      value)
#define KMP_STR_BUF_PRINT_UINT64                                               \
  __kmp_str_buf_print(buffer, "  %s %s='%" KMP_UINT64_SPEC "'\n",              \
                      KMP_I18N_STR(Device), name, value);
#define KMP_STR_BUF_PRINT_STR                                                  \
  __kmp_str_buf_print(buffer, "  %s %s='%s'\n", KMP_I18N_STR(Device), name,    \
                      value)
#else
#define KMP_STR_BUF_PRINT_NAME                                                 \
  __kmp_str_buf_print(buffer, "  %s %s", KMP_I18N_STR(Host), name)
#define KMP_STR_BUF_PRINT_NAME_EX(x)                                           \
  __kmp_str_buf_print(buffer, "  %s %s='", KMP_I18N_STR(Host), x)
#define KMP_STR_BUF_PRINT_BOOL_EX(n, v, t, f)                                  \
  __kmp_str_buf_print(buffer, "  %s %s='%s'\n", KMP_I18N_STR(Host), n,         \
                      (v) ? t : f)
#define KMP_STR_BUF_PRINT_BOOL                                                 \
  KMP_STR_BUF_PRINT_BOOL_EX(name, value, "TRUE", "FALSE")
#define KMP_STR_BUF_PRINT_INT                                                  \
  __kmp_str_buf_print(buffer, "  %s %s='%d'\n", KMP_I18N_STR(Host), name, value)
#define KMP_STR_BUF_PRINT_UINT64                                               \
  __kmp_str_buf_print(buffer, "  %s %s='%" KMP_UINT64_SPEC "'\n",              \
                      KMP_I18N_STR(Host), name, value);
#define KMP_STR_BUF_PRINT_STR                                                  \
  __kmp_str_buf_print(buffer, "  %s %s='%s'\n", KMP_I18N_STR(Host), name, value)
#endif

#endif // KMP_SETTINGS_H

// end of file //
