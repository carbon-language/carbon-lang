#!/usr/bin/env python
#===- lib/sanitizer_common/scripts/gen_dynamic_list.py ---------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# Generates the list of functions that should be exported from sanitizer
# runtimes. The output format is recognized by --dynamic-list linker option.
# Usage:
#   gen_dynamic_list.py libclang_rt.*san*.a [ files ... ]
#
#===------------------------------------------------------------------------===#
import os
import re
import subprocess
import sys

new_delete = set(['_ZdaPv', '_ZdaPvRKSt9nothrow_t',
                  '_ZdlPv', '_ZdlPvRKSt9nothrow_t',
                  '_Znam', '_ZnamRKSt9nothrow_t',
                  '_Znwm', '_ZnwmRKSt9nothrow_t'])

versioned_functions = set(['memcpy', 'pthread_attr_getaffinity_np',
                           'pthread_cond_broadcast',
                           'pthread_cond_destroy', 'pthread_cond_init',
                           'pthread_cond_signal', 'pthread_cond_timedwait',
                           'pthread_cond_wait', 'realpath',
                           'sched_getaffinity'])

def get_global_functions(library):
  functions = []
  nm_proc = subprocess.Popen(['nm', library], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  nm_out = nm_proc.communicate()[0].decode().split('\n')
  if nm_proc.returncode != 0:
    raise subprocess.CalledProcessError(nm_proc.returncode, 'nm')
  for line in nm_out:
    cols = line.split(' ')
    if (len(cols) == 3 and cols[1] in ('T', 'W')) :
      functions.append(cols[2])
  return functions

def main(argv):
  result = []

  library = argv[1]
  all_functions = get_global_functions(library)
  function_set = set(all_functions)
  for func in all_functions:
    # Export new/delete operators.
    if func in new_delete:
      result.append(func)
      continue
    # Export interceptors.
    match = re.match('__interceptor_(.*)', func)
    if match:
      result.append(func)
      # We have to avoid exporting the interceptors for versioned library
      # functions due to gold internal error.
      orig_name = match.group(1)
      if orig_name in function_set and orig_name not in versioned_functions:
        result.append(orig_name)
      continue
    # Export sanitizer interface functions.
    if re.match('__sanitizer_(.*)', func):
      result.append(func)

  # Additional exported functions from files.
  for fname in argv[2:]:
    f = open(fname, 'r')
    for line in f:
      result.append(line.rstrip())
  # Print the resulting list in the format recognized by ld.
  print('{')
  result.sort()
  for f in result:
    print('  ' + f + ';')
  print('};')

if __name__ == '__main__':
  main(sys.argv)
