#!/usr/bin/python
#
#===- check_clang_tidy.py - ClangTidy Test Helper ------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

r"""
ClangTidy Test Helper
=====================

This script runs clang-tidy in fix mode and verify fixes, messages or both.

Usage:
  check_clang_tidy.py <source-file> <check-name> <temp-file> \
    [optional clang-tidy arguments]

Example:
  // RUN: %check_clang_tidy %s llvm-include-order %t -- -isystem $(dirname %s)/Inputs/Headers
"""

import re
import subprocess
import sys


def write_file(file_name, text):
  with open(file_name, 'w') as f:
    f.write(text)
    f.truncate()

def main():
  if len(sys.argv) < 4:
    sys.exit('Not enough arguments.')

  input_file_name = sys.argv[1]
  extension = '.cpp'
  if (input_file_name.endswith('.c')):
    extension = '.c'

  check_name = sys.argv[2]
  temp_file_name = sys.argv[3] + extension

  clang_tidy_extra_args = sys.argv[4:]
  if len(clang_tidy_extra_args) == 0:
    clang_tidy_extra_args = ['--', '--std=c++11'] if extension == '.cpp' \
                       else ['--']

  with open(input_file_name, 'r') as input_file:
    input_text = input_file.read()

  has_check_fixes = input_text.find('CHECK-FIXES') >= 0
  has_check_messages = input_text.find('CHECK-MESSAGES') >= 0

  if not has_check_fixes and not has_check_messages:
    sys.exit('Neither CHECK-FIXES nor CHECK-MESSAGES found in the input')

  # Remove the contents of the CHECK lines to avoid CHECKs matching on
  # themselves.  We need to keep the comments to preserve line numbers while
  # avoiding empty lines which could potentially trigger formatting-related
  # checks.
  cleaned_test = re.sub('// *CHECK-[A-Z-]*:[^\r\n]*', '//', input_text)

  write_file(temp_file_name, cleaned_test)

  original_file_name = temp_file_name + ".orig"
  write_file(original_file_name, cleaned_test)

  args = ['clang-tidy', temp_file_name, '-fix', '--checks=-*,' + check_name] + \
        clang_tidy_extra_args
  print('Running ' + repr(args) + '...')
  try:
    clang_tidy_output = \
        subprocess.check_output(args, stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    print('clang-tidy failed:\n' + e.output.decode())
    raise

  print('------------------------ clang-tidy output -----------------------\n' +
        clang_tidy_output +
        '\n------------------------------------------------------------------')

  try:
    diff_output = subprocess.check_output(
        ['diff', '-u', original_file_name, temp_file_name],
        stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    diff_output = e.output

  print('------------------------------ Fixes -----------------------------\n' +
        diff_output.decode() +
        '\n------------------------------------------------------------------')

  if has_check_fixes:
    try:
      subprocess.check_output(
          ['FileCheck', '-input-file=' + temp_file_name, input_file_name,
           '-check-prefix=CHECK-FIXES', '-strict-whitespace'],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('FileCheck failed:\n' + e.output.decode())
      raise

  if has_check_messages:
    messages_file = temp_file_name + '.msg'
    write_file(messages_file, clang_tidy_output)
    try:
      subprocess.check_output(
          ['FileCheck', '-input-file=' + messages_file, input_file_name,
           '-check-prefix=CHECK-MESSAGES',
           '-implicit-check-not={{warning|error}}:'],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('FileCheck failed:\n' + e.output.decode())
      raise

if __name__ == '__main__':
  main()
