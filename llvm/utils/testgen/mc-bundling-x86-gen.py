#!/usr/bin/python

# Auto-generates an exhaustive and repetitive test for correct bundle-locked
# alignment on x86.
# For every possible offset in an aligned bundle, a bundle-locked group of every
# size in the inclusive range [1, bundle_size] is inserted. An appropriate CHECK
# is added to verify that NOP padding occurred (or did not occur) as expected.

# This script runs with Python 2.6+ (including 3.x)

from __future__ import print_function

BUNDLE_SIZE_POW2 = 4
BUNDLE_SIZE = 2 ** BUNDLE_SIZE_POW2

PREAMBLE = '''
# RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - \\
# RUN:   | llvm-objdump -triple i386 -disassemble -no-show-raw-insn - | FileCheck %s

# !!! This test is auto-generated from utils/testgen/mc-bundling-x86-gen.py !!!
#     It tests that bundle-aligned grouping works correctly in MC. Read the
#     source of the script for more details.

  .text
  .bundle_align_mode {0}
'''.format(BUNDLE_SIZE_POW2).lstrip()

ALIGNTO = '  .align {0}, 0x90'
NOPFILL = '  .fill {0}, 1, 0x90'

def print_bundle_locked_sequence(len):
  print('  .bundle_lock')
  print('  .rept {0}'.format(len))
  print('  inc %eax')
  print('  .endr')
  print('  .bundle_unlock')

def generate():
  print(PREAMBLE)
  
  ntest = 0
  for instlen in range(1, BUNDLE_SIZE + 1):
    for offset in range(0, BUNDLE_SIZE):
      # Spread out all the instructions to not worry about cross-bundle
      # interference.
      print(ALIGNTO.format(2 * BUNDLE_SIZE))
      print('INSTRLEN_{0}_OFFSET_{1}:'.format(instlen, offset))
      if offset > 0:
        print(NOPFILL.format(offset))
      print_bundle_locked_sequence(instlen)

      # Now generate an appropriate CHECK line
      base_offset = ntest * 2 * BUNDLE_SIZE
      inst_orig_offset = base_offset + offset  # had it not been padded...
      
      if offset + instlen > BUNDLE_SIZE:
        # Padding needed
        print('# CHECK: {0:x}: nop'.format(inst_orig_offset))
        aligned_offset = (inst_orig_offset + instlen) & ~(BUNDLE_SIZE - 1)
        print('# CHECK: {0:x}: incl'.format(aligned_offset))
      else:
        # No padding needed
        print('# CHECK: {0:x}: incl'.format(inst_orig_offset))

      print()
      ntest += 1

if __name__ == '__main__':
  generate()

