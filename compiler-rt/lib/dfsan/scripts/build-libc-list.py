#!/usr/bin/env python
#===- lib/dfsan/scripts/build-libc-list.py ---------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
# The purpose of this script is to identify every function symbol in a set of
# libraries (in this case, libc and libgcc) so that they can be marked as
# uninstrumented, thus allowing the instrumentation pass to treat calls to those
# functions correctly.

import os
import subprocess
import sys
from optparse import OptionParser

def defined_function_list(object):
  functions = []
  readelf_proc = subprocess.Popen(['readelf', '-s', '-W', object],
                                  stdout=subprocess.PIPE)
  readelf = readelf_proc.communicate()[0].split('\n')
  if readelf_proc.returncode != 0:
    raise subprocess.CalledProcessError(readelf_proc.returncode, 'readelf')
  for line in readelf:
    if (line[31:35] == 'FUNC' or line[31:36] == 'IFUNC') and \
       line[55:58] != 'UND':
      function_name = line[59:].split('@')[0]
      functions.append(function_name)
  return functions

p = OptionParser()
p.add_option('--lib', metavar='PATH',
             help='path to lib directory to use',
             default='/lib/x86_64-linux-gnu')
p.add_option('--usrlib', metavar='PATH',
             help='path to usr/lib directory to use',
             default='/usr/lib/x86_64-linux-gnu')
p.add_option('--gcclib', metavar='PATH',
             help='path to gcc lib directory to use',
             default='/usr/lib/gcc/x86_64-linux-gnu/4.6')
p.add_option('--with-libstdcxx', action='store_true',
             dest='with_libstdcxx',
             help='include libstdc++ in the list (inadvisable)')
(options, args) = p.parse_args()

libs = [os.path.join(options.lib, name) for name in
        ['ld-linux-x86-64.so.2',
         'libanl.so.1',
         'libBrokenLocale.so.1',
         'libcidn.so.1',
         'libcrypt.so.1',
         'libc.so.6',
         'libdl.so.2',
         'libm.so.6',
         'libnsl.so.1',
         'libpthread.so.0',
         'libresolv.so.2',
         'librt.so.1',
         'libthread_db.so.1',
         'libutil.so.1']]
libs += [os.path.join(options.usrlib, name) for name in
         ['libc_nonshared.a',
          'libpthread_nonshared.a']]
gcclibs = ['libgcc.a',
           'libgcc_s.so']
if options.with_libstdcxx:
  gcclibs += ['libstdc++.so']
libs += [os.path.join(options.gcclib, name) for name in gcclibs]

functions = []
for l in libs:
  if os.path.exists(l):
    functions += defined_function_list(l)
  else:
    print >> sys.stderr, 'warning: library %s not found' % l

functions = list(set(functions))
functions.sort()

for f in functions:
  print 'fun:%s=uninstrumented' % f
