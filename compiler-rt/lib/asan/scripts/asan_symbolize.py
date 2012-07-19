#!/usr/bin/env python
#===- lib/asan/scripts/asan_symbolize.py -----------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
import os
import re
import sys
import string
import subprocess

pipes = {}
filetypes = {}
DEBUG=False

def fix_filename(file_name):
  for path_to_cut in sys.argv[1:]:
    file_name = re.sub(".*" + path_to_cut, "", file_name)
  file_name = re.sub(".*asan_[a-z_]*.cc:[0-9]*", "_asan_rtl_", file_name)
  file_name = re.sub(".*crtstuff.c:0", "???:0", file_name)
  return file_name


# TODO(glider): need some refactoring here
def symbolize_addr2line(line):
  #0 0x7f6e35cf2e45  (/blah/foo.so+0x11fe45)
  match = re.match('^( *#([0-9]+) *0x[0-9a-f]+) *\((.*)\+(0x[0-9a-f]+)\)', line)
  if match:
    frameno = match.group(2)
    binary = match.group(3)
    addr = match.group(4)
    if not pipes.has_key(binary):
      pipes[binary] = subprocess.Popen(["addr2line", "-f", "-e", binary],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p = pipes[binary]
    try:
      print >>p.stdin, addr
      function_name = p.stdout.readline().rstrip()
      file_name     = p.stdout.readline().rstrip()
    except:
      function_name = ""
      file_name = ""
    file_name = fix_filename(file_name)

    print match.group(1), "in", function_name, file_name
  else:
    print line.rstrip()


def get_macho_filetype(binary):
  if not filetypes.has_key(binary):
    otool_pipe = subprocess.Popen(["otool", "-Vh",  binary],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    otool_line = "".join(otool_pipe.stdout.readlines())
    for t in ["DYLIB", "EXECUTE"]:
      if t in otool_line:
        filetypes[binary] = t
    otool_pipe.stdin.close()
  return filetypes[binary]


def symbolize_atos(line):
  #0 0x7f6e35cf2e45  (/blah/foo.so+0x11fe45)
  match = re.match('^( *#([0-9]+) *)(0x[0-9a-f]+) *\((.*)\+(0x[0-9a-f]+)\)', line)
  if match:
    #print line
    prefix = match.group(1)
    frameno = match.group(2)
    orig_addr = match.group(3)
    binary = match.group(4)
    offset = match.group(5)
    addr = orig_addr
    load_addr = hex(int(orig_addr, 16) - int(offset, 16))
    filetype = get_macho_filetype(binary)

    if not pipes.has_key(binary):
      # Guess which arch we're running. 10 = len("0x") + 8 hex digits.
      if len(addr) > 10:
        arch = "x86_64"
      else:
        arch = "i386"

    if filetype == "DYLIB":
      load_addr = "0x0"
    if DEBUG:
      print "atos -o %s -arch %s -l %s" % (binary, arch, load_addr)
    cmd = ["atos", "-o", binary, "-arch", arch, "-l", load_addr]
    pipes[binary] = subprocess.Popen(cmd,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
    p = pipes[binary]
    if filetype == "DYLIB":
      print >>p.stdin, "%s" % offset
    else:
      print >>p.stdin, "%s" % addr
    # TODO(glider): it's more efficient to make a batch atos run for each binary.
    p.stdin.close()
    atos_line = p.stdout.readline().rstrip()
    # A well-formed atos response looks like this:
    #   foo(type1, type2) (in object.name) (filename.cc:80)
    match = re.match('^(.*) \(in (.*)\) \((.*:\d*)\)$', atos_line)
    #print "atos_line: ", atos_line
    if match:
      function_name = match.group(1)
      function_name = re.sub("\(.*?\)", "", function_name)
      file_name = fix_filename(match.group(3))
      print "%s%s in %s %s" % (prefix, addr, function_name, file_name)
    else:
      print "%s%s in %s" % (prefix, addr, atos_line)
    del pipes[binary]
  else:
    print line.rstrip()

system = os.uname()[0]
if system in ['Linux', 'Darwin']:
  for line in sys.stdin:
    if system == 'Linux':
      symbolize_addr2line(line)
    elif system == 'Darwin':
      symbolize_atos(line)
else:
  print 'Unknown system: ', system
