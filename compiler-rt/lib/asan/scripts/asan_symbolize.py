#!/usr/bin/env python
#===- lib/asan/scripts/asan_symbolize.py -----------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
import bisect
import os
import re
import sys
import subprocess

symbolizers = {}
filetypes = {}
vmaddrs = {}
DEBUG = False


def fix_filename(file_name):
  for path_to_cut in sys.argv[1:]:
    file_name = re.sub(".*" + path_to_cut, "", file_name)
  file_name = re.sub(".*asan_[a-z_]*.cc:[0-9]*", "_asan_rtl_", file_name)
  file_name = re.sub(".*crtstuff.c:0", "???:0", file_name)
  return file_name


class Symbolizer(object):
  def __init__(self):
    pass


class LinuxSymbolizer(Symbolizer):
  def __init__(self, binary):
    super(LinuxSymbolizer, self).__init__()
    self.binary = binary
    self.pipe = self.open_addr2line()
  def open_addr2line(self):
    cmd = ["addr2line", "-f", "-e", self.binary]
    if DEBUG:
      print ' '.join(cmd)
    return subprocess.Popen(cmd,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  def symbolize(self, prefix, addr, offset):
    try:
      print >> self.pipe.stdin, offset
      function_name = self.pipe.stdout.readline().rstrip()
      file_name     = self.pipe.stdout.readline().rstrip()
    except Exception:
      function_name = ""
      file_name = ""
    file_name = fix_filename(file_name)
    return "%s%s in %s %s" % (prefix, addr, function_name, file_name)


class DarwinSymbolizer(Symbolizer):
  def __init__(self, addr, binary):
    super(DarwinSymbolizer, self).__init__()
    self.binary = binary
    # Guess which arch we're running. 10 = len("0x") + 8 hex digits.
    if len(addr) > 10:
      self.arch = "x86_64"
    else:
      self.arch = "i386"
    self.vmaddr = None
    self.pipe = None
  def get_binary_vmaddr(self):
    """
    Get the slide value to be added to the address.
    We're ooking for the following piece in otool -l output:
      Load command 0
      cmd LC_SEGMENT
      cmdsize 736
      segname __TEXT
      vmaddr 0x00000000
    """
    if self.vmaddr:
      return self.vmaddr
    cmdline = ["otool", "-l", self.binary]
    pipe = subprocess.Popen(cmdline,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    is_text = False
    vmaddr = 0
    for line in pipe.stdout.readlines():
      line = line.strip()
      if line.startswith('segname'):
        is_text = (line == 'segname __TEXT')
        continue
      if line.startswith('vmaddr') and is_text:
        sv = line.split(' ')
        vmaddr = int(sv[-1], 16)
        break
    self.vmaddr = vmaddr
    return self.vmaddr
  def write_addr_to_pipe(self, offset):
    slide = self.get_binary_vmaddr()
    print >> self.pipe.stdin, "0x%x" % (int(offset, 16) + slide)
  def open_atos(self):
    if DEBUG:
      print "atos -o %s -arch %s" % (self.binary, self.arch)
    cmdline = ["atos", "-o", self.binary, "-arch", self.arch]
    self.pipe = subprocess.Popen(cmdline,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
  def symbolize(self, prefix, addr, offset):
    self.open_atos()
    self.write_addr_to_pipe(offset)
    self.pipe.stdin.close()
    atos_line = self.pipe.stdout.readline().rstrip()
    # A well-formed atos response looks like this:
    #   foo(type1, type2) (in object.name) (filename.cc:80)
    match = re.match('^(.*) \(in (.*)\) \((.*:\d*)\)$', atos_line)
    if DEBUG:
      print "atos_line: ", atos_line
    if match:
      function_name = match.group(1)
      function_name = re.sub("\(.*?\)", "", function_name)
      file_name = fix_filename(match.group(3))
      return "%s%s in %s %s" % (prefix, addr, function_name, file_name)
    else:
      return "%s%s in %s" % (prefix, addr, atos_line)


# Chain two symbolizers so that the second one is called if the first fails.
class ChainSymbolizer(Symbolizer):
  def __init__(self, symbolizer1, symbolizer2):
    super(ChainSymbolizer, self).__init__()
    self.symbolizer1 = symbolizer1
    self.symbolizer2 = symbolizer2
  def symbolize(self, prefix, addr, offset):
    result = self.symbolizer1.symbolize(prefix, addr, offset)
    if result is None:
      result = self.symbolizer2.symbolize(prefix, addr, offset)
    return result


def BreakpadSymbolizerFactory(addr, binary):
  suffix = os.getenv("BREAKPAD_SUFFIX")
  if suffix:
    filename = binary + suffix
    if os.access(filename, os.F_OK):
      return BreakpadSymbolizer(filename)
  return None


def SystemSymbolizerFactory(system, addr, binary):
  if system == 'Darwin':
    return DarwinSymbolizer(addr, binary)
  elif system == 'Linux':
    return LinuxSymbolizer(binary)


class BreakpadSymbolizer(Symbolizer):
  def __init__(self, filename):
    super(BreakpadSymbolizer, self).__init__()
    self.filename = filename
    lines = file(filename).readlines()
    self.files = []
    self.symbols = {}
    self.address_list = []
    self.addresses = {}
    # MODULE mac x86_64 A7001116478B33F18FF9BEDE9F615F190 t
    fragments = lines[0].rstrip().split()
    self.arch = fragments[2]
    self.debug_id = fragments[3]
    self.binary = ' '.join(fragments[4:])
    self.parse_lines(lines[1:])
  def parse_lines(self, lines):
    cur_function_addr = ''
    for line in lines:
      fragments = line.split()
      if fragments[0] == 'FILE':
        assert int(fragments[1]) == len(self.files)
        self.files.append(' '.join(fragments[2:]))
      elif fragments[0] == 'PUBLIC':
        self.symbols[int(fragments[1], 16)] = ' '.join(fragments[3:])
      elif fragments[0] in ['CFI', 'STACK']:
        pass
      elif fragments[0] == 'FUNC':
        cur_function_addr = int(fragments[1], 16)
        if not cur_function_addr in self.symbols.keys():
          self.symbols[cur_function_addr] = ' '.join(fragments[4:])
      else:
        # Line starting with an address.
        addr = int(fragments[0], 16)
        self.address_list.append(addr)
        # Tuple of symbol address, size, line, file number.
        self.addresses[addr] = (cur_function_addr,
                                int(fragments[1], 16),
                                int(fragments[2]),
                                int(fragments[3]))
    self.address_list.sort()
  def get_sym_file_line(self, addr):
    key = None
    if addr in self.addresses.keys():
      key = addr
    else:
      index = bisect.bisect_left(self.address_list, addr)
      if index == 0:
        return None
      else:
        key = self.address_list[index - 1]
    sym_id, size, line_no, file_no = self.addresses[key]
    symbol = self.symbols[sym_id]
    filename = self.files[file_no]
    if addr < key + size:
      return symbol, filename, line_no
    else:
      return None
  def symbolize(self, prefix, addr, offset):
    res = self.get_sym_file_line(int(offset, 16))
    if res:
      function_name, file_name, line_no = res
      result = "%s%s in %s %s:%d" % (
          prefix, addr, function_name, file_name, line_no)
      print result
      return result
    else:
      return None


def symbolize_line(system, line):
  #0 0x7f6e35cf2e45  (/blah/foo.so+0x11fe45)
  match = re.match('^( *#([0-9]+) *)(0x[0-9a-f]+) *\((.*)\+(0x[0-9a-f]+)\)',
                   line)
  if match:
    if DEBUG:
      print line
    prefix = match.group(1)
    # frameno = match.group(2)
    addr = match.group(3)
    binary = match.group(4)
    offset = match.group(5)
    if not symbolizers.has_key(binary):
      p = BreakpadSymbolizerFactory(addr, binary)
      if p:
        symbolizers[binary] = p
      else:
        symbolizers[binary] = SystemSymbolizerFactory(system, addr, binary)
    result = symbolizers[binary].symbolize(prefix, addr, offset)
    if result is None:
      symbolizers[binary] = ChainSymbolizer(symbolizers[binary],
          SystemSymbolizerFactory(system, addr, binary))
    return symbolizers[binary].symbolize(prefix, addr, offset)
  else:
    return line


def main():
  system = os.uname()[0]
  if system in ['Linux', 'Darwin']:
    for line in sys.stdin:
      line = symbolize_line(system, line)
      print line.rstrip()
  else:
    print 'Unknown system: ', system


if __name__ == '__main__':
  main()
