#!/usr/bin/env python
# Merge or print the coverage data collected by asan's coverage.
# Input files are sequences of 4-byte integers.
# We need to merge these integers into a set and then
# either print them (as hex) or dump them into another file.
import array
import struct
import sys

prog_name = "";

def Usage():
  print >> sys.stderr, "Usage: \n" + \
      " " + prog_name + " merge file1 [file2 ...]  > output\n" \
      " " + prog_name + " print file1 [file2 ...]\n" \
      " " + prog_name + " unpack file1 [file2 ...]\n"
  exit(1)

def ReadOneFile(path):
  with open(path, mode="rb") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    s = set(array.array('I', f.read(size)))
  print >>sys.stderr, "%s: read %d PCs from %s" % (prog_name, size / 4, path)
  return s

def Merge(files):
  s = set()
  for f in files:
    s = s.union(ReadOneFile(f))
  print >> sys.stderr, "%s: %d files merged; %d PCs total" % \
    (prog_name, len(files), len(s))
  return sorted(s)

def PrintFiles(files):
  s = Merge(files)
  for i in s:
    print "0x%x" % i

def MergeAndPrint(files):
  if sys.stdout.isatty():
    Usage()
  s = Merge(files)
  a = array.array('I', s)
  a.tofile(sys.stdout)


def UnpackOneFile(path):
  with open(path, mode="rb") as f:
    print >> sys.stderr, "%s: unpacking %s" % (prog_name, path)
    while True:
      header = f.read(12)
      if not header: return
      if len(header) < 12:
        break
      pid, module_length, blob_size = struct.unpack('iII', header)
      module = f.read(module_length)
      blob = f.read(blob_size)
      assert(len(module) == module_length)
      assert(len(blob) == blob_size)
      extracted_file = "%s.%d.sancov" % (module, pid)
      print >> sys.stderr, "%s: extracting %s" % \
        (prog_name, extracted_file)
      # The packed file may contain multiple blobs for the same pid/module
      # pair. Append to the end of the file instead of overwriting.
      with open(extracted_file, 'ab') as f2:
        f2.write(blob)
    # fail
    raise Exception('Error reading file %s' % path)


def Unpack(files):
  for f in files:
    UnpackOneFile(f)


if __name__ == '__main__':
  prog_name = sys.argv[0]
  if len(sys.argv) <= 2:
    Usage();
  if sys.argv[1] == "print":
    PrintFiles(sys.argv[2:])
  elif sys.argv[1] == "merge":
    MergeAndPrint(sys.argv[2:])
  elif sys.argv[1] == "unpack":
    Unpack(sys.argv[2:])
  else:
    Usage()
