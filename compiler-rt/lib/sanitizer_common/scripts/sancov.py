#!/usr/bin/env python
# Merge or print the coverage data collected by asan's coverage.
# Input files are sequences of 4-byte integers.
# We need to merge these integers into a set and then
# either print them (as hex) or dump them into another file.
import array
import struct
import sys
import bisect
import os.path

prog_name = "";

def Usage():
  print >> sys.stderr, "Usage: \n" + \
      " " + prog_name + " merge file1 [file2 ...]  > output\n" \
      " " + prog_name + " print file1 [file2 ...]\n" \
      " " + prog_name + " unpack file1 [file2 ...]\n" \
      " " + prog_name + " rawunpack file1 [file2 ...]\n"
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

def UnpackOneRawFile(path, map_path):
  mem_map = []
  with open(map_path, mode="rt") as f_map:
    print >> sys.stderr, "%s: reading map %s" % (prog_name, map_path)
    bits = int(f_map.readline())
    for line in f_map:
      parts = line.rstrip().split()
      mem_map.append((int(parts[0], 16),
                  int(parts[1], 16),
                  int(parts[2], 16),
                  ' '.join(parts[3:])))
  mem_map.sort(key=lambda m : m[0])
  mem_map_keys = [m[0] for m in mem_map]

  with open(path, mode="rb") as f:
    print >> sys.stderr, "%s: unpacking %s" % (prog_name, path)

    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    if bits == 64:
      typecode = 'L'
    else:
      typecode = 'I'
    pcs = array.array(typecode, f.read(size))
    mem_map_pcs = [[] for i in range(0, len(mem_map))]

    for pc in pcs:
      if pc == 0: continue
      map_idx = bisect.bisect(mem_map_keys, pc) - 1
      (start, end, base, module_path) = mem_map[map_idx]
      assert pc >= start
      if pc >= end:
        print >> sys.stderr, "warning: %s: pc %x outside of any known mapping" % (prog_name, pc)
        continue
      mem_map_pcs[map_idx].append(pc - base)

    for ((start, end, base, module_path), pc_list) in zip(mem_map, mem_map_pcs):
      if len(pc_list) == 0: continue
      assert path.endswith('.sancov.raw')
      dst_path = module_path + '.' + os.path.basename(path)[:-4]
      print "writing %d PCs to %s" % (len(pc_list), dst_path)
      arr = array.array('I')
      arr.fromlist(sorted(pc_list))
      with open(dst_path, 'ab') as f2:
        arr.tofile(f2)

def RawUnpack(files):
  for f in files:
    if not f.endswith('.sancov.raw'):
      raise Exception('Unexpected raw file name %s' % f)
    f_map = f[:-3] + 'map'
    UnpackOneRawFile(f, f_map)

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
  elif sys.argv[1] == "rawunpack":
    RawUnpack(sys.argv[2:])
  else:
    Usage()
